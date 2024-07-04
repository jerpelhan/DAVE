import math

import numpy as np
import skimage
import torch
from PIL import Image
from numpy import linalg as LA
from scipy.sparse import csgraph
from sklearn.cluster import SpectralClustering
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou
from torchvision.ops import roi_align

from utils.helpers import mask_density, extend_bboxes
from .backbone import Backbone
from .box_prediction import FCOSHead, BoxList, boxlist_nms
from .feat_comparison import Feature_Transform
from .positional_encoding import PositionalEncodingsFixed
from .regression_head import DensityMapRegressor
from .transformer import TransformerEncoder, TransformerDecoder


class COTR(nn.Module):

    def __init__(
            self,
            image_size: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            num_objects: int,
            emb_dim: int,
            num_heads: int,
            kernel_dim: int,
            backbone_name: str,
            swav_backbone: bool,
            train_backbone: bool,
            reduction: int,
            dropout: float,
            layer_norm_eps: float,
            mlp_factor: int,
            norm_first: bool,
            activation: nn.Module,
            norm: bool,
            use_query_pos_emb: bool,
            zero_shot: bool,
            prompt_shot: bool,
            use_objectness: bool,
            use_appearance: bool,
            d_s: float,
            m_s: float,
            i_thr: float,
            d_t: float,
            s_t: float,
            egv: float,
            norm_s: bool,
            det_train: bool
    ):

        super(COTR, self).__init__()

        self.emb_dim = emb_dim
        self.plot = True
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot
        self.prompt_shot = prompt_shot
        self.use_query_pos_emb = use_query_pos_emb
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.use_objectness = use_objectness
        self.use_appearance = use_appearance
        self.d_s = d_s
        self.m_s = m_s
        self.i_thr = i_thr
        self.d_t = d_t
        self.s_t = s_t
        self.egv = egv
        self.det_train = det_train
        self.norm_s = norm_s
        self.upscale = nn.Upsample(scale_factor=(8, 8))
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
        self.backbone = Backbone(
            backbone_name, pretrained=True, dilation=False, reduction=reduction,
            swav=swav_backbone, requires_grad=train_backbone
        )
        self.cos_loss = nn.CosineEmbeddingLoss()
        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, emb_dim, kernel_size=1
        )
        if self.prompt_shot:
            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        if num_encoder_layers > 0:
            self.encoder = TransformerEncoder(
                num_encoder_layers, emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, norm
            )

        if num_decoder_layers > 0:
            self.decoder = TransformerDecoder(
                num_layers=num_decoder_layers, emb_dim=emb_dim, num_heads=num_heads,
                dropout=0, layer_norm_eps=layer_norm_eps,
                mlp_factor=mlp_factor, norm_first=norm_first,
                activation=activation, norm=norm,
                attn1=not self.zero_shot and self.use_appearance
            )

        self.regression_head = DensityMapRegressor(emb_dim, reduction)
        self.aux_heads = nn.ModuleList([
            DensityMapRegressor(emb_dim, reduction) for _ in range(num_decoder_layers - 1)
        ])
        self.box_predictor = FCOSHead(3584)
        self.idx = 0

        self.pos_emb = PositionalEncodingsFixed(emb_dim)
        if not self.det_train:
            self.feat_comp = Feature_Transform()
        if self.use_objectness:
            if not self.zero_shot:
                self.objectness = nn.Sequential(
                    nn.Linear(2, 64),
                    nn.ReLU(),
                    nn.Linear(64, emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, self.kernel_dim ** 2 * emb_dim)
                )
            else:
                self.objectness = nn.Parameter(
                    torch.empty((self.num_objects, self.kernel_dim ** 2, emb_dim))
                )
                nn.init.normal_(self.objectness)

    def clip_check_clusters(self, img, bboxes, category, img_name=None):
        bboxes = extend_bboxes(bboxes)

        C, H, W = img[0].shape
        mask_tensor = np.zeros((H, W, C))
        for box in bboxes.long():
            x1, y1, x2, y2 = box
            mask_tensor[y1:y2, x1:x2, :] = 1

        img_ = img[0].cpu().permute(1, 2, 0).numpy()
        non_zero_rows = np.any(img_ != 0, axis=(1, 2))
        non_zero_cols = np.any(img_ != 0, axis=(0, 2))
        top, bottom = np.where(non_zero_rows)[0][[0, -1]]
        left, right = np.where(non_zero_cols)[0][[0, -1]]

        img_ = img_ - np.min(img_)
        img_ = img_ / np.max(img_)
        img_ = img_ * mask_tensor
        img_ = img_[top:bottom + 1, left:right + 1]
        img_ = Image.fromarray(np.uint8(img_ * 255))
        inputs = self.clip_processor(text=[category[0]], images=img_, return_tensors="pt", padding=True).to(img.device)
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image

        return logits_per_image[0]

    def generate_bbox(self, density_map, tlrb, gt_dmap=None):
        if gt_dmap is not None:
            density_map = gt_dmap
        bboxes = []
        for i in range(density_map.shape[0]):
            density = np.array((density_map)[i][0].cpu())
            dmap = np.array((density_map)[i][0].cpu())

            mask = dmap < min(np.max(dmap) / self.d_t, self.s_t)
            dmap[mask] = 0
            a = skimage.feature.peak_local_max(dmap, exclude_border=0)

            boxes = []
            scores = []
            b, l, r, t = tlrb[i]

            for x11, y11 in a:
                box = [y11 - b[x11][y11].item(), x11 - l[x11][y11].item(), y11 + r[x11][y11].item(),
                       x11 + t[x11][y11].item()]
                boxes.append(box)
                scores.append(
                    (1 - math.fabs(density[max(0, int(box[1])): min(int(box[3]), dmap.shape[0]),
                                   max(int(box[0]), 0):min(int(box[2]), dmap.shape[1])].sum() - 1)) * self.d_s
                    + density[max(0, int(box[1])): min(int(box[3]), dmap.shape[0]),
                      max(int(box[0]), 0):min(int(box[2]), dmap.shape[1])].max().item() * self.m_s
                )

            b = BoxList(list(boxes), (density_map.shape[3], density_map.shape[2]))
            b.fields['scores'] = torch.tensor(scores, dtype=b.box.dtype)
            b = b.clip()
            if self.norm_s:
                b.fields['scores'] = torch.tensor(
                    [(float(i) - min(scores)) / (max(scores) - min(scores)) for i in b.fields['scores']])

            b = boxlist_nms(b, b.fields['scores'], self.i_thr)

            bboxes.append(b)
        return bboxes

    def eigenDecomposition(self, A):
        """
        References:
        https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
        http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
        """
        threshold = self.egv
        L = csgraph.laplacian(A, normed=True)
        eigenvalues, eigenvectors = LA.eig(L)

        index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:5]
        diffs = np.diff(eigenvalues)
        n_clusters = []

        for i in index_largest_gap:
            if diffs[i] > threshold:
                n_clusters.append(i)
        nb_clusters = np.array(n_clusters) + 1

        return nb_clusters[:2], eigenvalues, eigenvectors

    def compute_location(self, features):
        locations = []
        _, _, height, width = features.shape
        location_per_level = self.compute_location_per_level(
            height, width, 1, features.device
        )
        locations.append(location_per_level)

        return locations

    def compute_location_per_level(self, height, width, stride, device):
        shift_x = torch.arange(
            0, width * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y = torch.arange(
            0, height * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        location = torch.stack((shift_x, shift_y), 1) + stride // 2

        return location

    def predict_density_map(self, backbone_features, bboxes):
        bs, _, bb_h, bb_w = backbone_features.size()

        # # prepare the encoder input
        src = self.input_proj(backbone_features)
        bs, c, h, w = src.size()
        pos_emb = self.pos_emb(bs, h, w, src.device).flatten(2).permute(2, 0, 1)
        src = src.flatten(2).permute(2, 0, 1)

        # push through the encoder
        if self.num_encoder_layers > 0:
            if backbone_features.shape[2] * backbone_features.shape[3] > 6000:
                enc = self.encoder.cpu()
                memory = enc(src.cpu(), pos_emb.cpu(), src_key_padding_mask=None, src_mask=None).to(
                    backbone_features.device)
            else:
                memory = self.encoder(src, pos_emb, src_key_padding_mask=None, src_mask=None)
        else:
            memory = src

        # prepare the decoder input
        x = memory.permute(1, 2, 0).reshape(-1, self.emb_dim, bb_h, bb_w)

        bboxes_ = torch.cat([
            torch.arange(
                bs, requires_grad=False
            ).to(bboxes.device).repeat_interleave(self.num_objects).reshape(-1, 1),
            bboxes[:, :self.num_objects].flatten(0, 1),
        ], dim=1)

        # extract the objectness
        if self.use_objectness and not self.zero_shot:
            box_hw = torch.zeros(bboxes.size(0), bboxes.size(1), 2).to(bboxes.device)
            box_hw[:, :, 0] = bboxes[:, :, 2] - bboxes[:, :, 0]
            box_hw[:, :, 1] = bboxes[:, :, 3] - bboxes[:, :, 1]
            objectness = self.objectness(box_hw).reshape(
                bs, -1, self.kernel_dim ** 2, self.emb_dim
            ).flatten(1, 2).transpose(0, 1)
        elif self.zero_shot:
            objectness = self.objectness.expand(bs, -1, -1, -1).flatten(1, 2).transpose(0, 1)
        else:
            objectness = None

        # if not zero shot add appearance
        if not self.zero_shot and self.use_appearance:
            # reshape bboxes into the format suitable for roi_align
            bboxes = torch.cat([
                torch.arange(
                    bs, requires_grad=False
                ).to(bboxes.device).repeat_interleave(self.num_objects).reshape(-1, 1),
                bboxes.flatten(0, 1),
            ], dim=1)
            appearance = roi_align(
                x,
                boxes=bboxes, output_size=self.kernel_dim,
                spatial_scale=1.0 / self.reduction, aligned=True
            ).permute(0, 2, 3, 1).reshape(
                bs, self.num_objects * self.kernel_dim ** 2, -1
            ).transpose(0, 1)
        else:
            appearance = None

        if self.use_query_pos_emb:
            query_pos_emb = self.pos_emb(
                bs, self.kernel_dim, self.kernel_dim, memory.device
            ).flatten(2).permute(2, 0, 1).repeat(self.num_objects, 1, 1)
        else:
            query_pos_emb = None

        if self.num_decoder_layers > 0:
            weights = self.decoder(
                objectness if objectness is not None else appearance,
                appearance, memory, pos_emb, query_pos_emb
            )
        else:
            if objectness is not None and appearance is not None:
                weights = (objectness + appearance).unsqueeze(0)
            else:
                weights = (objectness if objectness is not None else appearance).unsqueeze(0)

        # prepare regression decoder input
        x = memory.permute(1, 2, 0).reshape(-1, self.emb_dim, bb_h, bb_w)

        outputs_R = list()
        for i in range(weights.size(0)):
            kernels = weights[i, ...].permute(1, 0, 2).reshape(
                bs, self.num_objects, self.kernel_dim, self.kernel_dim, -1
            ).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]
            if self.num_objects > 1 and not self.zero_shot:
                correlation_maps = F.conv2d(
                    torch.cat([x for _ in range(self.num_objects)], dim=1).flatten(0, 1).unsqueeze(0),
                    kernels,
                    bias=None,
                    padding=self.kernel_dim // 2,
                    groups=kernels.size(0)
                ).view(
                    bs, self.num_objects, self.emb_dim, bb_h, bb_w
                )
                softmaxed_correlation_maps = correlation_maps.softmax(dim=1)
                correlation_maps = torch.mul(softmaxed_correlation_maps, correlation_maps).sum(dim=1)
            else:
                correlation_maps = F.conv2d(
                    torch.cat([x for _ in range(self.num_objects)], dim=1).flatten(0, 1).unsqueeze(0),
                    kernels,
                    bias=None,
                    padding=self.kernel_dim // 2,
                    groups=kernels.size(0)
                ).view(
                    bs, self.num_objects, self.emb_dim, bb_h, bb_w
                ).max(dim=1)[0]

            # send through regression head
            if i == weights.size(0) - 1:
                # send through regression head
                _x = self.regression_head(correlation_maps)
                outputR = self.regression_head(correlation_maps)
            else:
                _x = self.aux_heads[i](correlation_maps)

            outputs_R.append(_x)
        return correlation_maps, outputs_R, outputR

    def forward(self, x_img, bboxes, name='', dmap=None, classes=None):
        self.num_objects = bboxes.shape[1]
        backbone_features = self.backbone(x_img)
        bs, _, bb_h, bb_w = backbone_features.size()

        #####################
        # DETECTION STAGE
        #####################

        # LOCA low-shot counter for density map prediction
        correlation_maps, outputs_R, outputR = self.predict_density_map(backbone_features, bboxes)

        if self.det_train:
            tblr = self.box_predictor(self.upscale(backbone_features), self.upscale(correlation_maps))
            location = self.compute_location(tblr)
            return outputs_R[-1], outputs_R[:-1], tblr, location

        if backbone_features.shape[2] * backbone_features.shape[3] > 8000:
            self.box_predictor = self.box_predictor.cpu()
            tblr = self.box_predictor(self.upscale(backbone_features.cpu()), self.upscale(correlation_maps.cpu()))
        else:
            tblr = self.box_predictor(self.upscale(backbone_features), self.upscale(correlation_maps))

        generated_bboxes = self.generate_bbox(outputR, tblr)[0]
        bboxes_p = generated_bboxes.box

        bboxes_pred = torch.cat([
            torch.arange(
                1, requires_grad=False
            ).to(bboxes_p.device).repeat_interleave(len(bboxes_p)).reshape(-1, 1),
            bboxes_p,
        ], dim=1).to(backbone_features.device)

        #####################
        # VERIFICATION STAGE
        #####################
        if not self.zero_shot:
            bboxes_ = torch.cat([
                torch.arange(
                    bs, requires_grad=False
                ).to(bboxes.device).repeat_interleave(self.num_objects).reshape(-1, 1),
                bboxes[:, :self.num_objects].flatten(0, 1),
            ], dim=1)
            bboxes_ = torch.cat([bboxes_, bboxes_pred])
        else:
            bboxes_ = bboxes_pred

        feat_vectors = roi_align(
            backbone_features,
            boxes=bboxes_, output_size=self.kernel_dim,
            spatial_scale=1.0 / self.reduction, aligned=True
        ).permute(0, 2, 3, 1).reshape(
            1, bboxes_.shape[0], 3, 3, -1
        ).permute(0, 1, 4, 2, 3)
        
        feat_pairs = self.feat_comp(feat_vectors.reshape(bs * bboxes_.shape[0], 3584, 3, 3))\
            .reshape(bs, bboxes_.shape[0], -1).permute(1, 0, 2)

        # Speed up, nothing changes
        if len(feat_pairs) > 500:
            return outputR, [], tblr, generated_bboxes

        # can be used to reduce memory consumption
        # dst_mtx = np.zeros((feat_pairs.shape[0], feat_pairs.shape[0]))
        # for f1, f2 in itertools.combinations(zip(feat_pairs, [i for i in range(feat_pairs.shape[0])]), 2):
        #     s=self.cosine_sim(f1[0], f2[0])
        #     dst_mtx[f1[1]][f2[1]] = s
        #     dst_mtx[f1[1]][f1[1]] = 1
        #     dst_mtx[f2[1]][f1[1]] = s

        feat_pairs = feat_pairs[:, 0]
        dst_mtx = self.cosine_sim(feat_pairs[None, :], feat_pairs[:, None]).cpu().numpy()
        dst_mtx[dst_mtx < 0] = 0

        if self.zero_shot and self.prompt_shot:
            preds = generated_bboxes

            k, _, _ = self.eigenDecomposition(dst_mtx)
            if len(k) > 1 or (len(k) > 1 and k[0] > 1):
                n_clusters_ = max(k)
                spectral = SpectralClustering(n_clusters=n_clusters_, affinity='precomputed')
                labels = spectral.fit_predict(dst_mtx)

                box_labels = labels
                labels = box_labels[box_labels >= 0]
                labels, counts = np.unique(np.array(labels), return_counts=True)
                correct_clusters = []
                probs = []
                for lab in labels:
                    mask = np.in1d(box_labels, lab).reshape(box_labels.shape)
                    probs.append(self.clip_check_clusters(x_img, bboxes_p[mask], classes, img_name=name).item())
                    correct_clusters.append(lab)
                thresh = max(probs) * 0.85
                correct = np.array(probs) > thresh
                correct_clusters = np.array(correct_clusters)[correct]
                mask = np.in1d(box_labels, correct_clusters).reshape(box_labels.shape)
                preds = generated_bboxes[mask]

                if len(preds) != len(generated_bboxes):
                    outputR[0][0] = mask_density(outputR[0], preds)

            return outputR, [], tblr, preds

        elif self.zero_shot and not self.prompt_shot:
            k, _, _ = self.eigenDecomposition(dst_mtx)
            preds = generated_bboxes
            if len(k) > 1 or (len(k) > 1 and k[0] > 1):
                n_clusters_ = max(k)
                spectral = SpectralClustering(n_clusters=n_clusters_, affinity='precomputed')
                labels = spectral.fit_predict(dst_mtx)

                box_labels = labels
                labels = box_labels[box_labels >= 0]
                labels, counts = np.unique(np.array(labels), return_counts=True)

                max_count = np.max(counts)
                proc_freq = counts / max_count
                correct_class_labels = labels[proc_freq > 0.50]

                mask = []
                for iii, box in enumerate(generated_bboxes.box):
                    if box_labels[iii] in correct_class_labels:
                        mask.append(iii)

                preds = generated_bboxes[mask]
                outputR[0][0] = mask_density(outputR[0], preds)
            return outputR, [], tblr, preds

        else:
            dst_mtx[dst_mtx < 0] = 0

            k, _, _ = self.eigenDecomposition(dst_mtx)
            exemplar_bboxes = generated_bboxes
            mask = None
            if len(k) > 1 or k[0] > 1:

                n_clusters_ = max(k)
                spectral = SpectralClustering(n_clusters=n_clusters_, affinity='precomputed')
                labels = spectral.fit_predict(dst_mtx)
                correct_class_labels = list(np.unique(np.array(labels[:self.num_objects])))

                for i in range(len(bboxes_p)):
                    box = bboxes_p[i]
                    if (box_iou(box.unsqueeze(0), bboxes_[:self.num_objects][:, 1:].cpu()) > 0.6).any():
                        correct_class_labels.append(labels[i + self.num_objects])

                mask = np.in1d(labels, correct_class_labels).reshape(labels.shape)

                exemplar_bboxes = generated_bboxes[mask[self.num_objects:]]

            if mask is not None and np.any(mask == False):
                outputR[0][0] = mask_density(outputR[0], exemplar_bboxes)

        return outputR, [], tblr, exemplar_bboxes


def build_model(args):
    return COTR(
        image_size=args.image_size,
        num_encoder_layers=args.num_enc_layers,
        num_decoder_layers=args.num_dec_layers,
        num_objects=args.num_objects,
        zero_shot=args.zero_shot,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        kernel_dim=args.kernel_dim,
        backbone_name=args.backbone,
        swav_backbone=args.swav_backbone,
        train_backbone=args.backbone_lr > 0,
        reduction=args.reduction,
        dropout=args.dropout,
        layer_norm_eps=1e-5,
        mlp_factor=8,
        norm_first=args.pre_norm,
        activation=nn.GELU,
        norm=True,
        use_query_pos_emb=args.use_query_pos_emb,
        use_objectness=args.use_objectness,
        use_appearance=args.use_appearance,
        d_s=args.d_s,
        m_s=args.m_s,
        i_thr=args.i_thr,
        d_t=args.d_t,
        s_t=args.s_t,
        norm_s=args.norm_s,
        egv=args.egv,
        prompt_shot=args.prompt_shot,
        det_train=args.det_train
    )
