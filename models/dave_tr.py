import itertools

import torch
from torch import nn
from torchvision.ops import roi_align

from .backbone import Backbone
from .feat_comparison import Feature_Transform


class COTR(nn.Module):

    def __init__(
            self,
            image_size: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            num_objects: int,
            emb_dim: int,
            kernel_dim: int,
            backbone_name: str,
            swav_backbone: bool,
            train_backbone: bool,
            reduction: int,
            use_query_pos_emb: bool,
            zero_shot: bool,
            use_objectness: bool,
            use_appearance: bool
    ):

        super(COTR, self).__init__()

        self.emb_dim = emb_dim
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot
        self.use_query_pos_emb = use_query_pos_emb
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.use_objectness = use_objectness
        self.use_appearance = use_appearance
        self.cosine_sim = nn.CosineSimilarity()
        self.backbone = Backbone(
            backbone_name, pretrained=True, dilation=False, reduction=reduction,
            swav=swav_backbone, requires_grad=train_backbone
        )
        self.cos_loss = nn.CosineEmbeddingLoss(margin=0.0)
        self.feat_comp = Feature_Transform()

    def forward(self, x, bboxes):
        # backbone
        backbone_features = self.backbone(x).detach()
        bs, _, bb_h, bb_w = backbone_features.size()

        bboxes_ = torch.cat([
            torch.arange(
                bs, requires_grad=False
            ).to(bboxes.device).repeat_interleave(bboxes.shape[1]).reshape(-1, 1),
            bboxes.flatten(0, 1),
        ], dim=1)

        feat_vectors = roi_align(
            backbone_features,
            boxes=bboxes_, output_size=self.kernel_dim,
            spatial_scale=1.0 / self.reduction, aligned=True
        ).permute(0, 2, 3, 1).reshape(
            bs, 6, 3, 3, -1
        ).permute(0, 1, 4, 2, 3)
        feat_pairs = self.feat_comp(feat_vectors.reshape(bs * 6, 3584, 3, 3)).reshape(bs, 6, -1).permute(1, 0, 2)
        sim = list()
        class_ = []
        loss = torch.tensor(0.0).to(feat_pairs.device)
        o = torch.tensor(1).to(feat_pairs.device)
        n = torch.tensor(-1).to(feat_pairs.device)

        for f1, f2 in itertools.combinations(zip(feat_pairs, [1, 1, 1, 2, 2, 2]), 2):
            for i in range(f1[0].shape[0]):
                loss += self.cos_loss(f1[0][i], f2[0][i], o if f1[1] == f2[1] else n)
            sim.append(self.cosine_sim(f1[0], f2[0]))
            class_.append([f1[1] == f2[1] for _ in range(bs)])

        return loss, sim, class_


def build_model(args):
    return COTR(
        image_size=args.image_size,
        num_encoder_layers=args.num_enc_layers,
        num_decoder_layers=args.num_dec_layers,
        num_objects=args.num_objects,
        zero_shot=args.zero_shot,
        emb_dim=args.emb_dim,
        kernel_dim=args.kernel_dim,
        backbone_name=args.backbone,
        swav_backbone=args.swav_backbone,
        train_backbone=args.backbone_lr > 0,
        reduction=args.reduction,
        use_query_pos_emb=args.use_query_pos_emb,
        use_objectness=args.use_objectness,
        use_appearance=args.use_appearance
    )
