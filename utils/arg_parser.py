import argparse


def get_argparser():

    parser = argparse.ArgumentParser("COTR parser", add_help=False)
    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument(
        '--data_path',
        default=r'C:\projects\DAVE\FSC147_384_V2',
        type=str
    )
    parser.add_argument(
        '--model_path',
        default='material/',
        type=str
    )
    parser.add_argument('--det_model_name', default='DAVE', type=str)
    parser.add_argument('--dataset', default='fsc147', type=str)
    parser.add_argument('--backbone', default='resnet18', type=str)
    parser.add_argument('--swav_backbone', action='store_true')
    parser.add_argument('--reduction', default=8, type=int)
    parser.add_argument('--image_size', default=512, type=int)
    parser.add_argument('--fcos_pred_size', default=512, type=int)
    parser.add_argument('--num_enc_layers', default=3, type=int)
    parser.add_argument('--num_dec_layers', default=3, type=int)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--kernel_dim', default=3, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--backbone_lr', default=0, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--max_grad_norm', default=0.1, type=float)
    parser.add_argument('--aux_weight', default=0.3, type=float)
    parser.add_argument('--tiling_p', default=0.5, type=float)
    parser.add_argument('--detection_loss_weight', default=0.01, type=float)
    parser.add_argument('--num_objects', default=3, type=int)
    parser.add_argument('--task', default='fscd147', type=str)
    parser.add_argument('--d_s', default=1.0, type=float)
    parser.add_argument('--m_s', default=0.0, type=float)
    parser.add_argument('--i_thr', default=0.55, type=float)
    parser.add_argument('--d_t', default=3, type=float)
    parser.add_argument('--s_t', default=0.008, type=float)
    parser.add_argument('--norm_s', action='store_true')
    parser.add_argument('--unseen', action='store_true')
    parser.add_argument('--egv', default=0.132, type=float)
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--det_train', action='store_true')
    parser.add_argument('--eval_multicat', action='store_true')
    parser.add_argument('--prompt_shot', action='store_true')
    parser.add_argument('--normalized_l2', action='store_true')
    parser.add_argument('--count_loss_weight', default=0, type=float)
    parser.add_argument('--min_count_loss_weight', default=0, type=float)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--use_query_pos_emb', action='store_true')
    parser.add_argument('--use_objectness', action='store_true')
    parser.add_argument('--use_appearance', action='store_true')
    parser.add_argument('--orig_dmaps', action='store_true')
    parser.add_argument('--skip_cars', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_test', action='store_true')

    return parser
