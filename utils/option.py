import argparse
import os
from typing import Any, Dict

try:
    import yaml
except Exception:
    yaml = None  # PyYAML is listed in req.txt; handle gracefully if missing at runtime


def _load_yaml_config(path: str) -> Dict[str, Any]:
    if path is None:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML not available. Please install pyyaml or remove --config.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def _apply_config_to_args(cfg: Dict[str, Any], args: argparse.Namespace, frozen_keys: set[str] | None = None) -> argparse.Namespace:
    """Map structured YAML config sections into existing argparse names."""
    if frozen_keys is None:
        frozen_keys = set()
    # model
    m = cfg.get('model', {}) or {}
    for k, v, dest in [
        ('architecture', m.get('architecture'), 'architecture'),
        ('head_type', m.get('head_type'), 'head_type'),
        ('mamba_scan_type', m.get('mamba_scan_type'), 'mamba_scan_type'),
        ('use_mamba', m.get('use_mamba'), 'use_mamba'),
        ('nb_cls', m.get('nb_cls'), 'nb_cls'),
        ('img_size', m.get('img_size'), 'img_size'),
        ('patch_size', m.get('patch_size'), 'patch_size'),
        ('proj', m.get('proj'), 'proj'),
        ('depth', m.get('depth'), 'depth'),
        ('attn_drop_rate', m.get('attn_drop_rate'), 'attn_drop_rate'),
        ('drop_path', m.get('drop_path'), 'drop_path'),
        ('bilstm_hidden_dim', m.get('bilstm_hidden_dim'), 'bilstm_hidden_dim'),
        ('bilstm_num_layers', m.get('bilstm_num_layers'), 'bilstm_num_layers'),
        ('bilstm_dropout', m.get('bilstm_dropout'), 'bilstm_dropout'),
        ('use_bimamba_arch_proj', m.get('use_bimamba_arch_proj'), 'use_bimamba_arch_proj'),
        ('use_bimamba_head_proj', m.get('use_bimamba_head_proj'), 'use_bimamba_head_proj')
    ]:
        if v is not None and dest not in frozen_keys:
            setattr(args, dest, v)

    # dataset
    d = cfg.get('dataset', {}) or {}
    if d.get('name') and not getattr(args, 'subcommand', None):
        args.subcommand = d.get('name')
    for k, v, dest in [
        ('data_path', d.get('data_path'), 'data_path'),
        ('train_data_list', d.get('train_data_list'), 'train_data_list'),
        ('val_data_list', d.get('val_data_list'), 'val_data_list'),
        ('test_data_list', d.get('test_data_list'), 'test_data_list'),
        ('nb_cls', d.get('nb_cls'), 'nb_cls'),
    ]:
        if v is not None and dest not in frozen_keys:
            setattr(args, dest, v)

    # optimizer
    o = cfg.get('optimizer', {}) or {}
    for k, v, dest in [
        ('max_lr', o.get('max_lr'), 'max_lr'),
        ('min_lr', o.get('min_lr'), 'min_lr'),
        ('initial_lr', o.get('initial_lr'), 'lr'),
        ('weight_decay', o.get('weight_decay'), 'weight_decay'),
        ('use_sam', o.get('use_sam'), 'use_sam'),
    ]:
        if v is not None and dest not in frozen_keys:
            setattr(args, dest, v)

    # training
    t = cfg.get('training', {}) or {}
    for k, v, dest in [
        ('total_iter', t.get('total_iter'), 'total_iter'),
        ('warm_up_iter', t.get('warm_up_iter'), 'warm_up_iter'),
        ('eval_iter', t.get('eval_iter'), 'eval_iter'),
        ('print_iter', t.get('print_iter'), 'print_iter'),
        ('seed', t.get('seed'), 'seed'),
        ('ema_decay', t.get('ema_decay'), 'ema_decay'),
        ('mask_ratio', t.get('mask_ratio'), 'mask_ratio'),
        ('attn_mask_ratio', t.get('attn_mask_ratio'), 'attn_mask_ratio'),
        ('max_span_length', t.get('max_span_length'), 'max_span_length'),
        ('spacing', t.get('spacing'), 'spacing'),
        ('alpha', t.get('alpha'), 'alpha'),
        ('cos_temp', t.get('cos_temp'), 'cos_temp'),
        ('mask_version', t.get('mask_version'), 'mask_version'),
        ('use_masking', t.get('use_masking'), 'use_masking'),
    ]:
        if v is not None and dest not in frozen_keys:
            setattr(args, dest, v)

    # dataloader
    dl = cfg.get('dataloader', {}) or {}
    for k, v, dest in [
        ('train_bs', dl.get('train_bs'), 'train_bs'),
        ('val_bs', dl.get('val_bs'), 'val_bs'),
        ('num_workers', dl.get('num_workers'), 'num_workers'),
    ]:
        if v is not None and dest not in frozen_keys:
            setattr(args, dest, v)

    # augmentation
    a = cfg.get('augmentation', {}) or {}
    aug_map = {
        'proba': 'proba',
        'dpi_min_factor': 'dpi_min_factor',
        'dpi_max_factor': 'dpi_max_factor',
        'perspective_low': 'perspective_low',
        'perspective_high': 'perspective_high',
        'elastic_distortion_min_kernel_size': 'elastic_distortion_min_kernel_size',
        'elastic_distortion_max_kernel_size': 'elastic_distortion_max_kernel_size',
        'elastic_distortion_max_magnitude': 'elastic_distortion_max_magnitude',
        'elastic_distortion_min_alpha': 'elastic_distortion_min_alpha',
        'elastic_distortion_max_alpha': 'elastic_distortion_max_alpha',
        'elastic_distortion_min_sigma': 'elastic_distortion_min_sigma',
        'elastic_distortion_max_sigma': 'elastic_distortion_max_sigma',
        'dila_ero_max_kernel': 'dila_ero_max_kernel',
        'dila_ero_iter': 'dila_ero_iter',
        'jitter_contrast': 'jitter_contrast',
        'jitter_brightness': 'jitter_brightness',
        'jitter_saturation': 'jitter_saturation',
        'jitter_hue': 'jitter_hue',
        'blur_min_kernel': 'blur_min_kernel',
        'blur_max_kernel': 'blur_max_kernel',
        'blur_min_sigma': 'blur_min_sigma',
        'blur_max_sigma': 'blur_max_sigma',
        'sharpen_min_alpha': 'sharpen_min_alpha',
        'sharpen_max_alpha': 'sharpen_max_alpha',
        'sharpen_min_strength': 'sharpen_min_strength',
        'sharpen_max_strength': 'sharpen_max_strength',
        'zoom_min_h': 'zoom_min_h',
        'zoom_max_h': 'zoom_max_h',
        'zoom_min_w': 'zoom_min_w',
        'zoom_max_w': 'zoom_max_w',
    }
    for k, dest in aug_map.items():
        if k in a and a[k] is not None and dest not in frozen_keys:
            setattr(args, dest, a[k])

    # output
    out = cfg.get('output', {}) or {}
    if out.get('out_dir') is not None and 'out_dir' not in frozen_keys:
        args.out_dir = out.get('out_dir')
    if out.get('exp_name') is not None and 'exp_name' not in frozen_keys:
        args.exp_name = out.get('exp_name')
    if out.get('use_wandb') is not None and 'use_wandb' not in frozen_keys:
        args.use_wandb = out.get('use_wandb')

    # pretrained
    pt = cfg.get('pretrained', {}) or {}
    if pt.get('path') and 'pretrained_path' not in frozen_keys:
        args.pretrained_path = pt.get('path')

    # Fallback: also allow flat, top-level keys to override known argparse attributes.
    # This makes cfgs flexible: either nested (recommended) or flat.
    section_keys = {'model', 'dataset', 'optimizer', 'training', 'dataloader', 'augmentation', 'output', 'pretrained'}
    for k, v in (cfg or {}).items():
        if k in section_keys:
            continue
        if v is None:
            continue
        # normalize hyphen-separated keys to argparse-style underscores
        k_norm = k.replace('-', '_')
        if hasattr(args, k_norm) and k_norm not in frozen_keys:
            setattr(args, k_norm, v)

    return args


def get_args_parser():
    parser = argparse.ArgumentParser(description='Origami for HTR',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    
    parser.add_argument('--head_type', type=str, default='linear', choices=['linear', 'bilstm', 'bimamba', 'bidimamba'],
                       help='Head type for the model: linear, bilstm, or bimamba')
    parser.add_argument('--bilstm_hidden_dim', type=int, default=512,
                       help='Hidden dimension for BiLSTM')
    parser.add_argument('--bilstm_num_layers', type=int, default=2,
                       help='Number of BiLSTM layers')
    parser.add_argument('--bilstm_dropout', type=float, default=0.1,
                       help='Dropout rate for BiLSTM')

    parser.add_argument('--mamba_scan_type', type=str, default='bidi', choices=['single', 'double', 'quad'],
                       help='Scan type for the model, single for single direction, double for double direction')
    
    parser.add_argument('--mask-version', default=0, type=int, help='mask version used for training (0 = legacy)')
    parser.add_argument('--use-masking', action='store_true', default=False, help='whether to use masking during training')
    parser.add_argument('--out-dir', type=str, default='./output', help='output directory')
    parser.add_argument('--use-sam', action='store_true', default=False, help='whether to use SAM optimizer')
    parser.add_argument('--train-bs', default=8, type=int, help='train batch size')
    parser.add_argument('--architecture', type=str, choices=['mamba', 'transformer', 'rwkv', 'hybrid', 'xlstm', 'bidimamba'], default='mamba', help='Use mamba, transformer, or RWKV architecture')
    parser.add_argument('--val-bs', default=1, type=int, help='validation batch size')
    parser.add_argument('--num-workers', default=8, type=int, help='nb of workers')
    parser.add_argument('--eval-iter', default=1000, type=int, help='nb of iterations to run evaluation')
    parser.add_argument('--total-iter', default=100000, type=int, help='nb of total iterations for training')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='nb of iterations for warm-up')
    parser.add_argument('--print-iter', default=100, type=int, help='nb of total iterations to print information')
    parser.add_argument('--max-lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--min-lr', default=1e-7, type=float, help='learning rate')
    parser.add_argument('--lr', default=1e-7, type=float, help='learning rate')
    parser.add_argument('--weight-decay', default=5e-1, type=float, help='weight decay')
    parser.add_argument('--use-wandb', action='store_true', default=False, help = 'wheteher use wandb, otherwise use tensorboard')
    parser.add_argument('--exp-name',type=str, default='IAM_HTR_ORIGAMI_NET', help='experimental name (save dir will be out_dir + exp_name)')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training. ')
    parser.add_argument('--pretrained_path', default='', type=str, help='If set it will load a pretrained before training')
    parser.add_argument('--use_mamba', default=False, type=bool, help='Use Mamba block (True) or Transformer block (False)')
    parser.add_argument('--depth', default=4, type=int, help='Depth of the model')

    parser.add_argument('--img-size', default=[512, 64], type=int, nargs='+', help='image size')
    parser.add_argument('--attn-mask-ratio', default=0., type=float, help='attention drop_key mask ratio')
    parser.add_argument('--patch-size', default=[4, 32], type=int, nargs='+', help='patch size')
    parser.add_argument('--mask-ratio', default=0.3, type=float, help='mask ratio')
    parser.add_argument('--cos-temp', default=8, type=int, help='cosine similarity classifier temperature')
    parser.add_argument('--max-span-length', default=4, type=int, help='max mask length')
    parser.add_argument('--spacing', default=0, type=int, help='the spacing between two span masks')
    parser.add_argument('--proj', default=8, type=float, help='projection value')
    parser.add_argument('--attn_drop_rate', default=0., type=float, help='attention drop path rate')
    parser.add_argument('--drop_path', default=0., type=float, help='drop path rate')

    parser.add_argument('--dpi-min-factor', default=0.5,type=float)
    parser.add_argument('--dpi-max-factor', default=1.5, type=float)
    parser.add_argument('--perspective-low', default=0., type=float)
    parser.add_argument('--perspective-high', default=0.4, type=float)
    parser.add_argument('--elastic-distortion-min-kernel-size', default=3, type=int)
    parser.add_argument('--elastic-distortion-max-kernel-size', default=3, type=int)
    parser.add_argument('--elastic_distortion-max-magnitude', default=20, type=int)
    parser.add_argument('--elastic-distortion-min-alpha', default=0.5, type=float)
    parser.add_argument('--elastic-distortion-max-alpha', default=1, type=float)
    parser.add_argument('--elastic-distortion-min-sigma', default=1, type=int)
    parser.add_argument('--elastic-distortion-max-sigma', default=10, type=int)
    parser.add_argument('--dila-ero-max-kernel', default=3, type=int )
    parser.add_argument('--jitter-contrast', default=0.4, type=float)
    parser.add_argument('--jitter-brightness', default=0.4, type=float )
    parser.add_argument('--jitter-saturation', default=0.4, type=float )
    parser.add_argument('--jitter-hue', default=0.2, type=float)

    parser.add_argument('--dila-ero-iter', default=1, type=int, help='nb of iterations for dilation and erosion kernel')
    parser.add_argument('--blur-min-kernel', default=3, type=int)
    parser.add_argument('--blur-max-kernel', default=5, type=int)
    parser.add_argument('--blur-min-sigma', default=3, type=int)
    parser.add_argument('--blur-max-sigma', default=5, type=int)
    parser.add_argument('--sharpen-min-alpha', default=0, type=int)
    parser.add_argument('--sharpen-max-alpha', default=1, type=int)
    parser.add_argument('--sharpen-min-strength', default=0, type=int)
    parser.add_argument('--sharpen-max-strength', default=1, type=int)
    parser.add_argument('--zoom-min-h', default=0.8, type=float)
    parser.add_argument('--zoom-max-h', default=1, type=float)
    parser.add_argument('--zoom-min-w', default=0.99, type=float)
    parser.add_argument('--zoom-max-w', default=1, type=float)
    parser.add_argument('--proba', default=0.5, type=float)

    parser.add_argument('--ema-decay', default=0.9999, type=float, help='Exponential Moving Average (EMA) decay')
    parser.add_argument('--alpha', default=0, type=float, help='kld loss ratio')

    # Optional: config file support
    parser.add_argument('--config', type=str, default=None, help='Path to YAML configuration file')

    subparsers = parser.add_subparsers(title="dataset setting", dest="subcommand")

    IAM = subparsers.add_parser("IAM",
                                description='Dataset parser for training on IAM',
                                add_help=True,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help="Dataset parser for training on IAM")

    IAM.add_argument('--train-data-list', type=str, default='./data/iam/train.ln',
                     help='train data list (gc file)(ln file)')
    IAM.add_argument('--data-path', type=str, default='./data/iam/lines/',
                     help='train data list')
    IAM.add_argument('--val-data-list', type=str, default='./data/iam/val.ln',
                     help='val data list')
    IAM.add_argument('--test-data-list', type=str, default='./data/iam/test.ln',
                     help='test data list')
    IAM.add_argument('--nb-cls', default=80, type=int, help='nb of classes, IAM=79+1, READ2016=89+1')
    
    IAM_OLD = subparsers.add_parser("IAM_OLD",
                                description='Dataset parser for training on IAM',
                                add_help=True,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help="Dataset parser for training on IAM")

    IAM_OLD.add_argument('--train-data-list', type=str, default='./data/iam/train.ln',
                     help='train data list (gc file)(ln file)')
    IAM_OLD.add_argument('--data-path', type=str, default='./data/iam/lines/',
                     help='train data list')
    IAM_OLD.add_argument('--val-data-list', type=str, default='./data/iam/val.ln',
                     help='val data list')
    IAM_OLD.add_argument('--test-data-list', type=str, default='./data/iam/test.ln',
                     help='test data list')
    IAM_OLD.add_argument('--nb-cls', default=80, type=int, help='nb of classes, IAM=79+1, READ2016=89+1')

    READ = subparsers.add_parser("READ",
                                 description='Dataset parser for training on READ',
                                 add_help=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 help="Dataset parser for training on READ")

    READ.add_argument('--train-data-list', type=str, default='./data/read2016/train.ln',
                      help='train data list (gc file)(ln file)')
    READ.add_argument('--data-path', type=str, default='./data/read2016/lines/',
                      help='train data list')
    READ.add_argument('--val-data-list', type=str, default='./data/read2016/val.ln',
                      help='val data list')
    READ.add_argument('--test-data-list', type=str, default='./data/read2016/test.ln',
                      help='test data list')
    READ.add_argument('--nb-cls', default=90, type=int, help='nb of classes, IAM=79+1, READ2016=89+1')

    LAM = subparsers.add_parser("LAM",
                                description='Dataset parser for training on LAM',
                                add_help=True,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help="Dataset parser for training on READ")

    LAM.add_argument('--train-data-list', type=str, default='./data/LAM/train.ln',
                     help='train data list (gc file)(ln file)')
    LAM.add_argument('--data-path', type=str, default='./data/LAM/',
                     help='train data list')
    LAM.add_argument('--val-data-list', type=str, default='./data/LAM/val.ln',
                     help='val data list')
    LAM.add_argument('--test-data-list', type=str, default='./data/LAM/test.ln',
                     help='test data list')
    LAM.add_argument('--nb-cls', default=90, type=int, help='nb of classes, IAM=79+1, READ2016=89+1')

    PONTALTO = subparsers.add_parser("PONTALTO",
                                description='Dataset parser for training on LAM',
                                add_help=True,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help="Dataset parser for training on READ")

    PONTALTO.add_argument('--train-data-list', type=str, default='./data/PONTALTO2/train.ln',
                     help='train data list (gc file)(ln file)')
    PONTALTO.add_argument('--data-path', type=str, default='./data/PONTALTO2/',
                     help='train data list')
    PONTALTO.add_argument('--val-data-list', type=str, default='./data/PONTALTO2/val.ln',
                     help='val data list')
    PONTALTO.add_argument('--test-data-list', type=str, default='./data/PONTALTO2/test.ln',
                     help='test data list')
    PONTALTO.add_argument('--nb-cls', default=90, type=int, help='nb of classes, IAM=79+1, READ2016=89+1')
    
    RIMES = subparsers.add_parser("RIMES",
                                description='Dataset parser for training on LAM',
                                add_help=True,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                help="Dataset parser for training on READ")

    RIMES.add_argument('--data-path', type=str, default='./data/',
                     help='train data list')
   
    RIMES.add_argument('--nb-cls', default=166, type=int, help='nb of classes, IAM=79+1, READ2016=89+1')
    
    

    # First parse defaults and CLI to detect explicit CLI overrides
    defaults = parser.parse_args([])
    args = parser.parse_args()

    # Keys explicitly provided on CLI (different from defaults) will be frozen
    frozen_keys = set()
    for k, v in vars(args).items():
        if not hasattr(defaults, k):
            continue
        dv = getattr(defaults, k)
        if v != dv:
            frozen_keys.add(k)
    # config itself should not be a frozen key for downstream mapping
    frozen_keys.discard('config')

    # Load and apply config if provided, honoring CLI overrides
    if getattr(args, 'config', None):
        cfg = _load_yaml_config(args.config)
        args = _apply_config_to_args(cfg, args, frozen_keys=frozen_keys)

    return args
