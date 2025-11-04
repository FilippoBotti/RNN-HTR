import argparse


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
    
    

    return parser.parse_args()
