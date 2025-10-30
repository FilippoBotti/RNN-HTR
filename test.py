import torch

import os
import re
import json
import valid
from utils import utils
from utils import option
from data import dataset
from model import HTR_VT
from collections import OrderedDict
from data.LAM.utils.dummy_main import LAM
from torchvision.transforms import ToTensor
from data.misc import collate_fn
from data.RIMES import build_RIMES
from data.PONTALTO.pontalto import PONTALTO

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1], args=args)

    pth_path = args.save_dir + '/best_CER.pth'
    logger.info('loading HWR checkpoint from {}'.format(pth_path))

    ckpt = torch.load(pth_path, map_location='cpu')
    model_dict = OrderedDict()
    pattern = re.compile('module.')

    for k, v in ckpt['state_dict_ema'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)
    model = model.cuda()

    if args.subcommand in ['READ', 'IAM']:
        logger.info('Loading test loader...')
        train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)

        test_dataset = dataset.myLoadDS(args.test_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.val_bs,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=args.num_workers)

        converter = utils.CTCLabelConverter(train_dataset.ralph.values())
        criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)
        
    elif args.subcommand == 'LAM':
        logger.info('Loading test loader...')
        train_dataset = LAM(args.data_path , 'basic', ToTensor(), img_size=args.img_size, nameset='train')
        test_dataset = LAM(args.data_path, 'basic', ToTensor(), nameset='test', img_size=args.img_size, charset=train_dataset.charset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_bs, shuffle=False, pin_memory=True,
                                               num_workers=args.num_workers)
        criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
        converter = utils.CTCLabelConverter(train_dataset.charset)
        
    elif args.subcommand == 'RIMES':
        train_dataset = build_RIMES(image_set='train', dataset_path=args.data_path, args=args)
        test_dataset = build_RIMES(image_set='test', dataset_path=args.data_path, args=args) 
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_bs, shuffle=False, pin_memory=True,
                                 drop_last=False,  num_workers=args.num_workers)
        criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
        converter = utils.CTCLabelConverter(train_dataset.charset)

    elif args.subcommand == 'PONTALTO':
        logger.info('Loading test loader...')
        train_dataset = PONTALTO(args.data_path , 'basic', ToTensor(), img_size=args.img_size, nameset='train')
        test_dataset = PONTALTO(args.data_path, 'basic', ToTensor(), nameset='test', img_size=args.img_size, charset=train_dataset.charset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_bs, shuffle=False, pin_memory=True,
                                               num_workers=args.num_workers)
        criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
        converter = utils.CTCLabelConverter(train_dataset.charset)
          
    else:
        assert("Dataset must be READ/IAM/LAM/RIMES/PONTALTO")

    model.eval()
    with torch.no_grad():
        val_loss, val_cer, val_wer, preds, labels = valid.validation(model,
                                                                     criterion,
                                                                     test_loader,
                                                                     converter)

    logger.info(
        f'Test. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} ')
    
    pth_path = args.save_dir + '/best_WER.pth'
    logger.info('loading HWR checkpoint from {}'.format(pth_path))

    ckpt = torch.load(pth_path, map_location='cpu')
    model_dict = OrderedDict()
    pattern = re.compile('module.')

    for k, v in ckpt['state_dict_ema'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        val_loss, val_cer, val_wer, preds, labels = valid.validation(model,
                                                                     criterion,
                                                                     test_loader,
                                                                     converter)

    logger.info(
        f'Test. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} ')

if __name__ == '__main__':
    args = option.get_args_parser()
    main()

