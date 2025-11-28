import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import os
import json
import valid
from utils import utils
from utils import sam
from utils import option
from data import dataset
from model import HTR_VT
from functools import partial
from tqdm import tqdm
from data.LAM.utils import dummy_main
from torchvision.transforms import ToTensor
from data.RIMES import build_RIMES
from data.PONTALTO.pontalto import PONTALTO
from data.format_iam import IAMDatasetFormatter, IAMDataset, build_charset

import time
import uuid

def compute_loss(args, model, image, batch_size, criterion, text, length):
    preds = model(image, args.mask_ratio, args.max_span_length, use_masking=True)
    preds = preds.float()
    preds_size = torch.IntTensor([preds.size(1)] * batch_size).cuda()
    preds = preds.permute(1, 0, 2).log_softmax(2)

    torch.backends.cudnn.enabled = False
    loss = criterion(preds, text.cuda(), preds_size, length.cuda()).mean()
    torch.backends.cudnn.enabled = True
    return loss


def main():

    args = option.get_args_parser()
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name, time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid.uuid4().hex[:8])
    os.makedirs(args.save_dir, exist_ok=True)

    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    writer = SummaryWriter(args.save_dir)

    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1], args=args)

    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('total_param is {}'.format(total_param))
    model.train()
    model = model.cuda()
    model_ema = utils.ModelEma(model, args.ema_decay)
    model.zero_grad()
    logger.info(model)
    if args.subcommand in ['READ', 'IAM_OLD']:
        logger.info('Loading train loader...')
        train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.train_bs,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=args.num_workers,
                                                collate_fn=partial(dataset.SameTrCollate, args=args))
        train_iter = dataset.cycle_data(train_loader)

        logger.info('Loading val loader...')
        val_dataset = dataset.myLoadDS(args.val_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=args.val_bs,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=args.num_workers)
        test_dataset = dataset.myLoadDS(args.test_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.val_bs,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=args.num_workers)
        if args.use_sam:
            optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
        converter = utils.CTCLabelConverter(train_dataset.ralph.values())
        
    elif args.subcommand == 'LAM':
        logger.info('Loading train loader...')
        train_dataset = dummy_main.LAM(args.data_path , 'basic', ToTensor(), img_size=args.img_size, nameset='train')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, pin_memory=True,
                                               num_workers=args.num_workers, collate_fn=partial(dataset.SameTrCollate, args=args))
        train_iter = dataset.cycle_data(train_loader)
        test_dataset = dummy_main.LAM(args.data_path, 'basic', ToTensor(), nameset='test', img_size=args.img_size, charset=train_dataset.charset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_bs, shuffle=False, pin_memory=True,
                                               num_workers=args.num_workers)
        
        logger.info('Loading val loader...')
        val_dataset = dummy_main.LAM(args.data_path, 'basic', ToTensor(), nameset='val', img_size=args.img_size, charset=train_dataset.charset)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_bs, shuffle=False, pin_memory=True,
                                               num_workers=args.num_workers)
        # optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        if args.use_sam:
            optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
        converter = utils.CTCLabelConverter(train_dataset.charset)
    elif args.subcommand == 'IAM':
        logger.info('Loading train loader...')
        xml_folder = "./data/iam_dataset/xml"
        image_folder = "./data/iam_dataset/lines"
        train_file = "./data/iam_dataset/trainset.txt"
        val_file = "./data/iam_dataset/validationset1.txt"
        test_file = "./data/iam_dataset/testset.txt"
        formatter = IAMDatasetFormatter(xml_folder, image_folder)
        train_dataset = IAMDataset(train_file, formatter, img_size=args.img_size)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.train_bs,
                                                shuffle=True,
                                                pin_memory=True,
                                                num_workers=args.num_workers,
                                                collate_fn=partial(dataset.SameTrCollate, args=args))
        train_iter = dataset.cycle_data(train_loader)

        logger.info('Loading val loader...')
        val_dataset = IAMDataset(val_file, formatter, img_size=args.img_size)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=args.val_bs,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=args.num_workers)
        test_dataset = IAMDataset(test_file, formatter, img_size=args.img_size)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=args.val_bs,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=args.num_workers)
        if args.use_sam:
            optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
        converter = utils.CTCLabelConverter(build_charset(formatter))
        
    elif args.subcommand == 'PONTALTO':
        logger.info('Loading train loader...')
        train_dataset = PONTALTO(args.data_path , 'basic', ToTensor(), img_size=args.img_size, nameset='train')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, pin_memory=True,
                                               num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
        train_iter = dataset.cycle_data(train_loader)
        
        logger.info('Loading val loader...')
        val_dataset = PONTALTO(args.data_path, 'basic', ToTensor(), nameset='test', img_size=args.img_size, charset=train_dataset.charset)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_bs, shuffle=False, pin_memory=True,
                                               num_workers=args.num_workers)
        
        if args.use_sam:
            optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
        converter = utils.CTCLabelConverter(train_dataset.charset)
    elif args.subcommand == 'RIMES':
        train_dataset = build_RIMES(image_set='train', dataset_path=args.data_path, args=args)
        val_dataset = build_RIMES(image_set='val', dataset_path=args.data_path, args=args) 
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_bs, shuffle=True, pin_memory=True,
                                   collate_fn=partial(dataset.SameTrCollate, args=args), num_workers=args.num_workers)
        train_iter = dataset.cycle_data(train_loader)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_bs, shuffle=False, pin_memory=True,
                                 drop_last=False, num_workers=args.num_workers)
        # optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        if args.use_sam:
            optimizer = sam.SAM(model.parameters(), torch.optim.AdamW, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        test_dataset = build_RIMES(image_set='test', dataset_path=args.data_path, args=args) 
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_bs, shuffle=False, pin_memory=True,
                                 drop_last=False,  num_workers=args.num_workers)
        criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True)
        converter = utils.CTCLabelConverter(train_dataset.charset)
    else:
        assert("Dataset must be READ/IAM/LAM/RIMES/PONTALTO")
        
    #Load pretrained model
    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path)
        model.load_state_dict(checkpoint['model'])
        logger.info(f'Loaded pretrained model from {args.pretrained_path}')

    
    best_cer, best_wer = 1e+6, 1e+6
    train_loss = 0.0

    #### ---- train & eval ---- ####

    for nb_iter in tqdm(range(1, args.total_iter)):
        if args.use_sam:
            optimizer, current_lr = utils.update_lr_cos(nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer, args.min_lr)
            optimizer.zero_grad()
            batch = next(train_iter)
            image = batch[0].cuda()
            text, length = converter.encode(batch[1])
            batch_size = image.size(0)
            loss = compute_loss(args, model, image, batch_size, criterion, text, length)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            compute_loss(args, model, image, batch_size, criterion, text, length).backward()
            optimizer.second_step(zero_grad=True)
            model.zero_grad()
            model_ema.update(model, num_updates=nb_iter / 2)
            train_loss += loss.item()
        else:
            optimizer, current_lr = utils.update_lr_cos(nb_iter, args.warm_up_iter, args.total_iter, args.max_lr, optimizer, args.min_lr)
            optimizer.zero_grad()
            batch = next(train_iter)
            image = batch[0].cuda()

            text, length = converter.encode(batch[1])
            batch_size = image.size(0)

            preds = model(image, args.mask_ratio, args.max_span_length, use_masking=args.use_masking, masking_version=args.mask_version).float()
            preds_size = torch.IntTensor([preds.size(1)] * batch_size).cuda()
            preds = preds.permute(1, 0, 2).log_softmax(2)

            # To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
            torch.backends.cudnn.enabled = False
            loss = criterion(preds, text.cuda(), preds_size, length.cuda()).mean()
            torch.backends.cudnn.enabled = True
            loss.backward()

            optimizer.step()

            model.zero_grad()
            model_ema.update(model, num_updates=nb_iter / 2)
            train_loss += loss.item()
        if nb_iter % args.print_iter == 0:
            train_loss_avg = train_loss / args.print_iter

            logger.info(f'Iter : {nb_iter} \t LR : {current_lr:0.5f} \t training loss : {train_loss_avg:0.5f} \t ' )

            writer.add_scalar('./Train/lr', current_lr, nb_iter)
            writer.add_scalar('./Train/train_loss', train_loss_avg, nb_iter)
            train_loss = 0.0

        if nb_iter % args.eval_iter == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_cer, val_wer, preds, labels = valid.validation(model_ema.ema,
                                                                             criterion,
                                                                             val_loader,
                                                                             converter)
                if val_cer < best_cer:
                    logger.info(f'CER improved from {best_cer:.4f} to {val_cer:.4f}!!!')
                    best_cer = val_cer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_CER.pth'))

                if val_wer < best_wer:
                    logger.info(f'WER improved from {best_wer:.4f} to {val_wer:.4f}!!!')
                    best_wer = val_wer
                    checkpoint = {
                        'model': model.state_dict(),
                        'state_dict_ema': model_ema.ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, 'best_WER.pth'))

                logger.info(
                    f'Val. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} \t ')

                writer.add_scalar('./VAL/CER', val_cer, nb_iter)
                writer.add_scalar('./VAL/WER', val_wer, nb_iter)
                writer.add_scalar('./VAL/bestCER', best_cer, nb_iter)
                writer.add_scalar('./VAL/bestWER', best_wer, nb_iter)
                writer.add_scalar('./VAL/val_loss', val_loss, nb_iter)

                test_loss, test_cer, test_wer, preds, labels = valid.validation(model_ema.ema,
                                                                     criterion,
                                                                     test_loader,
                                                                     converter)

                logger.info(
                    f'Test. loss : {test_loss:0.3f} \t CER : {test_cer:0.4f} \t WER : {test_wer:0.4f} ')
                model.train()


if __name__ == '__main__':
    main()