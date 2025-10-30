import os
import json
import torch
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import numpy as np
import skimage

def npThum(img, max_w, max_h):
    x, y = np.shape(img)[:2]

    y = min(int(y * max_h / x), max_w)
    x = max_h

    img = np.array(Image.fromarray(img).resize((y, x)))
    return img

def get_images(fname, max_w=500, max_h=500, nch=1):  # args.max_w args.max_h args.nch

    try:

        image_data = np.array(Image.open(fname).convert('L'))
        image_data = npThum(image_data, max_w, max_h)
        image_data = skimage.img_as_float32(image_data)

        h, w = np.shape(image_data)[:2]
        if image_data.ndim < 3:
            image_data = np.expand_dims(image_data, axis=-1)

        if nch == 3 and image_data.shape[2] != 3:
            image_data = np.tile(image_data, 3)

        image_data = np.pad(image_data, ((0, 0), (0, max_w - np.shape(image_data)[1]), (0, 0)), mode='constant',
                            constant_values=(1.0))

    except IOError as e:
        print('Could not read:', fname, ':', e)

    return image_data

class LAM(Dataset):
    def __init__(self, db_path, split, transforms, nameset='train', img_size=[512, 32], charset=None):
        set_path = os.path.join(db_path, 'lines', 'split', split, f'{nameset}.json')
        assert os.path.exists(set_path)
        self.img_size = img_size
        
        with open(set_path, 'r') as f:
            self.samples = json.load(f)
        self.transforms = transforms
        self.db_path = db_path
        self.imgs_path = os.path.join(db_path, 'lines', 'img')

        if charset is None:
            labels = [sample['text'] for sample in self.samples]
            charset = sorted(set(''.join(labels)))

        self.charset = charset
        self.char_to_idx = dict(zip(self.charset, range(len(self.charset))))
        self.idx_to_char = dict(zip(range(len(self.charset)), self.charset))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        img_name, text, decade_id = sample['img'], sample['text'], sample['decade_id']

        # img = Image.open(os.path.join(self.imgs_path, img_name))
        # if self.transforms is not None:
        #     img = self.transforms(img)
        # print(self.img_size)
        img = get_images(os.path.join(self.imgs_path, img_name), self.img_size[0], self.img_size[1])
        img = img.transpose((2, 0, 1))
        return img, text #, decade_id, img_name

    def collate_fn(self, samples):
        imgs = [sample[0] for sample in samples]
        texts = [sample[1] for sample in samples]
        # decades = [sample[2] for sample in samples]
        # names = [sample[3] for sample in samples]

        out_width = max([img.shape[-1] for img in imgs])
        out_height = max([img.shape[-2] for img in imgs])

        imgs = [F.pad(torch.tensor(img), pad=(0, out_width - img.shape[-1], 0, out_height - img.shape[-2])) for img in imgs]
        return torch.stack(imgs).float(), texts#, decades, names

if __name__ == '__main__':
    lam_path = './'

    # LAM(lam_path, 'leave_decade_out/leave_decade_1_out', ToTensor(), nameset='train')
    train_dataset = LAM(lam_path, 'basic', ToTensor(), nameset='train')
    valid_dataset = LAM(lam_path, 'basic', ToTensor(), nameset='val', charset=train_dataset.charset)
    test_dataset  = LAM(lam_path, 'basic', ToTensor(), nameset='test', charset=train_dataset.charset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True,
                                               num_workers=0, collate_fn=train_dataset.collate_fn)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True, pin_memory=True,
                                               num_workers=0, collate_fn=valid_dataset.collate_fn)

    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True, pin_memory=True,
                                               num_workers=0, collate_fn=test_dataset.collate_fn)

    for idx, (img, texts, decades, names) in enumerate(train_loader):
        # do stuff
        print('train', idx, len(train_loader), texts)
        exit()

    for idx, (img, texts, decades, names) in enumerate(valid_loader):
        # do stuff
        print('valid', idx, len(valid_loader), texts)

    for idx, (img, texts, decades, names) in enumerate(test_loader):
        # do stuff
        print('test', idx, len(test_loader), texts)

