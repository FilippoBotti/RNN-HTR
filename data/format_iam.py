import os
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import numpy as np
import skimage

class IAMDatasetFormatter:
    def __init__(self, xml_folder, image_folder):
        self.xml_folder = xml_folder
        self.image_folder = image_folder
        self.annotations = self._parse_xml_annotations()

    def _parse_xml_annotations(self):
        """
        Parses IAM XML files and returns a dictionary:
        line_id -> transcription text
        """
        print("Parsing XML annotations...")
        annotations = {}
        for xml_file in os.listdir(self.xml_folder):
            if not xml_file.endswith(".xml"):
                continue
            xml_path = os.path.join(self.xml_folder, xml_file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                handwritten_part = root.find('handwritten-part')
                if handwritten_part is None:
                    continue
                for line in handwritten_part.findall('line'):
                    line_id = line.attrib['id']
                    text = line.attrib.get('text', '').strip()
                    annotations[line_id] = text
            except Exception as e:
                print(f"Error parsing {xml_file}: {e}")
        print(f"Total annotated lines: {len(annotations)}")
        return annotations

    def id_to_image_path(self, line_id):
        """
        Example ID: 'a01-000u-00'
        Path: image_folder/a01/a01-000u/a01-000u-00.png
        """
        line_id_split = line_id.split('-')
        form = line_id_split[0] + '-' + line_id_split[1]  # a01-000u
        writer = line_id_split[0] # a01
        return os.path.join(self.image_folder, writer, form, f"{line_id}.png")

    def load_split_file(self, split_path):
        """
        Reads text file of line IDs
        """
        with open(split_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

class IAMDataset(Dataset):
    def __init__(self, split_file, formatter, img_size=[512, 32], transform=None):
        self.transform = transform
        self.formatter = formatter
        self.ids = formatter.load_split_file(split_file)
        self.samples = self._build_samples()
        self.img_size = img_size

    def _build_samples(self):
        samples = []
        for line_id in self.ids:
            text = self.formatter.annotations.get(line_id, None)
            img_path = self.formatter.id_to_image_path(line_id)
            if text is None:
                print(f"Warning: no text found for {line_id}")
                continue
            if not os.path.exists(img_path):
                print(f"Warning: image not found {img_path}")
                continue
            samples.append((img_path, text))
        print(f"Loaded {len(samples)} valid samples from {len(self.ids)} IDs")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        image = get_images(img_path, max_w=self.img_size[0], max_h=self.img_size[1])
        image = image.transpose((2, 0, 1))  # HWC to CHW
        return (image, text)
    

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

def get_dataloader(split_file, xml_folder, image_folder, batch_size=8, shuffle=True, num_workers=4, img_size=[512, 32]):
    """
    Helper to create a DataLoader directly
    """
    transform = T.Compose([
        T.Resize((128, 1024)),  # Example resizing; adjust as needed
        T.ToTensor(),
    ])
    formatter = IAMDatasetFormatter(xml_folder, image_folder)
    dataset = IAMDataset(split_file, formatter, img_size, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def build_charset(formatter):
    charset = set()
    for line_id, text in formatter.annotations.items():
        charset.update(set(text))
    return ''.join(sorted(charset))

if __name__ == "__main__":
    # Example usage
    xml_folder = "./iam_dataset/xml"
    image_folder = "./iam_dataset/lines"
    train_file = "./iam_dataset/validationset1.txt"

    formatter = IAMDatasetFormatter(xml_folder, image_folder)
    charset = build_charset(formatter)
    print("Character Set:", charset)
    dataloader = get_dataloader(train_file, xml_folder, image_folder, img_size=[512, 32])

    for batch_idx, (images, texts) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print("  Images tensor shape:", images.shape)
        print("  Text example:", texts[0])
        break  # remove this to iterate entire dataset
