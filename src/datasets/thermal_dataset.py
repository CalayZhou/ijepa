import glob
import os

from PIL import Image
from torch.utils.data import Dataset


def mysort(dataset_path, in1k=False):
    types = ['*.jpg', '*.jpeg', '*.JPEG', '*.png', '*.bmp', '*.tif']
    dataset_img_path = []
    for img_type in types:
        dataset_img_path.extend(glob.glob(os.path.join(dataset_path, img_type)))
    if not in1k:
        try:
            dataset_img_path = sorted(
                dataset_img_path,
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        except Exception:
            print(f'The name of images in {dataset_path} is not numerical!')
    return dataset_img_path


class ThermalDataset(Dataset):

    def __init__(self, dataset_path, transform=None, in1k=False):
        self.dataset_path = dataset_path
        self.transform = transform
        self.dataset_img_path = mysort(dataset_path=dataset_path, in1k=in1k)

    def __len__(self):
        return len(self.dataset_img_path)

    def __getitem__(self, idx):
        img_path = self.dataset_img_path[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
