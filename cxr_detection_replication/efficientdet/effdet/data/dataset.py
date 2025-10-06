""" Detection dataset

Hacked together by Ross Wightman
"""
import torch.utils.data as data
import numpy as np
import torchvision.utils
import torchvision.transforms.functional as TF
from PIL import Image
from .parsers import create_parser
import torch

class DetectionDatset(data.Dataset):
    """`Object Detection Dataset. Use with parsers for COCO, VOC, and OpenImages.
    Args:
        parser (string, Parser):
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``

    """

    def __init__(self, data_dir, parser=None, parser_kwargs=None, transform=None):
        super(DetectionDatset, self).__init__()
        parser_kwargs = parser_kwargs or {}
        self.data_dir = data_dir
        if isinstance(parser, str):
            self._parser = create_parser(parser, **parser_kwargs)
        else:
            assert parser is not None and len(parser.img_ids)
            self._parser = parser
        self._transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, annotations (target)).
        """
        img_info = self._parser.img_infos[index]
        target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']))
        if self._parser.has_labels:
            ann = self._parser.get_ann_info(index)
            target.update(ann)

        img_path = self.data_dir / img_info['file_name']
        img = Image.open(img_path)
        # print(f"[Dataset] BEFORE transform: {img_path}, type: {type(img)}, mode: {img.mode}")
        if img.mode == 'I':
            arr = np.array(img).astype(np.float32)
            arr = 255 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-5)
            arr = arr.astype(np.uint8)
            img = Image.fromarray(arr, mode='L')
            img = img.convert('RGB')
            # print(f"[Dataset] Normalized and converted 'I' mode image to RGB for {img_path}")
        if self.transform is not None:
            img, target = self.transform(img, target)
        # print(f"[Dataset] AFTER transform: {img_path}, type: {type(img)}, mode: {getattr(img, 'mode', 'n/a')}")
        # ADD THIS:
        # Save only the first valid sample
        if index == 0:
            if not isinstance(img, torch.Tensor):
                img_tensor = TF.to_tensor(img)  # PIL → FloatTensor [0,1], C×H×W
            else:
                img_tensor = img
            if img_tensor.ndim == 2:
                img_tensor = img_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
            if img_tensor.shape[0] == 1:
                img_tensor = img_tensor.repeat(3, 1, 1)  # (1, H, W) -> (3, H, W)
            if img_tensor.dtype != torch.float32:
                img_tensor = img_tensor.float().div(255)
            # Only save if shape is valid
            if img_tensor.ndim == 3 and img_tensor.shape[0] == 3:
                torchvision.utils.save_image(img_tensor, "debug_raw_after_transform.png", normalize=True)
            else:
                print(f"[Debug Save] Skipping image with shape {img_tensor.shape} for {img_path}")
        return img, target

    def __len__(self):
        return len(self._parser.img_ids)

    @property
    def parser(self):
        return self._parser

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, t):
        self._transform = t


class SkipSubset(data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        n (int): skip rate (select every nth)
    """
    def __init__(self, dataset, n=2):
        self.dataset = dataset
        assert n >= 1
        self.indices = np.arange(len(dataset))[::n]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    @property
    def parser(self):
        return self.dataset.parser

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, t):
        self.dataset.transform = t
