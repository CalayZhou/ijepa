# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import random
from logging import getLogger

from PIL import Image

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from src.datasets.imagenet1k import ImageNet

logger = getLogger()


def _apply_shared_transform(img1, img2, transform):
    if transform is None:
        return img1, img2

    seed = torch.randint(0, 2**31, (1,)).item()
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))
        random.seed(seed)
        out1 = transform(img1)

        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))
        random.seed(seed)
        out2 = transform(img2)
    finally:
        torch.set_rng_state(torch_state)
        np.random.set_state(np_state)
        random.setstate(py_state)

    return out1, out2


class MultiVideoFramePairDataset(Dataset):
    """Dataset that loads a random ordered frame pair from each per-video folder."""

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    def __init__(self, video_roots, transform=None):
        self.video_roots = [r for r in video_roots if r]
        self.transform = transform
        self.samples = self._gather_samples()

        if not self.samples:
            raise RuntimeError(
                f'No valid frame-pair samples found in video_roots={self.video_roots}'
            )

        logger.info(
            'Initialized MultiVideoFramePairDataset with %d samples from %d roots',
            len(self.samples),
            len(self.video_roots)
        )

    def _gather_samples(self):
        samples = []
        for root in self.video_roots:
            if not os.path.isdir(root):
                logger.warning('Skipping missing video root: %s', root)
                continue

            root_count = 0
            for entry in os.scandir(root):
                if not entry.is_dir():
                    continue

                frame_paths = self._find_frames(entry.path)
                if frame_paths is None:
                    continue

                samples.append(frame_paths)
                root_count += 1

            logger.info('Collected %d samples from %s', root_count, root)

        return samples

    def _find_frames(self, folder_path):
        frame_paths = [
            os.path.join(folder_path, f.name) for f in os.scandir(folder_path)
            if f.is_file() and f.name.lower().endswith(self.IMG_EXTENSIONS)
        ]
        if len(frame_paths) < 2:
            return None

        frame_paths.sort()
        return frame_paths

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        frame_paths = self.samples[index]
        first_idx = torch.randint(0, len(frame_paths) - 1, (1,)).item()
        second_idx = torch.randint(first_idx + 1, len(frame_paths), (1,)).item()
        img1 = Image.open(frame_paths[first_idx]).convert('RGB')
        img2 = Image.open(frame_paths[second_idx]).convert('RGB')

        img1, img2 = _apply_shared_transform(img1, img2, self.transform)

        return img1, img2, 0


class SiamImageNetAdapter(Dataset):
    """Adapter that turns single images into Siam-style (img1, img2, target)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        return img, img, target


def make_video_frame_loader(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    video_roots=None,
    drop_last=True,
):
    video_roots = video_roots or []
    dataset = MultiVideoFramePairDataset(
        video_roots=video_roots,
        transform=transform,
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
    )

    logger.info('Video frame-pair unsupervised data loader created')
    return dataset, data_loader, dist_sampler


def make_imagenet_and_video_frame_loader(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    copy_data=False,
    video_roots=None,
    drop_last=True,
):
    video_roots = video_roots or []

    imagenet_dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=True,
        copy_data=copy_data,
        index_targets=False,
    )
    video_dataset = MultiVideoFramePairDataset(
        video_roots=video_roots,
        transform=transform,
    )
    imagenet_dataset = SiamImageNetAdapter(imagenet_dataset)

    dataset = ConcatDataset([imagenet_dataset, video_dataset])
    logger.info(
        'Combined dataset created: ImageNet=%d, video-frame-pair=%d, total=%d',
        len(imagenet_dataset),
        len(video_dataset),
        len(dataset),
    )

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
    )

    logger.info('ImageNet + video frame-pair unsupervised data loader created')
    return dataset, data_loader, dist_sampler
