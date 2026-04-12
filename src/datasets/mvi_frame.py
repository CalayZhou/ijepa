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
from src.datasets.video_frames import (
    MultiVideoFramePairDataset,
    SiamImageNetAdapter,
)

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


class MultiViewFramePairDataset(Dataset):
    """Dataset that loads a random ordered frame pair from each sample folder."""

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    def __init__(self, view_roots, transform=None):
        self.view_roots = [r for r in view_roots if r]
        self.transform = transform
        self.samples = self._gather_samples()

        if not self.samples:
            raise RuntimeError(
                f'No valid frame-pair samples found in view_roots={self.view_roots}'
            )

        logger.info(
            'Initialized MultiViewFramePairDataset with %d samples from %d roots',
            len(self.samples),
            len(self.view_roots),
        )

    def _gather_samples(self):
        samples = []
        for root in self.view_roots:
            if not os.path.isdir(root):
                logger.warning('Skipping missing multi-view root: %s', root)
                continue

            root_count = 0
            for dirpath, _, filenames in os.walk(root):
                frame_names = [
                    name for name in filenames
                    if name.lower().endswith(self.IMG_EXTENSIONS)
                ]
                if len(frame_names) < 2:
                    continue

                frame_names.sort()
                samples.append([os.path.join(dirpath, fn) for fn in frame_names])
                root_count += 1

            logger.info('Collected %d samples from %s', root_count, root)

        return samples

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


def make_mvi_frame_loader(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    view_roots=None,
    drop_last=True,
):
    view_roots = view_roots or []
    dataset = MultiViewFramePairDataset(
        view_roots=view_roots,
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

    logger.info('Multi-view frame-pair unsupervised data loader created')
    return dataset, data_loader, dist_sampler


def make_imagenet_and_mvi_frame_loader(
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
    view_roots=None,
    drop_last=True,
):
    view_roots = view_roots or []

    imagenet_dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=True,
        copy_data=copy_data,
        index_targets=False,
    )
    mvi_dataset = MultiViewFramePairDataset(
        view_roots=view_roots,
        transform=transform,
    )
    imagenet_dataset = SiamImageNetAdapter(imagenet_dataset)

    dataset = ConcatDataset([imagenet_dataset, mvi_dataset])
    logger.info(
        'Combined dataset created: ImageNet=%d, multi-view-frame-pair=%d, total=%d',
        len(imagenet_dataset),
        len(mvi_dataset),
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

    logger.info('ImageNet + multi-view frame-pair unsupervised data loader created')
    return dataset, data_loader, dist_sampler


def make_imagenet_video_and_mvi_frame_loader(
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
    view_roots=None,
    drop_last=True,
):
    video_roots = video_roots or []
    view_roots = view_roots or []

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
    mvi_dataset = MultiViewFramePairDataset(
        view_roots=view_roots,
        transform=transform,
    )
    imagenet_dataset = SiamImageNetAdapter(imagenet_dataset)

    dataset = ConcatDataset([imagenet_dataset, video_dataset, mvi_dataset])
    logger.info(
        'Combined dataset created: ImageNet=%d, video-frame-pair=%d, '
        'multi-view-frame-pair=%d, total=%d',
        len(imagenet_dataset),
        len(video_dataset),
        len(mvi_dataset),
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

    logger.info(
        'ImageNet + video frame-pair + multi-view frame-pair '
        'unsupervised data loader created'
    )
    return dataset, data_loader, dist_sampler


def make_video_and_mvi_frame_loader(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    video_roots=None,
    view_roots=None,
    drop_last=True,
):
    video_roots = video_roots or []
    view_roots = view_roots or []

    video_dataset = MultiVideoFramePairDataset(
        video_roots=video_roots,
        transform=transform,
    )
    mvi_dataset = MultiViewFramePairDataset(
        view_roots=view_roots,
        transform=transform,
    )

    dataset = ConcatDataset([video_dataset, mvi_dataset])
    logger.info(
        'Combined dataset created: video-frame-pair=%d, '
        'multi-view-frame-pair=%d, total=%d',
        len(video_dataset),
        len(mvi_dataset),
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

    logger.info(
        'Video frame-pair + multi-view frame-pair '
        'unsupervised data loader created'
    )
    return dataset, data_loader, dist_sampler
