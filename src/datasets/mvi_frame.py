# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger

from PIL import Image

import torch
from torch.utils.data import ConcatDataset, Dataset

from src.datasets.imagenet1k import ImageNet
from src.datasets.video_frames import MultiVideoFirstFrameDataset

logger = getLogger()


class MultiViewFirstFrameDataset(Dataset):
    """Dataset that loads the first frame from each multi-view sample folder."""

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    def __init__(self, view_roots, transform=None):
        self.view_roots = [r for r in view_roots if r]
        self.transform = transform
        self.samples = self._gather_samples()

        if not self.samples:
            raise RuntimeError(
                f'No valid first-frame samples found in view_roots={self.view_roots}'
            )

        logger.info(
            'Initialized MultiViewFirstFrameDataset with %d samples from %d roots',
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
                if not frame_names:
                    continue

                frame_names.sort()
                samples.append(os.path.join(dirpath, frame_names[0]))
                root_count += 1

            logger.info('Collected %d samples from %s', root_count, root)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        frame_path = self.samples[index]
        img = Image.open(frame_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # Keep tuple shape compatible with existing collator/training loop.
        return img, 0


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
    dataset = MultiViewFirstFrameDataset(
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

    logger.info('Multi-view first-frame unsupervised data loader created')
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
    mvi_dataset = MultiViewFirstFrameDataset(
        view_roots=view_roots,
        transform=transform,
    )

    dataset = ConcatDataset([imagenet_dataset, mvi_dataset])
    logger.info(
        'Combined dataset created: ImageNet=%d, multi-view-first-frame=%d, total=%d',
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

    logger.info('ImageNet + multi-view first-frame unsupervised data loader created')
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
    video_dataset = MultiVideoFirstFrameDataset(
        video_roots=video_roots,
        transform=transform,
    )
    mvi_dataset = MultiViewFirstFrameDataset(
        view_roots=view_roots,
        transform=transform,
    )

    dataset = ConcatDataset([imagenet_dataset, video_dataset, mvi_dataset])
    logger.info(
        'Combined dataset created: ImageNet=%d, video-first-frame=%d, '
        'multi-view-first-frame=%d, total=%d',
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
        'ImageNet + video first-frame + multi-view first-frame '
        'unsupervised data loader created'
    )
    return dataset, data_loader, dist_sampler
