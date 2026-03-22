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

logger = getLogger()


class MultiVideoFirstFrameDataset(Dataset):
    """Dataset that loads the first frame from each per-video frame folder."""

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    def __init__(self, video_roots, transform=None):
        self.video_roots = [r for r in video_roots if r]
        self.transform = transform
        self.samples = self._gather_samples()

        if not self.samples:
            raise RuntimeError(
                f'No valid first-frame samples found in video_roots={self.video_roots}'
            )

        logger.info(
            'Initialized MultiVideoFirstFrameDataset with %d samples from %d roots',
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

                first_frame = self._find_first_frame(entry.path)
                if first_frame is None:
                    continue

                samples.append(first_frame)
                root_count += 1

            logger.info('Collected %d samples from %s', root_count, root)

        return samples

    def _find_first_frame(self, folder_path):
        frame_names = [
            f.name for f in os.scandir(folder_path)
            if f.is_file() and f.name.lower().endswith(self.IMG_EXTENSIONS)
        ]
        if not frame_names:
            return None

        frame_names.sort()
        return os.path.join(folder_path, frame_names[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        frame_path = self.samples[index]
        img = Image.open(frame_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # Keep tuple shape compatible with existing collator/training loop.
        return img, 0


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
    dataset = MultiVideoFirstFrameDataset(
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

    logger.info('Video first-frame unsupervised data loader created')
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
    video_dataset = MultiVideoFirstFrameDataset(
        video_roots=video_roots,
        transform=transform,
    )

    dataset = ConcatDataset([imagenet_dataset, video_dataset])
    logger.info(
        'Combined dataset created: ImageNet=%d, video-first-frame=%d, total=%d',
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

    logger.info('ImageNet + video first-frame unsupervised data loader created')
    return dataset, data_loader, dist_sampler
