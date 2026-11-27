"""Restorable training batches, including worker prefetch and augmentation.

Only batches delivered to the trainer advance the cursor. Prefetch may run
ahead without changing it. Epoch/index keyed transforms make replay independent
of worker scheduling and avoid consuming the model's dropout RNG.
"""

import hashlib
import random

import numpy as np
import torch

from .. import utils


def _seed(*parts):
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63)


class SeededDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        epoch, index = key
        if isinstance(self.dataset, torch.utils.data.TensorDataset):
            return self.dataset[index]
        seed = _seed(self.seed, epoch, index)
        with utils.fork_global_rng("cpu"):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            return self.dataset[index]


class BatchPlan:
    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.drop_last = True
        self.epoch = self.position = 0

    def __len__(self):
        return self.size // self.batch_size - self.position

    def __iter__(self):
        epoch, start = self.epoch, self.position
        generator = torch.Generator().manual_seed(_seed(self.seed, epoch))
        order = torch.randperm(self.size, generator=generator).tolist()
        for batch in range(start, self.size // self.batch_size):
            yield [
                (epoch, i)
                for i in order[batch * self.batch_size : (batch + 1) * self.batch_size]
            ]

    def state_dict(self):
        return {"epoch": self.epoch, "position": self.position}

    def load_state_dict(self, state):
        self.epoch, self.position = state["epoch"], state["position"]


def prepare_loaders(loaders, generator):
    result, plans = {}, {}
    for endpoint, loader in loaders.items():
        seed = _seed(generator.initial_seed(), endpoint)
        plan = plans[endpoint] = BatchPlan(len(loader.dataset), loader.batch_size, seed)
        result[endpoint] = torch.utils.data.DataLoader(
            SeededDataset(loader.dataset, seed),
            batch_sampler=plan,
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
            generator=generator,
        )
    return result, plans


class TrainingSource:
    def __init__(self, loader, plan, device):
        self.loader, self.plan, self.device = loader, plan, device
        self.iterator = None

    def close(self):
        iterator = self.iterator
        self.iterator = None
        if iterator is not None:
            if hasattr(iterator, "close"):
                iterator.close()
            # Accelerate's generator calls end() only after StopIteration;
            # a consumer stopping mid-pass must release that registration too.
            if hasattr(self.loader, "end"):
                self.loader.end()

    def __next__(self):
        if self.plan.position == self.plan.size // self.plan.batch_size:
            self.plan.epoch += 1
            self.plan.position = 0
            self.close()
        if self.iterator is None:
            # Worker base seeds are irrelevant to index-keyed transforms.
            # Creating an iterator must not move the training/global streams.
            generator = self.loader.generator
            state = generator.get_state()
            with utils.fork_global_rng(self.device):
                self.iterator = iter(self.loader)
                batch = next(self.iterator)
            generator.set_state(state)
        else:
            batch = next(self.iterator)
        self.plan.position += 1
        return batch
