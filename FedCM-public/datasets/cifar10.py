import numpy as np
from torch.utils.data import DataLoader, Dataset


class ClientImages(Dataset):
    def __init__(self, dataset, indices, rotation):
        self.dataset = dataset
        self.indices = indices
        self.rotation = rotation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[int(self.indices[index])]
        return image.rot90(self.rotation, dims=(-2, -1)), label


def cifar10_loaders(root, rotations, rng):
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor

    def split(train):
        dataset = CIFAR10(root=root, train=train, download=True, transform=ToTensor())
        shards = np.array_split(rng.permutation(len(dataset)), len(rotations))
        return [DataLoader(ClientImages(dataset, shard, rotation), batch_size=128, shuffle=train)
                for shard, rotation in zip(shards, rotations)]

    return split(train=True), split(train=False)
