"""CIFAR-10 data pipeline: SimCLR-style augmentation for contrastive pairs,
plain normalization for supervised training / evaluation."""
import torchvision.transforms as T
from torch.utils.data import Dataset

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def simclr_augment():
    """SimCLR augmentation recipe used in the paper's CIFAR-10 setup."""
    return T.Compose([
        T.RandomResizedCrop(32, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def eval_transform():
    """Plain (non-augmented) transform for supervised training and evaluation."""
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])


class ContrastivePairDataset(Dataset):
    """Wraps a CIFAR10 dataset of raw PIL images; returns two independent
    augmented views of the same image (a positive pair for InfoNCE)."""

    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self.transform(img), self.transform(img)
