import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from collections import Counter

dataset_path = 'data/PlantVillage'

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4591, 0.4753, 0.4116],
        std=[0.1812, 0.1573, 0.1956]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4591, 0.4753, 0.4116],
        std=[0.1812, 0.1573, 0.1956]
    )
])

def get_dataloaders(batch_size=32):
    full_dataset = datasets.ImageFolder(dataset_path)

    total = len(full_dataset)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_set.dataset.transform = train_transform
    val_set.dataset.transform = val_test_transform
    test_set.dataset.transform = val_test_transform

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, full_dataset, train_set

def get_class_weights(full_dataset, train_set, device):
    labels = [full_dataset.targets[i] for i in train_set.indices]
    class_counts = Counter(labels)
    total_samples = len(labels)

    class_weights = []
    for i in range(len(full_dataset.classes)):
        weight = total_samples / (len(full_dataset.classes) * class_counts[i])
        class_weights.append(weight)

    return torch.FloatTensor(class_weights).to(device)