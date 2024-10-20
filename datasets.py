import torchvision

def get_dataset(IMG_SIZE):
    transforms = torchvision.transforms.Compose(
        [
        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.ToTensor(),
        ]
    )
    dataset = torchvision.datasets.CIFAR100(root='.', download=True, transform=transforms)
    return dataset
