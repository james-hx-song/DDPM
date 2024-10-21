import torchvision

def get_dataset(IMG_SIZE):
    transforms = torchvision.transforms.Compose(
        [
        torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: (x * 2) - 1)
        ]
    )
    dataset = torchvision.datasets.MNIST(root='.', download=True, transform=transforms)
    return dataset
