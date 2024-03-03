import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y


# Assuming 'features' and 'labels' are your data and labels, respectively
# You would typically load these from your dataset files

# Define a transform if you need to preprocess your data (e.g., normalization)
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add more transformations as needed
])

# Create an instance of CustomDataset
dataset = CustomDataset(features, labels, transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Now you can iterate over the DataLoader in your training loop
