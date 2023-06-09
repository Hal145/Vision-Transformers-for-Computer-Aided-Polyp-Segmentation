from torch.utils.data import Dataset
from PIL import Image


class PolypDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask