from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class DatasetFromImagesPaths(Dataset):
    ### given a list of image_paths, it forms a dataset 
    ### that can be given to a DataLoader
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        return transforms.ToTensor()(img), 0.


class DatasetFromImages(Dataset):
    ### given a list of images, it forms a dataset 
    ### that can be given to a DataLoader
    def __init__(self, inputs, labels=None) -> None:
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (self.inputs[idx], 
            0 if self.labels is None else self.labels[idx])


def wrap_into_dataloader(inputs, labels, batch_size, pin_memory=True, workers=30, shuffle=False):
    ds = DatasetFromImages(inputs, labels)
    return DataLoader(ds, batch_size=batch_size, 
        num_workers=workers, pin_memory=pin_memory, shuffle=shuffle)

