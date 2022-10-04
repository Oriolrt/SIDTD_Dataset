import cv2
from torch.utils.data import DataLoader, Dataset 

from albumentations import Compose, Normalize, Resize 
from albumentations.pytorch import ToTensorV2 
from albumentations import RandomCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, CenterCrop, PadIfNeeded, RandomResizedCrop


def get_transforms(WIDTH, HEIGHT, mean, std, data):
    assert data in ('train', 'valid')
    
    if data == 'train':
        return Compose([
        RandomResizedCrop(WIDTH, HEIGHT, scale=(0.8, 1.0)),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomBrightnessContrast(p=0.2),
        Normalize(mean=mean, std=std),
        ToTensorV2(),
        ])
    
    elif data == 'valid':
        return Compose([
        Resize(WIDTH, HEIGHT),
        Normalize(mean=mean, std=std),
        ToTensorV2(),
        ]) 


class TrainDatasetConditionned(Dataset):
    def __init__(self, paths, ids, meta_class, transform=None, dataset_crop=False):
        self.paths = paths
        self.ids = ids
        self.meta_class = meta_class
        self.transform = transform
        self.dataset_crop = dataset_crop
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        file_path = self.paths[idx]
        id_meta_class = self.meta_class[idx]
        label = self.ids[idx]
        image = cv2.imread(file_path)
    
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(file_path)
        if self.dataset_crop:
            sx, sy, _ = image.shape
            sx2 = int(sx/2)
            sy2 = int(sy/2)
            image = image[sx2-250:sx2+250, sy2-250:sy2+250, :]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        img_meta_class = np.ones((image.shape[0], image.shape[1], 1)) * id_meta_class
        image = np.concatenate((image, img_meta_class), axis=2)
    
        return image, label
		

class TrainDataset(Dataset):

    def __init__(self, paths, ids, transform=None):
        self.paths = paths
        self.ids = ids
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        file_path = self.paths[idx]
        label = self.ids[idx]
        image = cv2.imread(file_path)
        #if the file path does not exist cv2 image is an empty variable
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(file_path)
           
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
    
        return image, label