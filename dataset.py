import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from config import cfg
    
class FaceDatasetFolder(Dataset):
    def __init__(self, root_dir, csv_file):
        super(FaceDatasetFolder, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.img_name, self.label = self.scan()
        
    def scan(self):
        labelencoder = LabelEncoder()
    
        df = pd.read_csv(self.csv_file, names=['img', 'label'])
        df['label2id'] = labelencoder.fit_transform(df['label'])
        df = df.values
        img_name = df[:, 0]
        label = df[:, 2]
        return img_name, label
    
    def readImage(self,path):
        return cv2.imread(os.path.join(self.root_dir, path))

    def __getitem__(self, index):
        path = self.img_name[index]
        img = self.readImage(path)
        label = self.label[index]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.img_name)

def get_dataset():
    
    train_dataset = FaceDatasetFolder(root_dir=cfg.IMG_DIR, csv_file=cfg.TRAIN_CSV)
    train_dataloader = DataLoader(train_dataset, batch_size=2,#args.bs_mult,
                        shuffle=True, num_workers=0)
    # for i, (sample, label) in enumerate(train_dataloader):
    #     print(sample.size(),
    #             label)
    #     break
    
    val_dataset = FaceDatasetFolder(root_dir=cfg.IMG_DIR, csv_file=cfg.VAL_CSV)
    val_dataloader = DataLoader(val_dataset, batch_size=1,#args.bs_mult_val,
                        shuffle=True, num_workers=0)
    # for i, (sample, label) in enumerate(val_dataloader):
    #     print(sample.size(),
    #             label)
    #     break
    return train_dataloader, val_dataloader

if __name__ == '__main__':
    get_dataset()
    