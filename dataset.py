import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from config import cfg

def label_encoding(train_csv, val_csv):
    labelencoder = LabelEncoder()
    
    df1 = pd.read_csv(train_csv, names=['img', 'label'])
    df2 = pd.read_csv(val_csv, names=['img', 'label'])
    df = df1.append(df2, ignore_index=True)
    df['label2id'] = labelencoder.fit_transform(df['label'])
    df.to_csv('label2id.csv')
    
    size = len(df1)
    df1 = df.iloc[:size, :]
    df2 = df.iloc[size:, :]
    return df1, df2
    
class FaceDatasetFolder(Dataset):
    def __init__(self, root_dir, csv_file):
        super(FaceDatasetFolder, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.img_name, self.label = self.scan()
        
    def scan(self):
        df = self.csv_file.values
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
    
# if __name__ == '__main__':
def get_dataset(args):
    train_csv, val_csv = label_encoding(train_csv = cfg.TRAIN_CSV, val_csv = cfg.VAL_CSV)
    train_dataset = FaceDatasetFolder(root_dir=cfg.IMG_DIR, csv_file=train_csv)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs_mult,
                        shuffle=True, num_workers=0)
    # for i, (sample, label) in enumerate(train_dataloader):
    #     print(sample.size(),
    #             label)
    #     break
    
    val_dataset = FaceDatasetFolder(root_dir=cfg.IMG_DIR, csv_file=val_csv)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs_mult_val,
                        shuffle=True, num_workers=0)
    # for i, (sample, label) in enumerate(val_dataloader):
    #     print(sample.size(),
    #             label)
    #     break
    return train_dataloader, val_dataloader
    