import pandas as pd
from torch.utils.data import Dataset
import os
from torchvision.io import read_image

from torch.utils.data import DataLoader


class MyData(Dataset):
    """
    Args:
        csv_file: 图像名及标签csv文件存放
        例如：tshirt1.jpg, 0
             tshirt2.jpg, 0
        img_dir：数据存放的路径
        transform: 是否对图像进行转换
        target_transform: 是否对图像进行转换
    """

    def __init__(self, csv_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        # os.path.join用于拼接路径
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[item, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[item, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.img_labels)


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)