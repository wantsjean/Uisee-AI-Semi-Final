import numpy as np
import torch
from torch.utils.data import Dataset
import os
import cv2 as cv

os.environ['DISPLAY'] = ':10'


class THOLoader():
    def __init__(self,
                 train_folder,
                 eval_folder,
                 label_path,
                 batch_size=4, 
                 num_workers=4, 
                 distributed=False):

        self.loaders = {
            "train": torch.utils.data.DataLoader(
                THODataset(
                    data_dir=train_folder,
                    label_path=label_path,
                    mode="train"),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=True
            ),
            "eval": torch.utils.data.DataLoader(
                THODataset(
                    data_dir=eval_folder,
                    label_path=label_path,
                    mode="eval"),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False
            )}


class THODataset(Dataset):
    def __init__(self, data_dir, label_path, mode="train"):
        self.data_dir = data_dir
        self.mode = mode
        self.file_paths = []
        self.angle = []
        self.speed = []
        self.pred_speed = []
        with open(label_path, 'r') as f:
            raw_data = f.read().strip().split("\n")
            count = 0
            for data in raw_data:
                data = data.split()
                self.file_paths.append(os.path.join(data_dir, data[1]+".png"))
                self.angle.append(float(data[-2]))
                self.speed.append(float(data[-1]))
                if count != len(raw_data)-1:
                    self.pred_speed.append(
                        float(raw_data[count+1].split()[-1]))
                else:
                    self.pred_speed.append(float(raw_data[count].split()[-1]))
                count += 1

            # for i in range(9000,len(raw_data)):
            #     img = cv.imread(self.file_paths[i])
            #     print(self.file_paths[i])
            #     cv.imshow("123",img)
            #     cv.waitKey(20)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = cv.imread(self.file_paths[idx])
        img = torch.from_numpy(img)
        speed = torch.tensor(self.speed[idx]).float() / 50
        target_speed = torch.tensor(self.pred_speed[idx]).float() / 50
        target_angle = torch.tensor(self.angle[idx]).float() / 10
        return img, speed, target_speed, target_angle


if __name__ == "__main__":
    dataset = THODataset('dataset/imgs', 'dataset/labels.txt')

    loader = THOLoader('dataset/imgs', 'dataset/imgs',
                       'dataset/labels.txt', batch_size=16, num_workers=0)

    for img, speed, pred_speed, pred_angle in loader.loaders['train']:
        print(img.size())
        print(speed, pred_speed, pred_angle)
