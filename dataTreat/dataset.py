from PIL import Image
import torch
import os
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import pathlib,numpy as np,cv2 as cv
#
class HouseDataset(Dataset):
    def __init__(self,root_dir,label_dir,is_transform=True):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.image_path_list=os.listdir(self.path)
        self.is_transform=is_transform

    def __getitem__(self, idx):
        img_name=self.image_path_list[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        if self.is_transform:
            img=trans(img)
            img=torch.squeeze(img)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.image_path_list)



# class HouseDataset(Dataset):
#     def __init__(self,root_dir,mean,std,is_transform=True):
#         data_root = pathlib.Path(root_dir)
#         all_image_paths = list(data_root.glob('*/*'))
#         self.all_image_paths = [str(path) for path in all_image_paths]
#         label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
#         label_to_index = dict((label, index) for index, label in enumerate(label_names))
#         self.all_image_labels = [label_to_index[path.parent.name] for path in all_image_paths]
#         self.mean = np.array(mean).reshape((1, 1, 3))
#         self.std = np.array(std).reshape((1, 1, 3))
#         self.is_transform=is_transform
#
#     def __getitem__(self, index):
#         img = cv.imread(self.all_image_paths[index])
#         # img = cv.resize(img, (224, 224))
#         img = img / 255.
#         img = (img - self.mean) / self.std
#         img = np.transpose(img, [2, 0, 1])
#         label = self.all_image_labels[index]
#         img = torch.tensor(img, dtype=torch.float32)
#         label = torch.tensor(label)
#         return img, label
#
#     def __len__(self):
#         return len(self.all_image_labels)



# trans = transforms.ToTensor()
trans=transforms.Compose([
    transforms.ToTensor(),  # 图片转张量，同时归一化0-255 ---》 0-1
    # transforms.Normalize(norm_mean, norm_std),  # 标准化均值为0标准差为1
])