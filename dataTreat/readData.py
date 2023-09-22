import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import os,cv2

def image2label(image, colormap):
    # 将标签转化为每个像素值为一类数据
    cm2lbl = np.zeros(256**3)
    for i,cm in enumerate(colormap):
        cm2lbl[(cm[0]*256+cm[1]*256+cm[2])] = i
    # 对一张图像转换
    image = np.array(image, dtype="int64")
    ix = (image[:,:,0]*256+image[:,:,1]*256+image[:,:,2])
    image2 = cm2lbl[ix]
    return image2

# 定义需要读取的数据路径的函数
def read_image_path(img_dir,label_dir):
# 原始图像路径输出为data，标签图像路径输出为label
    n=len(os.listdir(img_dir))
    data, label = [None]*n, [None]*n
    for i,fname in enumerate(os.listdir(img_dir)):
        # data[i] = r"C:\Users\22476\Desktop\VOC_format_dataset\JPEGImages\%s.jpg" % (fname)
        # label[i] = r"C:\Users\22476\Desktop\VOC_format_dataset\Segmentation_40\%s.png" % (fname)
        data[i] = os.path.join(img_dir,fname)
        img=cv2.imread(data[i])

        # try:
        #     img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
        #     cv2.imwrite(data[i], img)
        # except Exception as e:
        #     print(e)



    for i,fname in enumerate(os.listdir(label_dir)):
        label[i] = os.path.join(label_dir,fname)
        lab=cv2.imread(label[i])

        try:
            # lab = cv2.cvtColor(lab, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(label[i], lab)
        except Exception as e:
            print(e)


    return data, label

# 单组图像的转换操作
def img_transforms(data, label,mean,std):
# 将输入图像进行亮度、对比度、饱和度、色相的改变；将标记图像数据进行二维标签化的操作
# 输出原始图像和类别标签的张量数据
    if mean == None or std == None:
        composes=[transforms.ToTensor(),]
    else:
        composes=[transforms.ToTensor(),
                  transforms.Normalize(mean,std)]


    data_tfs = transforms.Compose(composes)
    data = data_tfs(data)
    label = data_tfs(label)
    return data, label

class LzgdDataset(Dataset):
    def __init__(self, img_dir,label_dir, img_transforms,mean=None,std=None):
        self.img_dir = img_dir
        self.label_dir=label_dir
        self.img_transforms = img_transforms
        data_list, label_list = read_image_path(self.img_dir,self.label_dir)
        self.data_list = data_list
        self.label_list = label_list
        self.mean=mean
        self.std=std

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        img = self.data_list[item]
        label = self.label_list[item]

        img = cv2.imread(img)

        # label = cv2.imread(label,cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label)

        img, label = self.img_transforms(img, label,self.mean,self.std)

        return img,label

class LzgdTestDataset(Dataset):
    def __init__(self,img_dir):
        self.img_dir = img_dir
        self.data_list=[]
        for i, fname in enumerate(os.listdir(img_dir)):
            self.data_list.append(os.path.join(img_dir, fname))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        img = self.data_list[item]

        img = cv2.imread(img)

        data_tfs = transforms.Compose([
            transforms.ToTensor(),
        ])
        img = data_tfs(img)
        return img