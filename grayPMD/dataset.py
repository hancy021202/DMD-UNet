import glob
import os
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
from torch.utils.data import Dataset
import groundtruth


class data_Loader_all(Dataset):
    def __init__(self, data_img, data_label1, data_label2, data_label3):
        # 初始化函数，读取所有data_path下的图片
        self.len = len(data_img)
        self.img1 = torch.from_numpy(data_img)
        self.img1 = self.img1.type(torch.FloatTensor).cuda()
        self.img2 = torch.from_numpy(data_img)
        self.img2 = self.img2.type(torch.FloatTensor).cuda()
        self.img3 = torch.from_numpy(data_img)
        self.img3 = self.img3.type(torch.FloatTensor).cuda()
        self.label1 = torch.from_numpy(data_label1)
        self.label1 = self.label1.type(torch.FloatTensor).cuda()
        self.label2 = torch.from_numpy(data_label2)
        self.label2 = self.label2.type(torch.FloatTensor).cuda()
        self.label3 = torch.from_numpy(data_label3)
        self.label3 = self.label3.type(torch.FloatTensor).cuda()
        # assert len(self.imgs_path) == len(self.label_path)  # 检查图片数量是否合理,一定要相等！！！

    def __getitem__(self, index):
        return self.img1[index], self.img2[index], self.img3[index], self.label1[index], self.label2[index], self.label3[index],

    def __len__(self):
        # 返回训练集大小
        return self.len


class data_Loader_one(Dataset):
    def __init__(self, data_img, data_label1):
        # 初始化函数，读取所有data_path下的图片
        self.len = len(data_img)
        self.img1 = torch.from_numpy(data_img)
        self.img1 = self.img1.type(torch.FloatTensor).cuda()
        # self.img2 = torch.from_numpy(data_img)
        # self.img2 = self.img2.type(torch.FloatTensor).cuda()
        # self.img3 = torch.from_numpy(data_img)
        # self.img3 = self.img3.type(torch.FloatTensor).cuda()
        self.label1 = torch.from_numpy(data_label1)
        self.label1 = self.label1.type(torch.FloatTensor).cuda()
        # self.label2 = torch.from_numpy(data_label2)
        # self.label2 = self.label2.type(torch.FloatTensor).cuda()
        # self.label3 = torch.from_numpy(data_label3)
        # self.label3 = self.label3.type(torch.FloatTensor).cuda()
        # assert len(self.imgs_path) == len(self.label_path)  # 检查图片数量是否合理,一定要相等！！！

    def __getitem__(self, index):
        return self.img1[index],  self.label1[index]

    def __len__(self):
        # 返回训练集大小
        return self.len
