import sys
sys.path.append('../../../pytorch-image-models-master/')
# 图片添加到搜索路径
from tqdm import tqdm
# 进度条库
import math
import random
import os
import pandas as pd
import numpy as np
from torch.nn import Parameter
# Visuals and CV2
import cv2

# albumentations for augs 图像增强
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

#torch
import torch
import timm 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader


import gc
# 垃圾回收
import matplotlib.pyplot as plt

from data import infer_ImgDataset
from model import model as _ImgNet
# import cudf
# import cuml
# import cupy
# from cuml.feature_extraction.text import TfidfVectorizer
# from cuml import PCA
# from cuml.neighbors import NearestNeighbors

import argparse

# 1. 定义命令行解析器对象
parser = argparse.ArgumentParser(description='Demo of argparse')

# 2. 添加命令行参数
parser.add_argument('--dimension', type=int, default=512)
parser.add_argument('--batch', type=int, default=64)
parser.add_argument('-m', '--model_path', required=True, help='训练模型路径')
parser.add_argument('-t', '--test_csv_path', required=True, help='测试集文件路径')
parser.add_argument('-i', '--test_image_path', required=True, help='测试集图片路径')

args = parser.parse_args()

NUM_WORKERS = 0
BATCH_SIZE = args.batch
# SEED = 2020？

device = torch.device('cuda')

# 注意修改！！！！！！！！
CLASSES = 12039
# CLASSES = 145？

################################################  ADJUSTING FOR CV OR SUBMIT ##############################################
# CHECK_SUB = False
# GET_CV = True
################################################# MODEL ####################################################################

model_name = 'efficientnet_b3' #efficientnet_b0-b7

################################################ MODEL PATH ###############################################################

# 修改
epoch = 50
weidu = args.dimension
DIM = (512, 512)
IMG_MODEL_PATH = args.model_path
# IMG_MODEL_PATH = r'C:\Users\Administrator\Documents\Tencent Files\2174661138\FileRecv\model_efficientnet_b3_IMG_SIZE_512_arcface (1).bin'

################################################ Metric Loss and its params #######################################################
# 针对img test
loss_module = 'arcface' # 'cosface' #'adacos' arcface

# 针对img_1w训练集
# loss_module = IMG_MODEL_PATH.split('_')[6]
# 修改
# s = 30
s = 15
# m = 0.5
????
ls_eps = 0.0
easy_margin = False


def get_test_transforms():

    return albumentations.Compose(
        [
            albumentations.Resize(DIM[0],DIM[1],always_apply=True),
            albumentations.Normalize(),
        ToTensorV2(p=1.0)
        ]
    )
# Resize：将图像的尺寸缩放到指定的大小。在这里，将图像缩放到了 DIM[0]×DIM[1] 的尺寸。

# Normalize：对图像进行标准化处理。在这里，采用了默认参数进行标准化，即对每个像素减去0.485，然后除以0.229。

# ToTensorV2：将图像转换为 PyTorch 中的张量形式。


def get_image_embeddings(image_paths):
#     Step 0. 预处理：图片全部编码为向量储存
# Step 1. 获取目标图片的嵌入向量
# Step 2. 在向量数据库中找到距离最近的向量
# Step 3. 根据搜索的结果，返回对应id的图片
# 加载预训练的图像分类模型，并将其设置为评估模式；
# 加载测试集图像，并将其转换为嵌入向量；
# 将所有嵌入向量拼接起来，返回一个大的矩阵。
    embeds = []

    model = _ImgNet.ImgNet(n_classes=CLASSES, model_name=model_name)
    model.eval()

    model.load_state_dict(torch.load(IMG_MODEL_PATH), strict=False)
    model = model.to(device)
# 在这段代码中，model.ImgNet 是一个自定义的 PyTorch 模型类，它包含一个卷积神经网络和一个全连接层，用于将输入的图像数据转换成一个向量。该模型的参数需要在训练过程中进行优化，以最小化损失函数并提高模型的性能。

# 根据代码，model.ImgNet 类的初始化参数中有一个 n_classes 参数，用于指定模型最后的输出类别数。在这里，CLASSES 是一个全局变量，指定了类别数。model_name 参数用于指定模型的名称，以便加载预训练的权重。

# 该模型的前向传播过程需要输入图像数据和标签。在代码中，这些数据是由 infer_ImgDataset.ImgDataset 加载器提供的，
# 该加载器基于 torch.utils.data.Dataset 类定义，并返回一对元组，包含输入图像和标签。在 model.ImgNet 的前向传播过程中，图像数据经过卷积和全连接层计算得到一个向量，作为该模型的输出。

# 在训练模型之前，需要使用预训练权重对模型进行初始化。这是通过调用 model.load_state_dict 方法来完成的
# ，该方法将预训练权重加载到模型中。在这里，预训练权重存储在 IMG_MODEL_PATH 文件中。
    image_dataset = infer_ImgDataset.ImgDataset(image_paths=image_paths, transforms=get_test_transforms())
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=True,
        drop_last=False,
        
        num_workers=NUM_WORKERS
    )

    with torch.no_grad():
        for img, label in tqdm(image_loader):
            img = img.cuda()
            label = label.cuda()
            feat, _ = model(img, label)
            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)
# 这段代码的作用是用预训练的深度学习模型提取输入图像的特征向量（embeddings）并将这些向量存储在一个列表中（embeds）。具体来说：

# 使用 torch.no_grad() 确保在推理期间不会计算梯度，以节省计算资源。
# 遍历 image_loader 中的每个图像（和标签），将它们移动到可用的 GPU 上，并将其输入到预训练模型中。
# 提取模型输出的特征向量 feat（在这里忽略了第二个输出参数 _），并将其转换为 NumPy 数组。
# 将特征向量存储在列表 embeds 中。
# 在循环完成后，使用 np.concatenate() 将所有特征向量拼接成一个 NumPy 数组 image_embeddings 并返回。
    del model

    image_embeddings = np.concatenate(embeds)
    
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings
# print(f'Our image embeddings shape is {image_embeddings.shape}')用于打印图片嵌入向量的形状，方便调试和查看输出结果。

# del embeds用于删除embeds列表，释放内存。

# gc.collect()用于清理内存，帮助程序在运行期间更好地管理内存。

# 修改 更清楚的测试集命名
df = pd.read_csv(args.test_csv_path)
image_paths = args.test_image_path + df['imgPath']
image_embeddings = get_image_embeddings(image_paths.values)

# 修改
name = args.model_path.replace('.bin', '')
np.save(f'embedding/{name}.npy', image_embeddings)
print(f'测试集嵌入向量保存至embedding/{name}.npy')
