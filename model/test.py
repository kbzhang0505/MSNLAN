
import torch
import pandas as pd
import torch.nn.functional as F
import numpy as np
import os
import cv2
from scipy.io import savemat
import scipy
import scipy.misc
from torchvision import transforms
def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    # rows,cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches
def conv_n(input,kernel,step=1):
    N = img.shape[0]  # 图片边长
    F = kernel.shape[0]  # filter长
    L = int((N - F) / step) + 1  # 输出结果边长
    res = torch.zeros(L, L, requires_grad=True)
    for row in range(0, L):
        for column in range(0, L):
            tmp_row, tmp_col = row * step, column * step
            res[row, column] = (imput[tmp_row: tmp_row + F, tmp_col: tmp_col + F] * kernel).sum().item()
    return res
# def conv(input,kernel,stride=1):

def saveFeature(feature, saveDir,i):
    # feature = feature.squeeze(0)
    if i!=-1:
        feature = feature.squeeze(i)
    out1 = feature.cpu().numpy()
    ori_path = saveDir
    for i,slice_2d in enumerate(out1):
        end_path =  str(i)+'.txt'
        if not os.path.isdir(ori_path):
            os.makedirs(ori_path, exist_ok=True)
        path = os.path.join(ori_path,end_path)
        # if not os.path.isdir(path):
        #     os.makedirs(path, exist_ok=True)

        with open(path,'w') as outfile:
            np.savetxt(outfile, slice_2d, fmt='%f', delimiter=',')


    # num = out1.shape[0]
    #
    # for index in range(1, num + 1):
    #     #     feature_res = feature[index-1]-feature2[index - 1]
    #     scipy.misc.imsave(saveDir + str(index) + '.png', out1[index - 1])

def saveMat(feature,datadir):

    b = feature.cpu().numpy()
    b = np.array(b)
    savemat(datadir,{'mat feature': b})
def saveExcel(feature,datadir,i):
    out = feature.squeeze(i)
    feature = out.cpu().numpy()


    for i, slice_2d in enumerate(feature):
        if not os.path.isdir(datadir):
            os.makedirs(datadir, exist_ok=True)
         # 关键1，将ndarray格式转换为DataFrame
        rows, cols = slice_2d.shape
        data_df = pd.DataFrame(slice_2d)
        # 更改表的索引
        data_index = []
        for i in range(rows):
            data_index.append(i)
        data_df.index = data_index
        # 更改表的索引
        data_index = []
        for i in range(cols):
            data_index.append(i)
        data_df.index = data_index
        data_df.columns = data_index

def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x
def Normalize(data):
    mean =reduce_mean(data,axis=[0,1],keepdim=True)
    std  =reduce_std(data,axis=[0,1],keepdim=True)
    x=data-mean
    x =x/std

    return x
def autoNorm(dataSet):
    minVals = dataSet.min(0) # 取每一列的最小值
    maxVals = dataSet.max(0) # 取每一列的最大值
    ranges = maxVals - minVals

    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet

def saveImage(img,dataDir,i):
    if i!=-1:
        img = img.squeeze(i)
    dataDir = dataDir+'_image'
    for i,slice_2d in enumerate(img):
        end_path =  str(i)+'.png'
        if not os.path.isdir(dataDir):
            os.makedirs(dataDir, exist_ok=True)
        path = os.path.join(dataDir,end_path)
        # if not os.path.isdir(path):
        #     os.makedirs(path, exist_ok=True)
        scipy.misc.imsave(path, slice_2d)
def saveSingleImage(img,dataDir,i):
    if i!=-1:
        img = img.squeeze(i)
    dataDir = dataDir+'_image'
    for i,slice_2d in enumerate(img):
        end_path =  str(i)+'.png'
        if not os.path.isdir(dataDir):
            os.makedirs(dataDir, exist_ok=True)
        path = os.path.join(dataDir,end_path)
        # if not os.path.isdir(path):
        #     os.makedirs(path, exist_ok=True)
        scipy.misc.imsave(path, slice_2d)
def conv_SingleStep(weights,data,savedir):
    weights = weights.squeeze(1)
    for i,w in enumerate(weights):
        w=w.view(-1,7,7)
        w = w.view(-1,1,7,7)
        yi = F.conv2d(data, w, stride=1)
        end_path = str(i) + '.png'
        if not os.path.isdir(savedir):
            os.makedirs(savedir, exist_ok=True)
        path = os.path.join(savedir, end_path)
        # if not os.path.isdir(path):
        #     os.makedirs(path, exist_ok=True)
        yi = yi.squeeze(0)
        yi = yi.squeeze(0)
        scipy.misc.imsave(path,yi)
def convTrans_SingleStep(weights,data,savedir):
    weights = weights.squeeze(1)
    data = data.squeeze(0)
    for i,w in enumerate(weights):
        w=w.view(-1,7,7)
        w = w.view(-1,1,7,7)
        d = data[i]
        d = d.view(-1,data.shape[1],data.shape[2])
        d=d.view(-1,1,data.shape[1],data.shape[2])
        yi = F.conv_transpose2d(d, w, stride=1,padding=3)
        end_path = str(i) + '.png'
        if not os.path.isdir(savedir):
            os.makedirs(savedir, exist_ok=True)
        path = os.path.join(savedir, end_path)
        # if not os.path.isdir(path):
        #     os.makedirs(path, exist_ok=True)
        yi = yi.squeeze(0)
        yi = yi.squeeze(0)
        scipy.misc.imsave(path,yi)
        oripath = os.path.join(savedir,'y2xt')
        end_path1 = str(i) + '.txt'
        if not os.path.isdir(oripath):
            os.makedirs(oripath, exist_ok=True)
        path = os.path.join(oripath, end_path1)
        # if not os.path.isdir(path):
        #     os.makedirs(path, exist_ok=True)

        with open(path, 'w') as outfile:
            np.savetxt(outfile, yi, fmt='%f', delimiter=',')


def getData(img,size):
    # image = cv2.resize(image, (25, 25), interpolation=cv2.INTER_LINEAR)
    TensorSelf = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    TensorSelf = torch.Tensor(TensorSelf)

    # ToImage  = transforms.ToPILImage
    # img = ToImage(TensorSelf)
    # x2 = torch.zeros(25,25)
    x = TensorSelf.view(TensorSelf.shape[0], TensorSelf.shape[1], -1)
    x = x.view(x.shape[0], x.shape[1], x.shape[2], -1)
    x = x.view(x.shape[3], x.shape[2], x.shape[0], x.shape[1])
    return x

def test():
    # TensorSelf = torch.rand([25, 25], out=None)
    image = cv2.imread('/opt/data/private/csnl/experiment/luna.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = autoNorm(image)

    # x = F.relu(getData(image,25))
    x  = getData(image,25)
    x_w =getData(image,20)
    w = x_w

    raw_k = x_w
    # saveMat(x,datadir='/opt/data/private/csnl/experiment/TestDemo/mat/x_mat')
    saveFeature(w,'/opt/data/private/csnl/experiment/test_last/x_w_future',1)
    saveImage(w,'/opt/data/private/csnl/experiment/test_last/image/x_w',1)

    saveFeature(x,'/opt/data/private/csnl/experiment/test_last/x_future',1)
    saveImage(x,'/opt/data/private/csnl/experiment/test_last/image/x',1)

    saveFeature(raw_k, '/opt/data/private/csnl/experiment/test_last/raw_future', 1)
    saveImage(raw_k, '/opt/data/private/csnl/experiment/test_last/image/raw_x', 1)
    # shape_input = list(x.size())
    raw = extract_image_patches(raw_k,ksizes=[7,7],strides=[1,1],rates=[1,1],padding='same')
    raw = raw.view(raw_k.shape[0],7,7,-1)
    raw = raw.permute(3,0,1,2)

    w = extract_image_patches(w, ksizes=[7, 7], strides=[1, 1], rates=[1, 1], padding='same')
    w = w.view(w.shape[0], 7, 7, -1)
    w = w.permute(3, 0, 1, 2)
    saveFeature(raw,'/opt/data/private/csnl/experiment/test_last/raw',1)
    saveImage(raw,'/opt/data/private/csnl/experiment/test_last/raw',i=1)

    saveFeature(w, '/opt/data/private/csnl/experiment/test_last/w', 1)
    saveImage(w, '/opt/data/private/csnl/experiment/test_last/w', i=1)

    xi =  same_padding(x, [7,7], [1, 1], [1, 1])
    y1 = F.conv2d(xi,w,stride=1)
    y11=F.relu(y1)
    # y11 = F.softmax(y1, dim=1)
    saveImage(y1, dataDir='/opt/data/private/csnl/experiment/test_last/image/y1', i=0)
    saveFeature(y1,'/opt/data/private/csnl/experiment/test_last/y1',i=0)

    saveImage(y11, dataDir='/opt/data/private/csnl/experiment/test_last/image/y11', i=0)
    saveFeature(y11, '/opt/data/private/csnl/experiment/test_last/y11', i=0)
    convTrans_SingleStep(raw,y11,'/opt/data/private/csnl/experiment/test_last/image/y2')
    y2 = F.conv_transpose2d(y11, raw, stride=1, padding=3)
    saveFeature(y2, '/opt/data/private/csnl/experiment/test_last/y2', i=0)
    saveImage(y2, dataDir='/opt/data/private/csnl/experiment/test_last/y2',i=1)


if __name__ == '__main__':
    test()



