import dicomsdl
def __dataset__to_numpy_image(self, index=0):
    info = self.getPixelDataInfo()
    dtype = info['dtype']
    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')
    else:
        shape = [info['Rows'], info['Cols']]
    outarr = np.empty(shape, dtype=dtype)
    self.copyFrameData(index, outarr)
    return outarr
dicomsdl._dicomsdl.DataSet.to_numpy_image = __dataset__to_numpy_image    

    
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from glob import glob
import copy
import time
import math
import command
import random

#os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

import cv2
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = 12, 8

from skimage import img_as_ubyte
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import *
from sklearn.metrics import *

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import timm

from transformers import get_cosine_schedule_with_warmup

import sys
sys.path.append('./')
from paths import PATHS

def glob_sorted(path):
    return sorted(glob(path), key=lambda x: int(x.split('/')[-1].split('.')[0]))

def get_rescaled_image(dcm):
    resI, resS = dcm.RescaleIntercept, dcm.RescaleSlope
    
    img = dcm.to_numpy_image()
    img = resS * img + resI
    
    return img

def get_windowed_image(img, WL=50, WW=400):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    
    return X

def load_volume(dcms):
    volume = []
    for dcm_path in dcms:
        #dcm = pydicom.read_file(dcm_path)
        #image = dcm.pixel_array
        
        dcm = dicomsdl.open(dcm_path)
        
        image = get_rescaled_image(dcm)
        image = get_windowed_image(image)
        
        #image = get_windowed_image(dcm)
        
        if np.min(image)<0:
            image = image + np.abs(np.min(image))
        
        image = image / image.max()
        
        volume.append(image)
        
    return np.stack(volume)

def get_volume_data(data, step=96, stride=1, stride_cutoff=200):
    volumes = []
    
    for gri, grd in tqdm(data.groupby('study')):
        
        # 同じStudyに属するデータがstride_cutoffより多いとき、strideの分だけ
        # 飛ばしてデータを取得する
        if len(grd)>stride_cutoff:
            grd = grd[::stride]
        
        take_last = False
        
        # 同じStudyに属するデータ数がstepの倍数でないとき、take_lastをTrueにする
        if not str(len(grd)/step).endswith('.0'):
            take_last = True
        
        started = False
        # 同じStudyに属するデータをstepで割った商が1以上のとき、以下の処理を繰り返す
        # len(grd)//stepが0となる場合、started=Falseのままとなる
        for i in range(len(grd)//step):
            
            # 同じStudyに属するデータに対し、stepの枚数だけスライス
            rows = grd[i*step:(i+1)*step]
            
            # rowsがstepとぴったり同じ数でなければ、重複を許して1/stepずつスライス
            if len(rows)!=step:
                rows = pd.DataFrame([rows.iloc[int(x*len(rows))] for x in np.arange(0, 1, 1/step)])
            
            volumes.append(rows)
            
            started = True
        
        # 同じStudyに属するデータをstepで割った商が0の場合、
        # 重複を許して1/stepずつスライス
        if not started:
            rows = grd
            rows = pd.DataFrame([rows.iloc[int(x*len(rows))] for x in np.arange(0, 1, 1/step)])
            volumes.append(rows)
        
        # take_lastがTrueの場合、最後のstep分を取得
        if take_last:
            rows = grd[-step:]
            # step分取得したときに不足がないとき、volumesに加える
            if len(rows)==step:
                volumes.append(rows)
                
    return volumes


study_level = pd.read_csv(f'{PATHS.BASE_PATH}/train.csv')

data = pd.read_csv(f'{PATHS.INFO_DATA_SAVE}')#[:4752]

OUTPUT_FOLDER = f'{PATHS.OURDATA_VOL_SAVE_PATH}'
os.makedirs(f"{OUTPUT_FOLDER}/", exist_ok=1)

# 100行ずつスライスしたデータ群のリストを取得
volume_data = get_volume_data(data, step=100, stride=2, stride_cutoff=400)

# データ群のリストのインデックスをもとに.dcmファイルを読み込み、np.saveを行う関数
def process(i):
    rows = volume_data[i]
    patient = rows.iloc[0].patient
    study = rows.iloc[0].study
    start = rows.iloc[0].instance
    end = rows.iloc[-1].instance
    
    files = np.array([f"{PATHS.BASE_PATH}/train_images/{row.patient}/{row.study}/{row.instance}.dcm" for i, row in rows.iterrows()])
    
    vol = load_volume(files)
    vol = (vol * 255).astype(np.uint8)
    
    np.save(f"{OUTPUT_FOLDER}/{patient}_{study}_{start}_{end}.npy", vol)
    
    return None

# 以下、process関数をmultiprocessingで処理する
import multiprocessing as mp

start = 0

with mp.Pool(processes=8) as pool:
    
    # スライスしたデータの個数を取得
    idxs = list(range(start, len(volume_data)))
    imap = pool.imap(process, idxs)
    _ = list(tqdm(imap, total=len(volume_data)-start))
