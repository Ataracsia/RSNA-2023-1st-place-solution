import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import os
import copy
import time

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import pydicom
import nibabel as nib

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import timm
import segmentation_models_pytorch as smp

import sys
sys.path.append('./')
from paths import PATHS

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

import sys
sys.path.append('./Models/')
from segmentation_model import Model, convert_3d


class CFG:
    model_name = 'resnet18d'
    
if type(CFG.model_name)!=str:
    
    CFG.model_name = 'resnet18d'
    
    # DOES NOT MATTER AS MOST LIKELY, I AM USING ONLY 1 MODEL FILE, 
    # AND "MODELS" IS NEVER USED

models = []
for F in [0,]:
    # Segmentationを行うモデル群
    model = convert_3d(Model())
    state_dict = torch.load(f'{PATHS.SEGMENTATION_MODEL_SAVE}/resnet18d_v1/{F}.pth')
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    import copy
    models.append(copy.deepcopy(model))
    
with torch.no_grad():
    outs = model(torch.zeros((2, 32, 128, 128)).cuda())
_ = [print(o.shape) for o in outs]

# Run Length Encodingを行うための関数
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    
    '''
    (0, 1, 1, 0, 1, 0) -> 2 2 5 1
    '''
    
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
# Run Length Encodingを復元するための関数
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# パス内のPatientまたはStudyを表すディレクトリをソートする関数
def glob_sorted(path):
    return sorted(glob(path), key=lambda x: int(x.split('/')[-1].split('.')[0]))

# Windowingするための関数
# CT画像ではWindowingすることが常であり、こうすることで関心のある臓器を目立たせる
# ことができる
def get_windowed_image(dcm, WL=50, WW=400):
    resI, resS = dcm.RescaleIntercept, dcm.RescaleSlope
    
    img = dcm.to_numpy_image()
    img = resS * img + resI
    
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    
    return X

# DCMファイルからCT画像を読み込み、np.stackして返す関数
# 途中でWindowingも行っている
def load_volume(dcms):
    volume = []
    for dcm_path in dcms:
        #dcm = pydicom.read_file(dcm_path)
        #image = dcm.pixel_array
        
        dcm = dicomsdl.open(dcm_path)
        
        # Windowing
        image = get_windowed_image(dcm)
        
        # Normalizing
        if np.min(image)<0:
            image = image + np.abs(np.min(image))
        
        image = image / image.max()
        
        volume.append(image)
        
    return np.stack(volume)

# Volumeに、リサイズ、32枚ごとのスライス、torch.Tensor化を行う関数
def process_volume(volume):
    volume = np.stack([cv2.resize(x, (128, 128)) for x in volume])
    
    volumes = []
    cuts = [(x, x+32) for x in np.arange(0, volume.shape[0], 32)[:-1]]
    
    # 32枚ずつ区切る
    if cuts:
        for cut in cuts:
            volumes.append(volume[cut[0]:cut[1]])
            
        volumes = np.stack(volumes)
    else:
        volumes = np.zeros((1, 32, 128, 128), dtype=np.uint8)
        volumes[0, :len(volume)] = volume
    
    # 端数があった場合の処理
    if cuts:
        last_volume = np.zeros((1, 32, 128, 128), dtype=np.uint8)
        last_volume[0, :volume[cuts[-1][1]:].shape[0]] =  volume[cuts[-1][1]:]
    
        volumes = np.concatenate([volumes, last_volume])
    
    volumes = torch.as_tensor(volumes).float()
    
    return volumes

'''
def predict_segmentation(volumes, model):
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(volumes.cuda())
            outputs = outputs.sigmoid().detach().cpu().numpy()
            outputs = (outputs>0.5).astype(np.float32)
    return outputs
#'''

# Segmentationを行い、スコアが0.5以上のピクセルをMaskする関数
def predict_segmentation(volumes, models):
    
    final_outputs = []
    
    for model in models:
    
        with torch.no_grad():
    
            with torch.cuda.amp.autocast(enabled=True):
    
                outputs = model(volumes.cuda())
                outputs = outputs.sigmoid().detach().cpu().numpy()
                #outputs = (outputs>0.5).astype(np.float32)
                final_outputs.append(outputs)
    
    # Segmentation Modelsの出力結果の平均値をとり、Maskを作成する
    final_outputs = np.stack(final_outputs).mean(0)
    final_outputs = (final_outputs>0.5).astype(np.float32)
    
    return final_outputs

# patient単位の全臓器の健康状態を読み込む
data = pd.read_csv(f'{PATHS.BASE_PATH}/train.csv')#[:10]

# CT画像一枚単位の[Active Extravasation|Bowel]情報を読み込む
image_level = pd.read_csv(f'{PATHS.BASE_PATH}/image_level_labels.csv')
image_level_extravasation = image_level[image_level.injury_name=='Active_Extravasation']
image_level_bowel = image_level[image_level.injury_name=='Bowel']

DAT = {
    'patient': [], 'study': [], 'instance': [],
    'extravasation': [], 'bowel': [],
    'liver_rle': [], 'right_kidney_rle': [], 'left_kidney_rle': [],
    'spleen_rle': [], 'bowel_rle': []
}

for patient in tqdm(data.patient_id):
    
    # studyレベルのディレクトリのリストを取得する
    studies = os.listdir(f'{PATHS.BASE_PATH}/train_images/{patient}/')
    
    for study in studies:
        
        # patient/studyディレクトリ内の.dcmファイルのリストを取得し、並び替えする
        dcms = glob_sorted(f'{PATHS.BASE_PATH}/train_images/{patient}/{study}/*.dcm')
        
        # ExtravasationもしくはBowelのラベルを持つinstance_numberをリスト化
        extravasation_instances = image_level_extravasation[image_level_extravasation.series_id==int(study)].instance_number.values
        bowel_instances = image_level_bowel[image_level_bowel.series_id==int(study)].instance_number.values
        
        # Volume化
        volume = load_volume(dcms)
        volumes = process_volume(volume)
        
        # Segmentation
        volumes_seg = predict_segmentation(volumes, models)
        
        # Segmentationの出力をconcatenateする
        volume_seg = np.concatenate(volumes_seg.transpose(0, 2, 1, 3, 4))[:len(volume)]
        
        for dcm, img, seg in zip(dcms, volume, volume_seg):
            
            instance = int(dcm.split('/')[-1].split('.')[0])
            
            # patient, study, .dcmファイルのstemの三つを記録
            DAT['patient'].append(patient)
            DAT['study'].append(study)
            DAT['instance'].append(instance)
            
            # extravasationまたはbowelのラベルがついていれば「1」とするフラグ変数を
            # 定義する
            extravasation_label, bowel_label = 0, 0
            if instance in extravasation_instances: extravasation_label = 1
            if instance in bowel_instances: bowel_label = 1
            
            DAT['extravasation'].append(extravasation_label)
            DAT['bowel'].append(bowel_label)
            
            # Segmentationのアウトプットに対し、Run Length Encodingを行う
            idx_to_key = {0: 'liver',
                          1: 'spleen',
                          2: 'right_kidney',
                          3: 'left_kidney',
                          4: 'bowel'}
            
            for idx in idx_to_key:
                
                # Segmentation ModelのOutput(5channels)に対し、idxで参照して
                # Run Length Encodingする
                rle = rle_encode(seg[idx])
                key = idx_to_key[idx]
                
                # Encodingしたものをそのままdictに入力
                DAT[key+'_rle'].append(rle)
            
# DATをFINAL_DAT(pd.DataFrame)に変換
FINAL_DAT = pd.DataFrame(DAT)

# Run Length Encoding済みのSegmentationデータに対し、
# 「1」となっているPixel数を合計して'xxx_size'として保存する
for col in FINAL_DAT.columns[-5:]:
    
    rles = FINAL_DAT[col].values
    pixels = []
    
    for rle in tqdm(rles):
    
        ps = np.sum([int(x) for x in rle.split(' ')[1::2]])
        pixels.append(ps)
    
    FINAL_DAT[col.replace('_rle', '_size')] = pixels

# 各臓器のPixel数をそれぞれstudyレベルでMin-Max Scallingし、0以上1以下に変換する
# 'xxx_size'はScallingされた値に上書きされる
for gri, grd in tqdm(FINAL_DAT.groupby('study')):
    for col in grd.columns[-5:]:
        grd[col] = grd[col] / grd[col].max()
    
    st = grd.study.values[0]
    FINAL_DAT[FINAL_DAT.study==st] = grd

# ↓変数名はpatient_levelの方が適当
study_level = pd.read_csv(f'{PATHS.BASE_PATH}/train.csv')

# 各patientのany_injuryラベルの値をdictで記録する
patient_to_injury = {pat: study_level[study_level.patient_id==pat].any_injury.values[0] for pat in study_level.patient_id.unique()}

# 各patientの各臓器が健康であることを示すラベルを辞書で取得する
cols = ['kidney_healthy', 'liver_healthy', 'spleen_healthy']

# patientごとに3つの臓器の健康状態を表すdictをつくる
for col in cols:
    exec(f'patient_to_{col}' + " = {pat: study_level[study_level.patient_id==pat]."+f"{col}"+".values[0] for pat in study_level.patient_id.unique()}")

# PROCESS DATA
# 以降、FINAL_DATをdataとして扱い、それに属性を追加する
data = FINAL_DAT

cols = ['kidney_healthy', 'liver_healthy', 'spleen_healthy']

# dataに、3つの臓器の健康状態を追加する
for col in cols:
    exec(f"data['{col}']"  + " = data.patient.apply(lambda x: " + f"patient_to_{col}[x])")

# dataに、'any_injury'を追加する
data['any_injury'] = data.patient.apply(lambda x: patient_to_injury[x])

# dataに、organs_sizeを追加する
cols = ['liver_size', 'right_kidney_size', 'left_kidney_size', 'spleen_size']

# 欠損値に0埋めを行う
for col in cols:
    data[col] = data[col].fillna(0)


study_boxes = {}
for gri, grd in tqdm(data.groupby('study')):
    
    # 欠損値に''埋め
    grd = grd.fillna('')
    
    # Run Length Decodingを行い、StudyレベルのSegmentationデータに対し、
    # 0より大なる値がある箇所に、ラベルをマッピングする
    add_rles = []
    cols = ['liver_rle', 'right_kidney_rle', 'left_kidney_rle', 'spleen_rle', 'bowel_rle']
    
    for col in cols:
    
        rle = np.stack([rle_decode(rle, (128, 128)) for rle in grd[col]]).max(0)
        add_rles.append(rle)
    
    # studyレベルで0より大なる値がある箇所に1でマッピングし、その箇所の位置を記録する
    # 位置は高さと幅の最大値を1とした割合で得る
    msk = np.stack(add_rles).max(0)
    ys, xs = np.where(msk)
    y1, y2, x1, x2 = np.min(ys) / 128, np.max(ys) / 128, np.min(xs) / 128, np.max(xs) / 128
    
    study_boxes[gri] = [y1, y2, x1, x2]

# Studyごとの位置を属性として追加
data['study_crop'] = data.study.apply(lambda x: study_boxes[x])

# 
dicom_tags = pd.read_parquet(f'{PATHS.BASE_PATH}/train_dicom_tags.parquet')

# parquetファイルにあるInstanceレベルのメタデータにあるImagePositionPatientから
# z軸の座標を得る
z_pos = {row.path: float(row.ImagePositionPatient.split(',')[-1].replace(']', '')) for i, row in tqdm(dicom_tags.iterrows())}

# "train_images/patient/study/instance.dcm"の形で.dcmのパスをdataに追加
data['patient_study_instance'] = 'train_images/' + data['patient'].astype(str) + '/' + data['study'].astype(str) + '/' + data['instance'].astype(str) +  '.dcm'
# z_posをdataに追加
data['z_pos'] = data.patient_study_instance.apply(lambda x: z_pos[x])

data.to_csv(f'{PATHS.INFO_DATA_SAVE}', index=False)
