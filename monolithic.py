import os
import cv2
from brisque import BRISQUE
from fiq.ser_fiq import calc_ser_fiq_score
from niqe import calc_niqe
import time
import random
# from pypiqe import piqe


def calc_brisque_score(img):
    obj = BRISQUE(url=False)
    score = min(obj.score(img=img) / 100, 1)
    norm_score = 1 - score
    return norm_score

def calc_niqe_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200,200))
    print(calc_niqe(gray))
    score = min(calc_niqe(gray), 1)
    return 1 - score

def calc_ser_score(img_path):
    # resize image to 112x112
    # img = cv2.resize(img, (112, 112))
    return calc_ser_fiq_score(img_path)

def calc_quality_score(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    brisque = calc_brisque_score(img)
    ser_fiq = calc_ser_score(img_path)
    q_brisque = 1 - brisque/100
    q_ser_fiq = ser_fiq #min(max(ser_fiq, 0.78), 0.91)
    qs = (q_brisque + q_ser_fiq)/2
    # print(f"BRISQUE: {1-q_brisque}, SER: {q_ser_fiq}")
    return qs

# def calc_PIQE(img):
#     img = cv2.resize(img, (160,160))
#     score = piqe(img)[0]
#     return 1 - score/100
