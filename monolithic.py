from brisque import BRISQUE
from fiq.ser_fiq import calc_ser_fiq_score
import time
import random
from pypiqe import piqe


def calc_brisque_score(img):
    obj = BRISQUE(url=False)
    return obj.score(img=img)

def calc_ser_score(img):
    # resize image to 112x112
    img = cv2.resize(img, (112, 112))
    return calc_ser_fiq_score(img)

def calc_quality_score(img):
    start_time = time.time()
    brisque = calc_brisque_score(img)
    ser_fiq = calc_ser_score(img)
    q_brisque = 1 - brisque/100
    q_ser_fiq = ser_fiq #min(max(ser_fiq, 0.78), 0.91)
    qs = (q_brisque + q_ser_fiq)/2
    # print(f"BRISQUE: {1-q_brisque}, SER: {q_ser_fiq}")
    return qs

def calc_PIQE(img):
    img = cv2.resize(img, (160,160))
    score = piqe(img)[0]
    return 1 - score/100
