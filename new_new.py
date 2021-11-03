from datetime import time
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import dtype, histogram
import pandas as pd
from pandas import json_normalize
import math

with open("image_log_update_final.json", 'r', encoding='utf-8-sig') as f:
    json_data = json.load(f)

j_data = json.dumps(json_data) #indent='\t')

def json_processing(json_data):
    conf_lst = []
    x_lst = []
    y_lst = []
    width_lst = []
    height_lst = []
    f_no_lst = []

    for dictionary in json_data['frames']:
        if dictionary['frame'] == []:
            conf_lst.append(None)
            x_lst.append(None)
            y_lst.append(None)
            height_lst.append(None)
            width_lst.append(None)
        else:
            conf_lst.append(round(dictionary['frame'][0]['confidence'],2))
            x_lst.append(round(dictionary['frame'][0]['center_x'],2))
            y_lst.append(round(dictionary['frame'][0]['center_y'],2))
            width_lst.append(round(dictionary['frame'][0]['width'],2))
            height_lst.append(round(dictionary['frame'][0]['height'],2))

        f_no_lst.append(dictionary['frame_no'])
    
    #총 시간(단위 : 초)
    total_time = len(f_no_lst)
    print("영상의 시간(초) : ", total_time)

    #처리를 위해 모든 데이터에 대한 데이터 프레임 생성 및 csv로 저장
    all_data =[]
    for idx in range(total_time):
        all_data.append({
            'time' : idx+1,
            'x' : x_lst[idx],
            'y' : y_lst[idx],
        })
    df1 = pd.DataFrame(all_data, columns=['time', 'x', 'y'])
    df1.to_csv('all_datas.csv')
    print("all_data is saved as CSV")

    #강아지를 detect 한 경우의 데이터만 모아서 데이터 프레임 생성 및 csv로 저장
    detect_x=[]
    detect_y=[]
    detect_t=[]
    out_time=0
    x_y_velo_act_lst = []

    for idx in range(total_time):
        if x_lst[idx] == None:
            out_time += 1
        else:
            detect_x.append(x_lst[idx])
            detect_y.append(y_lst[idx])
            detect_t.append(idx)
    
    act_time = []
    print('len(detect_x) = ', len(detect_x))

    for idx in range(len(detect_x)-1):
        #운동상태 기록 : 거리와 속력
        if idx==0:
            dist = None
            velocity = None
            act = None
            act_time.append(None)

        else:
            dist = round(math.sqrt((detect_x[idx+1]-detect_x[idx])**2 + (detect_y[idx+1] - detect_y[idx])**2),2)
            time_difference = detect_t[idx+1] - detect_t[idx]
            act_time.append(detect_t[idx]-detect_t[idx-1])
            velocity = round(dist / time_difference,2)
            if velocity > 35:
                act = 'run'
            elif velocity > 1 :
                act = 'walk'
            else:
                act = 'rest'
        
        is_dog = 'living room'
        #x : 650~720 / y : 107 ~ 118 >> 현관문 근처에 있음
        if detect_x[idx]>650 and detect_x[idx] < 730:
            if detect_y[idx] > 107 and detect_y[idx] < 120:
                is_dog = 'near door'

        x_y_velo_act_lst.append({
            't' : detect_t[idx],
            'x' : detect_x[idx],
            'y' : detect_y[idx],
            'is in' : is_dog,
            'dist' : dist,
            'velocity' : velocity,
            'act' : act,
            })
    detect_time = len(detect_t)
    print(f'미검출 시간(초) : {out_time} / 검출 시간(초) : {detect_time}')
    df2 = pd.DataFrame(x_y_velo_act_lst, columns=['t', 'x', 'y', 'is in','dist', 'velocity', 'act'])
    df2['act_time'] = act_time
    df2.to_csv('act_data.csv')
    df3 = df2[['t', 'is in','act', 'act_time']]

    act_lst = ['rest', 'walk', 'run']
    isin_lst = ['living room', 'near door']
    lst = []
    for i in act_lst:
        for j in isin_lst:
            lst.append({
                'act' : i,
                'is in' : j,
                'count' : len(df3[(df3['is in'] == j) & (df3['act'] == i)])
            })
    lst.append({
        'act' : 'Non_detect',
        'is in' : 'Non_detect',
        'count' : len(df1) - len(df2)})
    df4 = pd.DataFrame(lst, columns=['act', 'is in', 'count'])
    pivot_df = df4.pivot('act', 'is in', 'count')
    pivot_df = pivot_df.fillna(0).astype(int)

    return df1, df2, pivot_df


def draw_chart(df1, df2, df3):
    pass

if __name__ == '__main__':
    x, y, f = json_processing(json_data)
    #original_df, act_df, pivot_df = move_analyze_and_make_df(x, y, f)
    #print(original_df)
    #print(act_df)
    #print(pivot_df)