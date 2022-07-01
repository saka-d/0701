#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import matplotlib.pyplot as plt#グラフ描画に使用
"""
import pubchempy as pcp
from rdkit.Chem import AllChem, Draw, rdDistGeom
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolStandardize
from rdkit.Chem.Draw import IPythonConsole
import py3Dmol
from math import *
import subprocess
import pyautogui as pg
import pyperclip
import time
import re
import glob
import pandas as pd
import os
import json
import numpy as np
import copy
from sklearn.linear_model import LinearRegression#決定係数算出に使用
import sklearn.metrics as metrics#決定係数算出に使用

import culculate as cul
import carbonyl as car
import bond as bon
import atom_red as atr
import search as sea
import evaluate as eva
"""
def Rodrigues(Ang,n):#ロドリゲスの回転公式
    n/=np.linalg.norm(n)
    Ang=math.radians(Ang)
    c,s=np.cos(Ang),np.sin(Ang)
    C=1-c
    R = np.array([[c+n[0]*n[0]*C, n[0]*n[1]*C-n[2]*s, n[0]*n[2]*C+n[1]*s],
                  [n[1]*n[0]*C+n[2]*s, c+n[1]*n[1]*C, n[1]*n[2]*C-n[0]*s],
                  [n[2]*n[0]*C-n[1]*s, n[2]*n[1]*C+n[0]*s,c+n[2]*n[2]*C]])
    return R


def lotate_vec(BD_axis,FL_axis,car_vec,BD,FL):#COをBD角FL角回転させたvecを求める関数
    BD_vec=np.dot(Rodrigues(BD,BD_axis),car_vec)#car_vecを中心、BD_axisを軸としてBD°回転した位置(BD_vec)
    location_vec=np.dot(Rodrigues(FL,FL_axis),BD_vec)#BD_locationをcar_Cを中心、FL_axis°回転した位置(BD+FLlocation)
    location_vec/=np.linalg.norm(location_vec)#単位ベクトル化を軸としてFL
    #print(location_vec)
    return location_vec

def imaginary_radius_maker(h,radius_a,radius_b):#円錐台の半径
    return radius_a+radius_b*h

def Bd(x): return math.exp(-x/1.99/0.273)#ボルツマン分布
"""
def s(R,r,d):#2円重なり面積算出関数
    if d==0 or R+r<=d: return 0
    elif d+r<=R: return np.pi*r**2
    else:
        A=(R**2-r**2+d**2)/(2*d)
        B=d-A
        def ss(x,y):
            if x**2-y**2>0:
                A=(x**2-y**2)**(1/2)
                B=math.atan(y/A)
            elif y>0:
                B=np.pi/2
                A=0
            else:
                B=-np.pi/2
                A=0
            return 1/2*(np.pi/2*x**2-y*A-x**2*B)
        return ss(R,A)+ss(r,B)
"""
def s(R,r,d):#2円重なり面積算出関数
    if d==0 or R+r<=d: return 0
    elif d+r<=R: return np.pi*r**2
    else:
        A=(R**2-r**2+d**2)/(2*d)
        B=d-A
        def ss(x,y):
            if x**2-y**2>0:
                A=(x**2-y**2)**(1/2)
                B=math.atan(y/A)
            else:
                B=np.sign(y)*np.pi/2
                A=0
            return (np.pi/2*x**2-y*A-x**2*B)
        return ss(R,A)+ss(r,B)
    
def culcurate_cone2(paramater_cone,energies_list,name_list,model,car_C_list,cids_list):#BD角、FL角、距離の3変数。一点評価（エネルギーを利用する方向に拡張可能）
    BD,FL,low,high,radius_a,radius_b,FL_trans_paramater=paramater_cone#ここに変数を設定
    volume_rate=[]
    for i in [1,-1]:
        BD*=i
        volume_list=[]
        delta_h=(high-low)/10
        for name in name_list:
            energies=energies_list[name]
            mini_model=model[name]
            car_C=car_C_list[name]#car_Cの番号
            total_sqare=0
            for Idx,energy in zip(cids_list[name],energies):
                energy=energy[0]
                conf_coordinate,radius,BD_axis,FL_axis,car_vec=mini_model[Idx]
                car_C_location=conf_coordinate[car_C]+FL_trans_paramater*FL_axis/np.linalg.norm(FL_axis)
                location_vec=lotate_vec(BD_axis,FL_axis,car_vec,BD,FL)
                Idx_sqare=0
                for atom_location,atom_radius in zip(conf_coordinate,radius):
                    atom_to_car_C=atom_location-car_C_location#原子中心-car_C
                    inner=np.dot(location_vec,atom_to_car_C)#vとの内積
                    distance=np.linalg.norm(atom_to_car_C-inner*location_vec)#dを出す
                    for h in np.arange(low, high, delta_h):
                        circle_radius_sqare=atom_radius**2-(inner-h)**2#r^2を算出
                        if circle_radius_sqare>0:#r>0
                            circle_radius=math.sqrt(circle_radius_sqare)#rを出す
                            imaginary_radius=imaginary_radius_maker(h,radius_a,radius_b)#円錐台の半径をhの関数によって出す
                            sqare=s(circle_radius,imaginary_radius,distance)#面積算出
                            #sqare=circle_radius_sqare**2
                            Idx_sqare+=sqare*delta_h#Δhを掛ける#足す
                Idx_sqare*=Bd(energy)
                total_sqare+=Idx_sqare
                #print(total_sqare)
            volume_list.append(total_sqare/len(cids_list))
            #print(conf_coordinate[car_C])
            #print(FL_trans_paramater*FL_axis/np.linalg.norm(FL_axis))
            #print(car_C_location)
        volume_rate.append(volume_list)
    volume_rate=np.array(volume_rate[0])-np.array(volume_rate[1])#np.array(volume_rate[0])/sum(np.array(volume_rate))
    return volume_rate

"""バネモデルnew"""
def culcurate_cone2(paramater_cone,energies_list,name_list,model,car_C_list,cids_list):#BD角、FL角、距離の3変数。一点評価（エネルギーを利用する方向に拡張可能）
    BD,low,high,radius_a,radius_b,FL_trans_paramater=paramater_cone#ここに変数を設定
    FL=0
    volume_rate=[]
    for i in [1,-1]:
        BD*=i
        volume_list=[]
        #delta_h=(high-low)/10
        for name in name_list:
            energies=energies_list[name]
            mini_model=model[name]
            car_C=car_C_list[name]#car_Cの番号
            total_sqare=0
            #for Idx,energy in zip(cids_list[name],energies):
            for energy,Idx in energies:
                #energy=energy[0]
                conf_coordinate,radius,BD_axis,FL_axis,car_vec=mini_model[Idx]
                car_C_location=conf_coordinate[car_C]+FL_trans_paramater*FL_axis/np.linalg.norm(FL_axis)
                location_vec=lotate_vec(BD_axis,FL_axis,car_vec,BD,FL)
                Idx_sqare=0
                for atom_location,atom_radius in zip(conf_coordinate,radius):
                    atom_to_car_C=atom_location-car_C_location#原子中心-car_C
                    inner=np.dot(location_vec,atom_to_car_C)#vとの内積
                    distance=np.linalg.norm(inner*location_vec-atom_to_car_C)#dを出す
                    #d=np.linalg.norm(atom_location-inner*location_vec+conf_coordinate[car_C])
                    imaginary_radius=imaginary_radius_maker(inner,radius_a,radius_b)#円錐台の半径をhの関数によって出す
                    tortion=atom_radius**1.2+imaginary_radius-distance
                    if inner>low and inner<high and tortion>0:
                        Idx_sqare+=tortion**1*atom_radius**2
                Idx_sqare*=Bd(energy)
                total_sqare+=Idx_sqare
                #print(total_sqare)
            volume_list.append(total_sqare/len(cids_list[name]))
            #print(conf_coordinate[car_C])
            #print(FL_trans_paramater*FL_axis/np.linalg.norm(FL_axis))
            #print(car_C_location)
        volume_rate.append(volume_list)
    volume_rate=np.array(volume_rate[0])-np.array(volume_rate[1])#np.array(volume_rate[0])/sum(np.array(volume_rate))
    return volume_rate

"""バネモデルnew2#091713:57バックアップ"""
def culcurate_cone2(paramater_cone,energies_list,name_list,model,car_C_list,cids_list):#BD角、FL角、距離の3変数。一点評価（エネルギーを利用する方向に拡張可能）
    BD,low,high,radius_a,radius_b,FL_trans_paramater=paramater_cone#ここに変数を設定
    FL=0
    total_energy_list=[]
    for name in name_list:
        energies=energies_list[name]
        mini_model=model[name]
        car_C=car_C_list[name]#car_Cの番号
        total_energy=0
        for Idx,energy in zip(cids_list[name],energies):
            energy=energy[0]
            conf_coordinate,radius,BD_axis,FL_axis,car_vec=mini_model[Idx]
            car_C_location=conf_coordinate[car_C]+FL_trans_paramater*FL_axis/np.linalg.norm(FL_axis)
            
            
            local_energy_list=[]
            for BD_angle in [BD,-BD]:
                
                location_vec=lotate_vec(BD_axis,FL_axis,car_vec,BD_angle,FL)
                local_energy=0
                for atom_location,atom_radius in zip(conf_coordinate,radius):
                    atom_to_car_C=atom_location-car_C_location#原子中心-car_C
                    inner=np.dot(location_vec,atom_to_car_C)#vとの内積
                    distance=np.linalg.norm(inner*location_vec-atom_to_car_C)#dを出す
                    #d=np.linalg.norm(atom_location-inner*location_vec+conf_coordinate[car_C])
                    imaginary_radius=imaginary_radius_maker(inner,radius_a,radius_b)#円錐台の半径をhの関数によって出す
                    tortion=atom_radius**1.2+imaginary_radius-distance
                    if inner>low and inner<high and tortion>0:
                        local_energy+=tortion**1*atom_radius**2
                local_energy_list.append(local_energy)
            delta=local_energy_list[0]-local_energy_list[1]
            #total_energy+=(local_energy_list[0]-local_energy_list[1])*Bd(energy)
            #print(local_energy_list[0]-local_energy_list[1])
            print(delta)
            total_energy+=(1/(1+math.exp(-10*delta))-0.5)*Bd(energy)
        total_energy_list.append(total_energy)#/len(cids_list[name])
    print(total_energy_list)
    return np.array(total_energy_list)

"""バネモデルnew2#最新"""
def culcurate_cone(paramater_cone,energies_list,name_list,model,car_C_list,cids_list):#BD角、FL角、距離の3変数。一点評価（エネルギーを利用する方向に拡張可能）
    BD,low,high,radius_a,radius_b,FL_trans_paramater=paramater_cone#ここに変数を設定
    FL=0
    total_energy_list=[]
    for name in name_list:
        energies=energies_list[name]
        mini_model=model[name]
        car_C=car_C_list[name]#car_Cの番号
        total_energy=np.array([0.,0.])
        #print(energies)
        rate=np.array([Bd(energy[0]) for energy in energies])
        rate=rate/sum(rate)
        #print(rate)
        for Idx,energy in zip(cids_list[name],energies):
            energy=energy[0]
            conf_coordinate,radius,BD_axis,FL_axis,car_vec=mini_model[Idx]
            car_C_location=conf_coordinate[car_C]+FL_trans_paramater*FL_axis/np.linalg.norm(FL_axis)
            
            
            local_energy_list=[]
            for BD_angle in [BD,-BD]:
                
                location_vec=lotate_vec(BD_axis,FL_axis,car_vec,BD_angle,FL)
                local_energy=0
                for atom_location,atom_radius in zip(conf_coordinate,radius):
                    atom_to_car_C=atom_location-car_C_location#原子中心-car_C
                    inner=np.dot(location_vec,atom_to_car_C)#vとの内積
                    distance=np.linalg.norm(inner*location_vec-atom_to_car_C)#dを出す
                    #d=np.linalg.norm(atom_location-inner*location_vec+conf_coordinate[car_C])
                    imaginary_radius=imaginary_radius_maker(inner,radius_a,radius_b)#円錐台の半径をhの関数によって出す
                    tortion=atom_radius*2+imaginary_radius-distance
                    if inner>low and inner<high and tortion>0:
                        local_energy+=tortion**1*atom_radius**2
                #print(local_energy)
                #print(local_energy)
                local_energy_list.append(math.exp(-local_energy*2))
            #local_energy_list=np.array(local_energy_list)
            total_energy+=np.array(local_energy_list)*Bd(energy)
            #delta=local_energy_list[0]-local_energy_list[1]
            #total_energy+=(1/(1+math.exp(-10*delta))-0.5)*Bd(energy)
        total_energy_list.append(total_energy[0]/sum(total_energy))#/len(cids_list[name])
    print(total_energy_list)
    return np.array(total_energy_list)

"""バネモデル#ボツ"""
def culcurate_cone2(paramater_cone,energies_list,name_list,model,car_C_list,cids_list):#BD角、FL角、距離の3変数。一点評価（エネルギーを利用する方向に拡張可能）
    BD,low,high,radius_a,radius_b,power,times=paramater_cone#ここに変数を設定
    FL,FL_trans_paramater=0,0
    volume_rate=[]
    for i in [1,-1]:
        BD*=i
        volume_list=[]
        delta_h=(high-low)/20
        for name in name_list:
            energies=energies_list[name]
            mini_model=model[name]
            car_C=car_C_list[name]#car_Cの番号
            total_sqare=0
            for Idx,energy in zip(cids_list[name],energies):
                energy=energy[0]
                conf_coordinate,radius,BD_axis,FL_axis,car_vec=mini_model[Idx]
                car_C_location=conf_coordinate[car_C]+FL_trans_paramater*FL_axis/np.linalg.norm(FL_axis)
                location_vec=lotate_vec(BD_axis,FL_axis,car_vec,BD,FL)
                Idx_sqare=[]
                for h in np.arange(low, high, delta_h):
                    Idx_sqare_local=0
                    for atom_location,atom_radius in zip(conf_coordinate,radius):
                        imaginary_radius=imaginary_radius_maker(h,radius_a,radius_b)
                        d=np.linalg.norm(atom_location-h*location_vec+conf_coordinate[car_C])
                        tortion=atom_radius*times+imaginary_radius-d
                        if tortion>0:
                            Idx_sqare_local+=tortion**power/(10+h**2)#1/(1+d**2)#1/2*tortion#Δhを掛ける#足す
                    Idx_sqare.append(Idx_sqare_local)
                total_sqare+=sum(Idx_sqare)*Bd(energy)
                #print(total_sqare)
            volume_list.append(total_sqare)
            #print(conf_coordinate[car_C])
            #print(FL_trans_paramater*FL_axis/np.linalg.norm(FL_axis))
            #print(car_C_location)
        volume_rate.append(volume_list)
    volume_rate=np.array(volume_rate[0])-np.array(volume_rate[1])#np.array(volume_rate[0])/sum(np.array(volume_rate))
    return volume_rate


def sigmoid(a,x):
    return 80/(1+math.exp(a*(x)))+10#100/(1+math.exp(a*(x-0.5)))
def MSE_sigmoid(pro2,yld):
    da=0.1
    learning_rate=10
    times_sigmoid=200
    a=12#23.5
    def RMSE_sigomoid(a,pro2,yld):
        RMSE=math.sqrt(np.average([(sigmoid(a,i)-j)**2 for i,j in zip(pro2,yld)]))
        return RMSE
    for i in range(times_sigmoid):
        RMSE_sig_current=RMSE_sigomoid(a,pro2,yld)
        RMSE_sig_delta=RMSE_sigomoid(a+da,pro2,yld)-RMSE_sig_current
        a+=-RMSE_sig_delta*learning_rate
        #print(RMSE_sig_current,a)
    return RMSE_sig_current,a


def vol_opt(all_model,name_list,yld):
    model,car_C_list,cids_list,energies_list=all_model
    RMSE_sig_min=999#適当にでかい数字
    out_put="BAD"#初期値
    for BD in [-107]:#-105,-106,-107,-108,-109
        for FL in [0]:
            for bottom in [0.0,0.1,0.2,0.3,0.4]:#[0,0.1,0.2
                for top in [4]:#4,4.1,4.2,4.3,4.4
                    for radius_a in [0.9,1.0,1.1]:#1.4,1.5,1.6
                        for radius_b in [0]:#-0.01,0,0.01
                            for FL_trans_paramater in [0]:#-0.1,0,0.1
                                paramater_cone=[BD,bottom,top,radius_a,radius_b,FL_trans_paramater]#BD角、FL角、始点、終点1.3, 3.5, 0.1,-106,0,0.2,4.0
                                volume_rate=culcurate_cone(paramater_cone,energies_list,name_list,model,car_C_list,cids_list)
                                RMSE_sig_current,a=MSE_sigmoid(volume_rate,yld)
                                print(paramater_cone,RMSE_sig_current)
                                if RMSE_sig_min>RMSE_sig_current:
                                    out_put=paramater_cone,volume_rate,RMSE_sig_current,a
                                    RMSE_sig_min=RMSE_sig_current
    paramater_cone,volume_rate,RMSE_sig_min,a=out_put
    plt.title("cylinder radius vs R2")
    plt.plot( volume_rate,yld,linestyle='None',marker="o");
    x=[0.01*i for i in range(math.floor(min(volume_rate)*100),math.ceil(max(volume_rate)*100))]
    print(len(x))
    y=[sigmoid(a,i) for i in x]
    plt.plot(x,y,linestyle='solid',color = "orange")
    print(paramater_cone,RMSE_sig_min,a)
    plt.show()
    return out_put

def predict_graph(a,volume_rate,test_volume_rate,yld,test_yld):
    estimated=[sigmoid(a,i) for i in volume_rate]
    test_estimated=[sigmoid(a,i) for i in test_volume_rate]
    test_E=np.array(test_estimated)-np.array(test_yld)
    print(test_E)
    test_AE=[abs(i) for i in test_E]
    test_MAE=np.average(test_AE)
    print(test_MAE)
    plt.title("predicted yield vs experimental yield")
    plt.plot( estimated,yld,linestyle='None',marker="o");
    plt.plot( test_estimated,test_yld,linestyle='None',marker="o",color="red");
    #evaluate_liner=eva.kinji(volume_rate,yld)
    #print(evaluate_liner)

    x=[i for i in range(0,100)]
    print(len(x))
    #y=[cen.sigmoid(a,i) for i in x]
    plt.plot(x,x,linestyle='solid',color = "orange")
    #print(paramater_cone,RMSE_sig_min,a)
    plt.xlabel("predicted yield [%]")
    plt.ylabel("experimental yield [%]")
    plt.show()
    return "COMPLETE CULCULATION !"