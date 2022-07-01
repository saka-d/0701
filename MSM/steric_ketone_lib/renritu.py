#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import time
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""
def make_matrix(df_all,param):
    print("03011654")
    dim=param["dim"]
    num=param["num"]
    odds=param["odds"]
    x_num,y_num,z_num=param["xyz_num"]
    A=[]#np.zeros((len(df_all["cid"]),num**2))
    b=[]#[gibbs if not gibbs==np.inf else df_all["gibbs"]]
    #b=[df_all["gibbs"][i]*df_all["rate"][i][cid] for cid in df_all["cid"][i] for i in range(len(df_all["cid"]))]
    for i in range(len(df_all["cid"])):
        #print(i)
        for j,cid in enumerate(df_all["cid"][i]):
            a=np.zeros(num**2)
            
            a_3=np.zeros(x_num*y_num*z_num)
            a_div=np.zeros(x_num*y_num*z_num*2)
            #print(df_all["conf_coordinate"][i][cid])
            if True:
                for atom in df_all["sub_atom"][i]:#df_all['car_atom'][i]:#
                    coordinate=df_all["conf_coordinate"][i][cid][atom]*odds
                    x=round(coordinate[0]+x_num/2-odds*param["x_center"])
                    y=abs(coordinate[1])
                    y_3=round(abs(coordinate[1]))
                    z=round(abs(coordinate[2]))
                    weight=np.sign(coordinate[2])*df_all["rate"][i][cid]*df_all["charges"][i][atom]*100 if param["type"]=="electric" else np.sign(coordinate[2])*df_all["rate"][i][cid]#通常時、zの符号を加味する。
                    if dim==2 and 0<=x<=num-1 and 0<=z<=num-1 and y<=odds*10:
                        a[x+z*num]+=weight#np.sign(coordinate[2])*df_all["rate"][i][cid]
                    if dim==3 and 0<=x<=x_num-1 and z<=z_num-1 and y_3<=y_num-1:
                            l=x+y_3*x_num+z*x_num*y_num
                            if param["divCH"]:
                                if df_all["symbol"][i][atom]=="H": l+=x_num*y_num*z_num#0220_kをatomに変更
                                a_div[l]+=weight#np.sign(coordinate[2])*df_all["rate"][i][cid]
                            else: a_3[l]+=weight#np.sign(coordinate[2])*df_all["rate"][i][cid]#*weight
            if dim==2: A.append(a)
            if dim==3:
                A.append(a_div if param["divCH"] else a_3)
            b.append(df_all["gibbs"][i]*df_all["rate"][i][cid])
    return A,b
"""
def average_matrix(param):
    dim=param["dim"]
    odds=param["odds"]
    num3=param["ave"]
    num4=param["ave0"]
    num=param["num"]
    x_num,y_num,z_num=param["xyz_num"]
    A=[]
    b=[]
    """
    if dim==2:
        if False:#平均化#0126
            for n in range(3):
                for i, j in product(range(num-(1 if n==0 else 0)), range(num-(1 if n==1 else 0))):
                    a1=np.zeros(num**2)
                    l=i+j*num
                    if n!=2: a1[l],a1[l+num**n]=num3,-num3
                    else: a1[l]=num4
                    A.append(a1)
                    b.append(0)
        else:
            for i, j in product(range(num), range(num-1)):
                l=i*num+j
                a1=np.zeros(num**2)
                a1[l],a1[l+1]=num3,-num3
                A.append(a1)
                b.append(0)
            for i, j in product(range(num-1), range(num)):
                l=i*num+j
                a2=np.zeros(num**2)
                a2[l],a2[l+num]=num3,-num3
                A.append(a2)
                b.append(0)
            for i, j in product(range(num), range(num)):
                a3=np.zeros(num**2)
                a3[i*num+j]=num4
                A.append(a3)
                b.append(0)
    """
    """
    if dim==3 and False:
        
        if True:
            for n in range(4):#隣接セルを平均
                for i, j, k in product(range(num-(1 if n==0 else 0)), range(num-(1 if n==1 else 0)),range(num-(1 if n==2 else 0))):
                    a1=np.zeros(num**3)
                    l=i+j*num+k*num**2
                    if n!=3:a1[l],a1[l+num**n]=num3,-num3
                    else: a1[l]=num4
                    A.append(a1)
                    b.append(0)
        else:
            for n in range(3):#隣接セルを平均
                for i, j, k in product(range(num-(1 if n==0 else 0)), range(num-(1 if n==1 else 0)),range(num-(1 if n==2 else 0))):
                    a1=np.zeros(num**3)
                    l=i+j*num+k*num**2
                    a1[l],a1[l+num**n]=num3,-num3
                    A.append(a1)
                    b.append(0)
            for i, j, k in product(range(num), range(num),range(num)):
                a3=np.zeros(num**3)
                l=i*num+j+k*num**2
                a3[l]=num4
                A.append(a3)
                b.append(0)
    """
    """
    if False:
        if dim==3 and True:#ガウス関数型平均化
            n_elem=2 if param["divCH"] else 1
            for number in range(n_elem):
                for i,j,k in product(range(num), range(num),range(num)):
                    a1=np.zeros(num**3*n_elem)
                    l=i*num**2+j*num+k+number*num**3
                    for i2,j2,k2 in product(range(num), range(num),range(num)):
                        l2=i2*num**2+j2*num+k2+number*num**3
                        d=np.linalg.norm(np.array([i,j,k])-np.array([i2,j2,k2]))#/param["odds"]
                        if 0<d<2: a1[l2]=num3*np.exp(-d**2)
                        if 0<d<2: a1[l2]=1/(np.sqrt(2*np.pi)*param["gauss_c"])*num3*np.exp(-d**2/param["gauss_c"]**2)
                    a1[l]=-np.sum(a1)
                    A.append(a1)
                    b.append(0)
    """
    if True:
        if dim==3 and True:#ガウス関数型平均化
            n_elem=2 if param["divCH"] else 1
            for number in range(n_elem):
                for i,j,k in product(range(x_num), range(y_num),range(z_num)):
                    a1=np.zeros(x_num*y_num*z_num*n_elem)
                    l=i+j*x_num+k*x_num*y_num+number*x_num*y_num*z_num
                    if not param["edge"]:
                        for i2,j2,k2 in product(range(x_num), range(y_num),range(z_num)):
                            l2=i2+j2*x_num+k2*x_num*y_num+number*x_num*y_num*z_num
                            d=np.linalg.norm(np.array([i,j,k])-np.array([i2,j2,k2]))#/param["odds"]
                            #if 0<d<2: a1[l2]=num3*np.exp(-d**2)
                            if 0<d<1.1: a1[l2]=param["gauss_c"]/d*num3#a1[l2]=param["gauss_c"]**2/(d/param["odds"])**2*num3#1/(np.sqrt(2*np.pi)*param["gauss_c"])*num3*np.exp(-(d/param["odds"])**2/param["gauss_c"]**2)#0302_2→5へ
                        a1[l]=-np.sum(a1)
                    else:
                        sum_ans_=0
                        for i2,j2,k2 in product(range(-9,10), range(-9,10),range(-9,10)):#0607,3から10に書きかえ
                            d=np.linalg.norm(np.array([i2,j2,k2]))
                            #if 0<d<2: a1[l2]=num3*np.exp(-d**2)
                            if 0<d<10:#0607,3から10に書きかえ
                                ans_=param["gauss_c"]/(d**param["power"])*num3
                                #ans_=param["gauss_c"]**2/(d/param["odds"])**2*num3#1/(np.sqrt(2*np.pi)*param["gauss_c"])*num3*np.exp(-(d/param["odds"])**2/param["gauss_c"]**2)
                                if False:
                                    if 0<=j+j2: sum_ans_+=ans_
                                    if 0<=i+i2<x_num and 0<=j+j2<y_num and 0<=k+k2<z_num:
                                        l2=(i+i2)+(j+j2)*x_num+(k+k2)*x_num*y_num+number*x_num*y_num*z_num
                                        a1[l2]=ans_
                                else:
                                    
                                    if 0<=i+i2<x_num and j+j2<y_num and k+k2<z_num:
                                        l2=(i+i2)+abs(j+j2)*x_num+abs(k+k2)*x_num*y_num+number*x_num*y_num*z_num
                                        a1[l2]=ans_*np.sign(k+k2)
                                        sum_ans_+=ans_
                                    else:
                                        sum_ans_+=2*ans_#差を増大させる
                                    
                        a1[l]=-sum_ans_
                    A.append(a1)
                    b.append(0)
    """
    else:
        if dim==3 and not param["divCH"] and True:#ガウス関数型平均化
            for i,j,k in product(range(num), range(num),range(num)):
                a1=np.zeros(num**3)
                l=i*num**2+j*num+k
                for i2,j2,k2 in product(range(num), range(num),range(num)):
                    l2=i2*num**2+j2*num+k2
                    d=np.linalg.norm(np.array([i,j,k])-np.array([i2,j2,k2]))
                    if 0<d<2: a1[l2]=num3*np.exp(-d**2)
                a1[l]=-np.sum(a1)
                A.append(a1)
                b.append(0)
        if dim==3 and param["divCH"] and True:
            for number in range(2):
                for i,j,k in product(range(num), range(num),range(num)):
                    a1=np.zeros(num**3*2)
                    l=i*num**2+j*num+k+number*num**3
                    for i2,j2,k2 in product(range(num), range(num),range(num)):
                        l2=i2*num**2+j2*num+k2+number*num**3
                        d=np.linalg.norm(np.array([i,j,k])-np.array([i2,j2,k2]))
                        if 0<d<2: a1[l2]=num3*np.exp(-d**2)
                    a1[l]=-np.sum(a1)
                    A.append(a1)
                    b.append(0)
    """
    return A,b

def f(i,j,k,X,num,number):
    l=i+j*num+k*num**2+number*num**3
    for x, y, z in product(range(num), range(num),range(num)):
        m=x+y*num+z*num**2+number*num**3
        l[y,x,z]=X[int(m)]#25*x+5*y+z###
    return l
if False:
    def f(i,j,k,X,num,number):
        l=i+j*num+k*num**2+number*num**3
        for x, y, z in product(range(num), range(num),range(num)):
            m=x+y*num+z*num**2+number*num**3
            l[y,x,z]=X[int(m)]#25*x+5*y+z###
        return l

###↓0606追加
def dist_graph(A,param):
    X=sum(np.abs(A))
    len(X)
    fig = plt.figure()
    num=param["num"]
    number=2 if param["divCH"] else 0
    odds=param["odds"]
    x_num,y_num,z_num=param["xyz_num"]
    def f(i,j,k):
        #l=i+j*x_num+k*x_num*y_num+number*np.prod(param["xyz_num"])
        l=i+j*x_num+k*x_num*y_num+number*np.prod(param["xyz_num"])
        #print(l)
        for x, y, z in product(range(x_num), range(y_num),range(z_num)):
            m=x+y*x_num+z*x_num*y_num+number*np.prod(param["xyz_num"])
            #l[y,x,z]=X[int(m)]#25*x+5*y+z###
            l[x,y,z]=X[int(m)]
        return l
    data = np.fromfunction(lambda i,j,k : f(i,j,k), (x_num,y_num,z_num), dtype=float)

    #X_,Y_,Z_ = np.meshgrid(range(num), range(num), range(num))
    #Y_,Z_,X_ = np.meshgrid(range(x_num), range(y_num), range(z_num))
    X_,Y_,Z_ = np.meshgrid(range(x_num), range(y_num), range(z_num))
    X_=X_-(x_num/2-odds*param["x_center"])#-かｒ+に変更
    X_,Y_,Z_=X_/odds,Y_/odds,Z_/odds
    #X_=X_+param["x_center"]
    ax = fig.add_subplot(111, projection='3d', aspect='auto',alpha=1)
    #ax_ = fig.add_subplot(111, projection='3d', aspect='auto',alpha=1)
    cv=(X-np.min(X))/(np.max(X)-np.min(X))
    cmap = plt.cm.jet

    colors = [None for k in range(len(cv))]
    for i, j, k in product(range(x_num), range(y_num),range(z_num)):
        l=i+j*x_num+k*x_num*y_num+number*np.prod(param["xyz_num"])
        #print(cmap(cv[l]))
        c_r, c_g, c_b,_= cmap(cv[l])
        #l=j+i*x_num+k*x_num*y_num+number*np.prod(param["xyz_num"])
        l=i+j*x_num+k*x_num*y_num+number*np.prod(param["xyz_num"])
        colors[l] = (c_r, c_g, c_b, 0.5 if cv[l]!=0 else 0)
        #sc = ax.scatter(i,j,k, c=colors[l],s=100)
    sc = ax.scatter(X_, Y_, Z_, c=colors,s=100)#alpha=(data-min(data))/(max(data)-min(data))
    #sc = ax.scatter(Z_, X_, Y_, c=colors,s=100)#alpha=(data-min(data))/(max(data)-min(data))
    #sc=ax_.scatter(X_, Y_, Z_, c=data,alpha=1,s=100)
    fig.colorbar(sc,alpha=1)
    if True:
        ax.scatter(0,0,0, label='center',color="blue",alpha =1,s=400)
        ax.scatter(1.2,0,0, label='center',color="red",alpha =1,s=400)
        ax.scatter(-1,1,0, label='center',color="black",alpha =1,s=400)
        ax.scatter(-1,-1,0, label='center',color="black",alpha =1,s=400)
    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_zlabel('z [Å]')
    ax.view_init(elev=10, azim=-85)
    plt.show()
###↑0606追加



def graph(X,param):
    num=param["num"]
    odds=param["odds"]
    for number in range(1 if not param["divCH"] else 2):
        fig = plt.figure()
        #flag=2#0126
        #if flag==1:

        if param["dim"]==3:
            """
            if False:
                ax = Axes3D(fig)
                for i, j, k in product(range(num), range(num),range(num)):
                    l=i+j*num+k*num**2+number*num**3
                    x=(i-(num-1)//2)/odds
                    y=j/odds
                    z=k/odds
                    colorscale=(X[l]-np.min(X))/(np.max(X)-np.min(X))
                    #print(colorscale)
                    ax.scatter3D(x,y,z, label='Dataset',color=cm.jet(colorscale),alpha =0.5,s=100)
                ax.scatter3D(0,0,0, label='center',color="blue",alpha =0.5,s=400)
                ax.scatter3D(1.2,0,0, label='center',color="red",alpha =0.5,s=400)
                ax.scatter3D(-1,1,0, label='center',color="black",alpha =0.5,s=400)
                ax.scatter3D(-1,-1,0, label='center',color="black",alpha =0.5,s=400)
                #if flag==2:
            """
            if True:
                if True:
                    def f(i,j,k):
                        l=i+j*num+k*num**2+number*num**3
                        for x, y, z in product(range(num), range(num),range(num)):
                            m=x+y*num+z*num**2+number*num**3
                            l[y,x,z]=X[int(m)]#25*x+5*y+z###
                        return l
                    data = np.fromfunction(lambda i,j,k : f(i,j,k), (num,num,num), dtype=float)
                """
                elif False:
                    def f(i,j,k):
                        l=i+j*num+k*num**2+number*num**3
                        for x, y, z in product(range(num), range(num),range(num)):
                            m=x+y*num+z*num**2+number*num**3
                            l[x,y,z]=X[int(m)]#25*x+5*y+z###
                        return l
                    data = np.fromfunction(lambda i,j,k : f(i,j,k), (num,num,num), dtype=float)
                """
                """
                elif True:
                    cv=(X-np.min(X))/(np.max(X)-np.min(X))
                    cmap = plt.cm.jet
                    def f(i,j,k):
                        l=i+j*num+k*num**2+number*num**3
                        for x, y, z in product(range(num), range(num),range(num)):
                            m=x+y*num+z*num**2+number*num**3
                            c_r, c_g, c_b,_= cmap(cv[m])
                            l[y,x,z]=(c_r, c_g, c_b, 0.5 if cv[m]!=0 else 0)#25*x+5*y+z###
                        return l
                    data = np.fromfunction(lambda i,j,k : f(i,j,k), (num,num,num), dtype=float)
                else:
                    data = np.fromfunction(lambda i,j,k : f(i,j,k), (num,num,num), dtype=float)
                """
                #X_,Y_,Z_ = np.meshgrid(range(-((num-1)//2),num-((num-1)//2)), range(num), range(num))
                #X_,Y_,Z_ = np.meshgrid(range(-(num+param["x_center"])//2,num-(num+param["x_center"])//2), range(num), range(num))
                X_,Y_,Z_ = np.meshgrid(range(num), range(num), range(num))
                X_=X_-(num/2-odds*param["x_center"])
                X_,Y_,Z_=X_/odds,Y_/odds,Z_/odds
                ax = fig.add_subplot(111, projection='3d', aspect='auto',alpha=1)
                #ax_ = fig.add_subplot(111, projection='3d', aspect='auto')
                #data=(data-np.min(data))/(np.max(data)-np.min(data))
                #sc = ax.scatter(X_, Y_, Z_, c=data,alpha=1, cmap='jet',s=5)
                sc = ax.scatter(X_, Y_, Z_, c=data,alpha=0.2, cmap='jet',s=100)
                #sc=ax_.scatter(X_, Y_, Z_, c=data,alpha=1,s=100)
                #ax = fig.add_subplot(111, projection='3d', aspect='auto',alpha=1)
                fig.colorbar(sc)

                ax.scatter(0,0,0, label='center',color="blue",alpha =1,s=400)
                ax.scatter(1.2,0,0, label='center',color="red",alpha =1,s=400)
                ax.scatter(-1,1,0, label='center',color="black",alpha =1,s=400)
                ax.scatter(-1,-1,0, label='center',color="black",alpha =1,s=400)
            ax.set_xlabel('x [Å]')
            ax.set_ylabel('y [Å]')
            ax.set_zlabel('z [Å]')
            ax.view_init(elev=10, azim=-85)
            plt.show()
        """
        if param["dim"]==2:
            X2=np.reshape(X, (num, num))
            sns.heatmap(X2)
        """
            
            
def sigmoid(x): return 1/(1+np.exp(-0.2*x))
def activation(x): return sigmoid(x)
def activation_dash(x): return 0.2*sigmoid(x)*(1-sigmoid(x))# 活性化関数の微分

def forward(x,w1, w2, b1, b2): return w2 @ activation(w1 @ x + b1) + b2# 順方向。学習結果の利用。

# 逆方向。学習
def backward(x, diff,learn_rate,w1, w2, b1, b2):
    #print(w2)
    #global w1, w2, b1, b2
    v1 = (diff @ w2) * activation_dash(w1 @ x + b1)
    v2 = activation(w1 @ x + b1)
    w1 -= learn_rate * np.outer(v1, x)  # outerは直積
    b1 -= learn_rate * v1
    w2 -= learn_rate * np.outer(diff, v2)
    b2 -= learn_rate * diff
    return w1, w2, b1, b2

def NN(A,b,param):
    if False:
        dim_in = param["num"]**param["dim"]*(2 if param["divCH"] else 1)#1#入力は1次元
        
    else:
        dim_in = np.prod(param["xyz_num"])*(2 if param["divCH"] else 1)#1#入力は1次元
    dim_out = 1             # 出力は1次元
    hidden_count = 100     # 隠れ層のノードは1024個
    learn_rate = param["learn_rate"]#0.00001 if False else 0.001      # 学習率,線形ならTrue,sigmoidならFalse
    
    # 訓練データは x は -1～1、y は 2 * x ** 2 - 1
    train_count = len(b)        # 訓練データ数
    #train_x = np.arange(-1, 1, 2 / train_count).reshape((train_count, dim_in))
    train_x = np.array(A).reshape((train_count, dim_in))#np.arange(-1, 1, 2 / train_count).reshape((train_count//dim_in, dim_in))#
    #print(train_x)
    train_y = np.array(b).reshape((train_count, dim_out))#np.array([2 * x ** 2 - 1 for x in train_x]).reshape((train_count, dim_out))#
    #print(train_y)
    # 重みパラメータ。-0.5 〜 0.5 でランダムに初期化。この行列の値を学習する。
    w1 = np.random.rand(hidden_count, dim_in) - 0.5
    w2 = np.random.rand(dim_out, hidden_count) - 0.5
    b_1 = np.random.rand(hidden_count) - 0.5
    b_2 = np.random.rand(dim_out) - 0.5
    # メイン処理
    idxes = np.arange(train_count)#//dim_in)          # idxes は 0～63
    print(idxes)
    for epoc in range(100):                # 1000エポック
        np.random.shuffle(idxes)            # 確率的勾配降下法のため、エポックごとにランダムにシャッフルする
        error = 0                           # 二乗和誤差
        if True:
            for idx in idxes:
                y__ = forward(train_x[idx],w1, w2, b_1, b_2)       # 順方向で x から y を計算する

                diff = y__ - train_y[idx]         # 訓練データとの誤差
                error += diff ** 2              # 二乗和誤差に蓄積
                w1, w2, b1, b2=backward(train_x[idx], diff,learn_rate,w1, w2, b_1, b_2)    # 誤差を学習
            print((error/len(idxes))**0.5)#.sum())                  # エポックごとに二乗和誤差を出力。徐々に減衰して0に近づく。
        else:
            y__ = np.array([forward(train_x[idx],w1, w2, b_1, b_2) for idx in idxes])
            diff =( y__ - train_y)**2
            for idx in idxes:
                w1, w2, b1, b2=backward(train_x[idx], diff[idx],learn_rate,w1, w2, b_1, b_2)
    return w1, w2, b_1, b_2

###0606追加↓
from matplotlib.colors import Normalize
def graph(X,param):
    num=param["num"]
    odds=param["odds"]
    x_num,y_num,z_num=param["xyz_num"]
    for number in range(1 if not param["divCH"] else 2):
        fig = plt.figure()
        #flag=2#0126
        #if flag==1:

        if param["dim"]==3:
            """
            if False:
                ax = Axes3D(fig)
                for i, j, k in product(range(num), range(num),range(num)):
                    l=i+j*num+k*num**2+number*num**3
                    x=(i-(num-1)//2)/odds
                    y=j/odds
                    z=k/odds
                    colorscale=(X[l]-np.min(X))/(np.max(X)-np.min(X))
                    #print(colorscale)
                    ax.scatter3D(x,y,z, label='Dataset',color=cm.jet(colorscale),alpha =0.5,s=100)
                ax.scatter3D(0,0,0, label='center',color="blue",alpha =0.5,s=400)
                ax.scatter3D(1.2,0,0, label='center',color="red",alpha =0.5,s=400)
                ax.scatter3D(-1,1,0, label='center',color="black",alpha =0.5,s=400)
                ax.scatter3D(-1,-1,0, label='center',color="black",alpha =0.5,s=400)
                #if flag==2:
            """
            if True:#for elev_,azim_ in zip([5,5],[85,5]):
                ax = Axes3D(fig)
                if False:
                    for i, j, k in product(range(x_num), range(y_num),range(z_num)):
                        l=i+j*x_num+k*x_num*y_num+number*np.prod(param["xyz_num"])
                        x=(i-(x_num)//2)/odds+param["x_center"]

                        y=j/odds
                        z=k/odds
                        colorscale=(X[l]-np.min(X))/(np.max(X)-np.min(X))
                        #print(colorscale)
                        ax.scatter3D(x,y,z, label='Dataset',color=cm.jet(colorscale),alpha =(0 if colorscale==0 else 1),s=10)
                        #c=value, cmap=cm
                else:
                    
                    l_list,x_list,y_list,z_list,X_list,alpha_list=[],[],[],[],[],[]
                    for i, j, k in product(range(x_num), range(y_num),range(z_num)):
                        l=i+j*x_num+k*x_num*y_num+number*np.prod(param["xyz_num"])
                        x=(i-(x_num)//2)/odds+param["x_center"]
                        y=j/odds
                        z=k/odds
                        if X[l]!=0:
                            l_list.append(l)
                            x_list.append(x)
                            y_list.append(y)
                            z_list.append(z)
                            X_list.append(X[l])
                        #alpha_list.append((0 if X[l]==0 else 1))
                    # カラーマップを生成
                    cm = plt.cm.get_cmap("jet")#('RdYlBu')cool Reds
                    # axに散布図を描画、戻り値にPathCollectionを得る
                    mappable = ax.scatter(x_list, y_list, z_list, c=X_list, cmap=cm,norm=Normalize(vmin=-1, vmax=1),alpha=1,s=10)
                    #fig.colorbar(mappable, ax=ax)
                ax.scatter3D(0,0,0, label='center',color="blue",alpha =0.5,s=400)
                ax.scatter3D(1.2,0,0, label='center',color="red",alpha =0.5,s=400)
                ax.scatter3D(-1,1,0, label='center',color="black",alpha =0.5,s=400)
                ax.scatter3D(-1,-1,0, label='center',color="black",alpha =0.5,s=400)
                ax.set_xlabel('x [Å]')
                ax.set_ylabel('y [Å]')
                ax.set_zlabel('z [Å]')
                ax.view_init(elev=param["elev"], azim=param["azim"])#-85
                fig.colorbar(mappable, ax=ax)
                plt.show()
            if False:
                if True:
                    def f(i,j,k):
                        l=i+j*num+k*num**2+number*num**3
                        for x, y, z in product(range(x_num), range(y_num),range(z_num)):#product(range(num), range(num),range(num)):
                            #m=x+y*num+z*num**2+number*num**3
                            m=x+y*x_num+z*x_num*y_num+number*np.prod(param["xyz_num"])
                            l[y,x,z]=X[int(m)]#25*x+5*y+z###
                        return l
                    #data = np.fromfunction(lambda i,j,k : f(i,j,k), (num,num,num), dtype=float)
                    data = np.fromfunction(lambda i,j,k : f(i,j,k), (x_num,y_num,z_num), dtype=float)
            if True:
                if True:
                    def f(i,j,k):
                        l=i+j*x_num+k*x_num*y_num+number*np.prod(param["xyz_num"])
                        for x, y, z in product(range(x_num), range(y_num),range(z_num)):
                            m=x+y*x_num+z*x_num*y_num+number*np.prod(param["xyz_num"])
                            
                            l[x,y,z]=X[int(m)]#25*x+5*y+z###
                        return l
                    data = np.fromfunction(lambda i,j,k : f(i,j,k), (x_num,y_num,z_num), dtype=float)
                """
                elif False:
                    def f(i,j,k):
                        l=i+j*num+k*num**2+number*num**3
                        for x, y, z in product(range(num), range(num),range(num)):
                            m=x+y*num+z*num**2+number*num**3
                            l[x,y,z]=X[int(m)]#25*x+5*y+z###
                        return l
                    data = np.fromfunction(lambda i,j,k : f(i,j,k), (num,num,num), dtype=float)
                elif True:
                    cv=(X-np.min(X))/(np.max(X)-np.min(X))
                    cmap = plt.cm.jet
                    def f(i,j,k):
                        l=i+j*num+k*num**2+number*num**3
                        for x, y, z in product(range(num), range(num),range(num)):
                            m=x+y*num+z*num**2+number*num**3
                            c_r, c_g, c_b,_= cmap(cv[m])
                            l[y,x,z]=(c_r, c_g, c_b, 0.5 if cv[m]!=0 else 0)#25*x+5*y+z###
                        return l
                    data = np.fromfunction(lambda i,j,k : f(i,j,k), (num,num,num), dtype=float)
                else:
                    data = np.fromfunction(lambda i,j,k : f(i,j,k), (num,num,num), dtype=float)
                """
                
            #X_,Y_,Z_ = np.meshgrid(range(-((num-1)//2),num-((num-1)//2)), range(num), range(num))
            #X_,Y_,Z_ = np.meshgrid(range(-(num+param["x_center"])//2,num-(num+param["x_center"])//2), range(num), range(num))
            if False:
                X_,Y_,Z_ = np.meshgrid(range(num), range(num), range(num))
                X_=X_-(num/2-odds*param["x_center"])
            else:
                X_,Y_,Z_ = np.meshgrid(range(x_num), range(y_num), range(z_num))
                X_=X_-(x_num/2-odds*param["x_center"])
            X_,Y_,Z_=X_/odds,Y_/odds,Z_/odds
            ax = fig.add_subplot(111, projection='3d', aspect='auto',alpha=1)
            #ax_ = fig.add_subplot(111, projection='3d', aspect='auto')
            #data=(data-np.min(data))/(np.max(data)-np.min(data))
            sc = ax.scatter(X_, Y_, Z_, c=data,alpha=1, cmap='jet',s=5)
            #sc = ax.scatter(X_, Y_, Z_, c=data,alpha=0.2, cmap='jet',s=100)
            #sc=ax_.scatter(X_, Y_, Z_, c=data,alpha=1,s=100)
            #ax = fig.add_subplot(111, projection='3d', aspect='auto',alpha=1)
            fig.colorbar(sc)

            ax.scatter(0,0,0, label='center',color="blue",alpha =1,s=400)
            ax.scatter(1.2,0,0, label='center',color="red",alpha =1,s=400)
            ax.scatter(-1,1,0, label='center',color="black",alpha =1,s=400)
            ax.scatter(-1,-1,0, label='center',color="black",alpha =1,s=400)
            ax.set_xlabel('x [Å]')
            ax.set_ylabel('y [Å]')
            ax.set_zlabel('z [Å]')
            ax.view_init(elev=10, azim=-85)#-85
            plt.show()
        """
        if param["dim"]==2:
            X2=np.reshape(X, (num, num))
            sns.heatmap(X2)
        """
def dist_bar_graph(yld,df_all):
    bins = np.linspace(-50, 50, 11)
    #print(bins)

    freq = (yld-df_all["yld"]).value_counts(bins=bins, sort=False)
    #print(freq)
    class_value = (bins[:-1] + bins[1:]) / 2  # 階級値
    rel_freq = freq / df_all["yld"].count()  # 相対度数
    cum_freq = freq.cumsum()  # 累積度数
    rel_cum_freq = rel_freq.cumsum()  # 相対累積度数
    dist = pd.DataFrame(
        {
            "階級値": class_value,
            "度数": freq,
            "相対度数": rel_freq,
            "累積度数": cum_freq,
            "相対累積度数": rel_cum_freq,
        },
        index=freq.index
    )
    dist
    dist.plot.bar(x="階級値", y="度数", width=1, ec="k", lw=2)
    plt.show()
###0606追加↑


