#!/usr/bin/env python
# coding: utf-8
# In[ ]:
import math
import numpy as np
import matplotlib.pyplot as plt#グラフ描画に使用s
from sklearn.metrics import mean_squared_error
import time
from itertools import product
from scipy.optimize import minimize
from scipy.optimize import fmin
from math import *
def sphere(r,R,d):
    if d==0 or r+R<d: return 0
    elif d<abs(r-R): return 4/3*np.pi*min(r,R)**3
    return np.pi/12*(d**3-6*d*(r**2+R**2)+8*(r**3+R**3)-3/d*(r**2-R**2)**2)

def n_val(r,R,d,n):
    if d==0: return 0
    if r+R<d: return 0
    tortion=(r+R-d)/(r+R)
    volume_0=(4/3)*np.pi*min([r,R])**3
    return tortion**n*volume_0

def Bd(x,T):
    if T!=T: T=273.15+20
    return math.exp(-x/1.99/T*0.001)#ボルツマン分布
def coordinate_transformation(atom_list_2,df,i):
    c0num ,onum,c1num ,c2num =df["car_atom"][i]
    cx = atom_list_2[c0num][0]
    cy = atom_list_2[c0num][1]
    cz = atom_list_2[c0num][2]
    """回転"""
    atom_list_3=atom_list_2
    atom_list={}
    if True:
        for ang in [30,-30]:
            atom_list_2=atom_list_3
            ry=np.deg2rad(ang)if False  else 0
            for atom,ho1 in enumerate(atom_list_2):
                if atom not in [c0num ,onum]: (ho1[0],ho1[2]) = (ho1[0]*cos(ry)-ho1[2]*sin(ry),ho1[0]*sin(ry)+ho1[2]*cos(ry))
            atom_list[np.sign(ang)]=atom_list_2
    return atom_list
"""重なり体積リストを作成"""
def cul(df,sr,n):
    df["distance"]="nan"
    #distance0105={}
    #distance0107={}
    df["inner"]="nan"
    df["tortion"]="nan"
    df["sqare_local"]="nan"
    df["sqare_bottom"]="nan"
    df["sqare_top"]="nan"
    for i,name in enumerate(df["smiles"]):
        print(i)
        df["distance"][i]={}
        #distance0105[i]={}
        df["inner"][i]={}
        df["tortion"][i]={}
        df["sqare_local"][i]={}
        df["sqare_bottom"][i]={}
        df["sqare_top"][i]={}
        for cid in df["cid"][i]:
            if False:
                df["sqare_local"][i][cid]={}
                df["sqare_bottom"][i][cid]={}
                df["sqare_top"][i][cid]={}
            atom_list=coordinate_transformation(df["conf_coordinate"][i][cid],df,i)
            #for atom in range(len(df["mass"][i])):#0125削除
            for atom in df["sub_atom"][i]:
                atom_to_car_C=df["conf_coordinate"][i][cid][atom]#df["atom_from_car_C"][i][cid][atom]#df["conf_coordinate"][i][cid][atom]-car_C_location#原子中心-car_C
                #r,a,b=df["psi"][i][cid][atom]
                #for BD_angle in np.concatenate([sr["BD"],-1*sr["BD"]]):
                for BD_angle,ang in zip(sr["BD+-"],sr["ang+-"]):
                    inner=np.inner(sr["vec"][BD_angle],atom_to_car_C)
                    distance=np.linalg.norm(inner*sr["vec"][BD_angle]-atom_to_car_C)#dを出す
                    #df["distance"][i][cid][atom][BD_angle]=distance
                    df["distance"][i][cid,atom,BD_angle]=distance
                    df["inner"][i][cid,atom,BD_angle]=inner
                    #df["tortion"][i][cid][atom][BD_angle]={}
                    for radius in sr["radius_a"]:
                        #df["tortion"][i][cid][atom][BD_angle][radius]=tortion if (tortion:=radius+df["cube_radius"][i][cid][atom]-distance)>0 else 0
                        df["tortion"][i][cid,atom,BD_angle,radius]=tortion if (tortion:=radius+df["radius"][i][atom]-distance)>0 else 0
            for BD_angle in sr["BD"]:
                if False:
                    df["sqare_local"][i][cid][BD_angle]={}
                    df["sqare_bottom"][i][cid][BD_angle]={}
                    df["sqare_top"][i][cid][BD_angle]={}
                for radius in sr["radius_a"]:
                    if False:
                        df["sqare_local"][i][cid][BD_angle][radius]={}
                        df["sqare_bottom"][i][cid][BD_angle][radius]={}
                        df["sqare_top"][i][cid][BD_angle][radius]={}
                    atom_list={1:[],-1:[]}
                    #for atom in range(len(df["mass"][i])):
                    for atom in df["sub_atom"][i]:
                        for flag in [1,-1]:
                            if df["tortion"][i][cid,atom,BD_angle*flag,radius]>0: atom_list[flag].append(atom)
                    for flag in [1,-1]:
                        #df["sqare_local"][i][cid][BD_angle][radius][flag]={}#追加0105
                        df["sqare_local"][i][cid,BD_angle,radius,flag]={}
                        for atom in atom_list[flag]:
                            #inner=df["inner"][i][cid][atom][BD_angle*flag]
                            inner=df["inner"][i][cid,atom,BD_angle*flag]
                            #distance=df["distance"][i][cid][atom][BD_angle*flag]
                            distance=df["distance"][i][cid,atom,BD_angle*flag]
                            
                            #atom_radius=df["cube_radius"][i][cid][atom]#0117削除
                            atom_radius=df["radius"][i][atom] if True else df["cube_radius"][i][cid][atom]#0117削除
                            ans1221=sphere(radius,atom_radius,distance) if n=="volume" else n_val(radius,atom_radius,distance,n)
                            if False:#0128
                                if ans1221>0: df["sqare_local"][i][cid][BD_angle][radius][flag][inner]=ans1221 if n!=2 else ans1221/(1+df["path"][i][atom])#0119追加#追加0105
                            else:
                                if ans1221>0: df["sqare_local"][i][cid,BD_angle,radius,flag][inner]=ans1221 if n!=2 else ans1221/(1+df["path"][i][atom])#0119追加#追加0105
                        #for flag in [1,-1]:
                        for botortop,sqare_x in zip(["bottom","top"],["sqare_bottom","sqare_top"]):
                            #df[sqare_x][i][cid][BD_angle][radius][flag]={}#0128
                            for botop in sr[botortop]:
                                vol_=0
                                for atom in atom_list[flag]:
                                    if (bi:=(botop-inner)*(1 if botortop=="bottom" else -1))>0:
                                        dis=np.sqrt(distance**2+bi**2)
                                        vol_+=sphere(radius,atom_radius,dis) if n=="volume" else n_val(radius,atom_radius,dis,n)
                                if False:
                                    df[sqare_x][i][cid][BD_angle][radius][flag][botop]=vol_ if True else vol_/(1+df["path"][i][atom])#0119追加
                                else:
                                    df[sqare_x][i][cid,BD_angle,radius,flag,botop]=vol_ if True else vol_/(1+df["path"][i][atom])#0119追加
    return df

"""体積総和を取り、予測値を算出"""
#"""
def culcurate_cone(param,const_list,df,flag_,sum_flag):#BD角、FL角、距離の3変数。一点評価（エネルギーを利用する方向に拡張可能）#旧バージョン
    top=float(param["top"])
    bottom=float(param["bottom"])
    radius_a=float(param["radius_a"])
    BD_angle=float(param["BD"])
    predict={}
    predict2={}
    sqare_list={}
    for i,name in enumerate(df["smiles"]):
        #print(name)
        predict[i]={}
        predict2[i]={}
        sqare_list[i]={}
        for cid in df["cid"][i]:
            sqare=0
            volume=np.zeros(2)#{1:0,-1:0}
            predict[i][cid]={}
            predict2[i][cid]={}
            if sum_flag=="sum":
                for j,flag in enumerate([1,-1]):
                    #for key,value in zip(df["sqare_local"][i][cid][BD_angle][radius_a][flag].keys(),df["sqare_local"][i][cid][BD_angle][radius_a][flag].values()):#0128
                    for key,value in zip(df["sqare_local"][i][cid,BD_angle,radius_a,flag].keys(),df["sqare_local"][i][cid,BD_angle,radius_a,flag].values()):
                        if bottom<=float(key)<top:#float(key)>=bottom and float(key)<top:
                            sqare+=value*flag
                            volume[j]+=value
                    #volume[j]+=df["sqare_bottom"][i][cid][BD_angle][radius_a][flag][bottom]+df["sqare_top"][i][cid][BD_angle][radius_a][flag][top]
                    volume[j]+=df["sqare_bottom"][i][cid,BD_angle,radius_a,flag,bottom]+df["sqare_top"][i][cid,BD_angle,radius_a,flag,top]#0128
                    #sqare+=(df["sqare_bottom"][i][cid][BD_angle][radius_a][flag][bottom]+df["sqare_top"][i][cid][BD_angle][radius_a][flag][top])*flag
                    sqare+=(df["sqare_bottom"][i][cid,BD_angle,radius_a,flag,bottom]+df["sqare_top"][i][cid,BD_angle,radius_a,flag,top])*flag#0128
            elif sum_flag=="max":
                val_=0
                valx=0
                for key,value in zip(df["sqare_local"][i][cid][BD_angle][radius_a].keys(),df["sqare_local"][i][cid][BD_angle][radius_a].values()):
                    if bottom<=float(key)<top:# float(key)>=bottom and float(key)<top:
                        if value<min(0,val_): val_=value#value<0 and value<val_: val_=value
                        if value>max(0,valx): valx=value
                sqare+=val_+valx
                sqare+=max([i for i in df["sqare_bottom"][i][cid][BD_angle][radius_a][bottom] if not i<0])+min([i for i in df["sqare_bottom"][i][cid][BD_angle][radius_a][bottom] if not i>0])+max([i for i in df["sqare_top"][i][cid][BD_angle][radius_a][top] if not i<0])+min([i for i in df["sqare_top"][i][cid][BD_angle][radius_a][top] if not i>0])
            sqare_list[i][cid]=volume
            for const in const_list:
                x=sqare*const*1000/df["temperature"][i]#sqare*flag*const*1000/df["temperature"][i]
                if x>10: predict[i][cid][const]=0
                else: predict[i][cid][const]=1/(1+math.exp(x))
                x2=-volume*const*1000/df["temperature"][i]
                predict2[i][cid][const]=np.exp(x2)#[np.exp(x) for x in x2]##[np.exp(x) if x<100 else np.exp(100) for x in x2]
    param["predict_const_list"]={}
    param[flag_+"_rmse"] ={}
    param[flag_+"_me"] = {}
    param["predict_const_list_pentanone"]={}
    if False:
        for const in const_list:
            param["predict_const_list"][const]=[]
            param["predict_const_list_pentanone"][const]={}
            for i,name in enumerate(df["smiles"]):
                predict_local_list=[predict[i][cid][const] for cid in df["cid"][i] ]
                predict_yld=np.average(np.array(predict_local_list), weights=np.array(list(df["rate"][i].values())))
                predict_local_list2=[predict2[i][cid][const] for cid in df["cid"][i] ]
                predict_local_list2_up=[k[0] for k in predict_local_list2]
                predict_local_list2_dn=[k[1] for k in predict_local_list2]
                up_sum=np.average(predict_local_list2_up, weights=np.array(list(df["rate"][i].values())))
                dn_sum=np.average(predict_local_list2_dn, weights=np.array(list(df["rate"][i].values())))
                predict_yld=np.average(np.array(predict_local_list), weights=np.array(list(df["rate"][i].values())))
                predict_yld=up_sum/(up_sum+dn_sum)
                #print("yld"+str(predict_yld))
                #print(up_sum,dn_sum)
                param["predict_const_list"][const].append(predict_yld)
            param["predict_const_list"][const]=np.array(param["predict_const_list"][const])*100
            if len(df["yld"])!=0:
                param[flag_+"_rmse"][const] =np.sqrt(mean_squared_error(np.array(df["yld"]),param["predict_const_list"][const]))
                param[flag_+"_me"][const] = np.sqrt(np.average((np.array(df["yld"])-param["predict_const_list"][const])**2))
                #print(const,type(const))
    else:
        def pred(x_func):
            predict_yld_list=[]
            
            for i,name in enumerate(df["smiles"]):
                #print(type(-sqare_list[i][0]),type(x),type(1000/df["temperature"][i]))
                x3=[-sqare_list[i][cid]*x_func*1000/df["temperature"][i] for cid in df["cid"][i]]
                predict3=[np.exp(x3) for x3 in x3]
                #predict_local_list3=[predict3[cid] for cid in df["cid"][i] ]
                predict3up=[k[0] for k in predict3]
                predict3dn=[k[1] for k in predict3]
                up_sum=np.average(predict3up, weights=np.array(list(df["rate"][i].values())))
                dn_sum=np.average(predict3dn, weights=np.array(list(df["rate"][i].values())))
                """
                if dn_sum!=0 or up_sum!=0:
                    predict_yld=1/(1+dn_sum/up_sum)#up_sum/(up_sum+dn_sum)
                elif dn_sum!=0 or up_sum==0:
                    predict_yld=0
                elif dn_sum==0 or up_sum!=0:
                    predict_yld=1
                else:
                    predict_yld=0.5
                """
                if dn_sum==up_sum:
                    predict_yld=50#up_sum/(up_sum+dn_sum)
                elif 0==min(dn_sum,up_sum) or np.inf==max(dn_sum,up_sum):
                    predict_yld=(100 if dn_sum<up_sum else 0)
                else:
                    predict_yld=100/(1+dn_sum/up_sum)#up_sum/(up_sum+dn_sum)
                
                """
                elif 0==dn_sum<up_sum or dn_sum<up_sum==np.inf:
                    predict_yld=100
                elif 0==up_sum<dn_sum or up_sum<dn_sum==np.inf:
                    predict_yld=0
                """
                #print(up_sum,dn_sum)
                predict_yld_list.append(predict_yld)
            #print(predict_yld_list)
            return predict_yld_list
        
        def func(x):
            #print(np.array(df["yld"]),np.array(pred(x)))
            ans=np.sqrt(mean_squared_error(np.array(df["yld"]),np.array(pred(x)))) if len(df["yld"])!=0 else 0
            #print(ans)
            return ans
        if flag_=="train":
            #result = fmin(func, [0.1],  xtol=1e-5, ftol=1e-5, maxiter=1000, maxfun=1000)
            result=minimize(func, x0=10, method="Nelder-Mead",tol = 10**-1).x#"SLSQP"#最小にするxを返す[1]#
            print(result)
        else:
            result=[1]#const_list[0]
        RMSE=func(result)#funcの最小値を返す
        if False:
            RMSE=func(1)
            result=1
        #param["predict_const_list"]=[]
        print("RMSE="+str(RMSE))
        param["predict_yld_list"]=pred(result)
        param["opt_const"]=result
        if flag_=="train":
            for const in const_list:
                #if len(df["yld"])!=0:
                param[flag_+"_rmse"][const] =param[flag_+"_me"][const]=RMSE#result.fun
                param["predict_const_list"][const]=pred(result)#pred(result.fun)
                #param["predict_const_list"][const].append(result.fun)
        else:
            param[flag_+"_rmse"][result] =param[flag_+"_me"][result]=RMSE
            param["predict_const_list"][result]=pred(result)
    return param
def culcurate_cone(param,df,flag_):#BD角、FL角、距離の3変数。一点評価（エネルギーを利用する方向に拡張可能）#0425追加
    top=float(param["top"])
    bottom=float(param["bottom"])
    radius_a=float(param["radius_a"])
    BD_angle=float(param["BD"])
    predict={}
    predict2={}
    sqare_list={}
    for i,name in enumerate(df["smiles"]):
        #print(name)
        predict[i]={}
        predict2[i]={}
        sqare_list[i]={}
        for cid in df["cid"][i]:
            sqare=0
            volume=np.zeros(2)#{1:0,-1:0}
            predict[i][cid]={}
            predict2[i][cid]={}
            for j,flag in enumerate([1,-1]):
                #for key,value in zip(df["sqare_local"][i][cid][BD_angle][radius_a][flag].keys(),df["sqare_local"][i][cid][BD_angle][radius_a][flag].values()):#0128
                for key,value in zip(df["sqare_local"][i][cid,BD_angle,radius_a,flag].keys(),df["sqare_local"][i][cid,BD_angle,radius_a,flag].values()):
                    if bottom<=float(key)<top:#float(key)>=bottom and float(key)<top:
                        sqare+=value*flag
                        volume[j]+=value
                #volume[j]+=df["sqare_bottom"][i][cid][BD_angle][radius_a][flag][bottom]+df["sqare_top"][i][cid][BD_angle][radius_a][flag][top]
                volume[j]+=df["sqare_bottom"][i][cid,BD_angle,radius_a,flag,bottom]+df["sqare_top"][i][cid,BD_angle,radius_a,flag,top]#0128
                #sqare+=(df["sqare_bottom"][i][cid][BD_angle][radius_a][flag][bottom]+df["sqare_top"][i][cid][BD_angle][radius_a][flag][top])*flag
                sqare+=(df["sqare_bottom"][i][cid,BD_angle,radius_a,flag,bottom]+df["sqare_top"][i][cid,BD_angle,radius_a,flag,top])*flag#0128
            sqare_list[i][cid]=volume
            if False:
                for const in [0]:
                    x=sqare*const*1000/df["temperature"][i]#sqare*flag*const*1000/df["temperature"][i]
                    if x>10: predict[i][cid][const]=0
                    else: predict[i][cid][const]=1/(1+math.exp(x))
                    x2=-volume*const*1000/df["temperature"][i]
                    predict2[i][cid][const]=np.exp(x2)#[np.exp(x) for x in x2]##[np.exp(x) if x<100 else np.exp(100) for x in x2]
    #param["predict_const_list"]={}
    #param[flag_+"_rmse"] ={}
    #param[flag_+"_me"] = {}
    #param["predict_const_list_pentanone"]={}
    def pred(x_func):
        predict_yld_list=[]

        for i,name in enumerate(df["smiles"]):
            #print(type(-sqare_list[i][0]),type(x),type(1000/df["temperature"][i]))
            x3=[-sqare_list[i][cid]*x_func*1000/df["temperature"][i] for cid in df["cid"][i]]
            predict3=[np.exp(x3) for x3 in x3]
            #predict_local_list3=[predict3[cid] for cid in df["cid"][i] ]
            predict3up=[k[0] for k in predict3]
            predict3dn=[k[1] for k in predict3]
            up_sum=np.average(predict3up, weights=np.array(list(df["rate"][i].values())))
            dn_sum=np.average(predict3dn, weights=np.array(list(df["rate"][i].values())))
            """
            if dn_sum!=0 or up_sum!=0:
                predict_yld=1/(1+dn_sum/up_sum)#up_sum/(up_sum+dn_sum)
            elif dn_sum!=0 or up_sum==0:
                predict_yld=0
            elif dn_sum==0 or up_sum!=0:
                predict_yld=1
            else:
                predict_yld=0.5
            """
            if dn_sum==up_sum:
                predict_yld=50#up_sum/(up_sum+dn_sum)
            elif 0==min(dn_sum,up_sum) or np.inf==max(dn_sum,up_sum):
                predict_yld=(100 if dn_sum<up_sum else 0)
            else:
                predict_yld=100/(1+dn_sum/up_sum)#up_sum/(up_sum+dn_sum)

            """
            elif 0==dn_sum<up_sum or dn_sum<up_sum==np.inf:
                predict_yld=100
            elif 0==up_sum<dn_sum or up_sum<dn_sum==np.inf:
                predict_yld=0
            """
            #print(up_sum,dn_sum)
            predict_yld_list.append(predict_yld)
        #print(predict_yld_list)
        return predict_yld_list

    def func(x):
        #print(np.array(df["yld"]),np.array(pred(x)))
        ans=np.sqrt(mean_squared_error(np.array(df["yld"]),np.array(pred(x)))) if len(df["yld"])!=0 else 0
        #print(ans)
        return ans
    if "train" in flag_:
        result = fmin(func, [0],  xtol=1e-3, ftol=1e-3, maxiter=1000, maxfun=1000)
        #result=minimize(func, x0=10, method="Nelder-Mead",tol = 10**-10).x#"SLSQP"#最小にするxを返す[1]#
        #print(result)
    else:
        result=param["one_train_const"].tolist()
    RMSE=func(result)#funcの最小値を返す
    if False:
        RMSE=func(1)
        result=1
    #param["predict_const_list"]=[]
    #print("RMSE="+str(RMSE))
    #print(pred(result))
    #param["opt_const"]=result
    if True:
        #if len(df["yld"])!=0:
        param[flag_+"_rmse"]=RMSE#result.fun
        param[flag_+"_predict"]=[pred(result)]#pred(result.fun)
        #.loc[row_indexer,col_indexer] 
        param[flag_+"_const"]=result
        #param["predict_const_list"][const].append(result.fun)
    return param

def sigmoid(a,x): return 100/(1+math.exp(a*(x-0.5)))#100/(1+math.exp(a*(x-0.5)))
def MSE_sigmoid(pro2,yld):
    da=0.1
    learning_rate=1
    times_sigmoid=200
    a=-2#23.5
    def RMSE_sigomoid(a,pro2,yld):
        RMSE=math.sqrt(np.average([(sigmoid(a,i)-j)**2 for i,j in zip(pro2,yld)]))
        return RMSE
    for i in range(times_sigmoid):
        RMSE_sig_current=RMSE_sigomoid(a,pro2,yld)
        RMSE_sig_delta=RMSE_sigomoid(a+da,pro2,yld)-RMSE_sig_current
        a+=-RMSE_sig_delta*learning_rate
        #print(RMSE_sig_current,a)
    return RMSE_sig_current,a

def learning(param_list,const_list,dict_from_list,df_train,df_test,sum_flag):
    mme=np.inf
    #print(sr["const"])
    flag=0
    for i in range(len(param_list)):
        param=param_list[i].copy()
        #print(param)
        param=culcurate_cone(param,const_list,df_train,"train",sum_flag)
        const_=param["opt_const"]
        print(const_)
        print(param["train_me"],"train_me")
        BD=param["BD"]
        bottom=param["bottom"]
        top=param["top"]
        radius=param["radius_a"]
        for const in const_list:
            #print(param["predict_const_list"])
            mme=min(param["train_me"][const],mme)
            query_str = "const==@const & BD==@BD & bottom==@bottom & radius_a==@radius & top==@top"
            df_subset = dict_from_list.query(query_str)
            if mme==param["train_me"][const]:#.copy():
                df_train["predict"]=train_predict=param["predict_const_list"][const].copy()
                #param["const"]=const
                opt_param=param.copy()
                opt_param["const"]=const
                
                
                results=culcurate_cone(opt_param,const_,df_test,"test",sum_flag)
                #print(const,type(const))
                #dict_from_list["test_me"][dict_from_list["const"]==const]=results["test_me"][const]#.copy()
                dict_from_list.loc[df_subset.index, "test_me"]=results["test_me"][const_[0]]
                dict_from_list.loc[df_subset.index, "test_predict_list"]=results["predict_const_list"][const_[0]].copy()#results["test_me"][const_[0]]
                #test_predict=results["predict_const_list"][const].copy()
                df_test["predict"]=results["predict_const_list"][const_[0]].copy()
                print("!!!")
            else:
                dict_from_list.loc[df_subset.index,"test_me"]=dict_from_list["test_me"][flag-1]#.copy()
                #dict_from_list["test_me"][dict_from_list["const"]==const]=dict_from_list["test_me"][flag-1]#.copy()
            print(dict_from_list.query('const==@const & BD==@BD & bottom==@bottom & radius_a==@radius & top==@top')["test_me"])
            dict_from_list.loc[flag,"train_learned_me"]=mme
            dict_from_list.loc[flag,"num"]=flag
            dict_from_list.loc[flag,"train_me"]=param["train_me"][const]
            print(flag)
            flag+=1
    #print(df_train)
    return dict_from_list

def sr_dict_from_list(num,sr):
    import itertools
    import numpy as np
    import pandas as pd
    #sr={}
    #num=5
    #"""
    if False:
        sr["BD"]=np.array([list[0]])
        #sr["BD"]=np.array([107])#np.linspace(90,90,1)#[-100,-105,-110]
        sr["bottom"]=np.array([list[1]])#np.arange(1.2,2,0.1)#0から
        sr["top"]=np.array([100])#np.arange(3,5,0.5)#[4,4.5,5]
        sr["radius_a"]=np.arange(0.1,1,0.1)#np.array([1.2])#np.arange(1.2,1.7,0.1)
        sr["const"]=np.array([0])#np.arange(1000,2000,200)
    dict_from_list =[dict(zip(["BD","bottom","top","radius_a","const"],l)) for l in list(itertools.product( *sr.values() ) )]
    param_list=[dict(zip(["BD","bottom","top","radius_a"],l)) for l in list(itertools.product( sr["BD"],sr["bottom"],sr["top"],sr["radius_a"] ) )]
    sr["BD+-"]=np.concatenate([sr["BD"],-1*sr["BD"]])
    sr["ang+-"]=np.deg2rad(sr["BD+-"])
    sr["vec"]= dict(zip(sr["BD+-"], [np.array([np.cos(ang),0,np.sin(ang)]) for ang in sr["ang+-"]]))
    sr["rvec"]=dict(zip([(i,j) for i,j in product(sr["BD+-"],sr["bottom"])],[sr["vec"][i]*j for i,j in product(sr["BD+-"],sr["bottom"])]))
    dict_from_list=pd.DataFrame(dict_from_list)
    dict_from_list=dict_from_list.astype(float)
    dict_from_list["train_rmse"]="nan"
    dict_from_list["train_learned_rmse"]="nan"
    dict_from_list["train_learned_me"]="nan"
    dict_from_list["train_me"]="nan"
    dict_from_list["test_rmse"]="nan"
    dict_from_list["test_me"]="nan"
    #dict_from_list=dict_from_list.astype(float)
    dict_from_list["train_predict"]="nan"
    dict_from_list["test_predict"]="nan"
    dict_from_list["num"]="nan"
    #dict_from_list["opt_param"]="nan"
    print(len(dict_from_list))
    print(len(param_list))
    """
    vco.vol_opt(df_train,dict_from_list)
    """
    #pd.set_option("display.max_rows", len(dict_from_list))
    #dict_from_list.head(1000)
    return sr,dict_from_list,param_list

def opt_graph(sr,df_train_MeOH,df_train_iPrOH,dict_from_list_MeOH,dict_from_list_iPrOH):
    import matplotlib.pyplot as plt
    #min_num_MeOH=dict_from_list_MeOH["train_me"].astype(float).idxmin()
    plt.rcParams["font.size"] = 18
    name=["BD","bottom","top","radius_a","const"]
    sub=["ang","bot","top","rad","con"]
    unit=[" [deg.]"," [Å]"," [Å]"," [Å]"," [kJ mol"+rf'$^{{-1}}]$']#+" Å"+rf'$^{{-3}}]$'
    mark_size=10
    for name,unit,sub in zip(name,unit,sub):#重ねてグラフ化したい
        """
        if False:
            y_MeOH=[min(dict_from_list_MeOH[dict_from_list_MeOH[name]==x]["train_me"]) for x in sr[name]]
            plt.plot(sr[name],y_MeOH,"-",color="b",label=None,alpha = 1,linewidth=4)
            plt.legend()
            for x in sr[name]:
                y_MeOH_loc=min(dict_from_list_MeOH[dict_from_list_MeOH[name]==x]["train_me"])
                icon="x" if y_MeOH_loc==min(y_MeOH) else "o"
                plt.plot(x,y_MeOH_loc,icon ,markersize=mark_size,color="b",label=None,alpha = 1)
            plt.xlabel(str(sub)+unit,fontsize=20)
            plt.ylabel("RMSE [%]",fontsize=20)
            plt.show()
        """
        if True:
            for dict_from_list,solvent in zip([dict_from_list_MeOH,dict_from_list_iPrOH],["MeOH","iPrOH"]):
                y=[min(dict_from_list[dict_from_list[name]==x]["train_me"]) for x in sr[name]]
                color="b" if solvent=="MeOH" else "r"
                plt.plot(sr[name],y,"-",color=color,label=solvent,alpha = 1,linewidth=4)
                plt.legend()
                for x in sr[name]:
                    if False: 
                        y_loc=min(dict_from_list[dict_from_list[name]==x]["train_me"])
                        icon="x" if y_loc==min(y) else "o"

                        plt.plot(x,y_loc,icon ,markersize=mark_size,color=color,alpha = 1)
                    else:
                        y_loc=min(dict_from_list[dict_from_list[name]==x]["train_me"])
                        if y_loc==min(y):
                            plt.plot(x,y_loc,"o" ,markersize=mark_size,color=color,alpha = 1)
            plt.xlabel(str(sub)+unit,fontsize=20)
            plt.ylabel("RMSE [%]",fontsize=20)
            plt.show()
    #df_train_all=df_train.copy()
