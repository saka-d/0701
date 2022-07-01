#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import matplotlib.pyplot as plt#グラフ描画に使用
from sklearn.metrics import mean_squared_error
import time
def sphere(r,R,d):
    if d==0: return 0
    if r+R<d: return 0
    elif r+d<=R: return 4/3*np.pi*r**3#内包
    elif R+d<=r: return 4/3*np.pi*R**3#"error"#内包
    A=(-r**2+R**2+d**2)/(2*d)
    B=d-A
    C=d-r
    return 1/3*np.pi*(2*R**3-3*R**2*C+C**3)-abs(r-B)/2*np.pi*(R**2-C**2)

def n_val(r,R,d,n):
    if d==0: return 0
    if r+R<d: return 0
    tortion=(r+R-d)/(r+R)
    volume_0=(4/3)*np.pi*min([r,R])**3
    return tortion**n*volume_0

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

def imaginary_radius_maker(h,param):#円錐台の半径
    if float(param["bottom"])<=h and h<=float(param["top"]): return float(param["radius_a"])
    else: return np.sqrt(float(param["radius_a"])**2-max([float(param["bottom"])-h,h-float(param["top"])])**2)
def imaginary_radius_maker(h,top,bottom,radius_a):#円錐台の半径
    if bottom<=h and h<=top: return radius_a
    else: return np.sqrt(radius_a**2-max(bottom-h,h-top))

def Bd(x,T):
    if T!=T: T=273.15+20
    return math.exp(-x/1.99/T*0.001)#ボルツマン分布

def s(R,r,d):#2円重なり面積算出関数
    if d==0 or R+r<=d: return 0
    #if R+r<=d: return 0
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

"""学習効率化BD_angle一本化,const_list"""
#"""
def culcurate_cone(param,const_list,df,flag_):#BD角、FL角、距離の3変数。一点評価（エネルギーを利用する方向に拡張可能）
    top=float(param["top"])
    bottom=float(param["bottom"])
    radius_a=float(param["radius_a"])
    BD_angle=float(param["BD"])
    #const=float(param["const"])
    
    #h_list=np.arange(bottom-radius_a, top-radius_a, (delta_h:=0.2))
    delta_h=0.2
    predict={}
    for i,name in enumerate(df["smiles"]):
        #total_energy=np.array([0.,0.])
        predict[i]={}
        for cid in df["cid"][i]:
            sqare=0
            flag=float(df["cis_trans_flag"][i][cid])
            predict[i][cid]={}
            for key,value in zip(df["sqare_local"][i][cid][BD_angle][radius_a].keys(),df["sqare_local"][i][cid][BD_angle][radius_a].values()):
                if float(key)>=bottom and float(key)<top:
                    sqare+=value
            """
            val_=0
            valx=0
            for key,value in zip(df["sqare_local"][i][cid][BD_angle][radius_a].keys(),df["sqare_local"][i][cid][BD_angle][radius_a].values()):
                if float(key)>=bottom and float(key)<top:
                    if value<0 and value<val_:
                        val_=value
                    if value>0 and value>valx:
                        valx=value
            sqare+=val_+valx
            """
            #print(sqare,df["sqare_bottom"][i][cid][BD_angle][radius_a][bottom],df["sqare_top"][i][cid][BD_angle][radius_a][top])
            sqare+=df["sqare_bottom"][i][cid][BD_angle][radius_a][bottom]+df["sqare_top"][i][cid][BD_angle][radius_a][top]
            #print(sqare*delta_h)
            for const in const_list:
                x=sqare*flag*delta_h*const*1000/df["temperature"][i]
                if x>10: predict[i][cid][const]=0
                else: predict[i][cid][const]=1/(1+math.exp(x))
    param["predict_const_list"]={}
    param[flag_+"_rmse"] ={}
    param[flag_+"_me"] = {}
    param["predict_const_list_pentanone"]={}
    #df[flag+"_rmse"]={}#new
    #df[flag*"_me"]={}#new
    #df["predict_const_list"]={}#new
    for const in const_list:
        param["predict_const_list"][const]=[]
        param["predict_const_list_pentanone"][const]={}
        #df["predict_const_list"][const]=[]
        for i,name in enumerate(df["smiles"]):
            #print(predict,df["cid"][i])
            predict_local_list=[predict[i][cid][const] for cid in df["cid"][i] ]
            #predict_pentanone_local_list=
            predict_yld=np.average(np.array(predict_local_list), weights=np.array(list(df["rate"][i].values())))
            param["predict_const_list"][const].append(predict_yld)
            #param["predict_const_list_pentanone"][const].append(predict_pentanone_yld)
            #df["predict_const_list"][i][const]=predict_yld#new
        param["predict_const_list"][const]=np.array(param["predict_const_list"][const])*100
        if len(df["yld"])!=0:
            param[flag_+"_rmse"][const] =np.sqrt(mean_squared_error(np.array(df["yld"]),param["predict_const_list"][const]))
            param[flag_+"_me"][const] = np.average(np.abs(np.array(df["yld"])-param["predict_const_list"][const]))
            #param[flag*"_me_pentanone"]=np.average(np.abs(np.array(df["yld"][df["compound"]==4])-param["predict_const_list"][const]))
        #print(param["predict_const_list"][const])
    return  param
#"""

"""最大重なり体積"""
#"""
def culcurate_cone2(param,const_list,df,flag_):#BD角、FL角、距離の3変数。一点評価（エネルギーを利用する方向に拡張可能）
    top=float(param["top"])
    bottom=float(param["bottom"])
    radius_a=float(param["radius_a"])
    BD_angle=float(param["BD"])
    #const=float(param["const"])
    
    h_list=np.arange(bottom-radius_a, top-radius_a, (delta_h:=0.2))
    predict={}
    for i,name in enumerate(df["smiles"]):
        #total_energy=np.array([0.,0.])
        predict[i]={}
        for cid in df["cid"][i]:
            sqare=0
            
            flag=float(df["cis_trans_flag"][i][cid])
            predict[i][cid]={}
            """ここに最大重なり体積を算出"""
            """
            car_C_location=df["conf_coordinate"][i][cid][df["car_C_num"][i]]
            for atom in range(len(df["mass"][i])):
                atom_to_car_C=df["conf_coordinate"][i][cid][atom]-car_C_location#原子中心-car_C
                location_vec=lotate_vec(df["BD_axis"][i][cid],df["FL_axis"][i][cid],df["car_vec"][i][cid],BD_angle,0)
                inner=np.dot(location_vec,atom_to_car_C)#vとの内積
                distance=np.linalg.norm(inner*location_vec-atom_to_car_C)#dを出す
                df["tortion"][i][cid][atom][BD_angle][radius]=tortion if (tortion:=radius+df["cube_radius"][i][cid][atom]-distance)>0 else 0
            
            
            for key,value in zip(df["sqare_local"][i][cid][BD_angle][radius_a].keys(),df["sqare_local"][i][cid][BD_angle][radius_a].values()):
                if float(key)>=bottom and float(key)<top:
                    sqare+=value
            #print(sqare,df["sqare_bottom"][i][cid][BD_angle][radius_a][bottom],df["sqare_top"][i][cid][BD_angle][radius_a][top])
            sqare+=df["sqare_bottom"][i][cid][BD_angle][radius_a][bottom]+df["sqare_top"][i][cid][BD_angle][radius_a][top]
            """
            t=0
            for atom in range(len(df["mass"][i])):
                for flag2 in [1,-1]:
                    inner=df["inner"][i][cid][atom][BD_angle*flag2]
                    tortion=df["tortion"][i][cid][atom][BD_angle*flag2][radius_a]
                    distance=df["distance"][i][cid][atom][BD_angle*flag2]
                    radius=df["cube_radius"][i][cid][atom]
                    if bottom-inner>0:
                        dis=bottom-inner
                        distance=np.sqrt(distance**2+dis**2)
                    if inner-top>0:
                        dis=inner-top
                        distance=np.sqrt(distance**2+dis**2)

                    sqare+=sphere(radius_a,radius,distance)*flag2
                    if t<distance: t=distance
            #print(t,sqare)
            if sqare>100:
                sqare=100
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            for const in const_list:
                x=sqare*flag*const/df["temperature"][i]
                predict[i][cid][const]=1/(1+math.exp(x))
            
    param["predict_const_list"]={}
    param[flag_+"_rmse"] ={}
    param[flag_+"_me"] = {}
    param["predict_const_list_pentanone"]={}
    #df[flag+"_rmse"]={}#new
    #df[flag*"_me"]={}#new
    #df["predict_const_list"]={}#new
    for const in const_list:
        param["predict_const_list"][const]=[]
        param["predict_const_list_pentanone"][const]={}
        #df["predict_const_list"][const]=[]
        for i,name in enumerate(df["smiles"]):
            #print(predict,df["cid"][i])
            predict_local_list=[predict[i][cid][const] for cid in df["cid"][i] ]
            #predict_pentanone_local_list=
            predict_yld=np.average(np.array(predict_local_list), weights=np.array(list(df["rate"][i].values())))
            param["predict_const_list"][const].append(predict_yld)
            #param["predict_const_list_pentanone"][const].append(predict_pentanone_yld)
            #df["predict_const_list"][i][const]=predict_yld#new
        param["predict_const_list"][const]=np.array(param["predict_const_list"][const])*100
        if len(df["yld"])!=0:
            param[flag_+"_rmse"][const] =np.sqrt(mean_squared_error(np.array(df["yld"]),param["predict_const_list"][const]))
            param[flag_+"_me"][const] = np.average(np.abs(np.array(df["yld"])-param["predict_const_list"][const]))
            #param[flag*"_me_pentanone"]=np.average(np.abs(np.array(df["yld"][df["compound"]==4])-param["predict_const_list"][const]))
        #print(param["predict_const_list"][const])
    return  param
#"""


def sigmoid(a,x):
    return 100/(1+math.exp(a*(x-0.5)))#100/(1+math.exp(a*(x-0.5)))
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




def cul(df,sr):
    import numpy as np
    #df=df_all
    """
    l=["distance","inner",
      "tortion","sqare_local",
      "sqare_bottom","sqare_bottom"]
    df[l]="nan"
    """

    df["distance"]="nan"
    df["inner"]="nan"
    df["tortion"]="nan"
    df["sqare_local"]="nan"
    df["sqare_bottom"]="nan"
    df["sqare_top"]="nan"

    delta_h=0.2
    h_list=np.arange(min(sr["bottom"]), max(sr["top"]), delta_h)#topがminになってた！

    for i,name in enumerate(df["smiles"]):
        print(i)
        df["distance"][i]={}
        df["inner"][i]={}
        df["tortion"][i]={}
        df["sqare_local"][i]={}
        df["sqare_bottom"][i]={}
        df["sqare_top"][i]={}
        for cid in df["cid"][i]:
            df["distance"][i][cid]={}
            df["inner"][i][cid]={}
            df["tortion"][i][cid]={}
            df["sqare_local"][i][cid]={}
            df["sqare_bottom"][i][cid]={}
            df["sqare_top"][i][cid]={}
            car_C_location=df["conf_coordinate"][i][cid][df["car_C_num"][i]]
            for atom in range(len(df["mass"][i])):
                df["distance"][i][cid][atom]={}
                df["inner"][i][cid][atom]={}
                df["tortion"][i][cid][atom]={}
                #df["sqare_local"][i][cid][atom]={}
                atom_to_car_C=df["conf_coordinate"][i][cid][atom]-car_C_location#原子中心-car_C
                for BD_angle in np.concatenate([sr["BD"],-1*sr["BD"]]):
                    location_vec=lotate_vec(df["BD_axis"][i][cid],df["FL_axis"][i][cid],df["car_vec"][i][cid],BD_angle,0)
                    inner=np.dot(location_vec,atom_to_car_C)#vとの内積

                    distance=np.linalg.norm(inner*location_vec-atom_to_car_C)#dを出す
                    df["distance"][i][cid][atom][BD_angle]=distance
                    df["inner"][i][cid][atom][BD_angle]=inner

                    df["tortion"][i][cid][atom][BD_angle]={}
                    #df["sqare_local"][i][cid][BD_angle]={}
                    for radius in sr["radius_a"]:
                        df["tortion"][i][cid][atom][BD_angle][radius]=tortion if (tortion:=radius+df["cube_radius"][i][cid][atom]-distance)>0 else 0
                        

            for BD_angle in sr["BD"]:
                df["sqare_local"][i][cid][BD_angle]={}
                df["sqare_bottom"][i][cid][BD_angle]={}
                df["sqare_top"][i][cid][BD_angle]={}
                for radius in sr["radius_a"]:
                    df["sqare_local"][i][cid][BD_angle][radius]={}
                    df["sqare_bottom"][i][cid][BD_angle][radius]={}
                    df["sqare_top"][i][cid][BD_angle][radius]={}
                    #atom_list=[]
                    atom_list={1:[],-1:[]}
                    for atom in range(len(df["mass"][i])):
                        for flag in [1,-1]:
                            if df["tortion"][i][cid][atom][BD_angle*flag][radius]>0: atom_list[flag].append(atom)

                    for h in h_list:
                        df["sqare_local"][i][cid][BD_angle][radius][h]=0
                        break#←特殊処理
                        for flag in [1,-1]:
                            for atom in atom_list[flag]:
                                circle_radius_sqare=df["cube_radius"][i][cid][atom]**2-(df["inner"][i][cid][atom][BD_angle*flag]-h)**2#r^2を算出

                                if circle_radius_sqare>0:# and df["tortion"][i][cid][atom][BD_angle][radius]>0:#r>0
                                    circle_radius=np.sqrt(circle_radius_sqare)#rを出す
                                    df["sqare_local"][i][cid][BD_angle][radius][h]+=flag*s(circle_radius,radius,df["distance"][i][cid][atom][BD_angle*flag])#面積算出
                    for bottom in sr["bottom"]:
                        h_list_bottom=np.arange(bottom-radius, bottom, delta_h)
                        df["sqare_bottom"][i][cid][BD_angle][radius][bottom]=0
                        break#←特殊処理
                        for flag in [1,-1]:
                            for atom in atom_list[flag]:
                                for h in h_list_bottom:
                                    circle_radius_sqare=df["cube_radius"][i][cid][atom]**2-(df["inner"][i][cid][atom][BD_angle*flag]-h)**2#r^2を算出
                                    if circle_radius_sqare>0:# and df["tortion"][i][cid][atom][BD_angle][radius]>0:#r>0
                                        circle_radius=np.sqrt(circle_radius_sqare)#rを出す
                                        #print(circle_radius,radius,distance)
                                        #imaginary_radius=imaginary_radius_maker(h,top,bottom,radius_a)#円錐台の半径をhの関数によって出す
                                        sphere_radius=np.sqrt(radius**2-(bottom-h)**2)
                                        df["sqare_bottom"][i][cid][BD_angle][radius][bottom]+=flag*s(circle_radius,sphere_radius,df["distance"][i][cid][atom][BD_angle])#面積算出
                    for top in sr["top"]:
                        h_list_top=np.arange(top, top+radius, delta_h)
                        df["sqare_top"][i][cid][BD_angle][radius][top]=0
                        
                        #"""
                        for flag in [1,-1]:
                            for atom in atom_list[flag]:
                                for h in h_list_top:
                                    circle_radius_sqare=df["cube_radius"][i][cid][atom]**2-(df["inner"][i][cid][atom][BD_angle*flag]-h)**2#r^2を算出
                                    if circle_radius_sqare>0:# and df["tortion"][i][cid][atom][BD_angle][radius]>0:#r>0
                                        circle_radius=np.sqrt(circle_radius_sqare)#rを出す
                                        #print(circle_radius,radius,distance)
                                        #imaginary_radius=imaginary_radius_maker(h,top,bottom,radius_a)#円錐台の半径をhの関数によって出す
                                        if (sph_rad_sq:=radius**2-(h-top)**2)>0: sphere_radius=np.sqrt(sph_rad_sq)
                                        else: sphere_radius=0
                                        #print(radius**2-(h-top)**2)
                                        df["sqare_top"][i][cid][BD_angle][radius][top]+=flag*s(circle_radius,sphere_radius,df["distance"][i][cid][atom][BD_angle])#面積算出
                                        #print(circle_radius,sphere_radius,df["distance"][i][cid][atom][BD_angle])
                        """
                        for flag in [1]:
                            for atom in atom_list[flag]:
                                if np.linalg.norm(df["atom_from_car_C"][i][cid][atom])<radius:
                                    df["sqare_top"][i][cid][BD_angle][radius][top]+=np.sign(df["inner"][i][cid][atom][BD_angle])
                                    print(df["inner"][i][cid][atom][BD_angle])
                        """
    return df
"""最大重なり体積"""
def cul2(df,sr):
    import numpy as np
    #df=df_all
    """
    l=["distance","inner",
      "tortion","sqare_local",
      "sqare_bottom","sqare_bottom"]
    df[l]="nan"
    """

    df["distance"]="nan"
    df["inner"]="nan"
    df["tortion"]="nan"
    df["sqare_local"]="nan"
    df["sqare_bottom"]="nan"
    df["sqare_top"]="nan"

    delta_h=0.2
    h_list=np.arange(min(sr["bottom"]), max(sr["top"]), delta_h)#topがminになってた！

    for i,name in enumerate(df["smiles"]):
        print(i)
        df["distance"][i]={}
        df["inner"][i]={}
        df["tortion"][i]={}
        df["sqare_local"][i]={}
        df["sqare_bottom"][i]={}
        df["sqare_top"][i]={}
        for cid in df["cid"][i]:
            df["distance"][i][cid]={}
            df["inner"][i][cid]={}
            df["tortion"][i][cid]={}
            df["sqare_local"][i][cid]={}
            df["sqare_bottom"][i][cid]={}
            df["sqare_top"][i][cid]={}
            car_C_location=df["conf_coordinate"][i][cid][df["car_C_num"][i]]
            for atom in range(len(df["mass"][i])):
                df["distance"][i][cid][atom]={}
                df["inner"][i][cid][atom]={}
                df["tortion"][i][cid][atom]={}
                #df["sqare_local"][i][cid][atom]={}
                atom_to_car_C=df["conf_coordinate"][i][cid][atom]-car_C_location#原子中心-car_C
                for BD_angle in np.concatenate([sr["BD"],-1*sr["BD"]]):
                    location_vec=lotate_vec(df["BD_axis"][i][cid],df["FL_axis"][i][cid],df["car_vec"][i][cid],BD_angle,0)
                    inner=np.dot(location_vec,atom_to_car_C)#vとの内積

                    distance=np.linalg.norm(inner*location_vec-atom_to_car_C)#dを出す
                    df["distance"][i][cid][atom][BD_angle]=distance
                    df["inner"][i][cid][atom][BD_angle]=inner

                    df["tortion"][i][cid][atom][BD_angle]={}
                    #df["sqare_local"][i][cid][BD_angle]={}
                    for radius in sr["radius_a"]:
                        df["tortion"][i][cid][atom][BD_angle][radius]=tortion if (tortion:=radius+df["cube_radius"][i][cid][atom]-distance)>0 else 0
    return df
"""最大重なり体積２"""
def cul(df,sr,n):
    import numpy as np

    df["distance"]="nan"
    df["inner"]="nan"
    df["tortion"]="nan"
    df["sqare_local"]="nan"
    df["sqare_bottom"]="nan"
    df["sqare_top"]="nan"

    for i,name in enumerate(df["smiles"]):
        print(i)
        df["distance"][i]={}
        df["inner"][i]={}
        df["tortion"][i]={}
        df["sqare_local"][i]={}
        df["sqare_bottom"][i]={}
        df["sqare_top"][i]={}
        for cid in df["cid"][i]:
            df["distance"][i][cid]={}
            df["inner"][i][cid]={}
            df["tortion"][i][cid]={}
            df["sqare_local"][i][cid]={}
            df["sqare_bottom"][i][cid]={}
            df["sqare_top"][i][cid]={}
            car_C_location=df["conf_coordinate"][i][cid][df["car_C_num"][i]]
            for atom in range(len(df["mass"][i])):
                df["distance"][i][cid][atom]={}
                df["inner"][i][cid][atom]={}
                df["tortion"][i][cid][atom]={}
                #df["sqare_local"][i][cid][atom]={}
                atom_to_car_C=df["conf_coordinate"][i][cid][atom]-car_C_location#原子中心-car_C
                for BD_angle in np.concatenate([sr["BD"],-1*sr["BD"]]):
                    location_vec=lotate_vec(df["BD_axis"][i][cid],df["FL_axis"][i][cid],df["car_vec"][i][cid],BD_angle,0)
                    inner=np.dot(location_vec,atom_to_car_C)#vとの内積

                    distance=np.linalg.norm(inner*location_vec-atom_to_car_C)#dを出す
                    df["distance"][i][cid][atom][BD_angle]=distance
                    df["inner"][i][cid][atom][BD_angle]=inner

                    df["tortion"][i][cid][atom][BD_angle]={}
                    #df["sqare_local"][i][cid][BD_angle]={}
                    for radius in sr["radius_a"]:
                        df["tortion"][i][cid][atom][BD_angle][radius]=tortion if (tortion:=radius+df["cube_radius"][i][cid][atom]-distance)>0 else 0
                        
            for BD_angle in sr["BD"]:
                df["sqare_local"][i][cid][BD_angle]={}
                df["sqare_bottom"][i][cid][BD_angle]={}
                df["sqare_top"][i][cid][BD_angle]={}
                for radius in sr["radius_a"]:
                    df["sqare_local"][i][cid][BD_angle][radius]={}
                    df["sqare_bottom"][i][cid][BD_angle][radius]={}
                    df["sqare_top"][i][cid][BD_angle][radius]={}
                    #atom_list=[]
                    atom_list={1:[],-1:[]}
                    for atom in range(len(df["mass"][i])):
                        for flag in [1,-1]:
                            if df["tortion"][i][cid][atom][BD_angle*flag][radius]>0:
                                atom_list[flag].append(atom)
                    for flag in [1,-1]:
                        for atom in atom_list[flag]:
                            inner=df["inner"][i][cid][atom][BD_angle*flag]
                            distance=df["distance"][i][cid][atom][BD_angle*flag]
                            atom_radius=df["cube_radius"][i][cid][atom]
                            if n=="volume":
                                df["sqare_local"][i][cid][BD_angle][radius][inner]=flag*sphere(radius,atom_radius,distance)
                            else:
                                df["sqare_local"][i][cid][BD_angle][radius][inner]=flag*n_val(radius,atom_radius,distance,n)
                    for bottom in sr["bottom"]:
                        df["sqare_bottom"][i][cid][BD_angle][radius][bottom]=0
                        for flag in [1,-1]:
                            for atom in atom_list[flag]:
                                if (bi:=bottom-inner)>0:
                                    dis=np.sqrt(distance**2+bi**2)
                                    if n=="volume":
                                        df["sqare_bottom"][i][cid][BD_angle][radius][bottom]=+flag*sphere(radius,atom_radius,dis)
                                    else:
                                        df["sqare_bottom"][i][cid][BD_angle][radius][bottom]=+flag*n_val(radius,atom_radius,dis,n)
                    for top in sr["top"]:
                        df["sqare_top"][i][cid][BD_angle][radius][top]=0
                        for flag in [1,-1]:
                            for atom in atom_list[flag]:
                                if (it:=inner-top)>0:
                                    dis=np.sqrt(distance**2+it**2)
                                    if n=="volume":
                                        df["sqare_top"][i][cid][BD_angle][radius][top]=+flag*sphere(radius,atom_radius,dis)
                                    else:
                                        df["sqare_top"][i][cid][BD_angle][radius][top]=+flag*n_val(radius,atom_radius,dis,n)
    return df

def learning(param_list,const_list,dict_from_list,df_train,df_test):
    mme=99
    #print(sr["const"])
    flag=0
    for i in range(len(param_list)):
        param=param_list[i].copy()
        #print(param)
        param=culcurate_cone(param,const_list,df_train,"train")
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
            if mme==param["train_me"][const].copy():
                df_train["predict"]=train_predict=param["predict_const_list"][const].copy()
                #param["const"]=const
                opt_param=param.copy()
                opt_param["const"]=const
                results=culcurate_cone(opt_param,[const],df_test,"test")
                #dict_from_list["test_me"][dict_from_list["const"]==const]=results["test_me"][const]#.copy()
                dict_from_list.loc[df_subset.index, "test_me"]=results["test_me"][const]
                #test_predict=results["predict_const_list"][const].copy()
                df_test["predict"]=results["predict_const_list"][const].copy()
                print("!!!")
            else:
                dict_from_list.loc[df_subset.index, "test_me"]=dict_from_list["test_me"][flag-1]#.copy()
                #dict_from_list["test_me"][dict_from_list["const"]==const]=dict_from_list["test_me"][flag-1]#.copy()
            print(dict_from_list.query('const==@const & BD==@BD & bottom==@bottom & radius_a==@radius & top==@top')["test_me"])
            dict_from_list.loc[flag,"train_learned_me"]=mme
            dict_from_list.loc[flag,"num"]=flag
            dict_from_list.loc[flag,"train_me"]=param["train_me"][const]
            print(flag)
            flag+=1
    #print(df_train)
    return dict_from_list