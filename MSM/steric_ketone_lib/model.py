#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
from rdkit import *
from rdkit.Chem import AllChem
def Bd(x,T):
    return math.exp(-x/(1.99*T*0.001))#ボルツマン分布

def cul_rad_beta(bond_list,conf,const):#結合距離の半値三乗平均立方根
    radius=[]
    for i in range(const):
        L=np.array([Chem.rdMolTransforms.GetBondLength(conf, i, j) for j in bond_list[i]])
        radius_atom=np.cbrt(np.average((L/2)**3))
        #radius_atom=np.cbrt(np.average([(l/2)**3 for l in L]))
        radius.append(radius_atom)
    return radius

def van_der_waals_radius(symbol):
    radius_list={"H":1.20,"C":1.70,"O":1.4,"S":1.80,"Si":2.1, "N":1.6,"P":1.95}#https://japan2.wiki/wiki/Van_der_Waals_radius
    ans=[]
    for symbol in symbol:
        try:
            ans.append(radius_list[symbol])
        except:
            print("NO RADIUS DATA : "+symbol)
            ans.append(1.70)
    #ans=[radius_list[symbol] for symbol in symbol]
    return ans
def atom_mass(symbol):
    radius_list={"H":1,"C":4,"O":2,"S":2}#https://japan2.wiki/wiki/Van_der_Waals_radius
    ans=[radius_list[symbol] for symbol in symbol]
    return ans
def search(df,flag,i):#読み込んだ分子の配座探索
    print(i)
    temp=df["temperature"][i]
    carnum=df["carnum"][i]
    #mol=Chem.AddHs(Chem.MolFromSmiles(name))#SMILES→mol
    #cids = AllChem.EmbedMultipleConfs(mol, numConfs=100, randomSeed=1, pruneRmsThresh=0.1, numThreads=0)
    mol=df["mol"][i]
    cids=df["cid"][i]
    const=df["const"][i]
    atom_symbols=df["symbol"][i]#[atom.GetSymbol() for atom in atoms]
    #ランダム配座生成(mol:対象のmol形式データ, numConfs:生成配座数, randomSeed:乱数を決定する値, pruneRmsThresh:類似構造を排除する閾値, numThreads:何個で一セットにするか
    #print(name)
    energies=[]
    for cid in cids:#構造最適化
        try:
            mmff=AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid)#使用する力場
            mmff.Minimize()#MM計算
            energies.append([mmff.CalcEnergy(), cid])#計算した配座、エネルギーをenergyリストに追加
        except:
            print(i)
    #print(energies)
    if True:
        min_energy=sorted(energies)[0][0]
    if False:
        df.drop(df.index[i])
        print("energyERROR")
    for i in range(len(energies)):
        energies[i][0]-=min_energy
        #if energies[i][0]>: np.log(temp)*
    def num_cids(num,energies):
        if num=="opt":
            return [sorted(energies)[0]],[cids[0]]#return sorted(energies)
        if num=="all": return energies,cids
        if type(num)==float or type(num)==int:
            energies_new=[]
            cids_new=[]
            
            for i in range((leng:=len(energies))):
                rate_=num if True else num/leng
                if energies[i][0]<np.log(rate_)*-1*1.987e-3*temp:
                    
                    energies_new.append(energies[i])
                    cids_new.append(energies[i][1])
            return energies_new,cids_new
    def delete_cids(num,energies):
        energies_new=[]
        cids_new=[]
        for energy in energies:
            flag=True
            for energy_new in energies_new:
                if abs(energy_new[0]-energy[0])<num:
                    flag=False
            if flag:
                energies_new.append(energy)
                cids_new.append(energy[1])
        return energies_new,cids_new
    #energies,cids=delete_cids(0.1,energies)
    energies,cids=num_cids(flag,energies)
    #atoms=mol.GetAtoms()
    #const=len(atoms)#int(molblock[3].split()[0])
    carbonyl_sub=mol.GetSubstructMatch(Chem.MolFromSmarts("C(=O)(C)(C)"))
    
    
    if len(carbonyl_sub)==0:#隣接原子が芳香環炭素の場合の例外処理
        carbonyl_sub=mol.GetSubstructMatch(Chem.MolFromSmarts("C(=O)(C)(c)"))
    #print(carnum)

    #print(carbonyl_sub)
    #if carbonyl_sub[1]+2!=carbonyl_sub[2]: carbonyl_sub=list(carbonyl_sub[0:2])+list(reversed(carbonyl_sub[2:4]))
    if carbonyl_sub[2]>carbonyl_sub[3]: carbonyl_sub=list(carbonyl_sub[0:2])+list(reversed(carbonyl_sub[2:4]))
    if carnum==carnum:
        carbonyl_sub=[int(s) for s in carnum.split()]
    bond_list=[[] for i in range(const)]
    for bond in mol.GetBonds():
        #BeginAtomIdx=bond.GetBeginAtomIdx()
        #EndAtomIdx=bond.GetEndAtomIdx()
        #print(const,bond.GetBeginAtomIdx(),bond.GetEndAtomIdx())
        bond_list[(begin:=bond.GetBeginAtomIdx())].append((end:=bond.GetEndAtomIdx()))
        bond_list[end].append(begin)
    #df["energy"][i],df["mol"][i],df["cid"][i],df["const"][i],df["car_atom"][i],df["bond"][i],df["symbol"][i]=energies,mol,cids,const,carbonyl_sub,bond_list,atom_symbols
    return energies,mol,cids,const,carbonyl_sub,bond_list,atom_symbols
from math import *
def coordinate_transformation(atom_list_2,df,i):
    c0num ,onum,c1num ,c2num =df["car_atom"][i]
    #座標変換を行う
    #平行移動させて注目原子が原点にくるようにする
    atom_list_2=atom_list_2-atom_list_2[c0num]
    ry = atan(-1*atom_list_2[onum][2]/atom_list_2[onum][0])
    #回転によりxy平面に酸素原子を持ってくる
    for ho1 in atom_list_2: (ho1[0],ho1[2]) = (ho1[0]*cos(ry)-ho1[2]*sin(ry),ho1[0]*sin(ry)+ho1[2]*cos(ry))
    rz = atan(atom_list_2[onum][1]/atom_list_2[onum][0])
    #回転によりx軸上に酸素原子を持ってくる
    for ho2 in atom_list_2: (ho2[0],ho2[1]) = (ho2[0]*cos(rz)+ho2[1]*sin(rz),-1*ho2[0]*sin(rz)+ho2[1]*cos(rz))
    rx = atan((atom_list_2[c1num][2]-atom_list_2[c2num][2])/(atom_list_2[c2num][1]-atom_list_2[c1num][1]))
    #回転によりxy平面に注目原子を含む4つの原子が来るようにする
    for ho3 in atom_list_2: (ho3[1],ho3[2]) = (ho3[1]*cos(rx)-ho3[2]*sin(rx),ho3[1]*sin(rx)+ho3[2]*cos(rx))
    if atom_list_2[onum][0]<0:#Oをx正に
        for ho2 in atom_list_2: (ho2[0],ho2[1]) = (-ho2[0],-ho2[1])
    if atom_list_2[c1num][1]-atom_list_2[c2num][1]>0:#c1numをy負側に
        for ho3 in atom_list_2: (ho3[1],ho3[2]) = (-ho3[1],-ho3[2])
    if df["cis_trans"][i]!=[]:
        up,down=df["cis_trans"][i]
        #print(up,down)
        if atom_list_2[int(up)][2]-atom_list_2[int(down)][2]<0:#upの原子をz正側に
            #print(up,down)
            for ho3 in atom_list_2: (ho3[1],ho3[2]) = (-ho3[1],-ho3[2])
    return atom_list_2

def model_maker(df,flag):
    print("0218")
    model={}
    df["mol"]=[Chem.AddHs(Chem.MolFromSmiles(name)) for name in df["smiles"]]
    df["cid"]=[AllChem.EmbedMultipleConfs(mol, numConfs=10 if flag!="opt" else 10, randomSeed=1, pruneRmsThresh=0.1, numThreads=0) for mol in df["mol"]]#0.1#ちょっと変えた0301
    print("配座発生終了")
    df["car_C_num"]="nan"
    df["energy"]="nan"
    df["radius"]="nan"
    df["const"]=[len(mol.GetAtoms()) for mol in df["mol"]]
    df["mass"]="nan"
    df["car_atom"]="nan"
    df["bond"]="nan"
    df["symbol"]=[[atom.GetSymbol() for atom in mol.GetAtoms()] for mol in df["mol"]]
    if False:
        mmffs=[AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=cid) for mol,cid in zip(df["mol"],df["cid"])]#使用する力場
        mmff_minimize=[mmff.Minimize() for mmff in mmffs]
        energies=[mmff.CalcEnergy() for mmmf in mmffs]
        energies=np.array(energies)-min(energies)
        
    df["conf_coordinate"]="nan"
    if False:
        df["BD_axis"]="nan"
        df["FL_axis"]="nan"
    df["car_vec"]="nan"
    df["cube_radius"]="nan"
    df["rate"]="nan"
    df["cis_trans_flag"]="nan"
    df["atom_from_car_C"]="nan"
    df["inner"]="nan"
    df["psi"]="nan"
    df["sub_atom"]="nan"
    droplist=[]
    for i,name in enumerate(df["smiles"]):
        if False:
            i=df.index[df["smiles"] == name][0]
        print(i)
        #df["temperature"][i]=df["temperature"][i] if df["temperature"][i]==df["temperature"][i] else 273.15+20#nanの場合は20℃
        try:
            df["energy"][i],df["mol"][i],df["cid"][i],df["const"][i],df["car_atom"][i],df["bond"][i],df["symbol"][i]=search(df,flag,i)
        except:
            
            #条件にマッチしたIndexを取得
            drop_index = df.index[df["smiles"] == name][0]
            droplist.append(drop_index)
            if False:
                #条件にマッチしたIndexを削除
                df = df.drop(drop_index)
            print("model serch error")
            continue
        df["sub_atom"][i]=[j for j in range(df["const"][i]) if j not in df["car_atom"][i]]
        #df=search(df,flag,i)
        df["radius"][i]=van_der_waals_radius(df["symbol"][i].copy())
        #df["mass"][i]=atom_mass(df["symbol"][i])
        df["car_C_num"][i]=df["car_atom"][i][0]
        df["conf_coordinate"][i]={}
        #df["BD_axis"][i]={}
        #df["FL_axis"][i]={}
        df["car_vec"][i]={}
        df["cube_radius"][i]={}
        df["rate"][i]=(rate:=np.array([Bd(energy[0],df["temperature"][i]) for energy in df["energy"][i]]))/sum(rate)#{}
        df["rate"][i]=dict(zip(df["cid"][i], df["rate"][i]))
        df["cis_trans_flag"][i]={}
        df["atom_from_car_C"][i]={}
        df["inner"][i]={}
        df["psi"][i]={}
        for cid in df["cid"][i]:
            conf = df["mol"][i].GetConformer(cid)#ここにconfを算出
            df["conf_coordinate"][i][cid]=coordinate_transformation(conf.GetPositions(),df,i)#conf.GetPositions()
            if True: df["cube_radius"][i][cid]=cul_rad_beta(df["bond"][i],conf,df["const"][i])
            if False:
                df["psi"][i][cid]={}
                
                for j in range(len(df["conf_coordinate"][i][cid])):
                    coordinate=df["conf_coordinate"][i][cid][j]
                    df["psi"][i][cid][j]=[np.linalg.norm(coordinate),np.arccos(coordinate[2]/np.linalg.norm(coordinate)),np.sign(coordinate[1])*np.arccos(coordinate[0]/np.linalg.norm(coordinate[:2]))]#r,psi,theta

    df = df.drop(droplist)
    return df


def new_model_maker(df,param):
    df["mol"]=[Chem.AddHs(Chem.MolFromSmiles(name)) for name in df["smiles"]]
    df["symbol"]=[[atom.GetSymbol() for atom in mol.GetAtoms()] for mol in df["mol"]]
    df["cid"]=[AllChem.EmbedMultipleConfs(mol, numConfs=param["numConfs"], randomSeed=1, pruneRmsThresh=0.1, numThreads=0) for mol in df["mol"]]
    df["hetero"]=[sum([symbols.count(M) for M in ["C","H"]])+1!=len(symbols) for symbols in df["symbol"]]#ケトン置換基にヘテロがあるか
    df["temp."]=[temp if type(temp) in [float,int] and temp==temp else 273.15+20 for temp in df["temperature"]]#nanの場合は20℃
    df["yld_1to99"]=df["yld"].where(df["yld"]<99, 99).where(df["yld"]>1,1)
    df["gibbs"]=[param["R"]*temp*np.log(100/yld-1) for temp,yld in zip(df["temp."],df["yld_1to99"])]
    for mol in df["mol"]:
        AllChem.ComputeGasteigerCharges(mol)
    df["charges"]=[[float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()] for mol in df["mol"]]
    
    ratelist=[]
    energieslist=[]
    droplist=[]
    for i in df.index:#配座関係
        try:
            energies=[]
            for cid in df["cid"][i]:
                mmff=AllChem.MMFFGetMoleculeForceField(df["mol"][i], AllChem.MMFFGetMoleculeProperties(df["mol"][i]), confId=cid)#使用する力場
                if param["minimize"]: mmff.Minimize()
                energies.append([mmff.CalcEnergy(), cid])#計算した配座、エネルギーをenergyリストに追加
            min_energy=sorted(energies)[0][0]
            for _ in range(len(energies)):
                energies[_][0]-=min_energy
            if param["cid"]=="opt":
                energies=[sorted(energies)[0]]
            if False:
                def num_cids(num,energies):
                    if num=="opt":
                        return [sorted(energies)[0]],[cids[0]]#return sorted(energies)
                    if num=="all": return energies,cids
                    if type(num)==float or type(num)==int:
                        energies_new=[]
                        #cids_new=[]
                        for i in range((leng:=len(energies))):
                            rate_=num
                            if energies[i][0]<np.log(rate_)*-1*1.987e-3*temp:

                                energies_new.append(energies[i])
                                #cids_new.append(energies[i][1])
                        return energies_new#,cids_new
                energies=num_cids(0.1,energies)
            ene=[_[0] for _ in energies]
            df["cid"][i]=[_[1] for _ in energies]
            #df.loc[i,"cid"]=[_[1] for _ in energies]
            rate=np.array([math.exp(-e/(df["temp."][i]*param["R"])) for e in ene])
            rate=rate/sum(rate)
            ratelist.append(rate)#{}
            energieslist.append(ene)
        except:
            print(str(i)+" is deleted because conf calc. failed")
            droplist.append(i)
    df=df.drop(droplist)
    df["rate"]=ratelist
    df["energy"]=energieslist
    
    #カルボニルの位置取得
    df["[#6](=[#8])([c,C])([c,C])"]=[mol.GetSubstructMatch(Chem.MolFromSmarts("[#6](=[#8])([c,C])([c,C])")) for mol in df["mol"]]
    df["[#6](=[#8])([c,C])([c,C])"]=[list(carb[0:2])+list(reversed(carb[2:4])) if carb[2]>carb[3] else carb for carb in df["[#6](=[#8])([c,C])([c,C])"]]
    
    #置換基原子取得
    sublist=[]
    for mol,car in zip(df["mol"],df["[#6](=[#8])([c,C])([c,C])"]):
        sublist.append([atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in car])
    df["sub_atom"]=sublist
    
    #最短距離
    l=[]
    for i in df.index:
        mol=df["mol"][i]
        l_=[]
        for atom in mol.GetAtoms():
            if df["sub_atom"][i][0]==atom.GetIdx():
                l_.append(1)
            else:
                l_.append(len(Chem.rdmolops.GetShortestPath(mol, df["sub_atom"][i][0], atom.GetIdx())))
        l.append(l_)    
        #l.append(len([Chem.rdmolops.GetShortestPath(mol, df["sub_atom"][i][0], atom.GetIdx()) for atom in mol.GetAtoms()]))
    df["path"]=l
    
    #部分構造を持つかどうか
    for sub in ["c1ccccc1","O=C1CCCCCC1","O=C1CCCCC1","O=C1CCCC1","O=C1CCC1","O=C1CC1"]:
        df[sub]=[mol.HasSubstructMatch(Chem.MolFromSmarts(sub)) for mol in df["mol"]]
    
    #半径取得
    radlist=[]
    radius_list={"H":1.20,"C":1.70,"O":1.4,"S":1.80,"Si":2.1, "N":1.6,"P":1.95}#https://japan2.wiki/wiki/Van_der_Waals_radius
    for i in df.index:
        rad=[]
        for symbol in df["symbol"][i]:
            try:
                rad.append(radius_list[symbol])
            except:
                print("NO RADIUS DATA : "+symbol)
                rad.append(1.70)
        radlist.append(rad)
    df["radius"]=radlist
    
    
    
    #座標の定義
    conf__=[]
    for i in df.index:
        conf_=[]
        for cid in df["cid"][i]:
            conf=df["mol"][i].GetConformer(cid).GetPositions()
            c,o,c1,c2=df["[#6](=[#8])([c,C])([c,C])"][i]
            #平行移動させて注目原子が原点にくるようにする
            conf=conf-conf[c]
            ry = atan(-1*conf[o][2]/conf[o][0])
            #回転によりxy平面に酸素原子を持ってくる
            for ho1 in conf: (ho1[0],ho1[2]) = (ho1[0]*cos(ry)-ho1[2]*sin(ry),ho1[0]*sin(ry)+ho1[2]*cos(ry))
            rz = atan(conf[o][1]/conf[o][0])
            #回転によりx軸上に酸素原子を持ってくる
            for ho2 in conf: (ho2[0],ho2[1]) = (ho2[0]*cos(rz)+ho2[1]*sin(rz),-1*ho2[0]*sin(rz)+ho2[1]*cos(rz))
            rx = atan((conf[c1][2]-conf[c2][2])/(conf[c2][1]-conf[c1][1]))
            #回転によりxy平面に注目原子を含む4つの原子が来るようにする
            for ho3 in conf: (ho3[1],ho3[2]) = (ho3[1]*cos(rx)-ho3[2]*sin(rx),ho3[1]*sin(rx)+ho3[2]*cos(rx))
            if conf[o][0]<0:#Oをx正に
                for ho2 in conf: (ho2[0],ho2[1]) = (-ho2[0],-ho2[1])
            if conf[c1][1]-conf[c2][1]>0:#c1numをy負側に
                for ho3 in conf: (ho3[1],ho3[2]) = (-ho3[1],-ho3[2])
            try:
                up,down=df["cis_trans"][i]
                if conf[int(up)][2]-conf[int(down)][2]<0:#upの原子をz正側に
                    for ho3 in conf: (ho3[1],ho3[2]) = (-ho3[1],-ho3[2])
                    df.loc[i,"[#6](=[#8])([c,C])([c,C])"]=list(carb[0:2])+list(reversed(carb[2:4]))
            except: None
            conf_.append(conf)
        conf__.append(conf_)
    df["conf"] = conf__
    return df