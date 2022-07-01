#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import time
import pandas as pd
def data(name,sheet_name):
    import pandas as pd
    #df=pd.concat(pd.read_excel('data.xlsx',sheet_name =[name]))
    df=pd.concat(pd.read_excel(name,sheet_name =[sheet_name]))
    #print(df)
    load_start_time=time.time()
    if False:#テストと学習をランダムに設定
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd
        from sklearn.model_selection import train_test_split
        #df.head()
        train_x = df["smiles"]
        train_y = df['yld']
        (train_x, test_x ,train_y, test_y) = train_test_split(train_x, train_y, test_size = 0.3)
        #print(test_x)
        train_x
    df["temperature"]=[temp if temp==temp else 273.15+20 for temp in df["temperature"]]#nanの場合は20℃
    df["cis_trans"]=[cis_trans.split() if cis_trans==cis_trans else [] for cis_trans in df["cis_trans"]]
    
    #df_test=df[df["type"]=="test"]#[0:1]
    #df_train=df[df["type"]=="train"]#[0:4]#[0:4]
    #print(df_test)
    #df_all=pd.concat([df_test,df_train])
    try:
        df=df[df["type"].isin(["train","test"])]#[0:5]
    except:
        1
    #df_all=df[df["phenyl"]!=1]#[0:5]
    df["yld"]=df["yld"].astype(float)
    #df_all["gibbs"]=(8.31*df_all["temperature"]*np.log(100/df_all["yld"]-1)).replace([np.inf,-np.inf], [10000,-10000])
    #df_all=df_all[(df_all["compound_type"]!=4) & (df_all["compound_type"]!="hetero")]
    #df_train=df_train[(df_train["compound_type"]!=4) & (df_train["compound_type"]!="hetero")]
    load_end_time=time.time()
    print(load_end_time-load_start_time)
    return df#df_test,df_train#test_name_list,test_yld,name_list,yld,cis_trans_train