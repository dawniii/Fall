import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
f='E:/Fall/UPFall/data/CompleteDataSet.csv'
df=pd.read_csv(f,header=None,skiprows=2)
df=df[[1,2,3,4,5,6,8,9,10,11,12,13,15,16,17,18,19,20,22,23,24,25,26,27,29,30,31,32,33,34,43,44,45,46]]
columns=['Ankle_ACC_X','Ankle_ACC_Y','Ankle_ACC_Z','Ankle_ANG_X','Ankle_ANG_Y','Ankle_ANG_Z','RP_ACC_X','RP_ACC_Y','RP_ACC_Z','RP_ANG_X','RP_ANG_Y','RP_ANG_Z',
         'BELT_ACC_X', 'BELT_ACC_Y', 'BELT_ACC_Z', 'BELT_ANG_X', 'BELT_ANG_Y', 'BELT_ANG_Z','NECK_ACC_X','NECK_ACC_Y','NECK_ACC_Z','NECK_ANG_X','NECK_ANG_Y','NECK_ANG_Z','WRST_ACC_X','WRST_ACC_Y','WRST_ACC_Z','WRST_ANG_X','WRST_ANG_Y','WRST_ANG_Z',
         'Subject','Activity','Trial','Tag']
df = pd.DataFrame(df.values, columns=columns)
df=df.loc[df['Subject']==1]
df=df.loc[df['Activity']==1]
df=df.loc[df['Trial']==1]
df.reset_index(drop=True, inplace=True)
plt.plot(df['Ankle_ACC_X'],label='Ankle_ACC_X')
plt.plot(df['Activity'],label='Activity')
plt.plot(df['Trial'],label='Trial')
plt.legend()
plt.show()