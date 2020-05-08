import numpy as np
import pandas as pd
from functions import *

np.random.seed(0)
test=0

def pearson_correlation(vec1,vec2):
    mean_x=np.mean(vec1)
    mean_y=np.mean(vec2)
    upper_side=0
    lower_side1=0
    lower_side2=0
    for i in range(len(vec1)):
        upper_side+=(vec1[i]-mean_x)*(vec2[i]-mean_y)
        lower_side1+=(vec1[i]-mean_x)**2
        lower_side2+=(vec2[i]-mean_y)**2
    return upper_side/(np.sqrt(lower_side1*lower_side2))

#print(pearson_correlation([0.5,0.7],[1,1.4]))
#print("aa")

percent=0.20
def List_To_Dictionary(L):
    dic={}
    for i in range(len(L)):
        dic[L[i]]=i
    return dic
df=pd.read_csv("Audit_data.csv",encoding="utf_8")
all_keys=list(df.keys())

[all_keys.pop(i) for i in [9,6,3,2,0]]
df=df[all_keys]
if(test==1):
    df1=df[df["Audit_opinion"] == "Unqualified"][:140] #["Unqualified","Qualified","Adverse","Disclaimer"]
    df2=df[df["Audit_opinion"] == "Qualified"][:140]
    df3=df[df["Audit_opinion"] == "Adverse"][:70]
    df4=df[df["Audit_opinion"] == "Disclaimer"][:140]
else:
    df1=df[df["Audit_opinion"] == "Unqualified"][:int(percent*9410)]
    df2=df[df["Audit_opinion"] == "Qualified"][:int(percent*3706)]
    df3=df[df["Audit_opinion"] == "Adverse"][:int(percent*71)]
    df4=df[df["Audit_opinion"] == "Disclaimer"][:int(percent*644)]
df=pd.concat([df1,df2,df3,df4])

df=df.fillna(0)
df1=df["Legal_form"].values
df1_values=list(set(df1))
df2=[str(Str)[-4:] for Str in df["Year_of_establishment"].values]
#df["Year_of_establishment"]=df2
df2_values=list(set(df2))
df3=df['Auditor_name'].values
df3_values=list(set(df3))
df4=df['Audit_opinion'].values
df4_values=list(set(df4))
df5=df['Auditor_switch'].values
df5_values=list(set(df5))
#df=df.fillna("NaN")

df1_dic=List_To_Dictionary(df1_values)
df2_dic=List_To_Dictionary(df2_values)
df3_dic=List_To_Dictionary(df3_values)
df4_dic=List_To_Dictionary(df4_values)
df5_dic=List_To_Dictionary(df5_values)

df1=[df1_dic[df1[i]] for i in range(len(df1))]
df2=[df2_dic[df2[i]] for i in range(len(df2))]
df3=[df3_dic[df3[i]] for i in range(len(df3))]
df4=[df4_dic[df4[i]] for i in range(len(df4))]
df5=[df5_dic[df5[i]] for i in range(len(df5))]

df["Legal_form"]=df1
df["Year_of_establishment"]=df2
df['Auditor_name']=df3
df['Audit_opinion']=df4
df['Auditor_switch']=df5

all_keys.remove("Audit_opinion")
labels=df['Audit_opinion']



data=np.array(df)


doubles=[14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 30, 31, 32, 33, 34, 36, 37, 43, 45, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 65, 66, 68, 69, 70, 71, 73, 74, 75, 76, 79, 82, 83, 84, 86, 87, 88, 89, 94, 95, 98, 99, 100, 104, 105, 108, 109, 112, 114, 115, 116, 117, 118, 120, 122, 124, 127, 129, 130, 131, 132, 133, 134, 136, 137, 140, 142, 143, 146, 149, 153, 156, 159, 162, 163, 166, 169, 172, 174, 175, 179, 182, 185, 188, 189, 192, 194, 195, 198, 201, 205, 208, 211, 212, 214, 218, 220, 221, 224, 227, 228, 231, 232, 233, 234, 237, 239, 240, 241, 244, 247, 248, 250, 251, 253, 254, 257, 258, 259, 260, 261, 263, 264, 266, 267, 270, 271, 273, 275, 276, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 296, 299, 300, 302, 303, 305, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 325, 326, 327, 328, 337, 338, 339, 340, 341, 342, 343, 344, 345, 347, 348, 349, 350, 351, 352, 354, 355, 356, 357, 358, 360, 361, 362, 363, 365, 366, 367, 368, 369, 370, 371, 373, 374, 375, 376, 377, 381, 382, 383, 384, 385, 386, 389, 390, 393, 396, 397, 398, 400]
[all_keys.pop(i) for i in sorted(list(set(doubles)))[::-1]]
df=df[all_keys]


a=[np.array(data[i]) for i in range(len(data))]
db_vec=db_Scan(a,Eulician_Metric,305000)
print("")
print("db scan!")
print("")
label_orgenizer(db_vec,labels.values)

df=(df-df.min())/(df.max()-df.min())
df=df.fillna(0)




df=df[all_keys]
keys=df.keys()
n=len(keys)


"""
doubles=[]

for i in range(n-1):
    for j in range(i+1,n):
        tmp_value=pearson_correlation(df[keys[i]].values,df[keys[j]].values)
        if(np.abs(tmp_value)>0.7):
            doubles.append(j)
            print(i,j,tmp_value)
print(len(set(doubles)))
print(sorted(list(set(doubles))))

[all_keys.pop(i) for i in sorted(list(set(doubles)))[::-1]]
df=df[all_keys]
"""
"""
for i in range(n-1):
    for j in range(i+1,n):
        tmp_value=pearson_correlation(df[keys[i]].values,df[keys[j]].values)
        if(tmp_value>0.5):
            doubles.append(j)
            print(i,j,tmp_value)
print(len(set(doubles)))
print(sorted(list(set(doubles))))
#[all_keys.pop(i) for i in sorted(list(set(doubles)))[::-1]]
"""
print("aa")













data=df




print("preprocessing done, use 'data' as the clean data and 'labels' as the coresponding labels")