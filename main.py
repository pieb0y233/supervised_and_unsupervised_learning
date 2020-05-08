from functions import *
from preprocessing import *
import matplotlib.pyplot as plt
np.random.seed(0)
import numpy as np
import pandas as pd

def show(all_labels,inex1,index2,a):
    for i in range(len(a)):
        if(label_array[i]==all_labels[0]):
            plt.plot([a[i][inex1]], [a[i][index2]], "ro")
        elif(label_array[i]==all_labels[1]):
            plt.plot([a[i][inex1]],[a[i][index2]],"bo")
        elif(label_array[i]==all_labels[2]):
            plt.plot([a[i][inex1]],[a[i][index2]],"go")
        elif(label_array[i]==all_labels[3]):
            plt.plot([a[i][inex1]],[a[i][index2]],"yo")
    plt.show()
    plt.close()


def full_check(data, label_array):
    a = [np.array(data[i]) for i in range(len(data))]

    centers = k_means_centers(a[:8], a, Eulician_Metric)
    k_means_predictions = []
    for i in range(len(a)):
        k_means_predictions.append(k_means_predict(centers, a[i], Eulician_Metric))

    print("k means:")
    k_means_results = label_orgenizer(k_means_predictions, label_array)

    print("Hierarchical Clustering")
    HC = Hierarchical_Clustering(a, 4, Eulician_Metric)
    HC_results = label_orgenizer(HC, label_array)

    print("Girvan Newman:")
    GN = Girvan_Newman(a)
    GN_results = label_orgenizer(GN, label_array)


data_array=zero_centered(np.array(data))
label_array=np.array(labels)
all_labels=sorted(list(set(label_array)))

"""
data_after_pca=pca(data_array)
#show(all_labels,0,1,data_after_pca)#nice
print("#    DATA AFTER PCA")
full_check(data_after_pca,label_array)

data_after_pca=pca(data_array,Gaussian_kernel,2)
#show(all_labels,0,1,data_after_pca)
print("#    DATA AFTER Gaussian_kernel PCA")
full_check(data_after_pca,label_array)

data_after_pca=pca(data_array,Polynomial_kernel,4)
#show(all_labels,0,1,data_after_pca)
print("#    DATA AFTER Polynomial_kernel 4 PCA")
full_check(data_after_pca,label_array)

data_after_pca=pca(data_array,Polynomial_kernel,3)
#show(all_labels,0,1,data_after_pca)
print("#    DATA AFTER Polynomial_kernel 3 PCA")
full_check(data_after_pca,label_array)

data_after_pca=pca(data_array,Polynomial_kernel,2)
#show(all_labels,0,1,data_after_pca)
print("#    DATA AFTER Polynomial_kernel 2 PCA")
full_check(data_after_pca,label_array)
"""
data_after_pca=pca(data_array,Polynomial_kernel,1)
#show(all_labels,0,1,data_after_pca)
#print("#    DATA AFTER Polynomial_kernel 1 PCA")
#full_check(data_after_pca,label_array)

data_after_cmds,b=cmdscale(data_after_pca,Eulician_Metric)
print("#    DATA AFTER Polynomial_kernel 1 PCA and CMDS")
full_check(data_after_pca,label_array)

#show(all_labels,0,1,data_after_cmds)#pretty nice
#show(all_labels,0,2,data_after_cmds)#pretty nice
#show(all_labels,1,2,data_after_cmds)#pretty nice
#show(all_labels,2,3,data_after_cmds)#pretty nice



"""
data_after_cmds,b=cmdscale(data_array,Eulician_Metric)
show(all_labels,0,1,data_after_cmds)#pretty nice
show(all_labels,1,2,data_after_cmds)#pretty nice
show(all_labels,0,2,data_after_cmds)#pretty nice
#data_after_pca=pca(data_after_cmds)
#show(all_labels,0,1,data_after_pca)
data_after_pca=pca(data_after_cmds,Gaussian_kernel,2)
show(all_labels,0,1,data_after_pca1)
data_after_pca=pca(data_after_cmds,Gaussian_kernel,1)
show(all_labels,0,1,data_after_pca1)
data_after_pca=pca(data_after_cmds,Gaussian_kernel,-2)
show(all_labels,0,1,data_after_pca1)
data_after_pca=pca(data_after_cmds,Polynomial_kernel,1)
show(all_labels,0,1,data_after_pca)
"""

#a=pca(data_after_cmds,Gaussian_kernel,1)
#a=pca(data_array,Polynomial_kernel,1)
#a,b=cmdscale(a,Eulician_Metric)

#a=data_after_pca
#a=data_after_cmds

"""
print("#    DATA AFTER PCA")
full_check(data_after_pca,label_array)
print("#    DATA AFTER CMDS")
full_check(data_after_cmds,label_array)
"""



#show(all_labels,0,1,a)
#show(all_labels,0,2,a)
#show(all_labels,1,2,a)

#a,b=cmdscale(np.array(df),Eulician_Metric)


print("End of main.py")