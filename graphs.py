import matplotlib.pyplot as plt
import numpy as np


def plot(data,label):
    x = np.arange(len(data[0]))
    width = 0.2

    fig, axes = plt.subplots(ncols=1, nrows=1)
    plt.title('label '+str(label))
    #plt.xlabel('Parameters')
    plt.ylabel('Score')
    axes.bar(x-width, data[0],width=width, label="precision",color="r")
    for i, v in enumerate(data[0]):
        axes.text(x[i] - 1.5*width, v + 0.01, str(v)[:6])
    axes.bar(x, data[1],width=width, label="recall",color="g")
    for i, v in enumerate(data[1]):
        axes.text(x[i] - 0.5*width, v + 0.01, str(v)[:6])
    axes.bar(x+width, data[2],width=width, label="f1",color="b")
    for i, v in enumerate(data[2]):
        axes.text(x[i] + 0.5*width, v + 0.01, str(v)[:6])
    axes.set_xticks(x)
    axes.set_xticklabels(['db scan', 'k means', 'Hierarchical Clustering','Girvan Newman'])

    plt.legend()
    plt.show()
    print(label)





#label:  0
precision0 =[1.0,0,0,0.6785714285714286]
recall0 =[0.0625,0,0,0.1484375]
f10 =[0.11764705882352941,0,0,0.2435897435897436]

#label:  1
precision1= [0.9166666666666666,0.833530106257379,0.8,0.8527851458885941]
recall1= [0.029689608636977057,0.9527665317139001,0.016194331983805668,0.8677462887989204]
f11 =[0.05751633986928104,0.8891687657430731,0.031746031746031744,0.860200668896321]

#label:  2
precision2 =[0,0,1.0,0]
recall2=[0,0,0.07142857142857142,0]
f12= [0,0,0.13333333333333333,0]

#label:  3
precision3= [0.6882546652030735,0.9812304483837331,0.6835212804656239,0.9485627836611196]
recall3= [0.9994686503719448,1.0,0.9984059511158342,0.9994686503719448]
f13 =[0.8151679306608884,0.9905263157894737,0.8114877996113151,0.9733505821474774]

precision=[precision0,precision1,precision2,precision3]
recall=[recall0,recall1,recall2,recall3]
f1=[f10,f11,f12,f13]

for i in range(4):
    plot([precision[i],recall[i],f1[i]],i)