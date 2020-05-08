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
precision0 =[1.0,0,0,0.375]
recall0 =[0.8571428571428571,0,0,0.21428571428571427]
f10 =[ 0.923076923076923,0,0,0.2727272727272727]

#label:  1
precision1 =[0.9809523809523809,0,0,0.40540540540540543]
recall1 =[0.8046875,0,0,0.1171875]
f11 =[0.8841201716738197,0,0,0.1818181818181818]

#label:  2
precision2 =[0.9062348960850652,0.7273105745212323,0.6808972503617945,0.7661713286713286]
recall2 =[0.9962805526036131,0.9282678002125399,1.0,0.9314558979808714]
f12= [0.9491268033409264,0.8155929038281978,0.810159276797245,0.840767386091127]

#label:  3
precision3 =[0.9861830742659758,0.5371900826446281,1.0,0.6435185185185185]
recall3 =[0.7705802968960864,0.2631578947368421,0.001349527665317139,0.37516869095816463]
f13 =[0.8651515151515152,0.3532608695652174,0.0026954177897574125,0.473998294970162]

precision=[precision0,precision1,precision2,precision3]
recall=[recall0,recall1,recall2,recall3]
f1=[f10,f11,f12,f13]

for i in range(4):
    plot([precision[i],recall[i],f1[i]],i)