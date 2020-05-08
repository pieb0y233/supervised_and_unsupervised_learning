import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(0)







def Eulician_Metric(x,y):
    return np.linalg.norm(x-y)

def Cosine_Metric(x,y):
    summation=0
    for i in range(len(x)):
        summation+=x[i]*y[i]
    return summation/(np.linalg.norm(x)*np.linalg.norm(y))

###     k means



#   the prediction function of the k means algorithm
#
#   input:
#
#   centers:            one point for each cluster that suppose to represent it
#   point:              the data point
#   metricFunction:     the metric function
#
#   output:
#
#   clusterIndex:       the index of the nearest cluster in the centers list

def k_means_predict(centers,point,metric):
    distances = [metric(point, centers[j]) for j in range(len(centers))]
    clusterIndex = np.argmin(distances)
    return clusterIndex





#   the main function of the k means algorithm
#
#   input:
#
#   centers:            one point for each cluster that suppose to represent it
#   points:             the data points
#   metricFunction:     the metric function
#
#   output:
#
#   centers:            the new centers

def k_means_centers(centers,points,metricFunction):
    centers=np.array(centers)
    oldCenters=np.array([])
    for iteration in range(1000):
        groups=[[] for item in centers]
        for i in range(len(points)):
            clusterIndex=k_means_predict(centers,points[i],metricFunction)
            groups[clusterIndex].append(points[i])
        oldCenters=centers
        centers=np.array([np.mean(groups[i],axis=0) for i in range(len(centers))])
        if(np.linalg.norm(centers-oldCenters)<0.01):
            return centers


#   calling the k_means function
#centers=k_means_centers(points[:3],points,Eulician_Metric)
#print(points)
#print(centers)


###     PCA

def zero_centered(points):
    centers=np.zeros(len(points[0]))
    for i in range(len(points)):
        for j in range(len(points[0])):
            centers[j]+=points[i][j]
    centers/=len(points)

    for i in range(len(points)):
        for j in range(len(points[0])):
            points[i][j]-=centers[j]
    #print(points)
    return points

def No_kernel(data,param):
    return np.dot(data.T, data)

def Gaussian_kernel(data,betta):
    new_kernel=np.zeros([len(data[0]),len(data[0])])
    for i in range(len(new_kernel)):
        for j in range(len(new_kernel)):
            aa=data[j]
            #bb=data.T[:,i]
            bb=data[i]
            new_kernel[i][j]=np.exp(-betta*np.linalg.norm(bb-aa)**2)
    return new_kernel


def Polynomial_kernel(data,P):
    new_kernel=np.zeros([len(data[0]),len(data[0])])
    for i in range(len(new_kernel)):
        for j in range(len(new_kernel)):
            new_kernel[i][j]=np.power(1+np.dot(data.T[:,i],data[j]),P)
    return new_kernel






def pca(x,kernel=No_kernel,param=0):
  data=zero_centered(x)
  n, m = data.shape
  Cov = kernel(data,param) / (n-1)
  eigen_vals, eigen_vecs = np.linalg.eig(Cov)
  eigen_vals=np.real(eigen_vals)#[:372]
  eigen_vecs=np.real(eigen_vecs)#[:372]
  w, = np.where(eigen_vals > 0.01)
  eigen_vecs = eigen_vecs[:,w]
  X_pca = np.dot(data,eigen_vecs)
  return X_pca





#zero_centered_points=zero_centered(points)
#print(pca(zero_centered_points))
#print(pca(points,Gaussian_kernel,1))
#print(pca(points,Polynomial_kernel,1))


#   dbscan

#   removing a ndarray from a list of ndarrays, the List.remove() doesnt work that well
def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')


def db_Scan(data,metric,epsilon):
    groups=[]
    remaining_data=list(data)
    index_options=list(range(len(remaining_data)))
    total_scanned=[]
    while(len(total_scanned)<len(remaining_data)):
        #print(index_options)
        i=index_options[np.random.randint(len(index_options))]
        index_options.remove(i)
        total_scanned.append(i)
        ini_point=i
        scan=[ini_point]
        #removearray(remaining_data,remaining_data[ini_point])
        scanned=[]
        while(len(scan)>0):
            aaa=1
            for point_index in index_options[::-1]:
                aa=remaining_data[scan[0]]
                bb=remaining_data[point_index]
                #print(scan[0])
                distance=metric(aa,bb)
                if((distance<epsilon and (not (point_index==scan[0])) )and (not (point_index in total_scanned))):
                    scan.append(point_index)
                    #print(point_index)
                    total_scanned.append(point_index)
                    index_options.remove(point_index)
                    #removearray(remaining_data,remaining_data[point_index])
            scanned.append(scan[0])
            scan=list(set(scan[1:]))
        groups.append(sorted(list(set(scanned))))
    guess=[]
    for i in range(len(remaining_data)):
        for j in range(len(groups)):
            if(i in groups[j]):
                guess.append(j)
                break
    return guess

#print(points)
#db_resaults=db_Scan(points,Eulician_Metric,5)
#print("aa")


def Remove_Cluster(L,cluster):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind][0],cluster[0]):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('Cluster not found in list.')



def get_metric_matrix(points,metric,param=[0]):
    if(len(param)==1):
        n=len(points)
        D=np.zeros([n,n])
        for i in range(n-1):
            for j in range(i+1,n):
                dist=metric(np.array(points[i]),np.array(points[j]))
                #print(i,j)
                #print(dist)
                D[i][j]=dist
                D[j][i]=dist
        return D
    else:
        n=len(points)
        D=np.zeros([n,n])
        for i in range(n-1):
            for j in range(i+1,n):
                dist=metric(points[i],points[j],param)
                D[i][j]=dist
                D[j][i]=dist
        return D



def change_difference_matrix(points, diff, x, y, new_cluster, c):
    lower = min(x, y)
    higher = max(x, y)
    n = len(diff) - 1

    tmp_row = diff[n].copy()
    diff[n, :] = diff[higher]
    diff[higher] = tmp_row

    tmp_point = points[higher].copy()
    points[higher] = points[-1]
    points[-1] = tmp_point

    tmp_row = diff[n - 1].copy()
    diff[n - 1] = diff[lower]
    diff[lower] = tmp_row

    tmp_point = points[lower].copy()
    points[lower] = points[-2]
    points[-2] = tmp_point

    tmp_column = diff[:, n].copy()
    diff[:, n] = diff[:, higher]
    diff[:, higher] = tmp_column

    tmp_column = diff[:, n - 1].copy()
    diff[:, n - 1] = diff[:, lower]
    diff[:, lower] = tmp_column

    points = points[:-2]
    points.append(new_cluster)
    diff=diff[:,:-1]
    diff = diff[:-1, :]
    #distances=np.array([Centroid_Distance(points[-1],points[i],c) for i in range(len(points))])
    distances=np.array([Eulician_Metric(points[-1][1],points[i][1]) for i in range(len(points))])

    diff[:,-1]=distances
    diff[-1, :]=distances

    #print("aa")
    return diff,points


def unite_clusters(L,x,y,c,D):
    aa=np.concatenate((L[x][0],L[y][0]),axis=0)
    new_cluster=[aa,(len(L[x][0])*L[x][1]+len(L[y][0])*L[y][1])/(len(L[x][0])+len(L[y][0]))]
    D,L=change_difference_matrix(L,D,x,y,new_cluster,c)

    #Remove_Cluster(L, x)
    #Remove_Cluster(L, y)
    #L.append(new)
    return D,L


def Centroid_Distance(x,y,c,metric=Eulician_Metric):
    return len(x[0])/len(x[0]+len(y[0]))*(metric(x[1],c[1]))+len(y[0])/len(x[0]+len(y[0]))*(metric(y[1],c[1]))-(len(x)*len(y))/((len(x)+len(y))**2)*metric(x[1],y[1])

def Hierarchical_Preperation(data):
    new_data=[]
    for i in range(len(data)):
        new_data.append([np.array([i]),data[i]])
    return new_data

def biggest_norm(data):
    n=len(data)
    norms=np.zeros(n)
    for i in range(n):
        norms[i]=np.linalg.norm(data[i][1])
    return np.argmax(norms)




def Hierarchical_Clustering(points,clusters_amount,metric):
    data=Hierarchical_Preperation(points)
    tmp_points=points.copy()
    c = biggest_norm(data)
    c_point = data[c]
    data.pop(c)
    tmp_points.pop(c)
    #D=get_metric_matrix(tmp_points,Centroid_Distance,c_point)
    D=get_metric_matrix(tmp_points,Eulician_Metric)

    tmp_points=0
    #Remove_Cluster(data, c)
    while len(data)>clusters_amount:
        D+=10*np.eye(len(D))
        #tmp_data=data.copy()
        #c=tmp_data[np.random.randint(len(data))]#argmax norm
        min_distance=99999
        index1=0
        index2=1
        """
        for i in range(len(data)-1):
            for j in range(i+1,len(data),1):
                dist=Centroid_Distance(data[i],data[j],c_point,metric)
                if(dist<min_distance):
                    min_distance=dist
                    index1=i
                    index2=j
        """
        """
        for i in range(len(data)-1):
            for j in range(i+1,len(data),1):
                dist=D[i][j]
                if(dist<min_distance):
                    min_distance=dist
                    index1=i
                    index2=j
        """
        n=np.argmin(D)
        index1=int(n/len(D))
        index2=n-index1*len(D)


        D,data=unite_clusters(data,index1,index2,c_point,D)

    clusters=[]#lengths
    for i in range(len(data)):
        #print("cluster",i+1,data[i][0])
        clusters.append(data[i][0])
    guess=[]
    for i in range(len(points)):
        for j in range(len(clusters)):
            changed=0
            if(i in clusters[j]):
                guess.append(j)
                changed=1
                break
        if(changed==0):
            guess.append(len(data))

    return guess



#Hierarchical_Clustering(points,3,Eulician_Metric)


#   Girvan–Newman
def compare_points(point1,point2):
    for i in range(len(point1)):
        if(point1[i]!=point2[i]):
            return False
    return True

def connect2(point1,point2,verticies):

    for v in verticies:
        val=0
        if(point1==v[0] or point1==v[1]):
            val+=1
        if(point2==v[0] or point2==v[1]):
            val+=1
        if(val==2):
            return False
    return True

def shortest_path_graph(points,metric):
    verticies=[]
    for i in range(len(points)):
        dist = np.zeros(len(points))
        for j in range(len(points)):
            dist[j]=metric(points[i],points[j])
        dist[i]=999999999
        closest_point=np.argmin(dist)
        if(connect2(i,closest_point,verticies)):
            verticies.append([i,closest_point])
        dist[np.argmin(dist)]=999999999
        closest_point2=np.argmin(dist)
        if(connect2(i,closest_point2,verticies)):
            verticies.append([i,closest_point2])
        """
        dist[np.argmin(dist)]=999999999
        closest_point3=np.argmin(dist)
        if(connect2(i,closest_point3,verticies)):
            verticies.append([i,closest_point3])
        """
    return verticies


def find_index_of_array_v2(list1, array1):
    idx = np.nonzero([type(i).__module__ == np.__name__ for i in list1])[0]
    for i in idx:
        if np.all(list1[i]==array1):
            return i
    return -1

def key(item):
    global distances
    return distances[item]


def dikstra_with_path(start,points,verticies,metric):
    visited=[]
    next=[start]
    global distances
    distances={i:[999999,[]] for i in range(len(points))}
    distances[start]=[0,[]]
    while len(next)>0:
        for j in range(len(verticies)):
            if((next[0]==verticies[j][0]) or (next[0]==verticies[j][1])):
                current_index=next[0]
                if(current_index==verticies[j][0]):
                    next_index =verticies[j][1]
                elif(current_index==verticies[j][1]):
                    next_index = verticies[j][0]
                if ((not(next_index in next)) and (not(next_index in visited))):
                        next.append(next_index)
                if ((not(next_index in visited)) and distances[current_index][0] +
                    metric(points[verticies[j][0]], points[verticies[j][1]]) < distances[next_index][0]):
                    num=distances[current_index][0] + metric(points[verticies[j][0]], points[verticies[j][1]])
                    distances[next_index] = [num,distances[current_index][1] + [current_index]]
        visited.append(next[0])
        next=next[1:]
        #next_indexes=[find_index_of_array_v2(points,next[ii])for ii in range(len(next))]
        next=[ii for ii in sorted(next,key=key)]
    #print("aa")
    return distances



#edges=np.array([[0,0],[0,1],[1,1],[2,2],[2.5,4],[3,3],[4,4],[5,5],[5.5,5],[6,6],[7,7],[6,9],[8,9],[9,9],[9,9.5]])
#verticies=shortest_path_graph(edges,Eulician_Metric)


def Plot_Line(line,points):
    plt.plot([points[line[0]][0],points[line[1]][0]],[points[line[0]][1],points[line[1]][1]],"b")
    plt.plot([points[line[0]][0], points[line[1]][0]], [points[line[0]][1], points[line[1]][1]], "ro")
"""
for i in range(len(verticies)):
    Plot_Line(verticies[i])
plt.close()

for i in range(len(verticies)):
    plt.plot([verticies[i][0][0],verticies[i][1][0]],[verticies[i][0][1],verticies[i][1][1]],"b")
    plt.plot([verticies[i][0][0],verticies[i][1][0]],[verticies[i][0][1],verticies[i][1][1]],"ro")
plt.show()
"""
def in_betweeness(edges,verticies,metric,removed):
    scores={i:0 for i in range(len(edges))}
    options=list(range(len(edges)))
    [options.pop(i) for i in sorted(removed)[::-1]]
    for i in range(int(len(options)/20)):
        distances=dikstra_with_path(options[np.random.randint(len(options))],edges,verticies,metric)
        counts=[]
        for j in range(len(edges)):
            counts+=distances[j][1][1:]
        unique, counts = np.unique(counts, return_counts=True)
        for j in range(len(unique)):
            scores[unique[j]]+=counts[j]
    #print("a")
    return np.argmax([scores[i] for i in range(len(edges))])


def Girvan_Newman(data):
    verticies = shortest_path_graph(data, Eulician_Metric)
    initial_length=len(verticies)
    removed=[]
    while( len(verticies)>0.75*initial_length):
        next_point=in_betweeness(data,verticies,Eulician_Metric,removed)
        removed.append(next_point)
        tmp_verticies=[]
        for v in verticies:
            if((not(int(v[0])==int(next_point))) and (not(int(v[1])==int(next_point)))):
                tmp_verticies.append(v)
        verticies=tmp_verticies

    for v in verticies:
        Plot_Line(v,data)
    plt.close()

    options=[]
    [options.append(verticies[ii][0]) for ii in range(len(verticies))]
    [options.append(verticies[ii][1]) for ii in range(len(verticies))]
    options=sorted(list(set(options)))
    groups=[]
    done=[]
    #options.pop(index)
    while(len(options)>0):
        index = np.random.randint(len(options))
        next = [options[index]]
        sub_group = []
        while(len(next)>0):
            for j in range(len(verticies)):
                if(verticies[j][0]==next[0] or verticies[j][1]==next[0]):
                    if(verticies[j][0]==next[0] and not (verticies[j][1] in done) and not (verticies[j][1] in next)):
                        next.append(verticies[j][1])
                    elif(verticies[j][1]==next[0] and not (verticies[j][0] in done) and not (verticies[j][0] in next)):
                        next.append(verticies[j][0])
            sub_group.append(next[0])
            options.pop(options.index(next[0]))
            done.append(next[0])
            next=next[1:]

        groups.append(sub_group)
        sub_group=[]


    guess=[]
    for i in range(len(data)):
        val=0
        for j in range(len(groups)):
            if(i in groups[j]):
                guess.append(j)
                val=1
                break
        if(val==0):
            guess.append(len(groups))


    return guess
#Girvan_Newman(edges)


#   report
#print(points, len(points))
y_vector=[0,0,0,1,1,1,2,2]
y_vector2=[0,0,0,1,1,0,1,1]

def Result_Counter(true,guess):
    options=list(set(np.concatenate((true,guess))))
    n=len(options)
    confution_matrix=np.zeros([n,n])
    for i in range(len(true)):
        confution_matrix[int(guess[i])][int(true[i])]+=1
    return confution_matrix

def measuring(true,guess):
    mat=Result_Counter(true,guess)
    n=len(mat)
    unique, counts = np.unique(np.array(true), return_counts=True)
    stats=[]

    for i in unique:
        i=int(i)
        print("label: ",i)
        true_positive=mat[i][i]
        false_positive=sum(mat[i])-true_positive
        relevant=counts[i]
        if(true_positive+false_positive==0):
            print("not found any!")
            print("")
        else:
            precision=true_positive/(true_positive+false_positive)
            recall=true_positive/relevant
            f1=2*(precision*recall)/(precision+recall)
            stats.append([i,precision,recall,f1])
            if(not(np.nan in stats[-1])):
                print("precision",precision)
                print("recall",recall)
                print("f1",f1)
                print("")
    return stats

def label_orgenizer(guess,true):
    options=[[]for i in range(len(set(guess)))]
    for i in range(len(guess)):
        options[guess[i]].append(true[i])
    translator={}
    for j in range(len(options)):
        unique, counts = np.unique(options[j], return_counts=True)
        translator[j]=unique[np.argmax(counts)]
    new_guess=np.array([translator[guess[j]] for j in range(len(guess))])
    measuring(true,new_guess)




#resaults=measuring(y_vector,y_vector2)





def cmdscale(points,metric):
    D=get_metric_matrix(points,metric)

    # Number of points
    n = len(D)
    H = np.eye(n) - np.ones((n, n)) / n
    B = -H.dot(D ** 2).dot(H) / 2
    eigen_vals, eigen_vecs = np.linalg.eigh(B)
    eigen_vals = np.real(eigen_vals)
    eigen_vecs = np.real(eigen_vecs)
    idx = np.argsort(eigen_vals)[::-1]
    evals = eigen_vals[idx]
    evecs = eigen_vecs[:, idx]
    w, = np.where(evals > 0.01)
    L = np.diag(np.sqrt(evals[w]))
    V = evecs[:, w]
    Y = V.dot(L)

    return Y, evals


#a,b=cmdscale(points,Eulician_Metric)

print("function file loaded")


