import math
import matplotlib.pyplot as plt
import sklearn.preprocessing as sk
import numpy as np
import random

# 0 = undefined
#-1 = noise

def DBSCAN(data, eps, minPts):
    c = 0
    core = []
    classification = [0]*len(data)
    ini(data)
    plot_data(classification,data)
    for i in range(len(data)):
        neighbors =[]
        if (classification[i]==0):
            neighbors = rangeQuery(data, data[i], eps)
            if (len(neighbors)<minPts):
                classification[i] = -1
            else:
                c += 1
                classification[i]  = c
                core.append(i)
                s = neighbors.copy()
                s.remove(i)
                neighbors =[]
                for j in s:
                    classification[j]  = c
                while len(s)>0:
                    j = s[0]
                    results = rangeQuery(data, data[j], eps)
                    if(len(results)>=minPts):
                        core.append(j)
                        for i in range(len(results)):
                            point = results[i]
                            if ((classification[point]  == 0)or(classification[point]  == -1)):
                                if(classification[point]  == 0):
                                    s.append(point)
                                classification[point]  = c
                    s = s[1:]
                                
    return (classification, core)
    



def ini(data):
    for i in data:
        i.append(0)

def rangeQuery(data, p, eps):
    n = []
    for i in range(len(data)):
        if ((dist_func(p,data[i])<= eps)):
            n.append(i)
    return n

def dist_func(p,q):
    p = np.array(p)
    q = np.array(q)
    return np.linalg.norm(p[0:2]-q[0:2])

def normalization(data):
    #RobustScaler
    t = sk.RobustScaler().fit(data)
    data = t.transform(data)
    data = data.tolist()
    return data

def test(t_data,data, eps, core, cl):
    classif = [0]*len(t_data)
    plot_data(classif,t_data)
    for i in range(len(t_data)):
        neighbors = rangeQuery(data, t_data[i], eps)
        min_dis = None
        for j in neighbors:
            if(j in core):
                if (min_dis == None):
                    min_dis = j  
                else:
                          
                    if((dist_func(data[j],t_data[i])))<(dist_func(data[min_dis],t_data[i])):
                        min_dis = j
        if (min_dis != None):
            classif[i] = cl[min_dis]
        else: 
            classif[i] = -1
    return classif
    
def data_div(data):
    n_data = len(data)
    n_test = int(n_data * 0.1)
    test_data = []
    for i in range(n_test):
        j = random.randint(0,n_data-1)
        test_data.append(data.pop(j))
        n_data = len(data)
    return (data,test_data)

#plot 
def plot_data(cl,data):
    points_shape = ["b.","g.","c.","m.","y.","k.","bv","gv",
              "cv","mv","yv","kv","bs","gs","cs","ms","ys",
              "ks","b1","g1","c1","m1","y1","k1","b*",
              "g*","c*","m*","y*","k*","bo","go","co","mo",
              "yo","ko","b+","g+","c+","m+","y+","k+"]
    f = len(data[0])-1
    for i in range(len(data)):
        if(data[i][f] >= len(points_shape)):
            plt.plot(data[i][0], data[i][1],points_shape[cl[i] % len(points_shape)])
        else:
            plt.plot(data[i][0], data[i][1],points_shape[cl[i]])
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.show()

def main():
    data = [i.strip().split() for i in open("cluster.dat").readlines()]
    for i in range(len(data)):
        data[i] = [float(j) for j in data[i]]
    data = normalization(data)
    data, test_data = data_div(data)
    classif, core = DBSCAN(data, 0.2, 3)
    plot_data(classif,data)#plot clusters
    cl_test = test(test_data, data, 0.2, core, classif)
    plot_data(cl_test,test_data)
    
    
    
main()
