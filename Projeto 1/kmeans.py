import math
import matplotlib.pyplot as plt
import numpy as np
import random
import sklearn.preprocessing as sk

#Kmeans 
def kmeans(data, k):
    cl = [0]*len(data) #vetor com clusters de cada dado
    features = len(data[0])
    centers = find_center(data,k,features)#centros iniciais
    while True: 
        cl = clusters(data, centers ,cl) #monta os clusters
        old_centers = centers.copy() 
        new_centers(features, data, centers , cl)#calcula os novos centros
        if old_centers == centers:
            break
    return (centers,cl)
   
#Retorna posições aleatórias para os centros de cluster 
##range (menor valor entre os dados, menor valor entre os maiores valores de dado de cada feature)
def find_center(data,k,f):
    d = np.array(data)
    min_data = d.min(axis = 0)
    max_data = d.max(axis = 0)
    centers = []
    for i in range(k):
        centers.append([])
        for j in range(f):
            centers[i].append(random.randrange(int(min_data.min()), int(max_data.min())))
    return centers
    
#Percorre todos os dados escolhendo o cluster qual ele faz parte 
#de acordo com a menor distância entre ele e o centro dos clusters
def clusters(data, centers, cl):
    dist = [0]*len(centers)
    for i in range(len(data)):
        for j in range (len(centers)):
            dist[j] = math.dist(data[i],centers[j])
        cl[i] = dist.index(min(dist))
    return cl
        
    
#Percorre todos os dados de um cluster x e calcula o ponto médio entre esses dados
#tornando esse ponto o novo centro do cluster x
def new_centers(features, data, centers, cl):
    mid_point = np.zeros((len(centers),features))
    num = [0]*len(centers)
    for i in range(len(data)):
        a= cl[i]
        for j in range(features):
            mid_point[a][j] = mid_point[a][j] + data[i][j]
        num[a] += 1
 
    for i in range(len(centers)):
        if(num[i] != 0):
            centers[i]= [j/num[i] for j in mid_point[i]]

#Divide o conjunto de dados em treinamento e teste
#data -> conjunto de treinamento 90% dos dados
#test_data -> conjunto de teste 10% dos dados
def data_div(data):
    n_data = len(data)
    n_test = int(n_data * 0.1)
    test_data = []
    for i in range(n_test):
        j = random.randint(0,n_data-1)
        test_data.append(data.pop(j))
        n_data = len(data)
    return (data,test_data)

#Test
## Recebe o conjunto de dado teste e classifica eles de acordo com 
## clusters encontrados durante o treinamento na função kmeans
def test (data, centers):
    cl = [0]*len(data)
    clusters(data, centers, cl)
    return data, cl


#plota os dados de acordo com seus clusters
def plot_data(data,centers, cl):
    points_shape = ["b.","g.","c.","m.","y.","k.","bv","gv",
              "cv","mv","yv","kv","bs","gs","cs","ms","ys",
              "ks","b1","g1","c1","m1","y1","k1","b*",
              "g*","c*","m*","y*","k*","bo","go","co","mo",
              "yo","ko","b+","g+","c+","m+","y+","k+"]
    if (centers == None):
        plt.plot(list(zip(*data))[0], list(zip(*data))[1],points_shape[0])
    else:
        for i in range(len(data)):
            if(cl[i] >= len(points_shape)):
                plt.plot(data[i][0], data[i][1],points_shape[cl[i] % len(points_shape)])
            else:
                plt.plot(data[i][0], data[i][1],points_shape[cl[i]])
        plt.plot(list(zip(*centers))[0], list(zip(*centers))[1],"ro")
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.show()
    
def elbow_method(data,n):
    features = len(data[0])
    j_function = [0]*n
    for x in range(1,n+1):
        c,cl = kmeans(data, x)
        for j in range(x):
            for i in range(len(data)):
                if(cl[i] == j):
                    j_function[x-1] +=  (math.dist(data[i],c[j]))**2
    x = [i for i in range(1,n+1)]
    plt.plot(x, j_function)
    plt.ylabel('Cost of Function J')
    plt.xlabel('Number of clusters')
    plt.show()

def robustscaler(data):
    #RobustScaler
    t = sk.RobustScaler().fit(data)
    data = t.transform(data)
    data = data.tolist()
    return data

def parte1():
    data = [i.strip().split() for i in open("cluster.dat").readlines()]
    for i in range(len(data)):
        data[i] = [float(j) for j in data[i]]
    cl = [0]*len(data)
    data = robustscaler(data)
    plot_data(data,None,cl) #plota os dados padronizados
    data,test_data = data_div(data)#divide base de treinamento da base de teste
    elbow_method(data,9)#Plota método do cotovelo
    centers,cl = kmeans(data, 3)
    plot_data(data,centers,cl)#plota os clusters e os centros
    test_data,cl_test = test(test_data, centers)
    plot_data(test_data,centers,cl_test)#plota o teste
