import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from kmeans import kmeans, data_div, plot_data, elbow_method, test, normalization

#Handle with non-categorical data
def dummies(data, features):

	newData = pd.DataFrame(columns=['A'])

	#Split data 'Country' in comma
	num = data['country'].str.join(sep='').str.get_dummies(sep=',')
	newData = pd.concat([newData, num], axis=1)

	#Split data 'listed_in' in comma
	num = data['listed_in'].str.join(sep='').str.get_dummies(sep=',')
	newData = pd.concat([newData, num], axis=1)

	#Get dummies features for analyses
	for i in features:
		num = pd.get_dummies(data.values[:, i], prefix_sep='_', drop_first=False)
		newData = pd.concat([newData, num], axis=1)

	return newData.drop('A', axis=1)
	
#PCA training set
def applyPCAFit(rate, data):
	pca = PCA(n_components = rate)
	data = pca.fit_transform(data)
	var = pca.explained_variance_ratio_
	return data, var

def plotVariance(var):
	var = str(var).replace('\n', '').replace('[', '').replace(']', '').split(" ")
	var = ' '.join(var).split()

	for i in range(len(var)):
		var[i] = float(var[i])

	rates = range(len(var))
	plt.plot(rates, var)
	plt.ylabel('variance')
	plt.xlabel('n_components')
	plt.show()

def main():

	#Receive data
	data = pd.read_csv("netflix_titles.csv", sep=',')

	#plot_data(data, None) 
	featuresOH = [1, 3, 7, 8, 9]
	data = dummies(data, featuresOH)

	#Split data into sets
	data = data.values.tolist()
	dataTraining, dataTest = data_div(data)

	#Turn into numpy array
	dataTraining = np.array(dataTraining)
	dataTest = np.array(dataTest)

	#Finding best variance first 15
	dataPCATraining, var = applyPCAFit(15, dataTraining)
	plotVariance(var)

	#Find the best rate
	rate = int(input("What is the best variance? "))
	dataPCATraining, var = applyPCAFit(rate, dataTraining)

	#Run K-means for 50 differents k and plot elbow graphic
	dataPCATraining = dataPCATraining.tolist()
	elbow_method(dataPCATraining, 50)	

	#Find the best k
	k = input("What is the best k? ")

	#Apply PCA in training set
	dataPCATraining, var = applyPCAFit(int(rate), dataTraining)

	#Run K means in training set
	dataPCATraining = dataPCATraining.tolist()
	centers, clusters = kmeans(dataPCATraining, int(k))

	#Run in test set
	test_data,cl_test = test(dataPCATraining, centers)
	plot_data(test_data,centers,cl_test)

main()
