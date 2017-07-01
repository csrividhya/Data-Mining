"""
Created by Srividhya Chandrasekharan on 03/25/2017
K means clustering algorithm
"""
import math
import numpy as np
import csv
import sys
from collections import namedtuple
import random

def bss(data,overall_mean,true_centroid,k,num):
	bss=0.0
	for i in range(0,k):
		bss+=num[i]*math.pow(eucledian(true_centroid[i],overall_mean),2)
	print("True between-cluster sum of squres(SSB) = "+str(bss))


def square_of_distances(data,mean,k,truth): #truth=0 : estimated centroids and truth=1 : true centroids
	sse=0.0 #sum of squared errors
	if(k==1):	
		for i in range(0, len(data)):  # for each data point
			distance=eucledian(data[i],mean)
			distance=distance*distance
			sse=sse+distance
		print("True overall SSE = "+str(sse))
	else:
		for j in range(1,k+1):
			distance=0.0
			for i in range(0, n):  # for each data point
				if data[i][3] == j:
					distance=eucledian(data[i],mean[j-1])
					distance=distance*distance
					sse=sse+distance

		print("True cluster SSE = "+str(sse))

def sse(data, k):
	num = [0] * k
	sum1= [0.0]*k
	sum2= [0.0]*k
	overall_sum1=0.0
	overall_sum2= 0.0
	true_centroid=[]
	
	for j in range(1, k + 1):  # for each cluster
		new_record=[]
		avg1=0.0
		avg2=0.0
		for i in range(0, n):  # for each data point
			overall_sum1=overall_sum1+data[i][1]
			overall_sum2=overall_sum2+data[i][2]

			if data[i][3] == j:
				num[j - 1] = num[j - 1] + 1
				sum1[j-1]=sum1[j-1]+data[i][1]
				sum2[j-1]=sum2[j-1]+data[i][2]
		
		avg1=sum1[j-1]/num[j-1]
		avg2=sum2[j-1]/num[j-1]

		new_record.append(j)
		new_record.append(avg1)
		new_record.append(avg2)
		true_centroid.append(new_record)
	print("Analysis using True Clusters:-")
	for j in range(1,k+1):
		print(true_centroid[j-1])
		print("Number of data points in cluster "+str(j)+" is: "+str(num[j-1]))

	square_of_distances(data,true_centroid,k)
	#OVERALL MEAN
	overall_avg1= overall_sum1/n
	overall_avg2= overall_sum2/n
	overall_mean=[]
	overall_mean.append(00)
	overall_mean.append(avg1)
	overall_mean.append(avg2)
	square_of_distances(data,overall_mean,1)
	bss(data,overall_mean,true_centroid,k,num)

	
    
def assign_cluster(data, median):
	cluster = []
	for i in range(0, len(data)):
		min_dist = eucledian(data[i], median[0])
		centroid_index = 0
		for j in range(0, len(median)):
			distance = eucledian(data[i], median[j])
			if distance <= min_dist:
				min_dist = distance
				centroid_index = j
		cluster.append(centroid_index)
	return cluster


def restimate_median(data, median, cluster):
    new_median = []
    n = len(median)
    print(n)
    for k in range(0, n):
        sum1 = [0.0] * n
        sum2 = [0.0] * n
        count = [0.0] * n
        avg1 = [0.0] * n
        avg2 = [0.0] * n
        new_record = []
        for i in range(0, len(data)):
            if cluster[i] == k:
                sum1[k] = sum1[k] + data[i][1]
                sum2[k] = sum2[k] + data[i][2]
                count[k] = count[k] + 1
        avg1[k] = (sum1[k] / count[k])
        avg2[k] = (sum2[k] / count[k])
        new_record.append(k)
        new_record.append(avg1[k])
        new_record.append(avg2[k])
        new_median.append(new_record)
    return new_median


def min_max_normalization(l, new_min, new_max):
    mini = min(l)
    mini = float(mini)
    maxi = max(l)
    maxi = float(maxi)
    for i in range(0, len(l)):
        temp = (float(l[i]) - mini) / (maxi - mini)
        temp = temp * (new_max - new_min)
        temp = temp + new_min
        l[i] = temp
    return l


def eucledian(x, y):
    sum = 0.0
    for i in range(0, len(x)):
        if metadata[i][1] == 'num':
            temp = float(y[i]) - float(x[i])
            temp = pow(temp, 2)
            sum = sum + temp
            temp = 0
    return math.sqrt(sum)


def mean(l):
    sum = 0.0
    count = 0
    for x in l:
        if type(x) is float:
            sum = sum + x
            count = count + 1

    if count > 0:
        return (sum / count)
    else:
        return 0


def find_column(data, index):
    column = [item[index] for item in data]
    return column


def update_column(new_c, index, data):
    for i in range(0, len(new_c)):
        data[i][index] = new_c[i]
    return data


def normalize_data(data, metadata):
    l = len(metadata)
    for i in range(0, l):
        if metadata[i][1] == 'num':
            column = find_column(data, i)
            column = min_max_normalization(column, 0.0, 1.0)
            # print(column)
            data = update_column(column, i, data)
    return data

# main()
if __name__ == "__main__":

    if(len(sys.argv) != 3):
        print("Did not pass input csv file or k value")
        sys.exit()

    filename = sys.argv[1]
    k = int(sys.argv[2])
    metadata = []

    if filename == "TwoDimEasy.csv" or filename == "TwoDimHard.csv":
        metadata = [[0, 'tid'], [1, 'num'], [
            2, 'num'], [3, 'class'], [4, 'cluster']]

    data = []
    with open(filename) as csvfile:
        csvdata = csv.reader(csvfile)
        for row in csvdata:
            data.append(row)
        data.pop(0)

    n = len(data)

    for i in range(0, len(data)):
        data[i][0] = int(data[i][0])
        data[i][1] = float(data[i][1])
        data[i][2] = float(data[i][2])
        data[i][3]=int(data[i][3])

    # STEP2 - NORMALIZING DATA
    data = normalize_data(data, metadata)
    # to compute SSE and stuf
    sse(data,k)	

    # STEP3 - PICK INITIAL CENTROIDS
    count = 0
    pool = []  # to hold the generated random numbers
    median=[] #to hold the centroids
    while(count < k):
        random_index = random.randrange(0, n)
        while(random_index in pool):
            random_index = random.randrange(0, n)
        pool.append(random_index)
        median.append(data[random_index])
        count = count + 1
    print("Initial Centroids : " + str(median))

    cluster=[]
    old_median = []
    iter = 0
    # clustering once outside the loop
    cluster = assign_cluster(data, median)
    print("OUTSIDE WHILE LOOP" +str(cluster))
    iter=iter+1

    while(median != old_median):
            cluster=assign_cluster(data, median)
            print(cluster)
            old_median=median
            print(median)
            median = restimate_median(data, median, cluster)
            print("New Centroids : " + str(old_median))
            iter = iter + 1
    print("Number of iterations are "+str(iter))
	
