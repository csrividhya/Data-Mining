import math
import numpy as np
import csv
import sys
import collections
from collections import namedtuple
import random
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats


def cluster_entropy(data,k):
    total_entropy=0.0
    for i in range(0,k):
        c_data=[]
        for j in range(0,len(data[i])):
            c_data.append(data[i][j][12])

        p=0
        total=len(c_data)
        a=collections.Counter(c_data)
        a=dict(a)
        for xoxo in a.keys():
            yolo=(a[xoxo]/total)
            p=p+(yolo*(math.log2(yolo)))
       
        total_entropy= total_entropy + (-1)*(total/1599)*p
        #print("Entropy of cluster "+str(i)+" = "+str(entropyz))
    print("Total Entropy = "+str(total_entropy))



def find_cluster_data(data, cluster, k):
    c = []  # array of each cluster's data points' indices
    for i in range(0, k):
        cluster_data = []
        sse = 0.0
        for j in range(0, len(data)):
            if cluster[j] == i:
                cluster_data.append(data[j])
        c.append(cluster_data)
    return c

def find_overall_mean(data):
    overall_sum1 = 0.0 
    overall_sum2 = 0.0
    overall_sum3 = 0.0 
    overall_sum4 = 0.0
    overall_sum5 = 0.0 
    overall_sum6 = 0.0
    overall_sum7 = 0.0 
    overall_sum8 = 0.0
    overall_sum9 = 0.0 
    overall_sum10 = 0.0
    overall_sum11 = 0.0 
    overall_sum12 = 0.0 

    n = len(data)
    for i in range(0, n):  # for each data point
        overall_sum1 = overall_sum1 + data[i][1]
        overall_sum2 = overall_sum2 + data[i][2]
        overall_sum3 = overall_sum3 + data[i][3]
        overall_sum4 = overall_sum4 + data[i][4]
        overall_sum5 = overall_sum5 + data[i][5]
        overall_sum6 = overall_sum6 + data[i][6]
        overall_sum7 = overall_sum7 + data[i][7]
        overall_sum8 = overall_sum8 + data[i][8]
        overall_sum9 = overall_sum9 + data[i][9]
        overall_sum10 = overall_sum10 + data[i][10]
        overall_sum11 = overall_sum11 + data[i][11]
        overall_sum12 = overall_sum12 + data[i][12]

    overall_avg1 = overall_sum1 / n
    overall_avg2 = overall_sum2 / n
    overall_avg3 = overall_sum3 / n
    overall_avg4 = overall_sum4 / n
    overall_avg5 = overall_sum5 / n
    overall_avg6 = overall_sum6 / n
    overall_avg7 = overall_sum7 / n
    overall_avg8 = overall_sum8 / n
    overall_avg9 = overall_sum9 / n
    overall_avg10 = overall_sum10 / n
    overall_avg11 = overall_sum11 / n
    overall_avg12 = overall_sum12 / n
    overall_mean = []
    overall_mean.append(00)
    overall_mean.append(overall_avg1)
    overall_mean.append(overall_avg2)
    overall_mean.append(overall_avg3)
    overall_mean.append(overall_avg4)
    overall_mean.append(overall_avg5)
    overall_mean.append(overall_avg6)
    overall_mean.append(overall_avg7)
    overall_mean.append(overall_avg8)
    overall_mean.append(overall_avg9)
    overall_mean.append(overall_avg10)
    overall_mean.append(overall_avg11)
    overall_mean.append(overall_avg12)

    return overall_mean

def overall_SSE(data, centroid, k, cluster):  # for estimated cluster
    sse = 0.0
    for i in range(0, k):
        for j in range(0, len(data)):
            if cluster[j] == i:
                sse = sse + (eucledian(data[j], centroid[i])**2)
    print("Overall cluster SSE = " + str(sse))
    return sse

def estimated_SSB(data, overall_mean, centroid, k):  # for estimated cluster
    bss = 0.0
    num = [0.0] * k
    for j in range(0, len(data)):
        if cluster[j] == j:
            num[j] = num[j] + 1

    for i in range(0, k):
        bss += num[i] * math.pow(eucledian(centroid[i], overall_mean), 2)
    print("Between-cluster sum of squres(SSB) = " + str(bss))
    return bss

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

    for k in range(0, n ):
        sum1 = [0.0] * n
        sum2 = [0.0] * n
        sum3 = [0.0] * n
        sum4 = [0.0] * n
        sum5 = [0.0] * n
        sum6 = [0.0] * n
        sum7 = [0.0] * n
        sum8 = [0.0] * n
        sum9 = [0.0] * n
        sum10 = [0.0] * n
        sum11 = [0.0] * n
        sum12 = [0.0] * n
        
        count = [0.0] * n
        avg1 = [0.0] * n
        avg2 = [0.0] * n
        avg3 = [0.0] * n
        avg4 = [0.0] * n
        avg5 = [0.0] * n
        avg6 = [0.0] * n
        avg7 = [0.0] * n
        avg8 = [0.0] * n
        avg9 = [0.0] * n
        avg10 = [0.0] * n
        avg11 = [0.0] * n
        avg12 = [0.0] * n

        new_record = []
        for i in range(0, len(data)):
            if cluster[i] == k:
                sum1[k] = sum1[k] + data[i][1]
                sum2[k] = sum2[k] + data[i][2]
                sum3[k] = sum3[k] + data[i][3]
                sum4[k] = sum4[k] + data[i][4]
                sum5[k] = sum5[k] + data[i][5]
                sum6[k] = sum6[k] + data[i][6]
                sum7[k] = sum7[k] + data[i][7]
                sum8[k] = sum8[k] + data[i][8]
                sum9[k] = sum9[k] + data[i][9]
                sum10[k] = sum10[k] + data[i][10]
                sum11[k] = sum11[k] + data[i][11]
                sum12[k] = sum12[k] + data[i][12]                
                count[k] = count[k] + 1

        avg1[k] = (sum1[k] / count[k])
        avg2[k] = (sum2[k] / count[k])
        avg3[k] = (sum3[k] / count[k])
        avg4[k] = (sum4[k] / count[k])
        avg5[k] = (sum5[k] / count[k])
        avg6[k] = (sum6[k] / count[k])
        avg7[k] = (sum7[k] / count[k])
        avg8[k] = (sum8[k] / count[k])
        avg9[k] = (sum9[k] / count[k])
        avg10[k] = (sum10[k] / count[k])
        avg11[k] = (sum11[k] / count[k])
        avg12[k] = (sum12[k] / count[k])

        new_record.append(k)
        new_record.append(avg1[k])
        new_record.append(avg2[k])
        new_record.append(avg3[k])
        new_record.append(avg4[k])
        new_record.append(avg5[k])
        new_record.append(avg6[k])
        new_record.append(avg7[k])
        new_record.append(avg8[k])
        new_record.append(avg9[k])
        new_record.append(avg10[k])
        new_record.append(avg11[k])
        new_record.append(avg12[k])
        
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

    metadata=[[0, 'tid'], [1, 'num'], [2, 'num'] ,[3, 'num'], [4,'num'],[5, 'num'],[6, 'num'],[7, 'num'],[8, 'num'],[9, 'num'],[10, 'num'],[11, 'num'],[12, 'num'],[13, 'claas']]

    #0th index - row ID
    # 13th index - wine cluster label ('Low' , 'High')

    data = []
    with open(filename) as csvfile:
        csvdata = csv.reader(csvfile)
        for row in csvdata:
            data.append(row)
        data.pop(0)

    n = len(data)

    for i in range(0, len(data)):
       for j in range(1,13):
        data[i][j]=float(data[i][j])

    # STEP2 - NORMALIZING DATA
    data = normalize_data(data, metadata)

    # STEP3 - PICK INITIAL CENTROIDS
    count = 0
    pool = []  # to hold the generated random numbers
    median = []  # to hold the centroids
    while(count < k):
        random_index = random.randrange(0, n)
        while(random_index in pool):
            random_index = random.randrange(0, n)
        pool.append(random_index)
        median.append(data[random_index])
        count = count + 1
    print("Initial Centroids : ")
    print(pd.DataFrame(median))

    cluster = []
    old_median = []
    iter = 0
    # clustering once outside the loop
    cluster = assign_cluster(data, median)
    #print("OUTSIDE WHILE LOOP" + str(cluster))
    iter = iter + 1
    
    while(median != old_median):
        cluster = assign_cluster(data, median)
        # print(cluster)
        old_median = median
        #print(median)
        median = restimate_median(data, median, cluster)
        #print("New Centroids : " + str(old_median))
        iter = iter + 1
    
    print("Final Centroids : ")
    print(pd.DataFrame(median))


    overall_SSE(data, median, k, cluster)
    overall_mean = find_overall_mean(data)
    estimated_SSB(data, overall_mean, median, k)
    c_data=find_cluster_data(data,cluster,k)

    cluster_entropy(c_data,k)
