"""
Created by Srividhya Chandrasekharan on 03/25/2017
K means clustering algorithm
"""
import math
import numpy as np
import csv
import sys
import collections
from collections import namedtuple
import random
import pandas as pd
import matplotlib.pyplot as plt

row = namedtuple('r', 'name values')

def confusion(data,c_data,cluster,actual,k):
    winner=[]
    bro=[]
    matrix=np.zeros((actual,actual))
    title=[]
    if actual==2:
        title=['1','2']
    else:
        title=['1','2','3','4']

    for i in range(0,k):
        temp=[]
        for j in range(0,len(c_data[i])):
            temp.append(c_data[i][j][3])
        a=collections.Counter(temp).most_common(actual)
        a=dict(a)
        aw=sorted(a.items(), key=lambda a:a[1], reverse=True)
        winner.append(aw[0][0]-1)
        a=sorted(a.items(), key=lambda a:a[0])
        bro.append(a)

    for i in range(0,len(winner)):
        for j in range(0,len(bro[i])):
            matrix[bro[i][j][0]-1][winner[i]]=bro[i][j][1]

    print("\n CONFUSION MATRIX")
    if actual==2:
        print(pd.DataFrame.from_items([('1', matrix[0]), ('2', matrix[1])],orient='index', columns=title))
    elif actual==4:
        print(pd.DataFrame.from_items([('1', matrix[0]), ('2', matrix[1]),('3', matrix[2]),('4', matrix[3])],orient='index', columns=title))
    print("\n")
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

def hard_scatter(data,cluster,k):
    # correctly classified points
    cluster1_x1 = []
    cluster1_x2 = []
    cluster2_x1 = []
    cluster2_x2 = []
    cluster3_x1 = []
    cluster3_x2 = []
    cluster4_x1 = []
    cluster4_x2 = []

    # COLOR CODING FOR EACH CLUSTER
    colorz=['g','b','r','c']

    fig, ax = plt.subplots()
    for i in range(0, len(data)):
        if data[i][3] == 1:
            if cluster[i]==0:  # correct
                cluster1_x1.append(data[i][1])
                cluster1_x2.append(data[i][2])
            else: #misclassfied the point
                ax.scatter(data[i][1], data[i][2], color=colorz[cluster[i]],s=50, marker='^', alpha=.9)

        elif data[i][3] == 2:
            if cluster[i]==1:  # correct
                cluster2_x1.append(data[i][1])
                cluster2_x2.append(data[i][2])
            else: 
                ax.scatter(data[i][1], data[i][2], color=colorz[cluster[i]],s=50, alpha=.9)

        elif data[i][3] == 3:
            if cluster[i]==2:  # correct
                cluster3_x1.append(data[i][1])
                cluster3_x2.append(data[i][2])
            else:
                ax.scatter(data[i][1], data[i][2], color=colorz[cluster[i]],marker='d',s=50, alpha=.9)

        elif data[i][3] == 4:
            if cluster[i]==3:  # correct
                cluster4_x1.append(data[i][1])
                cluster4_x2.append(data[i][2])
            else:
                ax.scatter(data[i][1], data[i][2], color=colorz[cluster[i]],marker='*',s=50, alpha=.9)

    p1=ax.scatter(cluster1_x1, cluster1_x2, color='g',s=50, marker='^', alpha=.9)
    p2=ax.scatter(cluster2_x1, cluster2_x2, color='b',s=50, alpha=.9)
    p3=ax.scatter(cluster3_x1, cluster3_x2, color='r',s=50, marker='d', alpha=.9)
    p4=ax.scatter(cluster4_x1, cluster4_x2, color='c',s=50, marker='*', alpha=.9)
    if k==4:
        plt.legend([p1, p2, p3, p4], ['Cluster1', 'Cluster2','cluster3','Cluster4'],loc='best')
    elif k==3:
        plt.legend([p1, p2, p3], ['Cluster1', 'Cluster2','cluster3'],loc='best')
    elif k==2:
        plt.legend([p1, p2], ['Cluster1', 'Cluster2'],loc='best')
    
    plt.title('Points misclassified have same shape as original cluster and the color of the esimated cluster')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def bss(data, overall_mean, centroid, k, num):
    bss = 0.0
    for i in range(0, k):
        bss += num[i] * math.pow(eucledian(centroid[i], overall_mean), 2)
    print("Between-cluster sum of squares(SSB) = " + str(bss))
    return bss


# truth=0 : estimated centroids and truth=1 : true centroi
def square_of_distances(data, mean, k, flag, cluster, true_centroid):
    sse = 0.0  # sum of squared errors
    if(k == 1 and flag == True):
        for i in range(0, k):  # for each data point
            distance = eucledian(true_centroid[i], mean)
            distance = distance * distance
            sse = sse + distance
        print("True overall SSE = " + str(sse))
    elif k > 1 and flag == True:
        for j in range(1, k + 1):
            distance = 0.0
            for i in range(0, n):  # for each data point
                if data[i][3] == j:
                    distance = eucledian(data[i], mean[j - 1])
                    distance = distance * distance
                    sse = sse + distance
            print("SSE of cluster " + str(j) + " = " + str(sse))
    elif k > 1 and flag == False:
        for j in range(0, k):
            distance = 0.0
            for i in range(0, n):  # for each data point
                if cluster[i] == j:
                    distance = eucledian(data[i], mean[j])
                    distance = distance * distance
                    sse = sse + distance
            print("SSE of cluster " + str(j) + " = " + str(sse))


def find_true_centroids(data, k):
    num = [0] * k
    sum1 = [0.0] * k
    sum2 = [0.0] * k
    true_centroid = []
    output = []

    for j in range(1, k + 1):  # for each cluster
        new_record = []
        avg1 = 0.0
        avg2 = 0.0
        for i in range(0, len(data)):  # for each data point
            if data[i][3] == j:
                num[j - 1] = num[j - 1] + 1
                sum1[j - 1] = sum1[j - 1] + data[i][1]
                sum2[j - 1] = sum2[j - 1] + data[i][2]

        avg1 = sum1[j - 1] / num[j - 1]
        avg2 = sum2[j - 1] / num[j - 1]

        new_record.append(j)
        new_record.append(avg1)
        new_record.append(avg2)
        true_centroid.append(new_record)
        output.append(true_centroid)
        output.append(num)
    print("True centroids are: " + str(true_centroid))
    for j in range(1, k + 1):
        print("Number of data points in cluster " +
              str(j) + " is: " + str(num[j - 1]))
    return output


def find_overall_mean(data):
    overall_sum1 = 0.0  # dimension X1
    overall_sum2 = 0.0  # dimension X2
    n = len(data)
    for i in range(0, n):  # for each data point
        overall_sum1 = overall_sum1 + data[i][1]
        overall_sum2 = overall_sum2 + data[i][2]

    overall_avg1 = overall_sum1 / n
    overall_avg2 = overall_sum2 / n
    overall_mean = []
    overall_mean.append(00)
    overall_mean.append(overall_avg1)
    overall_mean.append(overall_avg2)

    return overall_mean


def true_sse(data, k):
    print("YOLO")
    output = find_true_centroids(data, k)
    true_centroid = output[0]
    num = output[1]
    square_of_distances(data, true_centroid, k, True, " ", true_centroid)

    overall_mean = find_overall_mean(data)
    square_of_distances(data, overall_mean, 1, True, " ", true_centroid)

    bss(data, overall_mean, true_centroid, k, num)


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
#***************************


def overall_SSE(data, centroid, k, cluster):  # for estimated cluster
    sse = 0.0
    for i in range(0, k):
        for j in range(0, len(data)):
            if cluster[j] == i:
                sse = sse + (eucledian(data[j], centroid[i])**2)
    print("Overall cluster SSE = " + str(sse))
    return sse


def within_cluster_SSE(cluster_data, mean, i):  # for estimated cluster
    sse = 0.0
    for i in range(0, len(cluster_data)):
        sse = sse + (eucledian(cluster_data[i], mean)**2)
    return sse


def invoke_within_cluster_sse(data, k, cluster, centroid):  # for estimated cluster
    for i in range(0, k):
        cluster_data = []
        sse = 0.0
        for j in range(0, len(data)):
            if cluster[j] == i:
                cluster_data.append(data[j])
        mean = centroid[i]
        sse = within_cluster_SSE(cluster_data, mean, i)
        print("Within cluster SSE of cluster " + str(i) + " = " + str(sse))


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


def compute_avg_distance(x, data):
    total_dist = 0.0
    avg = 0.0
    for i in range(0, len(data)):
        total_dist = total_dist + eucledian(x, data[i])
    avg = (total_dist / len(data))
    return avg


def find_silhouette_within_cluster(w, cluster, k):
    for i in range(0, k):
        total = []
        for j in range(0, len(cluster)):
            if cluster[j] == i:
                total.append(w[j])
        print("Average Silhouette for cluster " +
              str(i) + " = " + str(np.mean(total)))


def silhoute_width(data, c_data, cluster, k):
    #w = b(i) - a(i) / max(b - a)
    w = []
    count = 0
    for i in range(0, len(data)):  # DATA POINT IS data[i]
        b_temp = []
        cluster_num = cluster[i]
        own_cluster_data = c_data[cluster_num]
        a_i = compute_avg_distance(data[i], own_cluster_data)

        for j in range(0, len(c_data)):
            if j != cluster_num:
                b_temp.append(compute_avg_distance(data[i], c_data[j]))
        b_i = min(b_temp)
        width = (b_i - a_i / max(b_i, a_i))
        w.append(width)
        #print("Silhoute width of point " + str(i) + " is: " + str(width))
    find_silhouette_within_cluster(w, cluster, k)
    print("Average Silhouette for entire dataset = " + str(np.mean(w)))


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
#***************************

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
        data[i][3] = int(data[i][3])

    # STEP2 - NORMALIZING DATA
    data = normalize_data(data, metadata)

    # to compute SSE and stuf
    #true_sse(data, k)

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
    
    actual=0
    if filename=="TwoDimEasy.csv":
        actual=2
    elif filename=="TwoDimHard.csv":
        actual=4

    #SCATTERPLOT
    hard_scatter(data,cluster,k)
    c_data=find_cluster_data(data,cluster,k) 
    confusion(data,c_data,cluster,actual,k)
    
    # To print values regarding estimated clusters
    #print(median)
    invoke_within_cluster_sse(data, k, cluster, median)
    overall_SSE(data, median, k, cluster)
    overall_mean = find_overall_mean(data)
    estimated_SSB(data, overall_mean, median, k)
    c_data=find_cluster_data(data,cluster,k) #get indices of data points belonging to a cluster
    #print(c_data)

    silhoute_width(data,c_data,cluster,k)
    
