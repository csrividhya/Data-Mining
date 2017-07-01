"""
Created by Srividhya Chandrasekharan on 03/14/2017
K means clustering algorithm
"""

import math
import numpy as np
import csv
import sys
from collections import namedtuple

# GLOBAL VARIABLES
count = 0  # Number of centroids
mlist = []
centroid = namedtuple('m', 'x1 x2')
element = namedtuple('elt', 'id x1 x2 cluster')


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
        if metadata[i][1]=='num':
            temp = float(y[i-1]) - float(x[i])
            temp = pow(temp,2)
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


def find_centroid(data):
    one = find_column(data, 1)
    two = find_column(data, 2)
    m1 = mean(one)
    m2 = mean(two)
    m = centroid(m1, m2)
    return m


def find_closest_point(data, m): #used to compute the datapoint closest to the centroid
    min_dist = 100
    new_centroid = {'x1': 0.0, 'x2': 0.0}

    for i in range(0, len(data)):
        row_id=data[i][0]
        a = float(data[i][1])
        b = float(data[i][2])
        x = element(row_id,a, b, data[i][4])
        distance = eucledian(x, m)
        if distance <= min_dist and x not in mlist:
            min_dist = distance
            new_centroid[0] = a
            new_centroid[1] = b
            target_cluster = data[i][3]
    moo = centroid(float(new_centroid[0]), float(new_centroid[1]), target_cluster)
    return moo


def find_farthest_point(data, m): #used to compute the point that's farthest to centroid
    max_dist = 0
    new_centroid = {'x1': 0.0, 'x2': 0.0}

    for mean in mlist:
        for i in range(0, len(data)):
            row_id=data[i][0]
            a = float(data[i][1])
            b = float(data[i][2])
            x = element(row_id,a, b, data[i][3])
            distance = eucledian(x, mean)
            if distance >= max_dist and x not in mlist:
                max_dist = distance
                new_centroid[0] = a
                new_centroid[1] = b

    m = centroid(float(new_centroid[0]), float(new_centroid[1]))
    return m

def find_cluster(record):
    answer=1
    min_distance=eucledian(mlist[0],record)

    i=1
    while i<len(mlist):
        temp=eucledian(mlist[i],record)
        if temp<min_distance:
            min_distance=temp
            answer= i+1
        i=i+1
    return answer

# main()
if __name__ == "__main__":

    if(len(sys.argv) != 3):
        print("Did not pass input csv file or k value")
        sys.exit()

    filename = sys.argv[1]
    k = int(sys.argv[2])

    metadata = [[0, 'tid'], [1, 'num'], [2, 'num'], [3, 'class'], [4, 'cluster']]

    data = []
    # STEP1 - EXTRACTING DATA
    with open(filename) as csvfile:
        csvdata = csv.reader(csvfile, delimiter=',')
        for row in csvdata:
            row.append("00")
            data.append(row)
        data.pop(0)  # removes headings

        for i in range(0, len(data)):
            data[i][0] = int(data[i][0])
            data[i][1] = float(data[i][1])
            data[i][2] = float(data[i][2])
    

    # STEP2 - NORMALIZING DATA
    data = normalize_data(data, metadata)
     
    # STEP3 - PICK INITIAL CENTROIDS
    start = find_centroid(data)
    count = count + 2
    mlist.append(start)


    while(count<=k):
        m = find_farthest_point(data, mlist)
        print(m)
        mlist.append(m)
        count = count + 1

    print(mlist) 
    # STEP4 - iteratively cluster datapoints

    total_records=len(data)
    x=0
 
    
    while(x<total_records):
        cluster_assigned=find_cluster(data[x])
        print("Cluster assigned for data["+str(x)+" ] is"+str(cluster_assigned))
        x=x+1
    

    # STEP5 - reassign centroids
