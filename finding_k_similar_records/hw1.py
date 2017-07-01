"""
Mining Iris and Income datasets
Created by : Srividhya Chandrasekharan
Submitted on Feb 8th 2017
"""
import math
import numpy as np
import csv
import sys
from collections import namedtuple

similarity_record = namedtuple('x', 'similarity record')


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
    sum = 0
    for i in range(0, len(x)):
        temp = float(y[i]) - float(x[i])
        temp = pow(temp, 2)
        sum = sum + temp
        temp = 0
    return math.sqrt(sum)


def cosine_similarity(x, y):
    xlen = 0
    ylen = 0
    xtemp = 0
    ytemp = 0
    product = 0

    for i in range(0, len(x)):
        xtemp = xtemp + pow(x[i], 2)
        ytemp = ytemp + pow(y[i], 2)
        product = product + (x[i] * y[i])
    xlen = math.sqrt(xtemp)
    ylen = math.sqrt(ytemp)

    return product / (xlen * ylen)


def mean(l):
    sum = 0
    count = 0
    for x in l:
        if type(x) is int:
            sum = sum + x
            count = count + 1

    if count > 0:
        return (sum / count)
    else:
        return 0


def mode(l, category):
    max = 0
    if category == 'cat':
        item = 'yolo'
    elif category == 'num_cat':
        item = 0
    c = 0
    for x in l:
        c = l.count(x)
        if c > max:
            max = c
            item = x
    return item


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


def handle_missing_data(data, metadata):
    symbol = ' ?'
    count = 0

    for i, row in enumerate(data):
        if symbol in row:
            count = count + 1
            j = row.index(symbol)  # index
            c = metadata[j][1]  # category
            # finding column of jth attribute values
            column = find_column(data, j)

            if c == 'cat' or c == 'num_cat':
                missing = mode(column, c)
                row.remove(symbol)
                row.insert(j, missing)
                data.remove(row)
                data.insert(i - 1, row)

            elif c == 'num':
                missing = mean(column)
                row.remove(symbol)
                row.insert(j, missing)
                data.remove(row)
                data.insert(i - 1, row)
    #print(count," are missing values")
    if count:
        data = handle_missing_data(data, metadata)
    return data


def find_numeric_attributes(record, metadata):
    length = len(metadata)
    numbers = []

    for i in range(0, length):
        if metadata[i][1] == 'num':
            numbers.append(float(record[i]))
    return numbers


def find_categoric_attributes(record, metadata):
    length = len(metadata)
    cat = []

    for i in range(0, length):
        if metadata[i][1] == 'cat' or metadata[i][1] == 'num_cat':
            cat.append(record[i])
    return cat


def ret_sim(a):
    return a[0]


def decide_numeric_weight(metadata):
    count = 0

    if len(metadata) == 6:  # Iris.csv
        total = 4
    else:
        total = 14  # Income.csv

    for i in range(0, len(metadata)):
        if metadata[i][1] == 'num':
            count = count + 1
    return count / total


def compute_similarity(data, metadata, k, choice):
    length = len(metadata)

    f = open('output.csv','w+')
    writer = csv.writer(f)
    tex = ["Trans. ID"]
    for index in range(1,k+1):
    	tex.append(str(index)+" ID")
    	tex.append(str(index)+" Prox")
    writer.writerow(tex)

    # fixing weights
    if len(metadata) == 6:  # Iris.csv
        total = 4
        cat_w = 0
    else:
        total = 14  # Income.csv
        cat_w = (1.0 / 14)

    num_w = decide_numeric_weight(metadata)

    for i in range(0, len(data)):
        s = []
        # extracting numeric and categoric values
        n1 = find_numeric_attributes(data[i], metadata)
        c1 = find_categoric_attributes(data[i], metadata)

        for j in range(0, len(data)):
            if i != j:
                n2 = find_numeric_attributes(data[j], metadata)
                c2 = find_categoric_attributes(data[j], metadata)
                flag='cos'
                num_score=0
                if choice == 2:
                    num_score = cosine_similarity(n1, n2)

                elif choice == 1:
                    num_dist = eucledian(n1, n2)
                    num_score = 1.0 / (1 + num_dist)
                    
                # comparing categories
                cat_sim = 0
                for index in range(0, len(c1)):
                    if c1[index] == c2[index]:
                        cat_sim = cat_sim + (cat_w * 1.0)
                    else:
                        cat_sim = cat_sim + (cat_w * 0.0)

                sim = (num_w * num_score) + cat_sim
                new_record = similarity_record(sim, data[j][0])
                s.append(new_record)
        print(str(i+1)+"th record is close to:")
        s.sort(key=ret_sim, reverse=True)
        text = []
        text.append(data[i][0])
        for o in range(0, k):
            print(s[o])
            text.append(s[o][1])
            text.append(s[o][0])
        writer.writerow(text)
    f.close()


# main()
if __name__ == "__main__":

    ch = 2  # 1 - eucledean and 2 - cosine
    k = 5  # default value

    Iris = [[0, 'tid'], [1, 'num'], [2, 'num'],[3, 'num'], [4, 'num'], [5, 'class']]
    Income = [[0, 'tid'], [1, 'ID'], [2, 'num'], [3, 'cat'], [4, 'num'], [5, 'cat'], [6, 'num_cat'], [7, 'cat'], [
        8, 'cat'], [9, 'cat'], [10, 'cat'], [11, 'cat'], [12, 'num'], [13, 'num'], [14, 'num'], [15, 'cat'], [16, 'class']]

    if(len(sys.argv) != 2):
        print("Did not pass input csv file")
        sys.exit()

    filename = sys.argv[1]
    if filename == 'Iris.csv':
        metadata = Iris
    elif filename == 'Income.csv':
        metadata = Income
    else:
        print("Invalid filename")

    data = []
    with open(filename) as csvfile:
        csvdata = csv.reader(csvfile, delimiter=',')

        for i, row in enumerate(csvdata):
            data.append([i] + row)

        data.pop(0)  # removes headings

        # STEP1 - MISSING DATA HANDLING
        data = handle_missing_data(data, metadata)

        # STEP2 - NORMALIZING DATA
        data = normalize_data(data, metadata)
        # print(data)

        # STEP3 - COMPUTING SIMILARITY
        compute_similarity(data, metadata, k, ch)
