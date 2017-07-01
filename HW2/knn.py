"""
K Nearest Neighbours Iris and Income datasets
Created by : Srividhya Chandrasekharan
Submitted on Feb 22, 2017
"""
import math
import numpy as np
import pandas as pd
import csv
import sys
from collections import namedtuple
from scipy.stats import mode


similarity_record = namedtuple('x', 'similarity record')
rec = namedtuple('neighbour', 'id proximity class_label')


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


def test_min_max_normalization(l, new_min, new_max, mini, maxi):
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

meantuple = namedtuple('m', 'attribute_index mean')
modetuple = namedtuple('mode', 'attribute_index mode')
mintuple = namedtuple('min', 'attribute_index min')
maxtuple = namedtuple('max', 'attribute_index max')


def convert_to_float(c):
    x = []
    for i in range(0, len(c)):
        x.append(float(c[i]))
    return x


def calculate_min(d, m):
    x = []
    for i in range(0, len(m)):
        if m[i][1] == 'num':
            c = find_column(d, i)
            c = convert_to_float(c)
            min_of_c = min(c)
            temp = mintuple(i, min_of_c)
            x.append(temp)
    return x


def calculate_max(d, m):
    x = []
    for i in range(0, len(m)):
        if m[i][1] == 'num':
            c = find_column(d, i)
            c = convert_to_float(c)
            max_of_c = max(c)
            temp = mintuple(i, max_of_c)
            x.append(temp)
    return x


def calculate_means(d, m):
    means = []
    for i in range(0, len(m)):
        if m[i][1] == 'num':
            c = find_column(d, i)
            c = convert_to_float(c)
            mean_of_c = mean(c)
            temp = meantuple(i, mean_of_c)
            means.append(temp)
    return means


def calculate_modes(d, m):
    modezz = []
    for i in range(0, len(m)):
        if m[i][1] == 'cat' or m[i][1] == 'num_cat':
            c = find_column(d, i)
            mode_of_c = mode(c, m[i][1])
            temp = modetuple(i, mode_of_c)
            modezz.append(temp)
    return modezz


def test_missing_data(data, metadata, train_mean, train_mode):
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
                missing = ' '
                for ko in range(0, len(train_mode)):
                    if train_mode[ko][0] == j:
                        missing = train_mode[ko][1]
                row.remove(symbol)
                row.insert(j, missing)
                data.remove(row)
                data.insert(i - 1, row)

            elif c == 'num':
                missing = 0.0
                for ko in range(0, len(train_mean)):
                    if train_mode[ko][0] == j:
                        missing = train_mean[ko][1]
                missing = mean(column)
                row.remove(symbol)
                row.insert(j, missing)
                data.remove(row)
                data.insert(i - 1, row)
    #print(count," are missing values")
    if count:
        data = test_missing_data(data, metadata, train_mean, train_mode)
    return data


def test_normalize_data(test, metadata, train_min, train_max):
    l = len(metadata)
    for i in range(0, l):
        if metadata[i][1] == 'num':
            column = find_column(test, i)
            for ko in range(0, len(train_max)):
                if train_max[ko][0] == i:
                    training_data_max = train_max[ko][1]
            for ko in range(0, len(train_min)):
                if train_min[ko][0] == i:
                    training_data_min = train_min[ko][1]

            column = test_min_max_normalization(column, 0.0, 1.0, training_data_min, training_data_max)
            # print(column)
            test = update_column(column, i, test)
    return test


def mean(l):
    s = 0.0
    count = 0.0
    for x in l:
        if type(x) is float or type(x) is int:
            s = s + x
            count = count + 1

    if count > 0:
        return (s / count)
    else:
        return 0


def mode(l, category):
    maxii = 0
    if category == 'cat':
        item = 'yolo'
    elif category == 'num_cat':
        item = 0
    c = 0
    for x in l:
        c = l.count(x)
        if c > maxii:
            maxii = c
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
    return a[1]


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


def compute_similarity(test, data, metadata, k, choice):
    length = len(metadata)

    f = open('output.csv', 'w+')
    writer = csv.writer(f)
    tex = ["Test_record_ID", "Actual Class label"]

    for index in range(1, k + 1):
        tex.append(str(index) + "Training_record_ID")
        tex.append(str(index) + "Similarity")
        tex.append(str(index) + "claaz")
    writer.writerow(tex)
    class_label_index = 0
    # fixing weights
    if len(metadata) == 6:  # Iris.csv
        total = 4
        cat_w = 0
        class_label_index = 5
    else:
        total = 14  # Income.csv
        cat_w = (1.0 / 14)
        class_label_index = 16

    num_w = decide_numeric_weight(metadata)

    for i in range(0, len(test)):
        s = []
        # extracting numeric and categoric values
        n1 = find_numeric_attributes(test[i], metadata)
        c1 = find_categoric_attributes(test[i], metadata)

        for j in range(0, len(data)):
            n2 = find_numeric_attributes(data[j], metadata)
            c2 = find_categoric_attributes(data[j], metadata)
            num_score = 0
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
            new_record = rec(data[j][0], sim, data[j][class_label_index])
            s.append(new_record)
   # print(str(i+1)+"th record is close to:")
        s.sort(key=ret_sim, reverse=True)
        text = []
        text.append(test[i][0])
        text.append(test[i][class_label_index])
        for o in range(0, k):
            # print(s[o])
            text.append(s[o][0])
            text.append(s[o][1])
            text.append(s[o][2])
        writer.writerow(text)
    f.close()


def most_probable_class(a, k):
    neighbour_labelz = []
    for i in range(1, k+1):
        neighbour_labelz.append(a[1 + (3 * i)])

    MPC = mode(neighbour_labelz, 'cat')
    return MPC


def find_labels(a, k):
    neighbour_labelz = []
    for i in range(1, k + 1):
        neighbour_labelz.append(a[1 + (3 * i)])
    return neighbour_labelz


def posterior_probability(a, prediction, k):
    count = 0.0
    for i in range(0, len(a)):
        if a[i] == prediction:
            count = count + 1
    return(count / k)


def knn(k):
    data = []
    with open('output.csv') as csvfile:
        csvdata = csv.reader(csvfile, delimiter=',')

        for i, row in enumerate(csvdata):
            data.append(row)
    data.pop(0)  # remove headings bro

    f = open('knn.csv', 'w+')
    writer = csv.writer(f)
    tex = ["Trans. ID", "Actual Class label",
           " Predicted class", "Posterior Probabilty"]
    writer.writerow(tex)
    prediction = " "  # Class label predicted by Knn algorithm
    for i in range(0, len(data)):
        tex = []
   # print(data[i])
        tid = i + 1
        prediction = most_probable_class(data[i], k)
        actual_class = data[i][1]
        n = find_labels(data[i], k)
        prob = posterior_probability(n, prediction, k)
        tex.append(tid)
        tex.append(actual_class)
        tex.append(prediction)
        tex.append(prob)
        writer.writerow(tex)

    f.close()


def iris_confusion_matrix():
    l1 = 'Iris-setosa'
    l2 = 'Iris-versicolor'
    l3 = 'Iris-virginica'

    c11 = 0.0
    c12 = 0.0
    c13 = 0.0
    c21 = 0.0
    c22 = 0.0
    c23 = 0.0
    c31 = 0.0
    c32 = 0.0
    c33 = 0.0

    with open('knn.csv') as csvfile:
        csvdata = csv.reader(csvfile, delimiter=',')

        for i, row in enumerate(csvdata):
            data.append(row)
        data.pop(0)  # removes headings

        for i in range(0, len(data)):
            if data[i][1] == l1 and data[i][2] == l1:
                c11 = c11 + 1
            elif data[i][1] == l1 and data[i][2] == l2:
                c12 = c12 + 1
            elif data[i][1] == l1 and data[i][2] == l3:
                c13 = c13 + 1
            elif data[i][1] == l2 and data[i][2] == l1:
                c21 = c21 + 1
            elif data[i][1] == l2 and data[i][2] == l2:
                c22 = c22 + 1
            elif data[i][1] == l2 and data[i][2] == l3:
                c23 = c23 + 1
            elif data[i][1] == l3 and data[i][2] == l1:
                c31 = c31 + 1
            elif data[i][1] == l3 and data[i][2] == l2:
                c32 = c32 + 1
            elif data[i][1] == l3 and data[i][2] == l3:
                c33 = c33 + 1

        total1 = c11 + c12 + c13 
        total2=c21 + c22 + c23 
        total3=c31 + c32 + c33
        total=total1+total2+total3
        
        classification_rate=((c11+c22+c33)/total)
        error_rate=1-classification_rate

        print("CLASSIFICATION RATE = "+str(classification_rate))
        print("ERROR RATE = "+str(error_rate))

        print(pd.DataFrame.from_items([('Iris-setosa', [c11 / total1, c12 / total1, c13 / total1]), ('Iris-versicolor', [c21 / total2, c22 / total2, c23 / total2]),
                                       ('Iris-virginica', [c31 / total3, c32 / total3, c33 / total3])], orient='index', columns=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']))


def income_confusion_matrix():
    # open knn.csv file and proceed
    low = ' <=50K'
    high = ' >50K'
    # matrix components
    ll = 0.0
    lh = 0.0
    hl = 0.0
    hh = 0.0

    with open('knn.csv') as csvfile:
        csvdata = csv.reader(csvfile, delimiter=',')

        for i, row in enumerate(csvdata):
            data.append(row)
        data.pop(0)  # removes headings

        for i in range(0, len(data)):
            if data[i][1] == low and data[i][2] == low:
                ll = ll + 1
            elif data[i][1] == low and data[i][2] == high:
                lh = lh + 1
            elif data[i][1] == high and data[i][2] == low:
                hl = hl + 1
            elif data[i][1] == high and data[i][2] == high:
                hh = hh + 1

        l_total = lh + ll
        h_total=hl + hh
        total=lh+ll+hh+hl

        print(pd.DataFrame.from_items([('<=50K', [ll / l_total, lh / l_total]), ('>50K', [
            hl / h_total, hh / h_total])], orient='index', columns=['<=50K', '>50K']))

        classification_rate=(ll+hh)/total
        print("CLASSIFICATION RATE = "+str(classification_rate))
        error_rate=1-classification_rate
        print("ERROR RATE = "+str(error_rate))

        roc_curve(ll,lh,hl,hh)

def roc_curve(tp,fn,fp,tn):
    tpr=tp/(tp+fn)
    print("TRUE POSITIVE RATE = "+str(tpr))
    fpr=fp/(fp+tn)
    print("FALSE POSITIVE RATE = "+str(fpr))
    tnr=tn/(fp+tn)
    print("TRUE NEGATIVE RATE = "+str(tnr))
    fnr=fn/(tp+fn)
    print("FALSE NEGATIVE RATE = "+str(fnr))

    precision=tp/(tp+fp)
    print("PRECISION = "+str(precision))
    recall=tp/(tp+fn)
    print("RECALL = "+str(recall))

    fmeasure=((2*precision*recall)/(precision+recall))
    print("F-MEASURE = "+str(fmeasure))

# main()
if __name__ == "__main__":

    choice=1 #default : 1-eucledian 2-cosine

    Iris = [[0, 'tid'], [1, 'num'], [2, 'num'],
            [3, 'num'], [4, 'num'], [5, 'class']]
    Income = [[0, 'tid'], [1, 'ID'], [2, 'num'], [3, 'cat'], [4, 'num'], [5, 'cat'], [6, 'num_cat'], [7, 'cat'], [
        8, 'cat'], [9, 'cat'], [10, 'cat'], [11, 'cat'], [12, 'num'], [13, 'num'], [14, 'num'], [15, 'cat'], [16, 'class']]

    if(len(sys.argv) != 3):
        print("Did not pass input csv file or the value of k")
        sys.exit()

    filename = sys.argv[1]
    k = int(sys.argv[2])
    test_file = " "

    if filename == 'Iris.csv':
        metadata = Iris
        test_file = "Iris_Test.csv"
    elif filename == 'Income.csv':
        metadata = Income
        test_file = "Income_Test.csv"
    else:
        print("Invalid filename")

    # DEALING WITH TRAINING DATA
    data = []  # -training data records
    with open(filename) as csvfile:
        csvdata = csv.reader(csvfile, delimiter=',')

        for i, row in enumerate(csvdata):
            data.append([i] + row)

        data.pop(0)  # removes headings

        # STEP1 - MISSING DATA HANDLING
        data = handle_missing_data(data, metadata)  # TRAINING DATASET

        # STEP 1.5 -  CALCULATING TRAINING DATASET PARAMETERS

        train_mean = calculate_means(data, metadata)
        train_mode = calculate_modes(data, metadata)
        train_min = calculate_min(data, metadata)
        train_max = calculate_max(data, metadata)

        # STEP2 - NORMALIZING DATA
        data = normalize_data(data, metadata)

    # STEP 3 DEALING WITH TESTING DATASET
    test = []
    with open(test_file) as csvfile:
        testdata = csv.reader(csvfile, delimiter=',')

        for i, row in enumerate(testdata):
            test.append([i] + row)

        test.pop(0)  # to remove headings bro!
        test = test_missing_data(test, metadata, train_mean, train_mode)
        test = test_normalize_data(test, metadata, train_min, train_max)

        # STEP3 - COMPUTING SIMILARITY
        compute_similarity(test, data, metadata, k, choice)

        # STEP4 - FINDING KNN
        knn(k)

        # STEP 5 - CONFUSION MATRICES
        if filename == "Iris.csv":
            iris_confusion_matrix()
        elif filename == "Income.csv":
            income_confusion_matrix()
            
