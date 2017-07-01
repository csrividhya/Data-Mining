from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
import csv
import sys


def whos_my_neighbor(a):
	k = int(sys.argv[1])
	neighbour_labelz = []
	for i in range(1, k+1):
		neighbour_labelz.append(a[1+3*i])  
	count=0.0
	for i in range(0,len(neighbour_labelz)):
		if neighbour_labelz[i]==low:
			count=count+1
	return (count/len(neighbour_labelz))

data = []  # -training data records
low = ' <=50K' #positive
high = ' >50K' #negative

l=1
h=0

with open('output.csv') as csvfile:
    csvdata = csv.reader(csvfile, delimiter=',')

    for i, row in enumerate(csvdata):
        data.append(row)

    data.pop(0)  # removes headings

    actual=[] #binarizing the class labels to 1's and 0's
    for i in range(0,len(data)):
    	if data[i][1]==low:
    		actual.append(1)
    	if data[i][1]==high:
    		actual.append(0)

    predictions=[]
    for i in range(0,len(data)):
    	x= whos_my_neighbor(data[i])
    	predictions.append(x)

  

false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
         label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

