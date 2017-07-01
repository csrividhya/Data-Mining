from sklearn.neighbors import KNeighborsClassifier
import sys
import csv

def accuracy_score(a,p):
	count=0.0
	total=0.0
	for i in range(0,len(a)):
		total=total+1
		if a[i]==p[i]:
			count=count+1
	return count/total


def knn(k,traindata,train_labels,testdata,test_labels):
	X = traindata
	y = train_labels

	#neigh = KNeighborsClassifier(n_neighbors=k)
	
	# X - training data with features
	# Y - training data with class labels


	neigh= KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
	neigh.fit(X, y)
	y_pred=(neigh.predict(testdata))
	print(neigh.score(testdata,test_labels))
	print("Accuracy is ", accuracy_score(test_labels,y_pred)*100,"% for K-Value:",k)
	

if __name__ == "__main__":
	Iris = [[0, 'tid'], [1, 'num'], [2, 'num'],[3, 'num'], [4, 'num'], [5, 'class']]
	Income = [[0, 'tid'], [1, 'ID'], [2, 'num'], [3, 'cat'], [4, 'num'], [5, 'cat'], [6, 'num_cat'], [7, 'cat'], [8, 'cat'], [9, 'cat'], [10, 'cat'], [11, 'cat'], [12, 'num'], [13, 'num'], [14, 'num'], [15, 'cat'], [16, 'class']]

	filename = sys.argv[1]
	k = int(sys.argv[2])
	test_file = " "

	if filename == 'Iris.csv':
		metadata = Iris
		test_file = "Iris_Test.csv"
		class_index=4
	elif filename == 'Income.csv':
		metadata = Income
		test_file = "Income_Test.csv"
		class_index=15

	traindata = []  # -training data records
	with open(filename) as csvfile:
		csvdata = csv.reader(csvfile, delimiter=',')
		for i, row in enumerate(csvdata):
			traindata.append(row)
		traindata.pop(0)

	train_labels=[]

	for i in range(0,len(traindata)):
		train_labels.append(traindata[i][class_index])

	for i in range(0,len(traindata)):
		del traindata[i][-1]

	testdata = []
	copyTest=[]  # -training data records
	test_labels=[]

	with open(test_file) as csvfile:
		csvdata = csv.reader(csvfile, delimiter=',')
		for i, row in enumerate(csvdata):
			testdata.append(row)
		testdata.pop(0)
		copyTest=testdata

		for i in range(0,len(testdata)):
			test_labels.append(testdata[i][class_index])

		for i in range(0,len(testdata)):
			del testdata[i][-1]

	knn(k,traindata,train_labels,testdata,test_labels)
