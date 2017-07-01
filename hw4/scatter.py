import matplotlib.pyplot as plt
import numpy as np

# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api


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

	# incorrectly classified points
	wrong1_x1 = []
	wrong1_x2 = []
	wrong2_x1 = []
	wrong2_x2 = []
	wrong3_x1 = []
	wrong3_x2 = []
	wrong4_x1 = []
	wrong4_x2 = []

	fig, ax = plt.subplots()
	for i in range(0, len(data)):
		if data[i][3] == 1 and cluster[i] == 0:  # correct
			cluster1_x1.append(data[i][1])
			cluster1_x2.append(data[i][2])

		elif data[i][3] == 2 and cluster[i] == 1:  # correct
			cluster2_x1.append(data[i][1])
			cluster2_x2.append(data[i][2])

		elif data[i][3] == 3 and cluster[i] == 2:  # correct
			cluster3_x1.append(data[i][1])
			cluster3_x2.append(data[i][2])

		elif data[i][3] == 4 and cluster[i] == 3:  # correct
			cluster4_x1.append(data[i][1])
			cluster4_x2.append(data[i][2])

	ax.scatter(cluster1_x1, cluster1_x2, color='g', s=10,marker='^', alpha=.9)
	ax.scatter(cluster2_x1, cluster2_x2, color='b', alpha=.9)
	ax.scatter(cluster3_x1, cluster3_x2, color='r',marker='+', alpha=.9)
	ax.scatter(cluster4_x1, cluster4_x2, color='c',marker='*', alpha=.9)
	
	plt.xlabel('X1')
	plt.ylabel('X2')	
	plt.show()

data=[[1,2,3,1],[10,23,45,1],[4,6,7,2],[12,45,76,2],[56,32,89,2],[3,6,1,3],[1,34,8,4]]
cluster=[0,0,1,1,1,2,3]
hard_scatter(data,cluster,4)