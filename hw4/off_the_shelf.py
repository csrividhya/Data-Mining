import sklearn
from sklearn import cluster
from sklearn import metrics
from sklearn.cluster import k_means
import csv
import sys

# main()
if __name__ == "__main__":

    if(len(sys.argv) != 3):
        print("Did not pass input csv file or k value")
        sys.exit()

    filename = sys.argv[1]
    k = int(sys.argv[2])

    data = []
    with open(filename) as csvfile:
        csvdata = csv.reader(csvfile)
        for row in csvdata:
        	row.remove(row[13])
        	data.append(row)
        data.pop(0)

    """
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(data)
    labels = kmeans.labels_
    print(labels)
    centroids = kmeans.cluster_centers_
    print(centroids)
    """
    centroids,labels,inertia=sklearn.cluster.k_means(data, n_clusters=k, init='k-means++', precompute_distances='auto', n_init=10, max_iter=300, verbose=False, tol=0.0001, random_state=None, copy_x=True, n_jobs=1,return_n_iter=False)
   
    print(centroids)
    print(inertia)