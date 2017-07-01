import matplotlib.pyplot as plt
import csv
 
# Histogram3 &4 - Capital x1 and x2
x1 = []
x2 = []
data=[]
with open('TwoDimHard.csv') as csvfile:
    csvdata = csv.reader(csvfile, delimiter=',')
    for row in csvdata:
        data.append(row)
    data.pop(0)  # removes headings

    for i in range(0, len(data)):
        x = float(data[i][1])
        x1.append(x)
        y = float(data[i][2])
        x2.append(y)
    print(x1)
    print(x2)


print("Min x1 is: " + str(min(x1)))
print("Max x1 is: " + str(max(x1)))

print("Min x2 is: " + str(min(x2)))
print("Max x2 is: " + str(max(x2)))

n, bins, patches = plt.hist(x1, 10, normed=0, facecolor='crimson', alpha=0.75)
plt.xlabel('x1')
plt.ylabel('Frequency')
plt.title('Distribution of x1 in TwoDimHard dataset')
plt.show()
