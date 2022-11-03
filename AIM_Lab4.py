import csv
import sklearn
import pandas
import numpy
import matplotlib.pyplot as pyplot
import random
import time
def distance(xt1, xt2, xi1, xi2):
    return (abs(xt1 - xi1) ** 2 + abs(xt2 - xi2) ** 2) ** 0.5

start = time.time()
#header (row 0)
data =[['Food', 'Sweetness', 'Crustiness', 'Type'],
       #training set of 10 (row 1-10)
       ['Apple', '7', '7', '0'],
       ['Salad', '2', '5', '1'],
       ['Bacon', '1', '2', '2'],
       ['Nuts', '1', '5', '2'],
       ['Fish', '1', '1', '2'],
       ['Cheese', '1', '1', '2'],
       ['Banana', '9', '1', '0'],
       ['Carrot', '2', '8', '1'],
       ['Grape', '8', '1', '0'],
       ['Orange', '6', '1', '0'],
       #test set of 5 (row 11-16)
       ['Strawberry', '9', '1', '0'],
       ['Lettuce', '3', '7', '1'],
       ['Shashlik', '1', '1', '2'],
       ['Pear', '5', '3', '0'],
       ['Celery', '1', '5', '1']]

with open('sw_data_new.csv', 'w') as f:
    writer = csv.writer(f)
    for row in data:
        writer.writerow(row)

with open('sw_data_new.csv') as f:
    print(f.read())

training_size = 10
test_size = 5
new_dist = numpy.zeros((5, 10))
print(new_dist)
for i in range(test_size):
    for j in range (training_size):
        new_dist[i][j] = distance(int(data[training_size + i][1]), int(data[training_size + i][2]), int(data[j + 1][1]), int(data[j + 1][2])) #row 0 is header hence +1
print(new_dist)
err_k = [0]*training_size
types = [0]*training_size
for k in range(training_size): #k - number of neighbors
    max2 = 0
    print('Classification for k = ', k + 1)
    clas = [0, 0, 0, 0, 0]
    err = [0, 0, 0, 0, 0]
    for i in range(test_size):
        quant_dist = [0, 0, 0] #(OLD) how likely an element is to belong to each of three types
                               #how many times each of three types is seen among a point's neighbors
        print('Classification of ', data[training_size + i][0])
        tmp = numpy.array(new_dist[i, :]) #distances of a test point to other training points
        for j in range(k + 1):
            in_min = list(tmp).index(min(tmp)) #index of element with minimal distance
            print(in_min)
            quant_dist[int(data[in_min + i + 1][3])] += 1
            print(in_min + i + 1, int(data[in_min + i + 1][3])) #successively print index to nearest neighbors (0 is closest, 1 is second closest, etc) and their type
            tmp[in_min] = 1000
            print(quant_dist)
            max1 = max(quant_dist) #how frequently most seen type is seen
            max2 = list(quant_dist).index(max1) #most frequently seen type
            print(int(data[in_min + i + 1][3]), int(data[training_size + i][3])) #print estimated and actual types
            if quant_dist.count(max1) > 1: #if element equally belongs to several types; less of an issue hence 0.5
                err[i] = 0.5
            elif (int(data[in_min + i + 1][3]) != int(data[training_size + i][3])): #if estimated and actual types don't match; less of an issue hence 1
                err[i] = 1
            print(err) #error for each test element
        err_k[k] = numpy.mean(err) #mean error for each neighbor
    print('Errors: ', err_k)
    types[k] = max2
    print('Estimated type: ', types[k])

finish = time.time()
result = finish - start
print('Program time: ' + str(result) + ' seconds')

pyplot.plot([i for i in range(1, training_size + 1)], err_k)
pyplot.title('Dependence of error on k')
pyplot.xlabel('k')
pyplot.ylabel('Error')
pyplot.show()

pyplot.plot([i for i in range(1, training_size + 1)], types)
pyplot.title('Dependence of type on k')
pyplot.xlabel('k')
pyplot.ylabel('Types')
pyplot.show()

df = pandas.read_csv('sw_data_new.csv', encoding='cp1251')
print(df)
sweet = df['Sweetness']
crunchy = df['Crustiness']
print(sweet, crunchy)
pyplot.scatter(sweet, crunchy)
pyplot.show()