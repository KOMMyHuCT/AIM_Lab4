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
data =[['Food', 'Sweetness', 'Crustiness', 'Type'],
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

new_dist = numpy.zeros((5, 10))
print(new_dist)
for i in range(5):
    for j in range (10):
        new_dist[i][j] = distance(int(data[11 + i][1]), int(data[11 + i][2]), int(data[j + 1][1]), int(data[j + 1][2]))
print(new_dist)
err_k = [0]*10
for k in range(10):
    print('Classification for k = ', k + 1)
    clas = [0, 0, 0, 0, 0]
    err = [0, 0, 0, 0, 0]
    for i in range(5):
        quant_dist = [0, 0, 0]
        print('Classification of ', data[11 + i][0])
        tmp = numpy.array(new_dist[i,:])
        for j in range(k + 1):
            in_min = list(tmp).index(min(tmp))
            print(in_min)
            quant_dist[int(data[in_min + i + 1][3])] += 1
            print(in_min + i + 1, int(data[in_min + i + 1][3]))
            tmp[in_min] = 1000
            print(quant_dist)
            max1 = max(quant_dist)
            print(int(data[in_min + i + 1][3]), int(data[11 + i][3]))
            if quant_dist.count(max1) > 1:
                err[i] = 0.5
            elif (int(data[in_min + i + 1][3]) != int(data[11 + i][3])):
                err[i] = 1
            print(err)
        err_k[k] = numpy.mean(err)
    print('Errors ', err_k)

finish = time.time()
result = finish - start
print('Program time: ' + str(result) + ' seconds')

pyplot.plot([i for i in range(1, 11)], err_k)
pyplot.title('Dependence of error on k')
pyplot.xlabel('k')
pyplot.ylabel('Error')
pyplot.show()

df = pandas.read_csv('sw_data_new.csv', encoding='cp1251')
print(df)
sweet = df['Sweetness']
crunchy = df['Crustiness']
print(sweet, crunchy)

pyplot.scatter(sweet, crunchy)
pyplot.show()