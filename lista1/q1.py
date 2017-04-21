import numpy as np
import math
from heapq import heappush
from sklearn.neighbors import KDTree
from readers import read_breastcancer, read_sensorless, read_vicon

np.set_printoptions(threshold=np.nan)

###########################################################

def euclidean_distance(a, b):
    if len(a) == len(b):
        dist = 0
        for i in range(len(a)):
            dist = dist + math.pow(a[i] - b[i], 2)
        return math.sqrt(dist)

def count(labels):
    # if every entry is the same
    if all([x == labels[0] for x in labels]):
        return labels[0]

    table = dict()
    for label in labels:
        if label not in table:
            table[label] = 1
        else:
            table[label] = table[label] + 1
    maxidx = -1
    maxval = 0
    for entry in table:
        if table[entry] > maxval:
            maxval = table[entry]
            maxidx = entry
    return maxidx

def weighted_count(labels, distances):
    # if every entry is the same
    if all([x == labels[0] for x in labels]):
        return labels[0]

    table = dict()
    for i, label in enumerate(labels):
        if distances[i] == 0:
            weight = 1
        else:
            weight = float(1)/float(math.pow(distances[i],2))
        if label not in table:
            table[label] = weight
        else:
            table[label] = table[label] + weight
    maxidx = -1
    maxval = 0
    for entry in table:
        if table[entry] > maxval:
            maxval = table[entry]
            maxidx = entry
    return maxidx


def knn(training_data, training_labels, test_data, neighbors, use_kdtree = False, use_weights = False):
    estimated_labels = []
    if use_kdtree:
        tree = KDTree(training_data)
        distances, indices = tree.query(test_data, neighbors)
        for idx in indices:
            estimated_label = np.bincount(training_labels[idx]).argmax()
            estimated_labels.append(estimated_label)
    else:
        for test in test_data:
            distances = np.array([])
            for train in training_data:
                # distances.append(euclidean_distance(test, train))
                distances = np.append(distances, euclidean_distance(test, train))
            indices = np.argsort(distances)
            nlabels = indices[:neighbors]
            if (use_weights):
                distances = distances[indices]
                estimated_label = weighted_count(training_labels[nlabels], distances[:neighbors])
            else:
                estimated_label = count(training_labels[nlabels])
            estimated_labels.append(estimated_label)
    return estimated_labels

def weighted_knn(training_data, training_labels, test_data, neighbors, use_kdtree = False):
    return knn(training_data, training_labels, test_data, neighbors, use_kdtree, True)

def calculate_accuracy(labels, real_labels):
    correct = 0
    for i, label in enumerate(labels):
        if label == real_labels[i]:
            correct = correct + 1
    return float(correct) / float(len(labels))

###########################################################

data, labels = read_breastcancer()
# data, labels = read_sensorless()
# data, labels = read_vicon()
data = np.array(data)
labels = np.array(labels)

print "Data size: ", data.shape

# shuffling data to split in training and testing
p = np.random.permutation(len(data))
data = data[p]
labels = labels[p]

training_num = int(len(data) * 0.5)

training_data = data[:training_num]
training_labels = labels[:training_num]

test_data = data[training_num:]
test_labels = labels[training_num:]

acc = list()

kvalues = [1,2,3,5,7,9,11,13,15]
for k in kvalues:
    # labels = knn(training_data, training_labels, test_data, k, False)
    # labels = knn(training_data, training_labels, test_data, k, True)

    labels = weighted_knn(training_data, training_labels, test_data, k, False)
    # labels = weighted_knn(training_data, training_labels, test_data, k, True)

    accuracy = calculate_accuracy(labels, test_labels)
    acc.append(accuracy)

print acc

import matplotlib.pyplot as plt
plt.plot(kvalues, acc)
plt.ylabel('Acuracia')
plt.xlabel('K vizinhos')
plt.show()
