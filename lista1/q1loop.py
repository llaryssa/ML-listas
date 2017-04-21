from __future__ import division
import numpy as np
import math
from heapq import heappush
from sklearn.neighbors import KDTree
from readers import *
import time

np.set_printoptions(threshold=np.nan)

###########################################################

def euclidean_distance(a, b):
    if len(a) == len(b):
        dist = 0
        for i in range(len(a)):
            dist = dist + math.pow(a[i] - b[i], 2)
        return math.sqrt(dist)

def generate_vdm_table(data, labels):
    print "Generating VDM table..."
    table = dict()
    classes = list()
    for x, y in zip(data, labels):
        if y not in classes:
            classes.append(y)

        for i, xi in enumerate(x):
            if i not in table:
                table[i] = dict()
            col_table = table[i]

            if xi not in col_table:
                col_table[xi] = dict()
            class_col_table = col_table[xi]

            if y not in class_col_table:
                class_col_table[y] = 0
            class_col_table[y] = class_col_table[y] + 1
    return table, classes

def get_probability(table, i, xi, c):
    denominator = 0
    for cval in table[i][xi]:
        denominator = denominator + table[i][xi][cval]
    return table[i][xi][c]/denominator

def vdm(a,b,table,classes,q=1):
    if len(a) == len(b):
        summation = 0
        for i, (ai, bi) in enumerate(zip(a,b)):
            vdm_val = _vdm(i,ai,bi,table,classes,q)
            summation = summation + vdm_val
        return math.sqrt(summation)

def hvdm(a,b,table,classes,attribute_types,q=1):
    if len(a) == len(b):
        summation = 0
        for i, (ai, bi) in enumerate(zip(a,b)):
            if attribute_types[i] == 'cat':
                dist_val = _vdm(i,ai,bi,table,classes,q)
            elif attribute_types[i] == 'num':
                dist_val = _dif(i,ai,bi,table)
            # dist_val = _vdm(i,ai,bi,table,classes,q)
            summation = summation + dist_val
        return math.sqrt(math.pow(summation,2))


def _vdm(i,ai,bi,table,classes,q):
    summ = 0
    for c in classes:
        if c in table[i][ai]:
            p_iac = get_probability(table,i,ai,c)
        else:
            p_iac = 0

        if c in table[i][bi]:
            p_ibc = get_probability(table,i,bi,c)
        else:
            p_ibc = 0
        summ = summ + math.pow(abs(p_iac - p_ibc),q)
    return summ

def _dif(i,ai,bi,table):
    maxi = max([float(a) for a in table[i]])
    mini = min([float(a) for a in table[i]])
    ai = float(ai)
    bi = float(bi)
    return abs(ai-bi)/(maxi-mini)

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

def knn_all(training_data, training_labels, test_data, neighbors, use_weights = False, use_vdm = False, use_hvdm = False, attribute_types = None):
    estimated_labels = []
    all_labels1 = dict()
    all_labels2 = dict()

    for k in neighbors:
        all_labels1[k] = list()
        all_labels2[k] = list()

    if use_vdm or use_hvdm:
        table, classes = generate_vdm_table(training_data, training_labels)
        # print table, classes

    i = 0
    for test in test_data:
        if i%100==0:
            print i
        distances = np.array([])
        for train in training_data:
            start_time = time.time()
            if use_vdm:
                curr_dist = vdm(test, train, table, classes)
            elif use_hvdm:
                curr_dist = hvdm(test, train, table, classes, attribute_types)
            else:
                curr_dist = euclidean_distance(test, train)
            print time.time() - start_time
            distances = np.append(distances, curr_dist)
        for k in neighbors:
            indices = np.argsort(distances)
            nindices = indices[:k]

            # if (use_weights):
            dist = distances[indices[:k]]
            estimated_label2 = weighted_count(training_labels[nindices], dist)

            # else:
            estimated_label1 = count(training_labels[nindices])

            all_labels1[k].append(estimated_label1)
            all_labels2[k].append(estimated_label2)
        i = i + 1
        # print all_labels1
    return all_labels1, all_labels2

def calculate_accuracy(labels, real_labels):
    correct = 0
    for i, label in enumerate(labels):
        if label == real_labels[i]:
            correct = correct + 1
    return float(correct) / float(len(labels))

def mean_accuracy(acc):
    acc = np.array(acc)
    return np.mean(acc, axis=0)








###########################################################






data, labels = read_breastcancer()
# data, labels = read_vicon()
# data, labels = read_car()
# data, labels = read_velha()
# data, labels, attribute_types = read_contraceptive()
# data, labels, attribute_types = read_german()


data = np.array(data)
labels = np.array(labels)
print "Data size: ", data.shape


big_acc1 = list()
big_acc2 = list()
for times in range(0, 10):
    # shuffling data to split in training and testing
    p = np.random.permutation(len(data))
    data = data[p]
    labels = labels[p]

    # data = data[:20000]
    # labels = labels[:20000]

    training_num = int(len(data) * 0.7)

    training_data = data[:training_num]
    training_labels = labels[:training_num]

    test_data = data[training_num:]
    test_labels = labels[training_num:]

    acc1 = list()
    acc2 = list()

    kvalues = [1,2,3,5,7,9,11,13,15]

    # para a primeira questao: breast cancer e vicon
    labels_hat1, labels_hat2 = knn_all(training_data, training_labels, test_data, kvalues)

    # # para a segunda questao: car e velha
    # labels_hat1, labels_hat2 = knn_all(training_data, training_labels, test_data, kvalues, use_vdm=True)

    # # para a terceira questao: german e contraceptive
    # labels_hat1, labels_hat2 = knn_all(training_data, training_labels, test_data, kvalues, use_hvdm=True, attribute_types=attribute_types)

    for lb in labels_hat1:
        accuracy1 = calculate_accuracy(labels_hat1[lb], test_labels)
        acc1.append(accuracy1)

    for lb in labels_hat2:
        accuracy2 = calculate_accuracy(labels_hat2[lb], test_labels)
        acc2.append(accuracy2)

    big_acc1.append(acc1)
    big_acc2.append(acc2)


# plota tudo
import matplotlib.pyplot as plt
plt.plot(kvalues, mean_accuracy(big_acc1), 'b', kvalues, mean_accuracy(big_acc2), 'r--')
plt.ylabel('Acuracia')
plt.xlabel('K vizinhos')
plt.show()
