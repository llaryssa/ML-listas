from __future__ import division

import numpy as np
import math

from readers import *

np.set_printoptions(threshold=np.nan)



###########################################################

def euclidean_distance(a, b):
    if len(a) == len(b):
        dist = 0
        for i in range(len(a)):
            dist = dist + math.pow(a[i] - b[i], 2)
        return math.sqrt(dist)

def closest_point(sample, points, function, algorithm):
    distances = list()
    for point in points:
        dist = function(sample, point)
        distances.append(dist)
    distances = np.array(distances)
    indices = np.argsort(distances)
    if algorithm == 'lvq':
        return indices[0]
    elif algorithm == 'lvq21':
        return indices[:2]

def count_classes(labels):
    count = dict()
    for l in labels:
        if l not in count:
            count[l] = 0
        count[l] = count[l] + 1
    return count

def get_prototypes(data, labels, num):
    prototypes = list()
    prototypes_labels = list()

    # this is for getting the proportional amount of prototypes
    # according to the training data
    labels_count = count_classes(labels)
    for lab in labels_count:
        siz = labels_count[lab]
        num_classes_prototypes = int(math.ceil(num*(siz/len(labels))))
        labels_count[lab] = num_classes_prototypes

    # random choose the prototypes among classes
    perm = np.random.permutation(len(data))
    data = data[perm]
    labels = labels[perm]
    for x, y in zip(data, labels):
        if labels_count[y] > 0:
            prototypes.append(x)
            prototypes_labels.append(y)
            labels_count[y] -= 1
        # if all the prototypes were chosen exit loop
        if all(x == 0 for x in labels_count.values()):
            break

    return prototypes, prototypes_labels


def lvq1(training_data, training_labels, n_prototypes, alpha, max_iterations):
    prototypes, prototypes_labels = get_prototypes(training_data, training_labels, n_prototypes)
    for iteration in range(0, max_iterations):
        # updating the learning rate (smaller each iteraction)
        sum_error = 0
        learning_rate = alpha * (1 - (iteration/max_iterations))
        for sample, label in zip(training_data, training_labels):
            closest_idx = closest_point(sample, prototypes, euclidean_distance, algorithm='lvq')
            closest_label = prototypes_labels[closest_idx]
            for i in range(0,len(sample)):
                error = sample[i] - prototypes[closest_idx][i]
                sum_error = sum_error + error**2
                if closest_label == label:
                    prototypes[closest_idx][i] += learning_rate * error
                else:
                    prototypes[closest_idx][i] -= learning_rate * error
        print ">epoch:",iteration,"- lrate:",learning_rate,"- error:",sum_error
    return np.array(prototypes), np.array(prototypes_labels)

def lvq21(training_data, training_labels, n_prototypes, alpha, max_iterations):
    prototypes, prototypes_labels = get_prototypes(training_data, training_labels, n_prototypes)
    for iteration in range(0, max_iterations):
        # updating the learning rate (smaller each iteraction)
        sum_error = 0
        learning_rate = alpha * (1 - (iteration/max_iterations))
        for sample, label in zip(training_data, training_labels):
            closest_idx = closest_point(sample, prototypes, euclidean_distance, algorithm='lvq21')
            lab1 = prototypes_labels[closest_idx[0]]
            lab2 = prototypes_labels[closest_idx[1]]
            # so atualiza se cumprir as regras
            if (lab1 != lab2) and (lab1 == label or lab2 == label):
                for i in range(0,len(sample)):
                    error1 = sample[i] - prototypes[closest_idx[0]][i]
                    error2 = sample[i] - prototypes[closest_idx[1]][i]

                    sum_error = sum_error + error1**2 + error2**2

                    if lab1 == label:
                        prototypes[closest_idx[0]][i] += learning_rate * error1
                        prototypes[closest_idx[1]][i] -= learning_rate * error2
                    else:
                        prototypes[closest_idx[0]][i] -= learning_rate * error1
                        prototypes[closest_idx[1]][i] += learning_rate * error2

        print ">epoch:",iteration,"- lrate:",learning_rate,"- error:",sum_error
    return np.array(prototypes), np.array(prototypes_labels)

def lvq3(training_data, training_labels, n_prototypes, alpha, epsilon, max_iterations):
    prototypes, prototypes_labels = get_prototypes(training_data, training_labels, n_prototypes)
    for iteration in range(0, max_iterations):
        # updating the learning rate (smaller each iteraction)
        sum_error = 0
        learning_rate = alpha * (1 - (iteration/max_iterations))
        for sample, label in zip(training_data, training_labels):
            closest_idx = closest_point(sample, prototypes, euclidean_distance, algorithm='lvq21')
            lab1 = prototypes_labels[closest_idx[0]]
            lab2 = prototypes_labels[closest_idx[1]]
            # so atualiza se cumprir as regras
            if (lab1 != lab2) and (lab1 == label or lab2 == label):
                for i in range(0,len(sample)):
                    error1 = sample[i] - prototypes[closest_idx[0]][i]
                    error2 = sample[i] - prototypes[closest_idx[1]][i]
                    sum_error = sum_error + error1**2 + error2**2

                    if lab1 == label:
                        prototypes[closest_idx[0]][i] += learning_rate * error1
                        prototypes[closest_idx[1]][i] -= learning_rate * error2
                    else:
                        prototypes[closest_idx[0]][i] -= learning_rate * error1
                        prototypes[closest_idx[1]][i] += learning_rate * error2

            if (lab1 == lab2 == label):
                for i in range(0,len(sample)):
                    error1 = sample[i] - prototypes[closest_idx[0]][i]
                    error2 = sample[i] - prototypes[closest_idx[1]][i]
                    sum_error = sum_error + error1**2 + error2**2

                    prototypes[closest_idx[0]][i] += epsilon * learning_rate * error1
                    prototypes[closest_idx[1]][i] += epsilon * learning_rate * error2


        print ">epoch:",iteration,"- lrate:",learning_rate,"- error:",sum_error
    return np.array(prototypes), np.array(prototypes_labels)

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

def knn(training_data, training_labels, test_data, neighbors, use_weights = False):
    estimated_labels = []
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

###########################################################



# data, labels = read_breastcancer()
data, labels = read_wine(True)
# data, labels = read_yeast()

data = np.array(data)
labels = np.array(labels)
print "Data size: ", data.shape

p = np.random.permutation(len(data))
data = data[p]
labels = labels[p]

training_num = int(len(data) * 0.7)
training_data = data[:training_num]
training_labels = labels[:training_num]
test_data = data[training_num:]
test_labels = labels[training_num:]

print "\tTraining size: ", training_data.shape
print "\tTest size: ", test_data.shape

n_prototypes = 15
max_iterations = 10
alpha = 0.25
epsilon = 0.4

knn_neighbors = 3

# prototypes, prototypes_labels = lvq1(data,labels,n_prototypes,alpha,max_iterations)
# prototypes, prototypes_labels = lvq21(data,labels,n_prototypes,alpha,max_iterations)
prototypes, prototypes_labels = lvq3(data,labels,n_prototypes,alpha,epsilon,max_iterations)
print "Prototypes: ", prototypes.shape
labels_hat = knn(prototypes, prototypes_labels, test_data, knn_neighbors)

# labels_hat = knn(training_data, training_labels, test_data, knn_neighbors, use_weights=False)

right = 0
for hat, label in zip(labels_hat, test_labels):
    if hat == label:
        right += 1

print "Accuracy: ", right/len(test_labels)


# dataset = [[2.7810836,2.550537003],
# 	[1.465489372,2.362125076],
# 	[3.396561688,4.400293529],
# 	[1.38807019,1.850220317],
# 	[3.06407232,3.005305973],
# 	[7.627531214,2.759262235],
# 	[5.332441248,2.088626775],
# 	[6.922596716,1.77106367],
# 	[8.675418651,-0.242068655],
# 	[7.673756466,3.508563011]]
#
# labels = [0,0,0,0,0,0,1,1,1,1,1]
#
# dataset = np.array(dataset)
# labels = np.array(labels)
#
# prototypes, prototypes_labels = lvq1(dataset,labels,2,0.3,10)
