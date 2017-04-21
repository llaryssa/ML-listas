from __future__ import division

import numpy as np
import math

from readers import *

np.set_printoptions(threshold=np.nan)



###########################################################

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
            labels_count[y] = labels_count[y] - 1
        # if all the prototypes were chosen exit loop
        if all(x == 0 for x in labels_count.values()):
            break

    return prototypes, prototypes_labels


def lvq(training_data, training_labels, n_prototypes, alpha, max_iteractions):
    prototypes, prototypes_labels = get_prototypes(training_data, training_labels, n_prototypes)
    






###########################################################



data, labels = read_breastcancer()

data = np.array(data)
labels = np.array(labels)
print "Data size: ", data.shape

lvq(data,labels,100,10,10)
