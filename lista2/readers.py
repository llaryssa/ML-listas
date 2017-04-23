from __future__ import division

def normalize(data):
    lmax = [float("-inf")]*len(data[0])
    lmin = [float("+inf")]*len(data[0])
    for sample in data:
        for i in range(0,len(sample)):
            if sample[i] > lmax[i]:
                lmax[i] = sample[i]
            if sample[i] < lmin[i]:
                lmin[i] = sample[i]
    for x in range(0,len(data)):
        for y in range(0,len(data[0])):
            data[x][y] = (data[x][y] - lmin[y])/(lmax[y] - lmin[y])
    return data

def read_breastcancer():
    file = open("breast-cancer-wisconsin.data.txt", "r")

    data = list()
    labels = list()

    for line in file:
        instance = line.split(',')

        # ignoring the instances with missing data
        if '?' not in instance:
            label = int(instance[-1])
            instance = map(int, instance[1:len(instance) - 1])

            labels.append(label)
            data.append(instance)

    file.close()
    return data, labels

def read_wine(norm=False):
    file = open("wine.data.txt", "r")
    data = list()
    labels = list()

    for line in file:
        instance = line.split(',')

        # ignoring the instances with missing data
        if '?' not in instance:
            label = int(instance[0])
            instance = map(float, instance[1:len(instance)])

            labels.append(label)
            data.append(instance)

    file.close()
    if norm:
        normalize(data)
    return data, labels

def read_yeast(norm=False):
    file = open("yeast.data.txt", "r")
    data = list()
    labels = list()

    for line in file:
        instance = line.split('  ')
        label = instance[-1]

        if '' in instance:
            instance.remove('')

        instance = map(float, instance[1:len(instance)-1])

        labels.append(label)
        data.append(instance)

    file.close()
    if norm:
        normalize(data)
    return data, labels
