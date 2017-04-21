def read_sensorless():
    file = open("Sensorless_drive_diagnosis.txt", "r")

    data = list()
    labels = list()

    for line in file:
        instance = line.split()
        label = int(instance[-1])
        instance = map(float, instance[:len(instance) - 1])

        labels.append(label)
        data.append(instance)

    file.close()
    return data, labels

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

def read_vicon(num_of_folders = 1):
    dataset = "Vicon"
    folder = "sub"
    types = ["aggressive", "normal"]
    aggressive_types = ["Elbowing", "Frontkicking", "Hamering", "Headering", "Kneeing", "Pulling", "Punching", "Pushing", "Sidekicking", "Slapping"]
    normal_types = ["Bowing", "Clapping", "Handshaking", "Hugging", "Jumping", "Running", "Seating", "Standing", "Walking", "Waving"]

    data = list()
    labels = list()

    for i in range(1,num_of_folders+1):
        for aggt in aggressive_types:
            filename = dataset +'/'+ folder+str(i) +'/'+ types[0] +'/'+ aggt + '.txt'
            file = open(filename, 'r')
            for line in file:
                instance = line.split()
                instance = map(float, instance[1:])
                data.append(instance)
                labels.append(aggressive_types.index(aggt))

        for normt in normal_types:
            filename = dataset +'/'+ folder+str(i) +'/'+ types[1] +'/'+ normt + '.txt'
            file = open(filename, 'r')
            for line in file:
                instance = line.split()
                instance = map(float, instance[1:])
                data.append(instance)
                labels.append(len(aggressive_types) + normal_types.index(normt))

    file.close()
    return data, labels

def read_car():
    file = open("car.data.txt", "r")

    data = list()
    labels = list()

    for line in file:
        instance = line.split(',')

        # ignoring the instances with missing data
        if '?' not in instance:
            label = instance[-1]
            instance = instance[:len(instance) - 1]

            labels.append(label)
            data.append(instance)

    file.close()
    return data, labels

def read_velha():
    file = open("tic-tac-toe.data.txt", "r")

    data = list()
    labels = list()

    for line in file:
        instance = line.split(',')

        # ignoring the instances with missing data
        if '?' not in instance:
            label = instance[-1]
            instance = instance[:len(instance) - 1]

            labels.append(label)
            data.append(instance)

    file.close()
    return data, labels

def read_contraceptive():
    file = open("cmc.data.txt", "r")

    data = list()
    labels = list()

    atribute_types = ['num', 'cat', 'cat', 'num', 'cat', 'cat', 'cat', 'cat', 'cat']

    for line in file:
        instance = line.split(',')
        label = instance[-1]
        instance = instance[:len(instance) - 1]

        for idx in range(0,len(instance)):
            if atribute_types[idx] == 'num':
                instance[idx] = float(instance[idx])

        labels.append(label)
        data.append(instance)

    file.close()
    return data, labels, atribute_types

def read_german():
    file = open("german.data.txt", "r")

    data = list()
    labels = list()

    atribute_types = ['cat','num', 'cat', 'cat', 'num', 'cat', 'cat',
                    'num', 'cat', 'cat', 'num', 'cat', 'num', 'cat', 'cat',
                    'num', 'cat', 'num', 'cat', 'cat']

    for line in file:
        instance = line.split()
        label = instance[-1]
        instance = instance[:len(instance) - 1]

        for idx in range(0,len(instance)):
            if atribute_types[idx] == 'num':
                instance[idx] = float(instance[idx])

        labels.append(label)
        data.append(instance)

    file.close()
    return data, labels, atribute_types
