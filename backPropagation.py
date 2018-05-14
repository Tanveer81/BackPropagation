import numpy
import random
import math
import time

global layers, neurons, attributes, classes , error , learnRate
neurons = []
numberOfSample = 0
numberOfTest = 0
dataset = []
testset = []
nn = []
wt = [.15, .2, .25, .3, .4, .45, .5, .55]
class neuron:
    def __init__(self, layer, position, data):
        self.layer = layer
        self.position = position
        self.data = data
        #actual weight container
        self.weight = []
        self.delta = 1
        self.out = 1
        self.net = 1
        self.bias = 0
        self.error = 0

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def init():
    # in parameter.txt 1st , 2nd and 3rd line has layers,attributes and classes and next lines has number of neurons in each layer
    global dataset, neurons, layers, attributes, classes, numberOfSample, numberOfTest
    random.seed(4000)
    # 8562
    file = open("parameters.txt", "r")
    layers = int(file.readline())
    print("Number Of Layers", end=" : ")
    print(layers)
    attributes = int(file.readline())
    print("Number Of attributes", end=" : ")
    print(attributes)
    classes = int(file.readline())
    print("Number Of classes", end=" : ")
    print(classes)

    neurons.append(attributes)
    for i in range(0, layers):
        neurons.append(int(file.readline()))
    neurons.append(classes)
    print("Number Of Neurons in each layer : ")
    print(neurons)
    file.close()

    fp = open('train.txt')  # open file on read mode
    lines = fp.read().split("\n")  # create a list containing all lines
    fp.close()  # close file\
    for line in lines:
        tokenizer = line.split()
        dataset.append(tokenizer)
        numberOfSample = numberOfSample + 1
    print("Number Of samples", end=" : ")
    print(numberOfSample)
    
    fp = open('test.txt')  # open file on read mode
    lines2 = fp.read().split("\n")  # create a list containing all lines
    fp.close()  # close file\
    for line in lines2:
        tokenizer = line.split()
        testset.append(tokenizer)
        numberOfTest = numberOfTest + 1
    print("Number Of test", end=" : ")
    print(numberOfTest)


def createNeurons():
    global nn, layers, neurons, wt
    print("creating Neurons")
    t = 0
    for i in range(0, layers + 1):
        layerNeurons = []
        for j in range(0, neurons[i + 1]):
            n = neuron(i, j, neurons[i])
            n.bias = random.random()
            for a in range(0, n.data):
                n.weight.append(random.random())
                # n.weight.append(wt[t])
                t = t+1
            layerNeurons.append(n)
        nn.append(layerNeurons)


def learn():
    global error
    for data in dataset:
        error = 0
        target = [0] * classes
        e = int(data[attributes])-1
        target[e] = 1
        fwdPropagation(data, 0)

        for j in range(0, classes):
            nn[layers][j].error = .5 * ((target[j] - nn[layers][j].out) ** 2)
            error = error + nn[layers][j].error

        backPropagation(target)
        updateWeight(data, .1)

def fwdPropagation(data, g):
    for i in range(0, layers + 1):
        for j in range(0, neurons[i + 1]):
            sum = nn[i][j].bias
            if i == 0:
                for k in range(0, nn[i][j].data):
                    sum = sum + float(data[k]) * nn[i][j].weight[k]
            else:
                for k in range(0, nn[i][j].data):
                    sum = sum + nn[i - 1][k].out * nn[i][j].weight[k]

            nn[i][j].net = sum
            nn[i][j].out = sigmoid(sum)
            if i == layers:
                if g ==1:
                    print(j+1,end=": ")
                    print(nn[i][j].out,end=" ")
        if g == 1:
            print()

def backPropagation(target):
    """Output Layer"""
    for j in range(0, classes):
        nn[layers][j].delta = (nn[layers][j].out - target[j]) * nn[layers][j].out * (1 - nn[layers][j].out)

    """Hidden Layer"""
    for r in range (layers, 0, -1):
        for i in range(0, neurons[r]):
            nn[r - 1][i].delta = 0
            for j in range(0, neurons[r+1]):
                nn[r-1][i].delta += (nn[r][j].delta * nn[r][j].weight[i]) * nn[r - 1][i].out * (1 - nn[r - 1][i].out)


def updateWeight(data, rate):
    """Output Layer"""
    for j in range(0, classes):
        nn[layers][j].bias -= rate * nn[layers][j].delta
        for i in range(0, neurons[layers]):
            nn[layers][j].weight[i] -= rate * nn[layers][j].delta * nn[layers - 1][i].out

    """Hidden Layer"""
    for r in range (0,layers):
        for j in range(0, neurons[r + 1]):
            for i in range(0, neurons[r]):
                if r == 0:
                    nn[r][j].weight[i] -= rate * nn[r][j].delta * float(data[i])
                else:
                    nn[r][j].weight[i] -= rate * nn[r][j].delta * nn[r-1][i].out

    for r in range(0, layers):
        for j in range(0, neurons[r + 1]):
            nn[r][j].bias -= rate * nn[r][j].delta


def test():
    errorNo = 0
    for data in testset:
        fwdPropagation(data, 0)
        max = -100
        result = 0
        for j in range(0, neurons[layers + 1]):
            # print(j,end=" ")
            # print(nn[layers][j].out, end=" ")
            if nn[layers][j].out > max:
                # print(j)
                # print(nn[layers][j].out,end=" ")
                max = nn[layers][j].out
                result = j+1
        if float(data[2]) != result:
            errorNo+=1

    return errorNo

def main():
    init()
    createNeurons()

    start = time.clock()
    for i in range(1,300):
        learn()
        print("Error : ", end="")
        print(error)
    end = time.clock()
    print(test())
    print(end - start)


if __name__ == "__main__":
    main()
