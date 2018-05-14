import random
import math

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
        #temporary weight container to update weights in back propagation phase
        self.tempWeight = []
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
    
    # for ii in range(0, numberOfSample):
    #     for j in range(0, attributes +1):
    #         print(dataset[ii][j], end=" ")
    #     print("")
    # print(float(dataset[0][0]))
    # backPropagation(dataset)
    # print(globals())


def createNeurons():
    global nn, layers, neurons, wt
    print("creating Neurons")

    t = 0
    for i in range(0, layers + 1):
        layerNeurons = []
        for j in range(0, neurons[i + 1]):
            n = neuron(i, j, neurons[i])
            for a in range(0, neurons[i]):
                n.weight.append(random.random())
                n.bias = random.random()
                # n.weight.append(wt[t])
                n.tempWeight = n.weight.copy()
                t = t + 1
                # print(i, end="")
                # print(j, end="")
                # print(a)
            layerNeurons.append(n)
        nn.append(layerNeurons)
        # layerNeurons.clear()

    # for i in range(0, layers + 1):
    #     for j in range(0, neurons[i + 1]):
    #         print(nn[i][j].layer, end=" , ")
    #         print(nn[i][j].position, end=" , ")
    #         print(nn[i][j].data, end="  ")
    #         print(nn[i][j].weight)
    #     print()


def learn():
    global error
    for data in dataset:
        error = 0
        """must change it  to 0 #########################################################################################"""
        target = [0] * classes
        e = int(data[attributes])-1
        """must change it  to 1 #########################################################################################"""
        target[e] = 1
        # print(target)
        fwdPropagation(data)

        for j in range(0, classes):
            # print(target[j], end=" ")
            # print(nn[layers][j].out, end=" ")
            nn[layers][j].error = .5 * ((target[j] - nn[layers][j].out) ** 2)
            # print(nn[layers][j].error, end=" ")
            error = error + nn[layers][j].error
        # print("Error : ", end="")
        # print(error)
        backPropagation(data, target)
        updateWeight(data, .1)

def fwdPropagation(data):
    for i in range(0, layers + 1):
        for j in range(0, neurons[i + 1]):
            sum = nn[i][j].bias
            if i == 0:
                for k in range(0, attributes):
                    sum = sum + float(data[k]) * nn[i][j].weight[k]
            else:
                for k in range(0, neurons[i]):
                    sum = sum + nn[i - 1][k].out * nn[i][j].weight[k]

            nn[i][j].net = sum
            nn[i][j].out = sigmoid(sum)
        #     if i == layers:
        #         print(j+1,end=": ")
        #         print(nn[i][j].out,end=" ")
        # print()
def backPropagation(target):
    """Output Layer"""
    for j in range(0, neurons[layers+1]):
        nn[layers][j].delta = (nn[layers][j].out - target[j]) * nn[layers][j].out * (1 - nn[layers][j].out)
        # """Temporary Update"""
        # for i in range(0, len(nn[layers][j].tempWeight)):
        #     nn[layers][j].tempWeight[i] -= .5 * nn[layers][j].delta * nn[layers - 1][j].out

    """Hidden Layer"""
    for r in range (layers, 0, -1):
        for i in range(0, neurons[r]):
            for j in range (0 , neurons[r+1]):
                nn[r - 1][i].delta = 0
                nn[r-1][i].delta += (nn[r][j].delta * nn[r][j].weight[i]) * nn[r - 1][i].out * (1 - nn[r - 1][i].out)
                # """Temporary Update"""
                # if r != 1:
                #     nn[r][j].tempWeight[i] -= .5 * nn[r-1][j].delta * nn[r-2][j].out
                #     # print(nn[r - 1][j].tempWeight[i])
                # else:
                #     nn[r][j].tempWeight[i] -= .5 * nn[r - 1][j].delta * float(data[i])
                # # print(nn[r - 1][j].tempWeight[i])

    # """Updating Weights"""
    # for i in range(0, layers + 1):
    #     for j in range(0, neurons[i + 1]):
    #         nn[i][j].weight = nn[i][j].tempWeight.copy()


def updateWeight(data, rate):
    """Output Layer"""
    for j in range(0, neurons[layers+1]):
        for i in range(0, neurons[layers]):
            nn[layers][j].weight[i] -= rate * nn[layers][j].delta * nn[layers - 1][i].out
            nn[layers][j].bias -= rate * nn[layers][j].delta

    """Hidden Layer"""
    for r in range (0,layers-1):
        for i in range(0, neurons[r]):
            for j in range(0, neurons[r+1]):
                if r == 0:
                    nn[r][j].tempWeight[i] -= rate * nn[r][j].delta * float(data[i])
                else:
                    nn[r][j].tempWeight[i] -= rate * nn[r][j].delta * nn[r-1][i].out
                nn[r][j].bias -= rate * nn[r][j].delta
                
def test():
    for data in testset:
        fwdPropagation(data)
        max = -100
        result = 0
        for j in range(0, neurons[layers + 1]):
            if nn[layers][j].out > max:
                result = j + 1
        print(result)

def main():
    init()
    createNeurons()
    for i in range(1,200):
        learn()
        # print("Error : ", end="")
        print(i)


    test()
    # nn[0][0].bias = .35
    # nn[0][1].bias = .35
    # nn[1][0].bias = .6
    # nn[1][1].bias = .6
    # print(error / numberOfSample)
    # for i in range(0, layers+1):
    #     for j in range(0, neurons[i + 1]):
    #         print(nn[i][j].layer, end=" , ")
    #         print(nn[i][j].position, end="  ")
    #         # print(nn[i][j].data, end=" , ")
    #         print(nn[i][j].weight)
    #
    #     print(nn[1][0].out,end=" ")
    #     print(nn[1][1].out)


    # fwdPropagation(dataset[0],[.01,.99])
    # print(nn[1][1].net, end=" ")
    # print(nn[1][1].out, end=" ")

    # for i in range(0, layers + 1):
    #     for j in range(0, neurons[i + 1]):
    #         print(nn[i][j].layer, end=" ")
    #         print(nn[i][j].position, end=" ")
    #         for a in range(0, neurons[i]):
    #             print(nn[i][j].weight[a], end=" ")
    #         print("")
    #     print("")

if __name__ == "__main__":
    main()
