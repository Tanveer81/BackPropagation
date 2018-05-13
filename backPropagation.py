import random

global layers, neurons, attributes, classes
neurons = []
numberOfSample = 0
dataSet = []
nn = []

class neuron:
    def __init__(self, layer, position, inputs):
        self.layer = layer
        self.position = position
        self.inputs = inputs
        self.weight = []
        self.delta = None
        self.out = None
        self.net = None

def init():
    # in parameter.txt 1st , 2nd and 3rd line has layers,attributes and classes and next lines has number of neurons
    # in each layer
    global dataSet, neurons, layers, attributes, classes, numberOfSample
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

    fp = open('test.txt')  # open file on read mode
    lines = fp.read().split("\n")  # create a list containing all lines
    fp.close()  # close file\
    for line in lines:
        tokenizer = line.split()
        dataSet.append(tokenizer)
        numberOfSample = numberOfSample + 1
    print("Number Of samples", end=" : ")
    print(numberOfSample)
    # for ii in range(0, numberOfSample):
    #     for j in range(0, attributes + 1):
    #         print(dataSet[ii][j], end=" ")
    #     print("")
    # print(float(dataSet[0][0]))
    # backPropagation(dataset)
    # print(globals())

def createNeurons():
    global nn, layers, neurons
    print("creating Neurons")

    for i in range(0, layers+1):
        layerNeurons = []
        for j in range(0, neurons[i + 1]):
            n = neuron(i, j, neurons[i])
            for a in range(0, neurons[i]):
                n.weight.append(random.random())
            # print(n.weight)
            layerNeurons.append(n)
        nn.append(layerNeurons)
        # layerNeurons.clear()

    # for i in range(0, layers):
    #     for j in range(0, neurons[i + 1]):
    #         print(nn[i][j].layer, end=" , ")
    #         print(nn[i][j].position, end=" , ")
    #         print(nn[i][j].inputs, end=" , ")
    #         print(nn[i][j].weight)
    #     print()


def fwdPropagation(input):
    for i in range(0, layers+1):
        for j in range(0, neurons[i + 1]):
            sum = 0
            for k in neurons[i]:
                sum
            nn[i][j].net = sum


def backPropagation():
    # print(dat)
    print("Hello from a function")

def main():
    init()
    createNeurons()
    # for i in range (0,50):print(random.random())


if __name__ == "__main__":
    main()
