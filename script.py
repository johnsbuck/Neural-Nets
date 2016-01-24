#!/usr/bin/env python

import ForwardNN
import numpy as np
import argparse
import os.path

np.set_printoptions(threshold=np.nan)

# For doing inputs in both Python2 and Python3
try:
    input = raw_input
except NameError:
    pass

# Reads in file as an array of arrays.
def readFile(name):
    fileX = []
    for line in name:
        fileX.append([float(y) for y in line.split(' ')])
    return np.array(fileX)

def read_params(name):
    prev = []
    firstline = name.readline().replace("\n","").split(" ")
    print(firstline)
    layer = ()
    for arg in firstline:
        print("ARG: " + arg)
        layer = layer + (int(arg),)
    prev.append(layer)
    prev.append(np.fromstring(name.readline(), dtype=float, sep=" "))
    return prev

def forward(testfile, infile, outfile, NN):
    try:
        inputFile = open(testfile, 'r')
        input = readFile(inputFile)
        inputFile.close()
    except IOError:
        print("ERROR: Invalid file input")
        return False

    print(input)
    print(str(NN.forward(input)))
    print(np.around(NN.forward(input), decimals=2))

    # Additional checker tool, allows for a forwarded file to be added to test data.
    valid = raw_input("Is this the expected output? (y/n): ")
    if valid == "n":
        actualOutput = raw_input("What is the correct output: ")
        try:
            with open(infile, "a") as trainInput:
                with open(testfile, "r") as newInput:
                    trainInput.write(newInput.read())
                    trainInput.close()
                    newInput.close()
            with open(outfile, "a") as trainOutput:
                newData = ""
                trainOutput.write(actualOutput + "\n")
                trainOutput.close()
                print("Is added to training data. Will not be implemented until restart.")
        except ValueError:
            print("ERROR: Invalid input.")
            return False
    return True

def save(filename, NN, layerNodes):
    str_layer = str(layerNodes).replace("(", "").replace(")", "").replace(",","").replace("L","")
    if filename == "default":
        filename = "Weights-" + str_layer

    with open(filename, "w") as weights:
        print("Saving weights in " + filename)
        weights.write(str_layer)
        weights.write("\n")
        print(str(NN.get_params()))
        weights.write(str(NN.get_params()).replace("[ ", "").replace("[", "")
                        .replace("]","").replace("\n", "").replace("   "," ")
                        .replace("  ", " "))
        weights.close()
    return True

def train(max_count, NN, layerNodes, thresh, X, Y):
    bestNN = ForwardNN.ForwardNN(layerNodes, thresh)
    bestNN.set_params(NN.get_params())

    # This is our monte carlo. Continually trains networks until one statisfies our conditions.
    if max_count > 0:
        count = 0
        while np.isnan(bestNN.cost_function(X, Y)) or count < max_count:
            train = ForwardNN.Trainer(NN)
            train.train(X, Y)
            if bestNN.cost_function(X, Y) > NN.cost_function(X, Y):
                bestNN = ForwardNN.ForwardNN(layerNodes)
                bestNN.set_params(NN.get_params())
                print("New cost: " + str(bestNN.cost_function(X, Y)))
            count += 1
            NN = ForwardNN.ForwardNN(layerNodes)
            print("Current cycle: " + str(count))

        NN.set_params(bestNN.get_params())
    return NN


# In place of a main, as python lacks one. Call run to read in the training data and train,
# As well as run a monte carlo to find a satisfactory network. Then it will sit in a loop
# waiting for input commands.
def visual(argv):
    print(argv)
    X = readFile(argv.input[0])
    argv.input[0].close()
    Y = readFile(argv.output[0])
    argv.output[0].close()

    if not X.shape[0] == Y.shape[0]:
        print("ERROR: Need equal number of Inputs and Outputs")
        return False
    else:
        print(str(X.shape[0]) + " data points given.")

    # Creates a trainer and network, created as a X input to a set of hidden to Y output.
    print("This neural network will take in " + str(X.shape[1]) +
     " inputs and will output " + str(Y.shape[1]) + " floats.")

    layerNodes = None
    params = None

    if not argv.params is None:
        params = read_params(argv.params)
        argv.params.close()
        if params[0][0] == X.shape[1] and params[0][len(params[0]) - 1] == Y.shape[1]:
            layerNodes = params[0]
            NN = ForwardNN.ForwardNN(layerNodes, argv.thresh)
            NN.set_params(params[1])
        else:
            print("ERROR: Invalid layers given by parameters")
            return
    elif not argv.layers == None:
        layerNodes = (X.shape[1],) + tuple(argv.layers,) + (Y.shape[1],)
        NN = ForwardNN.ForwardNN(layerNodes, argv.thresh)
        print("Neural Network Layers: " + str(layerNodes))
    else:
        layerNodes = (X.shape[1],) + (Y.shape[1],)
        NN = ForwardNN.ForwardNN(layerNodes, argv.thresh)
        print("Neural Network Layers: " + str(layerNodes))


    # As the print states, runs a forward operation on the network with it's randomly generated weights.
    print("Now printing an initial run on the " + str(X.shape[0]) + " base inputs and their cost function.")
    test = NN.forward(X)
    print(type(test))
    print(str(test))
    print("Cost Function: " + NN.cost_function_type(X, Y))
    print("Threshold Function: " + NN.thresh_func_type())
    print("Cost: " + str(NN.cost_function(X, Y)))

    # Trains the network using the trainer and test data.
    NN = train(argv.cycle[0], NN, layerNodes, argv.thresh, X, Y)

    # Print the results of the training and monte carlo.
    print("Now printing the final match results.")
    print(np.around(NN.forward(X), decimals=2))
    print("Cost function: " + str(NN.cost_function(X, Y)))

    # Input control loop.
    while 1:
        ans = input("\nInput a one of the following commands: " +
        "\n\tforward <file>\n\tsave <file>\n\texit" +
        "\n\nCommand: ")

        # When a user inputs forward and a file, read in the file and run forward using it.
        if ans.split(' ')[0] == 'forward' and len(ans.split(' ')) > 1:
            forward(ans.split(' ')[1], argv.input[0].name, argv.output[0].name, NN)
        elif ans.split(' ')[0] == 'save':
            if len(ans.split(' ')) <= 1:
                save("default", NN, layerNodes)
            else:
                save(ans.split(' ')[1], NN, layerNodes)
        # Exit.
        elif ans.split(' ')[0] == 'exit':
            break
        # Completely invalid input.
        else:
            print("#NopeNopeNope.")


# If this script is being run, as opposed to imported, run the run function.
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Takes in information to give to Neural Network")
    parser.add_argument("input", metavar="I", type=argparse.FileType('r'), nargs=1, help="Input data points")
    parser.add_argument("output", metavar="O", type=argparse.FileType('r'), nargs=1, help="Expected output data")
    parser.add_argument("cycle", metavar="C", type=int, nargs=1,
                        help="Number of cycles used for training. Best cost function will be taken.")
    parser.add_argument("--params", metavar="W", type=argparse.FileType('r'), nargs='?',
                        help="File containing pre-existing weights and layers")
    parser.add_argument("--layers", type=int, nargs='*', const=None, default=None,
                        help="List of ints, containing the # of nodes for each hidden layer (Overwritten by params)")
    parser.add_argument("--visual", dest="visual", action="store_const", const=visual, default=visual,
                        help="Runs through Neural Network with visual")
    parser.add_argument("--sigmoid", dest="thresh", action="store_true")
    parser.add_argument("--tanh", dest="thresh", action="store_false")
    parser.set_defaults(thresh=True)

args = parser.parse_args()
args.visual(args)

# META
__author__ = 'Bill Clark & John Bucknam'
