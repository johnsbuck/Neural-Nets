#!/usr/bin/env python

import ForwardNN
import numpy as np
import sys
import os.path

np.set_printoptions(threshold='nan')

#Reads in file as an array of arrays.
def readFile(name):
    X = []
    if os.path.isfile(name):
        f = open(name,'r')
        for line in f:
            X.append([float(y) for y in line.split(' ')])
        f.close()
        return np.array(X)
    return False

def readWeights(name):
    if os.path.isfile(name):
        f = open(name, 'r')
        prevWeights = np.fromstring(f.readline(), dtype=float, sep=" ")
        f.close()
        return prevWeights
    return False

#In place of a main, as python lacks one. Call run to read in the training data and train,
#As well as run a monte carlo to find a satisfactory network. Then it will sit in a loop
#waiting for input commands.
def run():
    #reads in the test data files.
    X = readFile(sys.argv[1])
    Y = readFile(sys.argv[2])
    if len(sys.argv) >= 4:
        weights = readWeights(sys.argv[3])
    if not X.shape[0] == Y.shape[0]:
        print "Need equal number of Inputs and Outputs"
        return
    else:
        print str(X.shape[0]) + " data points given."

    #Creates a trainer and network, created as a X input to a set of hidden to Y output.
    print("This neural network will take in " + str(X.shape[1]) +
     " inputs and will output " + str(Y.shape[1]) + " floats.")

    numHiddenLayers = ""
    while not numHiddenLayers.isdigit():
        numHiddenLayers = raw_input("How many hidden layers do you want: ")
    numHiddenLayers = int(numHiddenLayers)

    layerNodes = (X.shape[1],)

    print("Enter the number of nodes for each hidden layer")
    for i in range(1, numHiddenLayers+1):
        hiddenLayerNodes = ""
        while not hiddenLayerNodes.isdigit():
            hiddenLayerNodes = raw_input("Hidden Layer " + str(i) + ": ")
        layerNodes = layerNodes + (int(hiddenLayerNodes),)
        print layerNodes

    layerNodes = layerNodes + (Y.shape[1],)
    print layerNodes

    NN = ForwardNN.ForwardNN(layerNodes)
    if len(sys.argv) >= 4:
        NN.set_params(weights)

    train = ForwardNN.Trainer(NN)

    # As the print states, runs a forward operation on the network with it's randomly generated weights.
    raw_input("Now printing an initial run on the " + str(X.shape[0]) + " base inputs and their cost function.")
    print(NN.forward(X))
    print("Cost Function: " + NN.cost_function_type(X, Y))
    print("Cost: " + str(NN.cost_function(X, Y)))

    #Trains the network using the trainer and test data.
    max_count = ""
    while not max_count.isdigit():
        max_count = raw_input("Training the network, then training on a monte carlo.\n# of Cycles: ")
    max_count = int(max_count)

    bestNN = ForwardNN.ForwardNN(layerNodes)
    bestNN.set_params(NN.get_params())

    #This is our monte carlo. Continually trains networks until one statisfies our conditions.
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
    #Print the results of the training and monte carlo.
    raw_input("Now printing the final match results.")
    print(np.around(NN.forward(X), decimals=2))
    print("Cost function: " + str(NN.cost_function(X, Y)))

    #Input control loop.
    while 1:
        ans = raw_input("input a command: forward <file>, save <file>, or exit: ")

        # When a user inputs forward and a file, read in the file and run forward using it.
        if ans.split(' ')[0] == 'forward' and len(ans.split(' ')) > 1:
                print ans.split(' ')
                input = readFile(ans.split(' ')[1])
                print(np.around(NN.forward(input), decimals=2))

                #Additional checker tool, allows for a forwarded file to be added to test data.
                valid = raw_input("Is this the expected output? (y/n): ")
                if valid == "n":
                    actualOutput = raw_input("What is the correct output: ")
                    try:
                        actualOutput = int(actualOutput)
                        with open(sys.argv[1], "a") as trainInput:
                            with open(ans.split(' ')[1], "r") as newInput:
                                trainInput.write(newInput.read())
                                trainInput.close()
                                newInput.close()
                        with open(sys.argv[2], "a") as trainOutput:
                            newData = ""
                            zeroes = 9 - actualOutput
                            while actualOutput > 0:
                                newData += "0 "
                                actualOutput = actualOutput - 1
                            newData += "1"
                            if zeroes == 0:
                                newData += "\n"
                            else:
                                newData += " "
                                while zeroes > 0:
                                    if zeroes == 1:
                                        newData += "0\n"
                                    else:
                                        newData += "0 "
                                    zeroes = zeroes - 1
                            trainOutput.write(newData)
                            trainOutput.close()
                            print("Is added to training data. Will not be implemented until restart.")
                    except ValueError:
                        print("Invalid input.")
        elif ans.split(' ')[0] == 'save' and len(ans.split(' ')) > 1:
            print ans.split(' ')

            filename = ans.split(' ')[1]
            if filename == "default":
                filename = "Weights-" + str(layerNodes)

            with open(filename, "w") as weights:
                print "Saving weights in " + filename
                weights.write(str(NN.get_params()).replace("[", "").replace("]","")
                              .replace("\n", "").replace("   "," ").replace("  ", " "))
                weights.close()
        # Exit.
        elif ans.split(' ')[0] == 'exit':
                break
        # Completely invalid input.
        else:
            print "#NopeNopeNope."

#If this script is being run, as opposed to imported, run the run function.
if __name__ == '__main__':
    if len(sys.argv) >= 2:
        run()
    else:
        print "Invalid number of inputs. (Requires expected input & expected output files)"

__author__ = 'Bill Clark & John Bucknam'
