# Neural Networks
A planned set of neural networks used to analyze different structures and different types over a range of problems.

## Required Modules & Programming Languages
This project requires the following programming languages and modules:

- Java (1.8)
- Python (2.7)
  * SciPy
  * NumPy

The modules NumPy and SciPy can be obtained through Anaconda at https://www.continuum.io/downloads.

## Script.py
Used to communicate with ForwardNN.py with a pair of Input and Output folders.

### Python Script Instructions
Our script can be ran by running in a command line `python script.py InputFile OutputFile Weights`

1. InputFile
  1. The InputFile is our list of training inputs that are each 35 characters long (each character separated by spaces). It matches the format of our GUI output having '0 1 0 0 1 ...' for each output.
2. OutputFile
  1. The OutputFile has many values as our InputFiles, except each being 10 characters with 0 defining not a specific and 1 defining as a specific digit (e.g. 0 1 0 0 0 0 0 0 0 0 is the digit 1)
3. Weights
  1. The Weights are a list of weights that can be used by the Neural Network. In script.py, our neural network uses float-type weights that has 450 weights total.

In the script program, it will pause to explain each step of the Neural Network in training and setting. After pressing enter for each pause and the Neural Network is trained, you may then submit one of the following comands to the script:

1. `forward guiInput`
  1. The *forward* command takes in one input file that was either created by the GUI or of the same format. This inserts the input into the Neural Network and returns an output of 10 numbers, each representing a possibility of being a specific number.
  2. If the GUI doesn't match the probability of what you expect, you may then use the prompt to define it as one of the 10 digits and have it be added to our input/output data to be trained on its next run.
2. `save Weights`
  1. The *save* command takes in one weight file that will save the current weights of our Neural Network. This file can then be used in the next run by adding it as the 3rd argument to our original python command.
  2. Can also use `save default` to save neural network weights as *Weights-(in,hidden...,out)*, where each number in the tuple is a layer with *n* nodes.
3. `exit`
  1. The *exit* command exits and closes the script.

# Problems
Used to test different Neural Networks.

## PatternCell
In a 7x5 grid (35 cells), classify symbols defined by the user using an algorithm that has pre-classified training. We will be aiming to train the 10 digits of base 10, and hope to implement in such a way that new patterns can be trained.

### GUI Instructions
There are 3 sections in the GUI:

1. File location *TextField*
  1. The top file location textfield defines where the grid will be stored after confirming the grid. This can be used to forward inputs in our script.py.
2. Grid
  1. The 7x5 grid has individual cells that may be clicked on to select and clicked again to deselect. By doing this, you can create different symbols, primarily the 10 digits (0, 1, ..., 9) that can then be stored to be run.
3. Confirm button
  1. Lastly, our confirmation button has the grid stored as a series of 0s and 1s separated by spaces in the file location specified. It also clears the grid for further use.
