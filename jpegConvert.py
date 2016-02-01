# !/usr/bin/env python

import argparse
import ForwardNN
import os.path
from PIL import Image

def visual(argv):

    for fn in os.listdir(os.getcwd() + '/TrainingSet/'):
        #print("what is up")
        print(fn)
        image = Image.open("TrainingSet/" + fn)
        inputs = argv.inputs[0]
        image.show()
        pix = list(image.getdata())
        first = True
        for pixel in pix:
            if first:
                first = False
            else:
                inputs.write(" ")
            if pixel[0] < 255:
                inputs.write("1")
            else:
                inputs.write("0")
        inputs.write("\n")

        number = int(fn[0])

        for x in range(1,number):
            argv.outputs[0].write("0 ")
        argv.outputs[0].write("1")
        for x in range(number, 9):
            argv.outputs[0].write(" 0")
        argv.outputs[0].write("\n")



# If this script is being run, as opposed to imported, run the run function.
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Takes in picture and shows it")
    parser.add_argument("picture", metavar="I", type=str, nargs=1, help="Input picture")
    parser.add_argument("number", metavar="Q", type=int, nargs=1, help="number represented")
    parser.add_argument("inputs", metavar="P", type=argparse.FileType('w'), nargs=1, help="Inputs file")
    parser.add_argument("outputs", metavar="R", type=argparse.FileType('w'), nargs=1, help="Outputs file")
    parser.add_argument("--visual", dest="visual", action="store_const", const=visual, default=visual, help="Runs through Neural Network with visual")
args = parser.parse_args()
args.visual(args)