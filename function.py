"""
This file is used as notes for how the arguments work in python command line.
"""
import sys, getopt
def main(args):
    input_file = ""
    output_file = ""
    try:
        opts, args = getopt.getopt(args, "i:o:", ["input=", "output="])
        for option, argument in opts:
            if option in ['-i', '--input']:
                input_file = argument
            elif option in ['-o', '--output']:
                output_file = argument
            else:
                print('option = {}, argument = {}'.format(option, argument))
        print('input file = {}'.format(input_file))
        print('output file = {}'.format(output_file))
    except getopt.GetoptError:
        print('Argument Error!')
        sys.exit(2)
    return True

if __name__ == "__main__":
    print(sys.argv[1:])
    main(sys.argv[1:])