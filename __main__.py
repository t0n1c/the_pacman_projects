import sys

from .autograder import main as autograder_main


__doc__ = "" # main function MANAGER

def main(argv):
    autograder_main(argv)


if __name__ == '__main__':
    main(sys.argv)
