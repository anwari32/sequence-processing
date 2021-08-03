#
#
#

import sys
if __name__ == "__main__":
    for arg in sys.argv[1:]:
        arr = arg.split('=')
        print('arg {} => {} {}'.format(arg, arr[0], arr[1]))