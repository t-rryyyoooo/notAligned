import sys
from pathlib import Path
import argparse
from functions import list_file
args = None

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("slicePath", nargs="+", help="~/Desktop/data/slice/summed_hist_1.0/path/validation1.txt")
    parser.add_argument("savePath", help="~/Desktop/data/textList")

    args = parser.parse_args()
    return args


def main(args):
    savePath = Path(args.savePath)
        
    if not savePath.parent.exists():
        print("Make ", str(savePath.parent))
        os.makedirs(str(savePath.parent), exist_ok = True)
    
    for path in args.slicePath:
        path = Path(path)
        if not path.exists():
            print("{} does not exist".format(str(path)))
            sys.exit()
        else:
            list_file(path, str(savePath))
            print("Writing {} to {}...".format(str(path), str(savePath)))
        

if __name__ == "__main__":
    args = parseArgs()
    main(args)
    
