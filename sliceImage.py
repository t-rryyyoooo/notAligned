import SimpleITK as sitk
import argparse
from slicer import slicer as sler
from pathlib import Path
from functions import getImageWithMeta

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("imageDirectory", help="$HOME/Desktop/data/kits19/case_00000")
    parser.add_argument("saveSlicePath", help="$HOME/Desktop/data/slice/hist_0.0", default=None)
    parser.add_argument("--outputImageSize", help="256-256-3", default="256-256-3")
    parser.add_argument("--widthSize", default=15, type=int)
    parser.add_argument("--paddingSize", default=100, type=int)
    parser.add_argument("--onlyCancer", action="store_true") 
    parser.add_argument("--noFlip", action="store_true")

    args = parser.parse_args()
    return args

def main(args):

    metaData = [{} for _ in range(2)]
    labelFile = Path(args.imageDirectory) / 'segmentation.nii.gz'
    imageFile = Path(args.imageDirectory) / 'imaging.nii.gz'

    """ Read image and label. """
    label = sitk.ReadImage(str(labelFile))
    image = sitk.ReadImage(str(imageFile))
    
    slicer = sler(image, label, outputImageSize = args.outputImageSize, widthSize = args.widthSize, paddingSize = args.paddingSize, onlyCancer = args.onlyCancer, noFlip = args.noFlip)

    patientID = args.imageDirectory.split("/")[-1]
    slicer.execute()
    slicer.save(args.saveSlicePath, patientID)

    """
    For restoration test. 

    predictArrayList, _ = slicer.output("Array")

    predictArray = slicer.restore(predictArrayList)

    predict = getImageWithMeta(predictArray, label)
    sitk.WriteImage(predict, "test/test.mha")
    """




if __name__ == '__main__':
    args = ParseArgs()
    main(args)
