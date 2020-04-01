import SimpleITK as sitk
import numpy as np
import argparse
from functions import createParentPath, write_file, resampleSize, getImageWithMeta
from cut import *
from pathlib import Path
import re
import sys
from tqdm import tqdm

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("imageDirectory", help="$HOME/Desktop/data/kits19/case_00000")
    parser.add_argument("-s", "--saveSlicePath", help="$HOME/Desktop/data/slice/hist_0.0", default=None)
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
    labelArray = sitk.GetArrayFromImage(label)
    imageArray = sitk.GetArrayFromImage(image)

    """ Get output size """
    matchobj = re.match("([0-9]+)-([0-9]+)-([0-9]+)", args.outputImageSize)
    if matchobj is None:
        print("[ERROR] Invalid patch size : {}".format(args.outputImageSize))
        sys.exit()

    outputImageSize = [int(s) for s in matchobj.groups()]
    outputLabelSize = outputImageSize[:2]
    print("Output patch size : ", outputImageSize)

    widthSize = args.widthSize
    paddingSize = args.paddingSize

    """ Get minimum value in image and label. """
    if image.GetNumberOfComponentsPerPixel() == 1:
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(image)
        imageMinval = minmax.GetMinimum()
    else:
        imageMinval = None

    if label.GetNumberOfComponentsPerPixel() == 1:
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(label)
        labelMinval = minmax.GetMinimum()
    else:
        labelMinval = None

    #print("Whole size: ",labelArray.shape)

    """ Extract the maximum region with a kidney. """
    kidneyStartIndex = []
    kidneyEndIndex = []
    kidneyStartIndex, kidneyEndIndex = searchBound(labelArray, "sagittal")
    #print(kidneyStartIndex)

    if len(kidneyStartIndex) != 2:
        print("The patient has horse shoe kidney")
        sys.exit() 
    
    #print(kidneyStartIndex, kidneyEndIndex)

    largestKidneyROILabel = [] #[[First kidney region], [Second kidney region],..]
    largestKidneyROIImage = []
    

    firstKidneyIndex = kidneyEndIndex[0]
    secondKidneyIndex = kidneyStartIndex[1]

    largestSlice = slice(firstKidneyIndex, labelArray.shape[0])
    metaData[0]["largestSlice"] = largestSlice
    largestKidneyROILabel.append(labelArray[largestSlice, ...])
    largestKidneyROIImage.append(imageArray[largestSlice , ...])

    largestSlice = slice(0, secondKidneyIndex)
    metaData[1]["largestSlice"] = largestSlice 
    largestKidneyROILabel.append(labelArray[largestSlice,  ...])
    largestKidneyROIImage.append(imageArray[largestSlice, ...])

    """ Slice module. """
    axialSize = labelArray.shape[2]
    roiLabelArrayList = [[] for _ in range(2)]
    roiStackedLabelArrayList = [[] for _ in range(2)]
    roiImageArrayList = [[] for _ in range(2)]
    for i in range(2):
        """ Reverse left kidney """
        if args.noFlip:
            print("The kidney doesn't flip.")
        else:
            if i == 1:
                largestKidneyROILabel[i] = largestKidneyROILabel[i][::-1,:,:]
                largestKidneyROIImage[i] = largestKidneyROIImage[i][::-1,:,:]

        p = (paddingSize, paddingSize)
        largestKidneyROILabel[i] = np.pad(largestKidneyROILabel[i], [p, p, (0, 0)], "minimum")
        largestKidneyROIImage[i] = np.pad(largestKidneyROIImage[i], [p, p, (0, 0)], "minimum")

        metaData[i]["paddingSize"] = paddingSize
        
        """ Clip image and label in axial direction. """
        axialTopIndex, axialBottomIndex = caluculateClipSize(largestKidneyROILabel[i], "axial")
        axialSlice = slice(axialTopIndex, axialBottomIndex)
        largestKidneyROILabel[i] = largestKidneyROILabel[i][..., axialSlice]
        largestKidneyROIImage[i] = largestKidneyROIImage[i][..., axialSlice]

        metaData[i]["axialSlice"] = axialSlice

        #print("cutted size_"+str(i)+": ",largestKidneyROILabel[i].shape)
        

        axialSize = largestKidneyROILabel[i].shape[2]

        """ Search the slice with the largest kidney. """
        area = []
        for x in range(axialSize):
            area.append(caluculateArea(largestKidneyROILabel[i][..., x]))
            maxArea = np.argmax(area)
        
        """ Caluculate the width and height of the largest kidney. """
        maxAreaLabelArray = largestKidneyROILabel[i][..., maxArea]
        s, e = caluculateClipSize(maxAreaLabelArray[..., np.newaxis], "sagittal")
        width = e - s
        s, e = caluculateClipSize(maxAreaLabelArray[..., np.newaxis], "coronal")
        height = e - s

        wh = max(width, height)
        

        margin = outputImageSize[2] // 2
       
        """ Align slices per patch. """
        top, bottom = caluculateClipSize(largestKidneyROILabel[i], "axial", widthSize=0)
        topSliceArray = largestKidneyROILabel[i][..., top]
        bottomSliceArray = largestKidneyROILabel[i][..., bottom - 1]

        check = False
        xMeta = []
        yMeta = []
        for x in tqdm(range(axialSize), desc="Slicing images...", ncols=60):
            a = caluculateArea(largestKidneyROILabel[i][..., x])
            if a == 0:
                if not check:
                    sliceLabelArray = topSliceArray
                else:
                    sliceLabelArray = bottomSliceArray
            else:
                sliceLabelArray = largestKidneyROILabel[i][..., x]
                check = True
                

            center = getCenterOfGravity(sliceLabelArray)
            x0 = center[0] - wh // 2
            x1 = center[0] + wh // 2
            y0 = center[1] - wh // 2
            y1 = center[1] + wh // 2

            minLabelArray = np.zeros((x1 - x0, y1 - y0)) + labelMinval
            minImageArray = np.zeros((x1 - x0, y1 - y0)) + imageMinval
     
            xSlice = slice(x0, x1)
            ySlice = slice(y0, y1)
            xMeta.append(xSlice)
            yMeta.append(ySlice)

            roiImageArray = []
            roiStackedLabelArray = []
            for y in range(-margin, margin + 1):
                if 0 <= x + y < axialSize:
                    roiImageArray.append(largestKidneyROIImage[i][xSlice, ySlice, x + y])
                    roiStackedLabelArray.append(largestKidneyROILabel[i][xSlice, ySlice, x + y])

                else:
                    roiImageArray.append(minImageArray)
                    roiStackedLabelArray.append(minLabelArray)

            roiLabelArray = largestKidneyROILabel[i][xSlice, ySlice, x]
            roiStackedLabelArray = np.dstack(roiStackedLabelArray)
            roiImageArray = np.dstack(roiImageArray)

            roiLabelArrayList[i].append(roiLabelArray)
            roiStackedLabelArrayList[i].append(roiStackedLabelArray)
            roiImageArrayList[i].append(roiImageArray)

        metaData[i]["xSlice"] = xMeta
        metaData[i]["ySlice"] = yMeta

    """ For resampling, get direction, spacing, origin and minimun value in image and label in 2D. """
    extractSliceFilter = sitk.ExtractImageFilter()
    size = list(image.GetSize())
    size[0] = 0
    index = (0, 0, 0)
    extractSliceFilter.SetSize(size)
    extractSliceFilter.SetIndex(index)
    sliceImage = extractSliceFilter.Execute(image)

    """ Transform array into image. """
    roiImageList = [[] for _ in range(2)]
    roiLabelList = [[] for _ in range(2)]
    for i in range(2):
        length = len(roiLabelArrayList[i])
        for x in tqdm(range(length), desc="Transforming images...", ncols=60):
            roiImageList[i].append(getImageWithMeta(roiImageArrayList[i][x], image))
            roiLabelList[i].append(getImageWithMeta(roiLabelArrayList[i][x], sliceImage))
           
            roiImageList[i][x] = resampleSize(roiImageList[i][x], outputImageSize[::-1])
            roiLabelList[i][x] = resampleSize(roiLabelList[i][x], outputLabelSize[::-1], is_label=True)

    """ Save module. """
    if args.saveSlicePath is not None:
        patientID = args.imageDirectory.split('/')[-1]
        saveImagePath = Path(args.saveSlicePath) / 'image' / patientID / "dummy.mha"
        saveTextPath = Path(args.saveSlicePath) / 'path' / (patientID + '.txt')

        """ Make parent path. """
        if not saveImagePath.parent.exists():
            createParentPath(str(saveImagePath))
        
        if not saveTextPath.parent.exists():
            createParentPath(str(saveTextPath))

        """ Save image and label. """
        for i in range(2):
            length = len(roiLabelList[i])
            for x in tqdm(range(length), desc="Saving images...", ncols=60):
                if args.onlyCancer and not (roiStackedLabelArrayList[i][x] == 2).any():
                    continue

                saveImagePath = Path(args.saveSlicePath) / 'image' / patientID / "image_{}_{:02d}.mha".format(i,x)
                saveLabelPath = Path(args.saveSlicePath) / 'image' / patientID / "label_{}_{:02d}.mha".format(i,x)
                saveTextPath = Path(args.saveSlicePath) / 'path' / (patientID + '.txt')

                sitk.WriteImage(roiImageList[i][x], str(saveImagePath), True)
                sitk.WriteImage(roiLabelList[i][x], str(saveLabelPath), True)

                write_file(str(saveTextPath), str(saveImagePath) + "\t" + str(saveLabelPath))
            if args.onlyCancer:
                print("Saving images with cancer.")

    else:
        print("Do not save image.")
        """
        About metaData[i]
        {largestSlice, padidngSize, axialSlice, xSlice, ySlice} * length
        """
        return roiImageList, roiLabelList, metaData



if __name__ == '__main__':
    args = ParseArgs()
    main(args)
