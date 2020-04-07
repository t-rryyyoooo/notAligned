import sys
import SimpleITK as sitk
from cut import searchBound, caluculateClipSize
import re
from functions import createParentPath, write_file, resampleSize, getImageWithMeta
from pathlib import Path
import numpy as np

class slicer():
    def __init__(self, image, label, outputImageSize="256-256-3", widthSize=15, paddingSize=100, onlyCancer=False, noFlip=False):
        self.image = image
        self.label = label
        self.outputImageSize = outputImageSize
        self.widthSize = widthSize
        self.paddingSize = paddingSize
        self.onlyCancer = onlyCancer
        self.noFlip = noFlip

    def execute(self):
        self.meta = [{} for _ in range(2)]

        imageArray = sitk.GetArrayFromImage(self.image)
        labelArray = sitk.GetArrayFromImage(self.label)

    
        matchobj = re.match("([0-9]+)-([0-9]+)-([0-9]+)", self.outputImageSize)
        self.outputImageSize = [int(s) for s in matchobj.groups()]
        self.outputLabelSize = self.outputImageSize[:2]

        startIndex, endIndex = searchBound(labelArray, "sagittal")
        
        if len(startIndex) != 2:
            print("The patient has horse shoe kidney")
            sys.exit()

        largestKidneyLabelArray = []
        largestKidneyImageArray = []

        largestSlice = slice(endIndex[0], labelArray.shape[0])
        largestKidneyLabelArray.append(labelArray[largestSlice, ...])
        largestKidneyImageArray.append(imageArray[largestSlice, ...])
        self.meta[0]["largestSlice"] = largestSlice

        largestSlice = slice(0, startIndex[1])
        largestKidneyLabelArray.append(labelArray[largestSlice, ...])
        largestKidneyImageArray.append(imageArray[largestSlice, ...])
        self.meta[1]["largestSlice"] = largestSlice

        if not self.noFlip:
            largestKidneyLabelArray[1] = largestKidneyLabelArray[1][::-1, ...]
            largestKidneyImageArray[1] = largestKidneyImageArray[1][::-1, ...]

        self.clippedLabelArrayList = [[] for _ in range(2)]
        self.clippedStackedLabelArrayList = [[] for _ in range(2)]
        self.clippedImageArrayList = [[] for _ in range(2)]
        for i in range(2):
            print("Clipping image...")

            p = (self.paddingSize, self.paddingSize)
            largestKidneyLabelArray[i] = np.pad(largestKidneyLabelArray[i], [p, p, (0, 0)], "minimum")
            largestKidneyImageArray[i] = np.pad(largestKidneyImageArray[i], [p, p, (0, 0)], "minimum")

            sagittalIndex = caluculateClipSize(largestKidneyLabelArray[i], "sagittal", widthSize = self.widthSize)
            coronalIndex = caluculateClipSize(largestKidneyLabelArray[i], "coronal", widthSize = self.widthSize)

            sagittalDiff = sagittalIndex[1] - sagittalIndex[0]
            coronalDiff = coronalIndex[1] - coronalIndex[0]

            """ To clip into square. """
            diff = abs(sagittalDiff - coronalDiff)
            if sagittalDiff >= coronalDiff:
                if diff % 2 == 0:
                    coronalIndex[0] -= (diff // 2)
                    coronalIndex[1] += (diff // 2)
                else:
                    coronalIndex[0] -= (diff // 2)
                    coronalIndex[1] += (diff // 2) + 1

            else:
                if diff % 2 == 0:
                    sagittalIndex[0] -= (diff // 2)
                    sagittalIndex[1] += (diff // 2)
                else:
                    sagittalIndex[0] -= (diff // 2)
                    sagittalIndex[1] += (diff // 2) + 1

            sagittalSlice = slice(sagittalIndex[0], sagittalIndex[1])
            coronalSlice = slice(coronalIndex[0], coronalIndex[1])

            largestKidneyLabelArray[i] = largestKidneyLabelArray[i][sagittalSlice, coronalSlice, :]
            largestKidneyImageArray[i] = largestKidneyImageArray[i][sagittalSlice, coronalSlice, :]

            self.meta[i]["sagittalSlice"] = sagittalSlice
            self.meta[i]["coronalSlice"] = coronalSlice

            axialIndex = caluculateClipSize(largestKidneyLabelArray[i], "axial")
            axialSlice = slice(axialIndex[0], axialIndex[1])

            largestKidneyLabelArray[i] = largestKidneyLabelArray[i][..., axialSlice]
            largestKidneyImageArray[i] = largestKidneyImageArray[i][..., axialSlice]

            self.meta[i]["axialSlice"] = axialSlice

            minLabelArray = np.zeros_like(largestKidneyLabelArray[i][..., 0]) + largestKidneyLabelArray[i].min()
            minImageArray = np.zeros_like(largestKidneyImageArray[i][..., 0]) + largestKidneyImageArray[i].min()
            length = largestKidneyLabelArray[i].shape[2]
            margin = self.outputImageSize[2] // 2
            for x in range(length):
                clippedLabelArray = []
                clippedImageArray = []
                for y in range(-margin, margin + 1):
                    if 0 <= x + y < length:
                        clippedLabelArray.append(largestKidneyLabelArray[i][..., x + y])
                        clippedImageArray.append(largestKidneyImageArray[i][..., x + y])
                    else:
                        clippedLabelArray.append(minLabelArray)
                        clippedImageArray.append(minImageArray)

                clippedLabelArray = np.dstack(clippedLabelArray)
                clippedImageArray = np.dstack(clippedImageArray)

                self.clippedStackedLabelArrayList[i].append(clippedLabelArray)
                self.clippedImageArrayList[i].append(clippedImageArray)
                self.clippedLabelArrayList[i].append(largestKidneyLabelArray[i][..., x])

            print("Done")

        """ For resampling, get direction, spacing, origin and minimun value in image and label in 2D. """
        extractSliceFilter = sitk.ExtractImageFilter()
        size = list(self.image.GetSize())
        size[0] = 0
        index = (0, 0, 0)
        extractSliceFilter.SetSize(size)
        extractSliceFilter.SetIndex(index)
        self.sliceImage = extractSliceFilter.Execute(self.image)


        """ Transform image. """
        self.clippedLabelList = [[] for _ in range(2)]
        self.clippedImageList = [[] for _ in range(2)]
        for i in range(2):
            print("Transforming image...")
            size = []
            length = len(self.clippedLabelArrayList[i])
            for x in range(length):
                size.append(self.clippedLabelArrayList[i][x].shape)
                clippedLabel = getImageWithMeta(self.clippedLabelArrayList[i][x], self.sliceImage)
                clippedImage = getImageWithMeta(self.clippedImageArrayList[i][x], self.image)
                
                clippedLabel = resampleSize(clippedLabel, self.outputLabelSize[::-1], is_label = True)
                clippedImage = resampleSize(clippedImage, self.outputImageSize[::-1])
                self.clippedLabelList[i].append(clippedLabel)
                self.clippedImageList[i].append(clippedImage)

                clippedLabelArray = sitk.GetArrayFromImage(clippedLabel)
                clippedImageArray = sitk.GetArrayFromImage(clippedImage)

                self.clippedLabelArrayList[i][x] = clippedLabelArray
                self.clippedImageArrayList[i][x] = clippedImageArray

            self.meta[i]["size"] = size

            print("Done")


    
    def output(self, kind = "Array"):
        if kind == "Array":
            return self.clippedLabelArrayList, self.clippedImageArrayList

        elif kind == "Image":
            return self.clippedLabelList, self.clippedImageList

        else:
            print("Argument error kind = [Array / Image]")
            sys.exit()

    def save(self, savePath, patientID):
        if self.onlyCancer:
            print("Saving only images with cancer.")


        savePath = Path(savePath)
        saveImagePath = savePath / "image" / patientID / "dummy.mha"
        saveTextPath = savePath / "path" / (patientID + ".txt")

        if not saveImagePath.parent.exists():
            createParentPath(str(saveImagePath))

        if not saveTextPath.parent.exists():
            createParentPath(str(saveTextPath))

        for i in range(2):
            print("Saving images...")
            length = len(self.clippedLabelList[i])
            for x in range(length):
                if self.onlyCancer and not (self.clippedStackedLabelArrayList[i][x] == 2).any():
                    continue

                saveImagePath = savePath / "image" / patientID / "image_{}_{:02d}.mha".format(i, x)
                saveLabelPath = savePath / "image" / patientID / "label_{}_{:02d}.mha".format(i, x)

                sitk.WriteImage(self.clippedLabelList[i][x], str(saveLabelPath), True)
                sitk.WriteImage(self.clippedImageList[i][x], str(saveImagePath), True)

                write_file(str(saveTextPath) ,str(saveImagePath) + "\t" + str(saveLabelPath))

            print("Done")


    def restore(self, predictArrayList):
        """
        predictArray = [[] * length for _ in range(2)]
        predictArray.shape == clippedLabelArrayList.shape
        """

        labelArray = sitk.GetArrayFromImage(self.label)
        predictedArray = np.zeros_like(labelArray)

        print("Restoring image...")
        for i in range(2):
            largestSlice = self.meta[i]["largestSlice"]
            largestArray = predictedArray[largestSlice, ...]
            paddingSize = self.paddingSize
            p = (paddingSize, paddingSize)
            largestArray = np.pad(largestArray, [p, p, (0, 0)], "minimum")

            length = len(predictArrayList[i])
            for x in range(length):
                pre = getImageWithMeta(predictArrayList[i][x], self.sliceImage)
                size = self.meta[i]["size"][x]
                pre = resampleSize(pre, size, is_label=True)
                preArray = sitk.GetArrayFromImage(pre)

                predictArrayList[i][x] = preArray
                

            preArray = np.dstack(predictArrayList[i])

            sagittalSlice = self.meta[i]["sagittalSlice"]
            coronalSlice = self.meta[i]["coronalSlice"]
            axialSlice = self.meta[i]["axialSlice"]

            largestArray[sagittalSlice, coronalSlice, axialSlice] = preArray

            largestArray = largestArray[paddingSize : -paddingSize, paddingSize : -paddingSize, :]

            if not self.noFlip:
                if i == 1:
                    largestArray = largestArray[::-1, ...]

            predictedArray[largestSlice, ...] += largestArray

        print("Done")

        return predictedArray











            

