import numpy as np
import cv2
import SimpleITK as sitk
import sys

def searchBound(labelArray, axis):
    '''
    Argument
    axis: "sagittal" or "coronal" or "axial"
    
    Return the kidney borders in axis derection.
    
    startIndex: [Index of the beginning of the first kidney area, second, ...]
    endIndex: [Index of the ending of the first kidney area, second, ...]
    
    '''
    encounter = False
    startIndex= []
    endIndex = []
    if axis == 'sagittal':
        length, _, _ = labelArray.shape

        for l in range(length):
            is_something = (labelArray[l, ...] != 0).any()
            if is_something and not encounter:
                startIndex.append(l)
                encounter = True
            elif not is_something and encounter:
                endIndex.append(l)
                encounter = False

        if not startIndex:
            print("Nothing is in this array")
            sys.exit()

        if encounter:
            endIndex.append(length - 1)

    if axis == 'coronal':
        _, length, _ = labelArray.shape

        for l in range(length):
            is_something = (labelArray[:, l, :] != 0).any()
            if is_something and not encounter:
                startIndex.append(l)
                encounter = True
            elif not is_something and encounter:
                endIndex.append(l)
                encounter = False

        if not startIndex:
            print("Nothing is in this array")
            sys.exit()

        if encounter:
            endIndex.append(length - 1)


    if axis == 'axial':
        _, _, length = labelArray.shape

        for l in range(length):
            is_something = (labelArray[..., l] != 0).any()
            if is_something and not encounter:
                startIndex.append(l)
                encounter = True
            elif not is_something and encounter:
                endIndex.append(l)
                encounter = False

        if not startIndex:
            print("Nothing is in this array")
            sys.exit()

        if encounter:
            endIndex.append(length - 1)

    return np.array(startIndex), np.array(endIndex)

def caluculateClipSize(labelArray, axis, widthSize=None):
    startIndex, endIndex = searchBound(labelArray, axis)
    start = startIndex[0]
    end = endIndex[-1]
    if widthSize is None:
        widthSize = (end - start) // 10

    if axis == "sagittal":
        axis = 0
    elif axis == "coronal":
        axis = 1
    elif axis == "axial":
        axis = 2
    else:
        print("Error : {} direction doesn't exist.".format(axis))
        sys.exit()

    if len(labelArray.shape) <= axis:
        print("Dimension error.")
        sys.exit()

    length = labelArray.shape[axis]

    start -= widthSize
    if start < 0:
        start = 0
    end += widthSize
    if end >= length:
        end = length

    return start, end

def caluculateArea(imageArray):
    area = (imageArray > 0).sum()

    return area


def getCenterOfGravity(imageArray):
    """ 
    imageArray : labelArray
    """
    moment = np.where(imageArray > 0)
    size = (imageArray > 0).sum()
    center = [m.sum() // size for m in moment]

    return np.array(center)


