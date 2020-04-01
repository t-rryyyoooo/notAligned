import os
import numpy as np
import SimpleITK as sitk
import random
import tensorflow as tf
from pathlib import Path

def ReadSliceDataList(filename):
    datalist = []
    with open(filename) as f:
        for line in f:
            imagefile, labelfile = line.strip().split('\t')
            datalist.append((imagefile, labelfile))

    return datalist

### Ignore upper and lower limiits slices
# def ReadSliceDataList3ch_1ch(filename):
#     datalist = []
#     with open(filename) as f:
#         for line in f:
#             labelfile, imagefile = line.strip().split('\t')
#             datalist.append((imagefile, labelfile))
            
            
#     pathDicImg = {}#{~/case_00000/image0 : ~/case_00000/image0_00.mha}
#     labellist = {}
#     pathList = []#3枚ごとにまとめられたリスト(image,label)

#     #パスを同じ腎臓ごとにまとめる
#     for path in datalist:
#         dicPathI, filePathI = os.path.split(path[0])
#         fI,nameI = filePathI.split("_")
#         fPathI = os.path.join(dicPathI, fI)
        

#         if fPathI not in pathDicImg:
#             pathDicImg[fPathI] = []
#             labellist[fPathI] = []

#         pathDicImg[fPathI].append(path[0])

#         labellist[fPathI].append(path[1])

#     #同じ腎臓の中で、あるスライスと前後2枚をくっつける(path)

#     for (keyI, valueI),(labkey, labvalue) in zip(pathDicImg.items(), labellist.items()):
#         valueI = sorted(valueI)
#         labvalue = sorted(labvalue)
#         for x in range(1,len(valueI)-1):
#             pathList.append((valueI[x-1:x+2], labvalue[x]))
   
#     return pathList

def ReadSliceDataList3ch_1ch(filename):
    datalist = []
    with open(filename) as f:
        for line in f:
            labelfile, imagefile = line.strip().split('\t')
            datalist.append((imagefile, labelfile))
            
            
    pathDicImg = {}#{~/case_00000/image0 : ~/case_00000/image0_00.mha}
    labellist = {}
    pathList = []#3枚ごとにまとめられたリスト(image,label)

    #パスを同じ腎臓ごとにまとめる
    for path in datalist:
        dicPathI, filePathI = os.path.split(path[0])
        fI,nameI = filePathI.split("_")
        fPathI = os.path.join(dicPathI, fI)
        

        if fPathI not in pathDicImg:
            pathDicImg[fPathI] = []
            labellist[fPathI] = []

        pathDicImg[fPathI].append(path[0])

        labellist[fPathI].append(path[1])

    #同じ腎臓の中で、あるスライスと前後2枚をくっつける(path)

    for (keyI, valueI),(labkey, labvalue) in zip(pathDicImg.items(), labellist.items()):
        valueI = sorted(valueI)
        labvalue = sorted(labvalue)
        for x in range(len(valueI)):
            if x==0:
                
                pathList.append(([None] + valueI[x:x+2], labvalue[x]))
            elif x == (len(valueI) - 1):
                pathList.append((valueI[x-1:x+1] + [None], labvalue[x]))
            
            else:
                pathList.append((valueI[x-1:x+2], labvalue[x]))
   
    return pathList

def makeDict(datalist):
    imageDict = {}#{~/case_00000/image0 : ~/case_00000/image0_00.mha}
    labelDict = {}
    #パスを同じ腎臓ごとにまとめる

    for data in datalist:
        parentPath = Path(data[0]).parent        
        
        if str(parentPath) not in imageDict:
            imageDict[str(parentPath)] = []
            labelDict[str(parentPath)] = []

        imageDict[str(parentPath)].append(data[0])
        labelDict[str(parentPath)].append(data[1])
    return imageDict, labelDict

def make3ch_1chList(imageList, labelList):
    pathList = []
    length = len(imageList)

    imageList = sorted(imageList)
    labelPath = sorted(labelList)
    
    for x in range(length):
        if x == 0:
            pathList.append(([None] + imageList[x : x + 2], labelList[x]))
        
        elif x == length - 1:
            pathList.append((imageList[x - 1 : x + 1] + [None], labelList[x]))
        
        else:
            pathList.append((imageList[x - 1 : x + 2], labelList[x]))
        
    return pathList

def ReadSliceDataList6ch_1ch(filename1, filename2):
    dataList1 = []
    dataList2 = []
    
    pathList = []
    #Read datalist
    with open(filename1) as f:
        for line in f:
            labelfile, imagefile = line.strip().split('\t')
            dataList1.append((imagefile, labelfile))
    
    with open(filename2) as f:
        for line in f:
            labelfile, imagefile = line.strip().split('\t')
            dataList2.append((imagefile, labelfile))

    imageDict1, labelDict1 = makeDict(dataList1)
    imageDict2, labelDict2 = makeDict(dataList2)

    for id1 in imageDict1.keys():
        for id2 in imageDict2.keys():
            lastDict1 = id1.split("/")
            lastDict2 = id2.split("/")

            if lastDict1[-2] == lastDict2[-2]:
                if lastDict1[-1] != lastDict2[-1]:
                    pathList1 = make3ch_1chList(imageDict1[id1], labelDict1[id1])
                    pathList2 = make3ch_1chList(imageDict2[id2], labelDict2[id2])


                    for path1, path2 in zip(pathList1, pathList2):
                        imageList = path1[0] + path2[0]
                        labelList = path1[1]

                        pathList.append((imageList, labelList))
                        
    return pathList

def ReadSliceDataList3ch3ch_1ch(filename1, filename2):
    dataList1 = []
    dataList2 = []
    
    pathList = []
    #Read datalist
    with open(filename1) as f:
        for line in f:
            labelfile, imagefile = line.strip().split('\t')
            dataList1.append((imagefile, labelfile))
    
    with open(filename2) as f:
        for line in f:
            labelfile, imagefile = line.strip().split('\t')
            dataList2.append((imagefile, labelfile))

    imageDict1, labelDict1 = makeDict(dataList1)
    imageDict2, labelDict2 = makeDict(dataList2)

    for id1 in imageDict1.keys():
        for id2 in imageDict2.keys():
            lastDict1 = id1.split("/")
            lastDict2 = id2.split("/")

            if lastDict1[-2] == lastDict2[-2]:
                if lastDict1[-1] != lastDict2[-1]:
                    pathList1 = make3ch_1chList(imageDict1[id1], labelDict1[id1])
                    pathList2 = make3ch_1chList(imageDict2[id2], labelDict2[id2])


                    for path1, path2 in zip(pathList1, pathList2):
                        imageList = [path1[0], path2[0]]
                        labelList = path1[1]

                        pathList.append((imageList, labelList))
                        
    return pathList


def ImportImage(filename):
    image = sitk.ReadImage(filename)
    imagearry = sitk.GetArrayFromImage(image)
#    if image.GetNumberOfComponentsPerPixel() == 1:
#        imagearry = imagearry[..., np.newaxis]
    if len(imagearry.shape) != 3:
        imagearry = imagearry[..., np.newaxis]

    return imagearry

# Ignore the upper and lower slices
# def ImportImage3ch(pList):#[["case_00000/image0_00.mha", "case_00000/image0_01.mha", "case_00000/image0_02.mha"],\
#                           # ["case_00000/image0_01.mha", "case_00000/image0_02.mha", "case_00000/image0_03.mha"]...]
    
#     check = False
#     for x in pList:
#         img = sitk.ReadImage(x)
#         imgArray = sitk.GetArrayFromImage(img)
        
#         if not check:
#             check = True
#             stackedArray = imgArray

#         else:
#             stackedArray = np.dstack([stackedArray, imgArray])
    
    
#     return stackedArray

def ImportImage3ch(pList):#[["case_00000/image0_00.mha", "case_00000/image0_01.mha", "case_00000/image0_02.mha"],\
                          # ["case_00000/image0_01.mha", "case_00000/image0_02.mha", "case_00000/image0_03.mha"]...]
    
    stacked = False
    dummy = sitk.GetArrayFromImage(sitk.ReadImage(pList[1]))

    for x in pList:
        if x is None:
            imgArray = np.zeros_like(dummy) - 1024.0
            minval = -1024.0
        else:
            img = sitk.ReadImage(x)
            imgArray = sitk.GetArrayFromImage(img)

        if not stacked:
            stacked = True
            stackedArray = imgArray

        else:
            stackedArray = np.dstack([stackedArray, imgArray])
    
    
    return stackedArray

def ImportImage3ch3ch(pList):#[["case_00000/image0_00.mha", "case_00000/image0_01.mha", "case_00000/image0_02.mha"],\
                          # ["case_00000/image0_01.mha", "case_00000/image0_02.mha", "case_00000/image0_03.mha"]...]
    
    stacked = False
    dummy = sitk.GetArrayFromImage(sitk.ReadImage(pList[0][1]))
    stack = []
    for imageFile in pList:
        check = False
        for img in imageFile:
            if img is None:
                imagearry = np.zeros_like(dummy) - 1024.0
                minval = -1024.0
            else:
                image = sitk.ReadImage(img)
                minval = GetMinimumValue(image)

                imagearry = sitk.GetArrayFromImage(image)
            
            if not check:
                check = True
                stackedArray = imagearry

            else:
                stackedArray = np.dstack([stackedArray, imagearry])

        stack.append(stackedArray)

    stackedArray = np.stack(stack, axis=-1)
 
   
    
    return stackedArray

def GetInputShapes(filenamepair):
    image = ImportImage(filenamepair[0])
    label = ImportImage(filenamepair[1])
    return (image.shape, label.shape)

def GetInputShapes3ch3ch(filenamepair):
    image = ImportImage3ch3ch(filenamepair[0])
    label = ImportImage(filenamepair[1])
    return (image.shape, label.shape)


def GetMinimumValue(image):
    minmax = sitk.MinimumMaximumImageFilter()
    minmax.Execute(image)
    return minmax.GetMinimum()


def Affine(t, r, scale, shear, c, dimension):
    a = sitk.AffineTransform(dimension)
    a.SetCenter(c)
    
    if dimension == 2:
        a.Rotate(0,1,r)
        a.Shear(0,1,shear[0])
        a.Shear(1,0,shear[1])
        a.Scale(scale)
        
    elif dimension == 3:
        a.Rotate(1, 0, r)
        a.Shear(1, 0, shear[0])
        a.Shear(0, 1, shear[1])
        a.Scale((scale, scale, 1))
        
    a.Translate(t)
    return a


def Transforming(image, bspline, affine, interpolator, minval):
    # B-spline transformation
    if bspline is not None:
        transformed_b = sitk.Resample(image, bspline, interpolator, minval)

    # Affine transformation
        transformed_a = sitk.Resample(transformed_b, affine, interpolator, minval)

    else:
        transformed_a = sitk.Resample(image, affine, interpolator, minval)

    return transformed_a

# original ImportImageTransformed
#def ImportImageTransformed(imagefile, labelfile):
#    sigma = 4
#    translationrange = 5 # [mm]
#    rotrange = 5 # [deg]
#    shearrange = 1/16 
#    scalerange = 0.05
#
#    image = sitk.ReadImage(imagefile)
#    label = sitk.ReadImage(labelfile)
#
#    # B-spline parameters
#    bspline = None
#    bspline = sitk.BSplineTransformInitializer(image, [5,5])
#    p = bspline.GetParameters()
#    numbsplineparams = len(p)
#    coeff = np.random.normal(0, sigma, numbsplineparams)
#    bspline.SetParameters(coeff)
#
#    # Affine parameters
#    translation = np.random.uniform(-translationrange, translationrange, 2)
#    rotation = np.radians(np.random.uniform(-rotrange, rotrange))
#    shear = np.random.uniform(-shearrange, shearrange, 2)
#    scale = np.random.uniform(1-scalerange, 1+scalerange)
#    center = np.array(image.GetSize()) * np.array(image.GetSpacing()) / 2
#    affine = Affine(translation, rotation, scale, shear, center)
#
#    minval = GetMinimumValue(image)
#
#    transformed_image = Transforming(image, bspline, affine, sitk.sitkLinear, minval)
#    transformed_label = Transforming(label, bspline, affine, sitk.sitkNearestNeighbor, 0)
#
#    imagearry = sitk.GetArrayFromImage(transformed_image)
#    imagearry = imagearry[..., np.newaxis]
#    labelarry = sitk.GetArrayFromImage(transformed_label)
#
#    return imagearry, labelarry

def makeAffineParameters(image, translationRange, rotateRange, shearRange, scaleRange):
    dimension = image.GetDimension()
    translation = np.random.uniform(-translationRange, translationRange, dimension)
    rotation = np.radians(np.random.uniform(-rotateRange, rotateRange))
    shear = np.random.uniform(-shearRange, shearRange, 2)
    scale = np.random.uniform(1-scaleRange, 1+scaleRange)
    center = (np.array(image.GetSize()) * np.array(image.GetSpacing()) / 2)[::-1]
    
    
    return [translation, rotation, scale, shear, center, dimension]

# image 3ch label 1ch
def ImportImageTransformed(imagefile, labelfile):
    sigma = 0
    translationrange = 0 # [mm]
    rotrange = 180 # [deg]
    shearrange = 0
    scalerange = 0.05
    bspline = None
    
    image = sitk.ReadImage(imagefile)
    
    imageParameters = makeAffineParameters(image, translationrange, rotrange, shearrange, scalerange)
    imageAffine = Affine(*imageParameters)

    minval = GetMinimumValue(image)
    transformed_image = Transforming(image, bspline, imageAffine, sitk.sitkLinear, minval)
    imageArray = sitk.GetArrayFromImage(transformed_image)
    
    if image.GetDimension() != 3:
        imagearry = imagearry[..., np.newaxis]
        
        
    label = sitk.ReadImage(labelfile)   
    
    
    labelParameters = imageParameters
    dimension = label.GetDimension()
    labelParameters[0] = np.random.uniform(-translationrange, translationrange, dimension)
    labelParameters[4] = np.array(label.GetSize()) * np.array(label.GetSpacing()) / 2
    labelParameters[5] = dimension

    
    
    labelAffine = Affine(*labelParameters)
    transformed_label = Transforming(label, bspline, labelAffine, sitk.sitkNearestNeighbor, 0)
    labelArray = sitk.GetArrayFromImage(transformed_label)

    
    return imageArray, labelArray



# only rotatation and scaling and do not use b-spline
# def ImportImageTransformed(imagefile, labelfile):
#     sigma = 0
#     translationrange = 0 # [mm]
#     rotrange = 15 # [deg]
#     shearrange = 0
#     scalerange = 0.05

#     image = sitk.ReadImage(imagefile)
#     label = sitk.ReadImage(labelfile)
    
 
#     bspline = None

#     # Affine parameters
#     translation = np.random.uniform(-translationrange, translationrange, 2)
#     rotation = np.radians(np.random.uniform(-rotrange, rotrange))
#     shear = np.random.uniform(-shearrange, shearrange, 2)
#     scale = np.random.uniform(1-scalerange, 1+scalerange)
#     center = np.array(image.GetSize()) * np.array(image.GetSpacing()) / 2
#     affine = Affine(translation, rotation, scale, shear, center)

#     minval = GetMinimumValue(image)

#     transformed_image = Transforming(image, bspline, affine, sitk.sitkLinear, minval)
#     transformed_label = Transforming(label, bspline, affine, sitk.sitkNearestNeighbor, 0)

#     imagearry = sitk.GetArrayFromImage(transformed_image)
#     if len(imagearry.shape) != 3:
#         imagearry = imagearry[..., np.newaxis]
#     labelarry = sitk.GetArrayFromImage(transformed_label)

#     return imagearry, labelarry

def Transforming3d(image, affine, interpolator, minval):
    #Affine transformation
    #transformed_a = sitk.Resample(transformed_b, affine, interpolator, minval)
    transformed_a = sitk.Resample(image, affine, interpolator, minval)

    return transformed_a

###Ignore upper and lower limit slices
# def ImportImageTransformed3d(imagefile, labelfile):
#     translationrange = 0 # [mm]
#     rotrange = 15 # [deg]
#     shearrange = 0
#     scalerange = 0.05

#     # Affine parameters
#     translation = np.random.uniform(-translationrange, translationrange, 2)
#     rotation = np.radians(np.random.uniform(-rotrange, rotrange))
#     shear = np.random.uniform(-shearrange, shearrange, 2)
#     scale = np.random.uniform(1-scalerange, 1+scalerange)
    
#     #label(1ch)
#     label = sitk.ReadImage(labelfile)
#     center = np.array(label.GetSize()) * np.array(label.GetSpacing()) / 2

    
#     affine = Affine(translation, rotation, scale, shear, center)
#     transformed_label = Transforming3d(label, affine, sitk.sitkNearestNeighbor, 0)
#     labelarry = sitk.GetArrayFromImage(transformed_label)
    
#     #image(3ch)
#     check = False
#     for img in imagefile:
#         image = sitk.ReadImage(img)
#         minval = GetMinimumValue(image)

#         transformed_image = Transforming3d(image, affine, sitk.sitkLinear, minval)
#         imagearry = sitk.GetArrayFromImage(transformed_image)
        
#         if not check:
#             check = True
#             stackedArray = imagearry

#         else:
#             stackedArray = np.dstack([stackedArray, imagearry])
    
    

#     return stackedArray, labelarry


def ImportImageTransformed3d(imagefile, labelfile):
    translationrange = 0 # [mm]
    rotrange = 15 # [deg]
    shearrange = 0
    scalerange = 0.05

    # Affine parameters
    translation = np.random.uniform(-translationrange, translationrange, 2)
    rotation = np.radians(np.random.uniform(-rotrange, rotrange))
    shear = np.random.uniform(-shearrange, shearrange, 2)
    scale = np.random.uniform(1-scalerange, 1+scalerange)
    
    #label(1ch)
    label = sitk.ReadImage(labelfile)
    center = np.array(label.GetSize()) * np.array(label.GetSpacing()) / 2

    
    affine = Affine(translation, rotation, scale, shear, center)
    transformed_label = Transforming3d(label, affine, sitk.sitkNearestNeighbor, 0)
    labelarry = sitk.GetArrayFromImage(transformed_label)
    
    #image(3ch)
    check = False
    for img in imagefile:
        if img is None:
            imagearry = np.zeros_like(labelarry) - 1024.0
            minval = -1024.0
        else:
            image = sitk.ReadImage(img)
            minval = GetMinimumValue(image)

            transformed_image = Transforming3d(image, affine, sitk.sitkLinear, minval)
            imagearry = sitk.GetArrayFromImage(transformed_image)
        
        if not check:
            check = True
            stackedArray = imagearry

        else:
            stackedArray = np.dstack([stackedArray, imagearry])
    
    

    return stackedArray, labelarry

def ImportImageTransformed3ch3ch(imageFileList, labelfile):
    translationrange = 0 # [mm]
    rotrange = 15 # [deg]
    shearrange = 0
    scalerange = 0.05

    # Affine parameters
    translation = np.random.uniform(-translationrange, translationrange, 2)
    rotation = np.radians(np.random.uniform(-rotrange, rotrange))
    shear = np.random.uniform(-shearrange, shearrange, 2)
    scale = np.random.uniform(1-scalerange, 1+scalerange)
    
    #label(1ch)
    label = sitk.ReadImage(labelfile)
    center = np.array(label.GetSize()) * np.array(label.GetSpacing()) / 2

    
    affine = Affine(translation, rotation, scale, shear, center)
    transformed_label = Transforming3d(label, affine, sitk.sitkNearestNeighbor, 0)
    labelarry = sitk.GetArrayFromImage(transformed_label)
    
    #image(3ch * 3ch)
    stack = []
    for imageFile in imageFileList:
        check = False
        for img in imageFile:
            if img is None:
                imagearry = np.zeros_like(labelarry) - 1024.0
                minval = -1024.0
            else:
                image = sitk.ReadImage(img)
                minval = GetMinimumValue(image)

                transformed_image = Transforming3d(image, affine, sitk.sitkLinear, minval)
                imagearry = sitk.GetArrayFromImage(transformed_image)
            
            if not check:
                check = True
                stackedArray = imagearry

            else:
                stackedArray = np.dstack([stackedArray, imagearry])

        stack.append(stackedArray)

    stackedArray = np.stack(stack, axis=-1)
    print(stackedArray.shape)


    return stackedArray, labelarry

# original make 3ch when reading images
#def ImportBatchArray(datalist, batch_size = 32, apply_augmentation = False):
#    while True:
#        indices = list(range(len(datalist)))
#        random.shuffle(indices)
#        
#
#        if apply_augmentation:
#            for i in range(0, len(indices), batch_size):
#               imagelabellist = [ ImportImageTransformed3d(datalist[idx][0], datalist[idx][1]) for idx in indices[i:i+batch_size] ]
#               imagelist, labellist = zip(*imagelabellist)
#               onehotlabellist = tf.keras.utils.to_categorical(labellist,num_classes=3)
#               
#               yield (np.array(imagelist), np.array(onehotlabellist))
#
#        else:
#            for i in range(0, len(indices), batch_size):
#                imagelist = np.array([ ImportImageh(datalist[idx][0]) for idx in indices[i:i+batch_size] ])
#
#                onehotlabellist = np.array([ tf.keras.utils.to_categorical(ImportImage(datalist[idx][1]),num_classes=3) for idx in indices[i:i+batch_size] ])
#
#                yield (imagelist, onehotlabellist)

# Read images made as 3ch
def ImportBatchArray(datalist, batch_size = 32, apply_augmentation = False):
    while True:
        indices = list(range(len(datalist)))
        random.shuffle(indices)
        

        if apply_augmentation:
            for i in range(0, len(indices), batch_size):
               imagelabellist = [ ImportImageTransformed(datalist[idx][0], datalist[idx][1]) for idx in indices[i:i+batch_size] ]
               imagelist, labellist = zip(*imagelabellist)
               onehotlabellist = tf.keras.utils.to_categorical(labellist,num_classes=3)
               
               yield (np.array(imagelist), np.array(onehotlabellist))

        else:
            for i in range(0, len(indices), batch_size):
                imagelist = np.array([ ImportImage(datalist[idx][0]) for idx in indices[i:i+batch_size] ])

                onehotlabellist = np.array([ tf.keras.utils.to_categorical(ImportImage(datalist[idx][1]),num_classes=3) for idx in indices[i:i+batch_size] ])

                yield (imagelist, onehotlabellist)

def ImportBatchArray3ch3ch(datalist, batch_size = 32, apply_augmentation = False):
    while True:
        indices = list(range(len(datalist)))
        random.shuffle(indices)
        

        if apply_augmentation:
            for i in range(0, len(indices), batch_size):
               imagelabellist = [ ImportImageTransformed3ch3ch(datalist[idx][0], datalist[idx][1]) for idx in indices[i:i+batch_size] ]
               #print("apply_augmentation")
               imagelist, labellist = zip(*imagelabellist)
               onehotlabellist = tf.keras.utils.to_categorical(labellist,num_classes=3)
               #print("patch shape1 :",imagelist[0].shape)
               yield (np.array(imagelist), np.array(onehotlabellist))

        else:
            for i in range(0, len(indices), batch_size):
                imagelist = np.array([ ImportImage3ch3ch(datalist[idx][0]) for idx in indices[i:i+batch_size] ])

                onehotlabellist = np.array([ tf.keras.utils.to_categorical(ImportImage(datalist[idx][1]),num_classes=3) for idx in indices[i:i+batch_size] ])

                yield (imagelist, onehotlabellist)
