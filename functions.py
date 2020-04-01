import numpy as np
import os
import SimpleITK as sitk
import tensorflow as tf
import matplotlib.pyplot as plt

def getImageWithMeta(imageArray, refImage, spacing=None, origin=None, direction=None):
    image = sitk.GetImageFromArray(imageArray)
    if spacing is None:
        spacing = refImage.GetSpacing()
    if origin is None:
        origin = refImage.GetOrigin()
    if direction is None:
        direction = refImage.GetDirection()

    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction)

    return image


def DICE(trueLabel, result):
    intersection=np.sum(np.minimum(np.equal(trueLabel,result),trueLabel))
    union = np.count_nonzero(trueLabel)+np.count_nonzero(result)
    dice = 2 * intersection / (union + 10**(-9))
   
    return dice

def DICEVersion2(trueLabel1, result1, trueLabel2, result2):
    
    intersection1 = np.sum(np.minimum(np.equal(trueLabel1, result1), trueLabel1))
    union1 = np.count_nonzero(trueLabel1) + np.count_nonzero(result1)

    intersection2 = np.sum(np.minimum(np.equal(trueLabel2, result2), trueLabel2))
    union2 = np.count_nonzero(trueLabel2) + np.count_nonzero(result2)
    
    dice = 2 * (intersection1 + intersection2) / (union1 + union2 + 10**(-9))
    return dice


def caluculateAVG(num):
    if len(num) == 0:
        return 1.0
    
    else: 
        nsum = 0
        for i in range(len(num)):
            nsum += num[i]

        return nsum / len(num)
    
def outputMedium(num):
    number = sorted(num)
    l = len(number)
    half = l//2
    if l % 2 != 0:
        
        return number[half]
    
    else:
        
        return (number[half - 1] + number[half]) / 2
    
def outputMax(num):
    number = sorted(num)
    
    return number[-1]


def createParentPath(filepath):
    head, _ = os.path.split(filepath)
    if len(head) != 0:
        os.makedirs(head, exist_ok = True)

def write_file(file_name, text):
    if not os.path.exists(file_name):
        createParentPath(file_name)
    with open(file_name, mode='a') as file:
        #print(text)
        file.write(text + "\n")

# 3D -> 3D or 2D -> 2D
def Resampling(image, newsize, roisize, origin = None, is_label = False):
   #isize = image.GetSize()
    ivs = image.GetSpacing()
    
    if image.GetNumberOfComponentsPerPixel() == 1:
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(image)
        minval = minmax.GetMinimum()
    else:
        minval = None
    
    osize = newsize
    

    ovs = [ vs * s / os for vs, s, os in zip(ivs, roisize, osize) ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(osize)
    if origin is not None:
        resampler.SetOutputOrigin(origin)
    else:
        resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputSpacing(ovs)
    if minval is not None:
        resampler.SetDefaultPixelValue(minval)
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    resampled = resampler.Execute(image)

    return resampled

# 3D -> 3D or 2D -> 2D
def resampleSize(image, newSize, is_label = False):
    originalSpacing = image.GetSpacing()
    originalSize = image.GetSize()

    if image.GetNumberOfComponentsPerPixel() == 1:
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(image)
        minval = minmax.GetMinimum()
    else:
        minval = None


    newSpacing = [osp * os / ns for osp, os, ns in zip(originalSpacing, originalSize, newSize)]
    newOrigin = image.GetOrigin()
    newDirection = image.GetDirection()

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(newSize)
    resampler.SetOutputOrigin(newOrigin)
    resampler.SetOutputDirection(newDirection)
    resampler.SetOutputSpacing(newSpacing)
    if minval is not None:
        resampler.SetDefaultPixelValue(minval)
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    resampled = resampler.Execute(image)

    return resampled

#3D -> 2D in axis direction
def ResamplingInAxial(image, idx, newSize, is_label = False):
    extractSliceFilter = sitk.ExtractImageFilter()
    size = list(image.GetSize())
    size[0] = 0
    extractSliceFilter.SetSize(size)
    index = (idx, 0, 0)
    extractSliceFilter.SetIndex(index)
    sliceImage = extractSliceFilter.Execute(image)

    if image.GetNumberOfComponentsPerPixel() == 1:
        minmax = sitk.MinimumMaximumImageFilter()
        minmax.Execute(image)
        minval = minmax.GetMinimum()
    else:
        minval = None
 

    originalSpacing = sliceImage.GetSpacing()
    originalSize = sliceImage.GetSize()

    newSpacing = [originalSize[0] * originalSpacing[0] / newSize[0], 
                  originalSize[1] * originalSpacing[1] / newSize[1]]

    resampler = sitk.ResampleImageFilter()

    if minval is not None:
        resampler.SetDefaultPixelValue(minval)
        
    resampler.SetOutputSpacing(newSpacing)
    resampler.SetOutputOrigin(sliceImage.GetOrigin())
    resampler.SetOutputDirection(sliceImage.GetDirection())
    resampler.SetSize(newSize)
    
    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    sliceImageResampled = resampler.Execute(sliceImage)

    return sliceImageResampled


def readlines_file(file_name):
    # 行毎のリストを返す
    with open(file_name, 'r') as file:
        return file.readlines()


def save_file(file_name, text):
    with open(file_name, 'a') as file:
        file.write(text + "\n")
        
def list_file(file_name,savefile):
    cal1 = readlines_file(file_name)
    # 改行を削除
    cal1 = list(map(lambda x: x.strip("\n"), cal1))
    for line in cal1:
        save_file(savefile, line)

def caluculateTime( start, end):
    tt = end-start
    hour = int(tt/3600)
    mini = int((tt-hour*3600)/60)
    sec = int(tt - hour*3600 - mini*60)
    print("time: {}:{:2d}:{:2d}".format(hour, mini, sec))

def dice(y_true, y_pred):
    K = tf.keras.backend

    eps = K.constant(1e-6)
    truelabels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(predictions, truelabels), tf.int32), truelabels)), tf.float32)
    union = tf.count_nonzero(predictions, dtype=tf.float32) + tf.count_nonzero(truelabels, dtype=tf.float32)
    dice = 2 * intersection / (union + eps)
    return dice

def cancer_dice(y_true, y_pred):
    K = tf.keras.backend

    eps = K.constant(1e-6)
    truelabels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    
    truelabel = K.cast(K.equal(truelabels, 2), tf.int32)##ガンだけ
    prediction = K.cast(K.equal(predictions, 2), tf.int32)

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(prediction, truelabel), tf.int32), truelabel)), tf.float32)
    union = tf.count_nonzero(prediction, dtype=tf.float32) + tf.count_nonzero(truelabel, dtype=tf.float32)
    dice = 2 * intersection / (union + eps)
    return dice

def kidney_dice(y_true, y_pred):#canver
    K = tf.keras.backend

    eps = K.constant(1e-6)
    truelabels = tf.argmax(y_true, axis=-1, output_type=tf.int32)
    predictions = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    
    truelabel = K.cast(K.equal(truelabels, 1), tf.int32)##腎臓だけ
    prediction = K.cast(K.equal(predictions, 1), tf.int32)

    intersection = K.cast(K.sum(K.minimum(K.cast(K.equal(prediction, truelabel), tf.int32), truelabel)), tf.float32)
    union = tf.count_nonzero(prediction, dtype=tf.float32) + tf.count_nonzero(truelabel, dtype=tf.float32)
    dice = 2 * intersection / (union + eps)
    return dice

def penalty_categorical(y_true,y_pred):
    K = tf.keras.backend
    
    array_tf = tf.convert_to_tensor(y_true,dtype=tf.float32)
    pred_tf = tf.convert_to_tensor(y_pred,dtype=tf.float32)

    epsilon = K.epsilon()

    result = tf.reduce_sum(array_tf,[0,1,2,3])
    #result = tf.reduce_sum(array_tf,[0,1,2])

    #result_pow = tf.pow(result,1.0/3.0)
    result_pow = tf.math.log(result)

    weight_y = result_pow / tf.reduce_sum(result_pow)

    k_dice = kidney_dice(y_true, y_pred)
    c_dice = cancer_dice(y_true, y_pred)

    return (-1) * tf.reduce_sum( 1 / (weight_y + epsilon) * array_tf * tf.math.log(pred_tf + epsilon),axis=-1) #+ (1 - k_dice) + (1 - c_dice)

def advancedSettings(xlabel, ylabel, fontsize=20):
    #plt.figure(figsize=(10,10))
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    #plt.xticks(left + width/2,left)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()
    
    return 

def saveImage(saveImg, img, savePath):
    #saveImg = sitk.GetImageFromArray(saveImgArray)
    saveImg.SetDirection(img.GetDirection())
    saveImg.SetOrigin(img.GetOrigin())
    saveImg.SetSpacing(img.GetSpacing())
    
    print('Saving image to {}...'.format(savePath))
    sitk.WriteImage(saveImg, savePath, True)
    print("Done.")
    return

def printchk(x):
    for s, v in globals().items():
        if id(v) == id(x):
            print('{} : {}'.format(s, x))
            break
            
    return 

def CenterOfGravity(imgArray):
    y, x, z = imgArray.shape
    
    Y, X, Z = np.mgrid[:y, :x, :z]
    
    Sum = np.where(imgArray > 0, True, False).sum()
    
    Ax = np.sum(imgArray * X) / Sum
    Ay = np.sum(imgArray * Y) / Sum
    Az = np.sum(imgArray * Z) / Sum
    
    return np.array([Ay, Ax, Az])
