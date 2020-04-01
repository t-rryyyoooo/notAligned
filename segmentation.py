import SimpleITK as sitk
import numpy as np
import argparse
import os
import sys
import tensorflow as tf
from functions import createParentPath, getImageWithMeta, resampleSize, cancer_dice, kidney_dice, penalty_categorical
from pathlib import Path
from sliceImage import main as sliceImage
from tqdm import tqdm

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()
    """ For main in sliceImage.py """
    parser.add_argument("imageDirectory", help="$HOME/Desktop/data/kits19/case_00000")
    parser.add_argument("--widthSize", default=15, type=int)
    parser.add_argument("--paddingSize", default=100, type=int)
    parser.add_argument("--outputImageSize", help="256-256-3", default="256-256-3")
    parser.add_argument("-s", "--saveSlicePath", help="For main in sliceImage.py, it is not used, but needs writing.", default=None)
    parser.add_argument("--onlyCancer", action="store_true", help="You do not use it.")
    parser.add_argument("--noFlip", action="store_true")

    """ For main in segmentation.py """
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("savePath", help="Segmented label file.(.mha)")
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    parser.add_argument("-b", "--batchsize", help="Batch size", default=1, type=int)

    
    args = parser.parse_args()
    return args

def main(_):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    modelweightfile = os.path.expanduser(args.modelweightfile)

    with tf.device('/device:GPU:{}'.format(args.gpuid)):
        print('loading U-net model {}...'.format(modelweightfile), end='', flush=True)
        #with open(args.modelfile) as f:
        #     model = tf.compat.v1.keras.models.model_from_yaml(f.read())
        # model.load_weights(args.modelweightfile)
        model = tf.compat.v1.keras.models.load_model(modelweightfile,
        custom_objects={'penalty_categorical' : penalty_categorical, 'kidney_dice':kidney_dice, 'cancer_dice':cancer_dice})

        print('done')


    """ Slice image. """
    imageList, labelList, metaData = sliceImage(args)

    """ Segmentation module """
    segmentedArrayList = [[] for _ in range(2)]
    for i in range(2):
        length = len(imageList[i])
        for x in tqdm(range(length), desc="Segmenting images...", ncols=60):
            imageArray = sitk.GetArrayFromImage(imageList[i][x])

            imageArray = imageArray[np.newaxis, ...]
            
            segmentedArray = model.predict(imageArray, batch_size=args.batchsize, verbose=0)
            segmentedArray = np.squeeze(segmentedArray)
            segmentedArray = np.argmax(segmentedArray, axis=-1).astype(np.uint8)
            segmentedArrayList[i].append(segmentedArray)
            

            """ For test 
            labelArray = sitk.GetArrayFromImage(labelList[i][x])
            segmentedArrayList[i].append(labelArray)
            """


    """ Restore image. """
    imagePath = Path(args.imageDirectory) / "imaging.nii.gz"
    image = sitk.ReadImage(str(imagePath))
    imageArray = sitk.GetArrayFromImage(image)
    outputArray = np.zeros_like(imageArray)
    for i in range(2):
        length = len(segmentedArrayList[i])
        largestSlice = metaData[i]["largestSlice"]
        axialSlice = metaData[i]["axialSlice"]
        paddingSize = metaData[i]["paddingSize"]
        largestArray = outputArray[largestSlice, ...]
        p = (paddingSize, paddingSize)
        largestArray = np.pad(largestArray, [p, p, (0, 0)], "minimum")
        largestArray = largestArray[..., axialSlice]
        for x in tqdm(range(length), desc="Restoring images...", ncols=60):
            segmented = getImageWithMeta(segmentedArrayList[i][x], labelList[i][x])
            xSlice = metaData[i]["xSlice"][x]
            ySlice = metaData[i]["ySlice"][x]
            restoreSizeX = int(xSlice.stop - xSlice.start)
            restoreSizeY = int(ySlice.stop - ySlice.start)
            restoreSize = [restoreSizeX, restoreSizeY]
            segmented = resampleSize(segmented, restoreSize[::-1], is_label=True)
            segmentedArray = sitk.GetArrayFromImage(segmented)
            largestArray[xSlice, ySlice, x] = segmentedArray

        largestArray = largestArray[paddingSize : -paddingSize, 
                                    paddingSize : -paddingSize, :]
        
        if not args.noFlip:
            if i == 1:
                largestArray = largestArray[::-1, ...]

        outputArray[largestSlice, :, axialSlice] += largestArray

    output = getImageWithMeta(outputArray, image)
    createParentPath(args.savePath)
    print("Saving image to {}".format(args.savePath))
    sitk.WriteImage(output, args.savePath, True)


if __name__ == '__main__':
    args = ParseArgs()
    tf.compat.v1.app.run(main=main, argv=sys.argv)
