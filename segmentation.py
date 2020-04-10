import SimpleITK as sitk
import numpy as np
import argparse
import os
import sys
import tensorflow as tf
from functions import createParentPath, getImageWithMeta, penalty_categorical, kidney_dice,cancer_dice 
from pathlib import Path
from slicer import slicer as sler
from tqdm import tqdm

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("imageDirectory", help="$HOME/Desktop/data/kits19/case_00000")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("savePath", help="Segmented label file.(.mha)")
    parser.add_argument("--widthSize", default=15, type=int)
    parser.add_argument("--paddingSize", default=100, type=int)
    parser.add_argument("--outputImageSize", help="256-256-3", default="256-256-3")
    parser.add_argument("--noFlip", action="store_true")

    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    parser.add_argument("-b", "--batchsize", help="Batch size", default=1, type=int)

    
    args = parser.parse_args()
    return args

def main(_):
    """ Slice module. """
    labelFile = Path(args.imageDirectory) / "segmentation.nii.gz"
    imageFile = Path(args.imageDirectory) / "imaging.nii.gz"

    label = sitk.ReadImage(str(labelFile))
    image = sitk.ReadImage(str(imageFile))

    slicer = sler(image, label, outputImageSize = args.outputImageSize, widthSize = args.widthSize, paddingSize = args.paddingSize, noFlip = args.noFlip)

    slicer.execute()
    _, cuttedImageArrayList = slicer.output("Array")

    """ Segmentation module. """
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


    segmentedArrayList = [[] for _ in range(2)]
    for i in range(2):
        length = len(cuttedImageArrayList[i])
        for x in tqdm(range(length), desc="Segmenting images...", ncols=60):
            imageArray = cuttedImageArrayList[i][x]
            imageArray = imageArray[np.newaxis, ...]
            
            segmentedArray = model.predict(imageArray, batch_size=args.batchsize, verbose=0)
            segmentedArray = np.squeeze(segmentedArray)
            segmentedArray = np.argmax(segmentedArray, axis=-1).astype(np.uint8)
            segmentedArrayList[i].append(segmentedArray)

    """ Restore module. """
    segmentedArray = slicer.restore(segmentedArrayList)

    segmented = getImageWithMeta(segmentedArray, label)
    createParentPath(args.savePath)
    print("Saving image to {}".format(args.savePath))
    sitk.WriteImage(segmented, args.savePath, True)


if __name__ == '__main__':
    args = ParseArgs()
    tf.compat.v1.app.run(main=main, argv=sys.argv)
