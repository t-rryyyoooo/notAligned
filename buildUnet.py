import sys
import os
import numpy as np
import tensorflow as tf
import argparse
import SimpleITK as sitk
import random
#import keras
#import keras.backend as K
import time
from functions import createParentPath, caluculateTime, dice, cancer_dice, kidney_dice, penalty_categorical
from loader import ReadSliceDataList, GetInputShapes, ImportBatchArray
from Unet import ConstructModel
from saver import *

args = None


def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("trainingdatafile", help="Input Dataset file for training")
    parser.add_argument("--bestfile", help="The filename of the best weights.")
    parser.add_argument("--initialfile", help="For Reproducibility")
    parser.add_argument("--latestfile", help="The filename of the latest weights.")
    #parser.add_argument("modelfile", help="Output trained model file in HDF5 format (*.hdf5).")
    parser.add_argument("-t","--testfile", help="Input Dataset file for validation")
    parser.add_argument("-e", "--epochs", help="Number of epochs", default=1000, type=int)
    parser.add_argument("-b", "--batchsize", help="Batch size", default=10, type=int)
    parser.add_argument("-l", "--learningrate", help="Learning rate", default=1e-3, type=float)
    parser.add_argument("--nobn", help="Do not use batch normalization layer", action='store_true')
    parser.add_argument("--nodropout", help="Do not use dropout layer", action='store_true')
    parser.add_argument("--noaugmentation", help="Do not use training data augmentation", action='store_true')
    parser.add_argument("--magnification", help="Magnification coefficient for data augmentation", default=10, type=int)
    parser.add_argument("--weightinterval", help="The interval between checkpoint for weight saving.", type=int)
    parser.add_argument("--weightfile", help="The filename of the trained weight parameters file for fine tuning or resuming.")
    parser.add_argument("--premodel", help="The filename of the previously trained model")
    parser.add_argument("--initialepoch", help="Epoch at which to start training for resuming a previous training", default=0, type=int)
    #parser.add_argument("--idlist", help="The filename of ID list for splitting input datasets into training and validation datasets.")
    #parser.add_argument("--split", help="Fraction of the training data to be used as validation data.", default=0.0, type=float)
    parser.add_argument("--logdir", help="Log directory", default='log')
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    parser.add_argument("--history")

    args = parser.parse_args()
    return args


def main(_):
    config = tf.compat.v1.ConfigProto(
    
    #allow_soft_placement=True, log_device_placement=True
    )
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    trainingdatafile = os.path.expanduser(args.trainingdatafile)

    trainingdatalist = ReadSliceDataList(trainingdatafile)
    print(trainingdatalist[0])
    testdatalist = None
    if args.testfile is not None:
        testfile = os.path.expanduser(args.testfile)

        testdatalist = ReadSliceDataList(testfile)

        #testdatalist = random.sample(testdatalist, int(len(testdatalist)*0.1))

    (imageshape, labelshape) = GetInputShapes(trainingdatalist[0])
    nclasses = 3 # Number of classes
    print("image shapes: ",imageshape)
    print("label shapes: ",labelshape)

    with tf.device('/device:GPU:{}'.format(args.gpuid)):
        x = tf.keras.layers.Input(shape=imageshape, name="x")
        segmentation = ConstructModel(x, nclasses, not args.nobn, not args.nodropout)
        model = tf.compat.v1.keras.Model(x, segmentation)
        #model = tf.keras.models.Model(x, segmentation)
        model.summary()

        optimizer = tf.keras.optimizers.Adam(lr=args.learningrate)

        model.compile(loss=penalty_categorical, optimizer=optimizer, metrics=[kidney_dice, cancer_dice])

    #createParentPath(args.modelfile)
    # with open(args.modelfile, 'w') as f:
    #     f.write(model.to_yaml())
    #tf.compat.v1.keras.models.save_model(model, args.modelfile)

    createParentPath(args.initialfile)
    model.save_weights(args.initialfile)

    if args.weightfile is None:
        initial_epoch = 0
    else:
        model.load_weights(args.weightfile)
        initial_epoch = args.initialepoch

    if testdatalist is not None:
        if args.bestfile is None:
            logdir = os.path.expanduser(args.logdir)
            bestfile = logdir + '/bestweights.hdf5'

        else:
            bestfile = os.path.expanduser(args.bestfile)
            createParentPath(bestfile)

    if args.latestfile is None:
        latestfile = args.logdir + '/latestweights.hdf5'

    else:
        latestfile = os.path.expanduser(args.latestfile)
        createParentPath(latestfile)

    #latest_cbk = LatestWeightSaver(latestfile)
    
    createParentPath(args.logdir + "/model/dummy.hdf5")
    tb_cbk = tf.keras.callbacks.TensorBoard(log_dir=args.logdir)
    latest_cbk = tf.keras.callbacks.ModelCheckpoint(filepath = latestfile, save_best_only = False, save_weights_only = False)
    best_cbk = tf.keras.callbacks.ModelCheckpoint(filepath = bestfile, save_best_only = True, save_weights_only = False)
    #every_cbk = tf.keras.callbacks.ModelCheckpoint(filepath = (args.logdir + "/model/model_{epoch:02d}_{val_loss:.3f}.hdf5"), save_best_only = False, save_weights_only = False)

    callbacks = [tb_cbk, latest_cbk, best_cbk]#, every_cbk]

    if args.weightinterval is not None:
        periodic_cbk = PeriodicWeightSaver(logdir=args.logdir, interval=args.weightinterval)

        callbacks.append(periodic_cbk)
    

    steps_per_epoch = len(trainingdatalist) / args.batchsize 
    print ("Batch size: {}".format(args.batchsize))
    print ("Number of Epochs: {}".format(args.epochs))
    print ("Number of Steps/epoch: {}".format(steps_per_epoch))
    

    with tf.device('/device:GPU:{}'.format(args.gpuid)):
        if not args.noaugmentation:
            applyAugmentation = True
        else:
            applyAugmentation = False

        if testdatalist is not None:
            
            historys = model.fit_generator(ImportBatchArray(trainingdatalist, batch_size = args.batchsize, apply_augmentation = applyAugmentation),
                    steps_per_epoch = int(steps_per_epoch), epochs = args.epochs,
                    callbacks=callbacks,
                    validation_data = ImportBatchArray(testdatalist, batch_size = args.batchsize),
                    validation_steps = len(testdatalist),
                    initial_epoch = int(initial_epoch))
        else:
            historys = model.fit_generator(ImportBatchArray(trainingdatalist, batch_size = args.batchsize, apply_augmentation = applyAugmentation),
                    steps_per_epoch = int(steps_per_epoch), epochs = args.epochs,
                    callbacks=callbacks,
                    initial_epoch = int(initial_epoch))
        
    
    cancer = historys.history['cancer_dice']
    val_cancer = historys.history['val_cancer_dice']
    kid = historys.history['kidney_dice']
    val_kid = historys.history['val_kidney_dice']
    epochs = len(cancer)
    
    
    
    if args.history is not None:
        historyy = os.path.expanduser(args.history)
        createParentPath(historyy)
        history_file = open(historyy,"a")
        for x in range(epochs):

            print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(cancer[x],val_cancer[x],kid[x],val_kid[x]),file = history_file)
        print("\n",file=history_file)
        history_file.close()
    

if __name__ == '__main__':
    t1 = time.time()

    args = ParseArgs()
    
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]])

    t2 = time.time()
    caluculateTime(t1, t2)

    
