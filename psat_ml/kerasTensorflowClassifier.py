#!/usr/bin/env python
"""Read the training set and train a machine.

Usage:
  %s <trainingset> [--classifierfile=<classifierfile>] [--trainer=<trainer>] [--outputcsv=<outputcsv>]
  %s (-h | --help)
  %s --version

Options:
  -h --help                          Show this screen.
  --version                          Show version.
  --classifierfile=<classifierfile>  Classifier file [default: /tmp/atlas.model.best.hdf5].
  --trainer=<trainer>                Training file [default: PSAT-D].
  --outputcsv=<outputcsv>            Output file [default: /tmp/output.csv].


"""
import sys
import importlib
__doc__ = __doc__ % (sys.argv[0], sys.argv[0], sys.argv[0])
from docopt import docopt
import os, shutil, re
from gkutils.commonutils import Struct, cleanOptions
import h5py
import numpy as np
import scipy.io as sio

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint  

from rocCurve import roc_curve

def one_percent_mdr(y_true, y_pred):
    t = 0.01
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, step=0.001)
    return fpr[np.where(1-tpr<=t)[0]][-1]

def one_percent_fpr(y_true, y_pred):
    t = 0.01
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, step=0.001)
    return 1-tpr[np.where(fpr<=t)[0]][0]

def load_data(filename):
    #data = sio.loadmat(filename)
    data = h5py.File(filename,'r')
    X = data['X']
    y_train = np.squeeze(data['y'])
    ascii_train_files = data['train_files']
    train_files = np.squeeze([n.decode("utf-8") for n in ascii_train_files])
    m, n = X.shape
    image_dim = int(np.sqrt(n))
    x_train = np.zeros((m, image_dim, image_dim, 1))
    for i in range(m):
        x_train[i,:,:,0] += np.reshape(X[i], (image_dim, image_dim), order='F')

    X = data['testX']
    y_test = np.squeeze(data['testy'])
    ascii_test_files = data['test_files']
    test_files = np.squeeze([n.decode("utf-8") for n in ascii_test_files])
    m, n = X.shape
    x_test = np.zeros((m, image_dim, image_dim, 1))
    for i in range(m):
        x_test[i,:,:,0] += np.reshape(X[i], (image_dim, image_dim), order='F')
    
    return (x_train, y_train, train_files), (x_test, y_test, test_files), image_dim

def kerasTensorflowClassifier(opts):

    # Use utils.Struct to convert the dict into an object for compatibility with old optparse code.
    if type(opts) is dict:
        options = Struct(**opts)
    else:
        options = opts

    filename = options.trainingset

    #filename = 'andrei_20x20_skew3_signpreserve_f200000b600000.mat'
    train_data, test_data, image_dim = load_data(filename)

    num_classes = 2

    x_train = train_data[0]
    y_train = np_utils.to_categorical(train_data[1], num_classes)

    m = x_train.shape[0]
    split_frac = int(.75*m)
    (x_train, x_valid) = x_train[:split_frac], x_train[split_frac:]
    (y_train, y_valid) = y_train[:split_frac], y_train[split_frac:]

    x_test = test_data[0]
    #y_test = np_utils.to_categorical(test_data[1], num_classes)
    y_test = test_data[1]
    
    
    trainer = importlib.import_module(options.trainer)
    create_model = trainer.create_model
    
    model = create_model(num_classes, image_dim)
    """  
    checkpointer = ModelCheckpoint(filepath=options.classifierfile, \
                                   verbose=1, save_best_only=True)

    model.fit(x_train, y_train, batch_size=128, epochs=20, \
              validation_data=(x_valid, y_valid), \
              callbacks=[checkpointer], verbose=1, shuffle=True)
    """

    if not os.path.exists(options.classifierfile):
        # If we don't already have a trained classifier, train a new one.
        checkpointer = ModelCheckpoint(filepath=options.classifierfile, \
                                   verbose=1, save_best_only=True)
        print(checkpointer)

        model.fit(x_train, y_train, batch_size=128, epochs=20, \
              validation_data=(x_valid, y_valid), \
              callbacks=[checkpointer], verbose=1, shuffle=True)


    model.load_weights(options.classifierfile)

    (y_train, y_valid) = train_data[1][:split_frac], train_data[1][split_frac:]

    print('[+] Training Set Error:')
    pred = model.predict(x_train, verbose=0)
    print((one_percent_mdr(y_train, pred[:,1])))
    print((one_percent_fpr(y_train, pred[:,1])))

    print('[+] Validation Set Error:')
    pred = model.predict(x_valid, verbose=0)
    print((one_percent_mdr(y_valid, pred[:,1])))
    print((one_percent_fpr(y_valid, pred[:,1])))

    print('[+] Test Set Error:')
    pred = model.predict(x_test, verbose=0)
    print((one_percent_mdr(y_test, pred[:,1])))
    print((one_percent_fpr(y_test, pred[:,1])))

    output = open(options.outputcsv,"w")
    for i in range(len(pred[:,1])):
        output.write("%s,%d,%.3lf\n"%(test_data[2][i], y_test[i], pred[i,1]))
    output.close()


def main():
    opts = docopt(__doc__, version='0.1')
    opts = cleanOptions(opts)

    # Use utils.Struct to convert the dict into an object for compatibility with old optparse code.
    options = Struct(**opts)
    kerasTensorflowClassifier(options)


if  __name__ == '__main__':
    main()

