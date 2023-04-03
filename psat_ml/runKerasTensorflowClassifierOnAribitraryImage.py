#!/usr/bin/env python
"""Run the Keras/Tensorflow classifier.

Usage:
  %s <image>... [--classifier=<classifier>] [--outputcsv=<outputcsv>] [--fitsextension=<fitsextension>] [--keepfilename]
  %s (-h | --help)
  %s --version

Options:
  -h --help                          Show this screen.
  --version                          Show version.
  --classifier=<classifier>          Classifier file.
  --outputcsv=<outputcsv>            Output file.
  --fitsextension=<fitsextension>    Which default FITS extension? [default: 0]
  --keepfilename                     Keep the filename - don't truncate it.

Example:
  python %s /tmp/image1.fits /tmp/image2.fits --classifier=/data/db4data1/scratch/kws/training/ps1/20190115/ps1_20190115_400000_1200000.best.hdf5 --outputcsv=/tmp/output.csv

"""
import sys
__doc__ = __doc__ % (sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0])
from docopt import docopt
from gkutils.commonutils import Struct, cleanOptions, readGenericDataFile, dbConnect
import sys, csv, os
from TargetImage import *
import numpy as np
from kerasTensorflowClassifier import create_model, load_data
from collections import defaultdict, OrderedDict


def getRBValues(imageFilenames, classifier, extension = 0, keepfilename = None):
    num_classes = 2
    image_dim = 20
    numImages = len(imageFilenames)
    images = np.zeros((numImages, image_dim,image_dim,1))
    #print images
    # loop through and fill the above matrix, remembering to correctly scale the
    # raw pixels for the specified sparse filter.
    for j,imageFilename in enumerate(imageFilenames):
        print(imageFilename)
        vector = np.nan_to_num(TargetImage(imageFilename, extension=extension).signPreserveNorm())
        #print vector
        #print vector.shape
        images[j,:,:,0] += np.reshape(vector, (image_dim,image_dim), order="F")

    #print images.shape


    model = create_model(num_classes, image_dim)
    model.load_weights(classifier)

    pred = model.predict(images, verbose=0)
    # Collect the predictions from all the files, but aggregate into objects
    objectDict = defaultdict(list)
    for i in range(len(pred[:,1])):
        if keepfilename:
            candidate = os.path.basename(imageFilenames[i])
        else:
            candidate = os.path.basename(imageFilenames[i]).split('.')[0]
        # Each candidate will end up with a list of predictions.
        objectDict[candidate].append(pred[i,1])

        #print "%s,%.3lf"%(imageFilenames[i], pred[i,1])

    return objectDict


def runKerasTensorflowClassifier(opts, processNumber = None):

    # Use utils.Struct to convert the dict into an object for compatibility with old optparse code.
    if type(opts) is dict:
        options = Struct(**opts)
    else:
        options = opts

    imageFilenames = options.image
    fitsExtension = int(options.fitsextension)

    objectDictPS1 = getRBValues(imageFilenames, options.classifier, extension = fitsExtension, keepfilename = options.keepfilename)
    objectScores = defaultdict(dict)
    for k, v in list(objectDictPS1.items()):
        objectScores[k]['ps1'] = np.array(v)
    finalScores = {}

    objects = list(objectScores.keys())
    for object in objects:
        finalScores[object] = np.median(objectScores[object]['ps1'])

    finalScoresSorted = OrderedDict(sorted(list(finalScores.items()), key=lambda t: t[1]))

    if options.outputcsv is not None:
        prefix = options.outputcsv.split('.')[0]
        suffix = options.outputcsv.split('.')[-1]

        if suffix == prefix:
            suffix = ''

        if suffix:
            suffix = '.' + suffix

        processSuffix = ''

        if processNumber is not None:
            processSuffix = '_%02d' % processNumber

        # Generate the insert statements
        with open('%s%s%s' % (prefix, processSuffix, suffix), 'w') as f:
            for k, v in list(finalScoresSorted.items()):
                print(k, finalScoresSorted[k])
                f.write('%s,%f\n' % (k, finalScoresSorted[k]))

    scores = list(finalScoresSorted.items())

    return scores


def main():
    opts = docopt(__doc__, version='0.1')
    opts = cleanOptions(opts)

    # Use utils.Struct to convert the dict into an object for compatibility with old optparse code.
    options = Struct(**opts)
    objectsForUpdate = runKerasTensorflowClassifier(options)

if __name__=='__main__':
    main()
