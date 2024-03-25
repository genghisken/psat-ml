#!/usr/bin/env python
"""Run the Keras/Tensorflow classifier on Pan-STARRS and ATLAS images.

Usage:
  %s <configFile> [<candidate>...] [--hkoclassifier=<hkoclassifier>] [--mloclassifier=<mloclassifier>] [--sthclassifier=<sthclassifier>] [--chlclassifier=<chlclassifier>] [--ps1classifier=<ps1classifier>] [--ps2classifier=<ps2classifier>] [--outputcsv=<outputcsv>] [--listid=<listid>] [--imageroot=<imageroot>] [--update] [--tablename=<tablename>] [--columnname=<columnname>] [--candidatesinfiles] [--magicNumber=<magicNumber>]
  %s (-h | --help)
  %s --version

Options:
  -h --help                          Show this screen.
  --version                          Show version.
  --listid=<listid>                  List ID [default: 4].
  --hkoclassifier=<hkoclassifier>    HKO Classifier file.
  --mloclassifier=<mloclassifier>    MLO Classifier file.
  --sthclassifier=<sthclassifier>    STH Classifier file.
  --chlclassifier=<chlclassifier>    CHL Classifier file.
  --ps1classifier=<ps1classifier>    PS1 Classifier file. This option will cause the ATLAS classifiers to be ignored.
  --ps2classifier=<ps2classifier>    PS2 Classifier file. This option will cause the ATLAS classifiers to be ignored.
  --outputcsv=<outputcsv>            Output file [default: /tmp/update_eyeball_scores.csv].
  --imageroot=<imageroot>            Root location of the actual images [default: /db4/images/].
  --update                           Update the database.
  --tablename=<tablename>            Database table name to update [default: atlas_diff_objects].
  --columnname=<columnname>          Database column name to update [default: zooniverse_score].
  --candidatesinfiles                Interpret the inline candidate IDs as a files containing candidates.
  --magicNumber=<magicNumber>        Magic number used to mask bad pixels in integer image files (ATLAS only).

Example:
  python %s ~/config.pso3.gw.warp.yaml --ps1classifier=/data/db4data1/scratch/kws/training/ps1/20190115/ps1_20190115_400000_1200000.best.hdf5 --listid=4 --outputcsv=/tmp/pso3_list_4.csv
  python %s ../ps13pi/config/config.yaml --ps1classifier=/data/db4data1/scratch/kws/training/ps1/20190115/ps1_20190115_400000_1200000.best.hdf5 --listid=4 --outputcsv=/tmp/ps13pi_list_4.csv
  python %s /usr/local/ps1code/gitrelease/atlas/config/config4_db1_readonly.yaml /tmp/candidates.txt --hkoclassifier=/usr/local/ps1code/gitrelease/tf_trained_classifiers/02a_asteroids_good330000_bad990000_s3_20230405_classifier.h5 --mloclassifier=/usr/local/ps1code/gitrelease/tf_trained_classifiers/asteroids136521_good13479_bad450000_20x20_skew3_signpreserve_20200819mlo_classifier.h5 --sthclassifier=/usr/local/ps1code/gitrelease/tf_trained_classifiers/03a_asteroids_good320000_bad960000_s3_20230303_classifier.h5 --chlclassifier=/usr/local/ps1code/gitrelease/tf_trained_classifiers/04a_asteroids_good380000_bad1140000_s3_20230213_classifier.h5 --outputcsv=/db4/tc_logs/atlas4/ml_tf_keras_20230502_1017.csv --update --candidatesinfiles
  python %s /usr/local/ps1code/gitrelease/atlas/config/config4_db1_readonly.yaml /tmp/candidates_20230906.txt --hkoclassifier=/usr/local/ps1code/gitrelease/tf_trained_classifiers/02a_asteroids_good330000_bad990000_s3_20230405_20x20_nomagic_classifier.h5 --mloclassifier=/usr/local/ps1code/gitrelease/tf_trained_classifiers/01a_asteroids_20x20_good15000_bad45000_s3_20191125_nomagic_classifier.h5 --sthclassifier=/usr/local/ps1code/gitrelease/tf_trained_classifiers/03a_asteroids_good320000_bad960000_s3_20230303_20x20_nomagic_classifier.h5 --chlclassifier=/usr/local/ps1code/gitrelease/tf_trained_classifiers/04a_asteroids_good380000_bad1140000_s3_20230213_20x20_nomagic_classifier.h5 --outputcsv=/db4/tc_logs/atlas4/ml_tf_keras_20230906_1800.csv --update --candidatesinfiles --magicNumber=-31415
  python %s ../../ps13pi/config/config.yaml --ps1classifier=/data/db4data1/scratch/kws/training/ps1/20190115/ps1_20190115_400000_1200000.best.hdf5 --ps2classifier=/data/db0jbod05/training/ps2/ps2_good95824_bad287472_s3_20230707_20x20_classifier.h5 --listid=4 --imageroot=/db0/images/ --update


"""
import sys
__doc__ = __doc__ % (sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0])
from docopt import docopt
from gkutils.commonutils import Struct, cleanOptions, readGenericDataFile, dbConnect
import sys, csv, os
from TargetImage import *
import numpy as np
from kerasTensorflowClassifier import create_model, load_data
from collections import defaultdict, OrderedDict

# 2019-05-05 KWS Limit the number of CPUs to 4 for each process. Should still overuse the CPUs
#                but should get away with this because of I/O.
#from keras import backend as K
#K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)))

def getObjectsByList(conn, dbName, listId = 4, imageRoot='/db4/images/', ps1Data = False):
    # First get the candidates
    import MySQLdb
    try:
        cursor = conn.cursor (MySQLdb.cursors.DictCursor)

        if ps1Data:
            cursor.execute ("""
                select id
                  from tcs_transient_objects
                 where detection_list_id = %s
                   and confidence_factor is null
                   and tcs_images_id is not null
              order by followup_id desc
            """, (listId,))
        else:
            cursor.execute ("""
                select id
                  from atlas_diff_objects
                 where detection_list_id = %s
                   and zooniverse_score is null
            """, (listId,))
        resultSet = cursor.fetchall ()
        cursor.close ()

    except MySQLdb.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    return resultSet

# 2019-05-02 KWS Separated out the acquisiton of images so that can do
#                this multithreaded. Also so we can pass a user defined
#                list of objects to the processing.

def getImages(conn, dbName, objectList, imageRoot='/psdb3/images/', ps1Data = False):
    import MySQLdb
    images = []
    # Now, for each candidate, get the image
    for row in objectList:
        try:
            cursor = conn.cursor (MySQLdb.cursors.DictCursor)
            # 2022-01-01 KWS Use mjd_obs for Pan-STARRS but use filename to get MJD if ATLAS.
            #                Fixes issue with South Africa (and Chile) night number and MJD
            #                mismatch.
            if ps1Data:
                cursor.execute ("""
                select concat(%s ,%s,'/',truncate(mjd_obs,0), '/', image_filename,'.fits') as filename, filter from tcs_postage_stamp_images
                 where image_filename like concat(%s, '%%')
                   and image_filename not like concat(%s, '%%4300000000%%')
                   and image_type = 'diff'
                   and image_filename is not null
                   and pss_error_code = 0
                   and mjd_obs is not null
                """, (imageRoot, dbName, row['id'], row['id']))
            else:
                cursor.execute ("""
                select concat(%s ,%s,'/',if(instr(pss_filename,'skycell'),truncate(mjd_obs,0),substr(pss_filename,4,5)), '/', image_filename,'.fits') as filename, filter from tcs_postage_stamp_images
                 where image_filename like concat(%s, '%%')
                   and image_filename not like concat(%s, '%%4300000000%%')
                   and image_type = 'diff'
                   and image_filename is not null
                   and pss_error_code = 0
                   and mjd_obs is not null
                """, (imageRoot, dbName, row['id'], row['id']))

            imageResultSet = cursor.fetchall ()
            cursor.close ()
            for row in imageResultSet:
                # Only append images that actually exist!
                if os.path.exists(row['filename']):
                    images.append(row)

        except MySQLdb.Error as e:
            print("Error %d: %s" % (e.args[0], e.args[1]))

    return images

# Update the database.
def updateTransientRBValue(conn, objectId, realBogusValue, ps1Data = False):
    import MySQLdb

    rowsUpdated = 0

    try:
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)

        if ps1Data:
            # It's Pan-STARRS data
            cursor.execute ("""
                 update tcs_transient_objects
                 set confidence_factor = %s
                 where id = %s
            """, (realBogusValue, objectId))
        else:
            # It's ATLAS data
            cursor.execute ("""
                 update atlas_diff_objects
                 set zooniverse_score = %s
                 where id = %s
            """, (realBogusValue, objectId))

        rowsUpdated = cursor.rowcount

        # Did we update any transient object rows? If not issue a warning.
        if rowsUpdated == 0:
            print ("WARNING: No transient object entries were updated.")

        cursor.close ()


    except MySQLdb.Error as e:
        print ("Error %d: %s" % (e.args[0], e.args[1]))

    return rowsUpdated


def updateObjectRBFactors(conn, objectId, realBogusValue, tableName, columnName):

    import MySQLdb
    rowsUpdated = 0

    try:
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)

        statement = """
             update %s
             set %s = %s
             where id = %s
            -- and %s is null
        """ % (tableName, columnName, realBogusValue, objectId, columnName)
        cursor.execute(statement)

        rowsUpdated = cursor.rowcount

        # Did we update any transient object rows? If not issue a warning.
        if rowsUpdated == 0:
            print("WARNING: No transient object entries were updated.")

        cursor.close ()


    except MySQLdb.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    return rowsUpdated


def getRBValues(imageFilenames, classifier, extension = 0, magicNumber = None):
    num_classes = 2
    image_dim = 20
    numImages = len(imageFilenames)
    images = np.zeros((numImages, image_dim,image_dim,1))
    #print images
    # loop through and fill the above matrix, remembering to correctly scale the
    # raw pixels for the specified sparse filter.
    for j,imageFilename in enumerate(imageFilenames):
        vector = np.nan_to_num(TargetImage(imageFilename, extension=extension, magicNumber = magicNumber).signPreserveNorm())
        #print vector
        #print vector.shape
        images[j,:,:,0] += np.reshape(vector, (image_dim,image_dim), order="F")

    #print images.shape


    model = create_model(num_classes, image_dim)
    model.load_weights(classifier)

    pred = model.predict(images, verbose=0)
    print("PRED = ", pred)
    # Collect the predictions from all the files, but aggregate into objects
    objectDict = defaultdict(list)
    for i in range(len(pred[:,1])):
        candidate = os.path.basename(imageFilenames[i]).split('_')[0]
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

    import yaml
    with open(options.configFile) as yaml_file:
        config = yaml.load(yaml_file)

    username = config['databases']['local']['username']
    password = config['databases']['local']['password']
    database = config['databases']['local']['database']
    hostname = config['databases']['local']['hostname']

    conn = dbConnect(hostname, username, password, database)
    if not conn:
        print("Cannot connect to the database")
        return 1

    # 2023-03-25 KWS MySQLdb disables autocommit by default. Switch it on globally.
    conn.autocommit(True)

    # 2018-07-31 KWS We have PS1 data. Don't bother with the HKO/MLO ATLAS data.
    ps1Data = False
    if options.ps1classifier or options.ps2classifier:
        ps1Data = True

    if options.listid is not None:
        try:
            detectionList = int(options.listid)
            if detectionList < 0 or detectionList > 8:
                print ("Detection list must be between 0 and 8")
                return 1
        except ValueError as e:
            sys.exit("Detection list must be an integer")

    objectList = []
    imageFilenames = []

    # if candidates are specified in the options, then override the list.
    if len(options.candidate) > 0:
        if options.candidatesinfiles:
            candidates = []
            for f in options.candidate:
                with open(f) as fp:
                    content = fp.readlines()
                    content = [c.strip() for c in content]
                candidates += content
            objectList = [{'id': int(candidate)} for candidate in candidates]
        else:
            objectList = [{'id': int(candidate)} for candidate in options.candidate]
    else:
        # Only collect by the list ID if we are running in single threaded mode
        if processNumber is None:
            objectList = getObjectsByList(conn, database, listId = int(options.listid), ps1Data = ps1Data)

    if len(objectList) > 0:
        imageFilenames = getImages(conn, database, objectList, imageRoot=options.imageroot)
        if len(imageFilenames) == 0:
            print("NO IMAGES")
            conn.close()
            return []

    magicNumber = None
    if options.magicNumber:
        magicNumber = int(options.magicNumber)

    if ps1Data:
        # 2023-07-24 KWS Split the PS1 and PS2 images like the ATLAS ones.
        #                The filter column can easily be used for this.
        ps1Filenames = []
        for row in imageFilenames:
            if '00000' in row['filter']:
                ps1Filenames.append(row['filename'])

        ps2Filenames = []
        for row in imageFilenames:
            if '00002' in row['filter']:
                ps2Filenames.append(row['filename'])


        if ps1Filenames:
            objectDictPS1 = getRBValues(ps1Filenames, options.ps1classifier, extension = 1)
        if ps2Filenames:
            objectDictPS2 = getRBValues(ps2Filenames, options.ps2classifier, extension = 1)

        # Now we have two dictionaries. Combine them.

        objectScores = defaultdict(dict)

        if ps1Filenames:
            for k, v in list(objectDictPS1.items()):
                objectScores[k]['ps1'] = np.array(v)
        if ps2Filenames:
            for k, v in list(objectDictPS2.items()):
                objectScores[k]['ps2'] = np.array(v)

        finalScores = {}

        objects = list(objectScores.keys())
        for object in objects:
            objectKeys = objectScores[object].keys()
            lengths = {}
            for key in objectKeys:
                #print(object, key, objectScores[object][key]) 
                lengths[key] = len(objectScores[object][key])
            finalScores[object] = np.median(objectScores[object][max(lengths, key=lambda key: lengths[key])])


    else:
        # Split the images into HKO and MLO data so we can apply the HKO and MLO machines separately.
        hkoFilenames = []
        for row in imageFilenames:
            if '02a' in row['filename']:
                hkoFilenames.append(row['filename'])
        mloFilenames = []
        for row in imageFilenames:
            if '01a' in row['filename']:
                mloFilenames.append(row['filename'])
        sthFilenames = []
        for row in imageFilenames:
            if '03a' in row['filename']:
                sthFilenames.append(row['filename'])
        chlFilenames = []
        for row in imageFilenames:
            if '04a' in row['filename']:
                chlFilenames.append(row['filename'])

        if hkoFilenames:
            objectDictHKO = getRBValues(hkoFilenames, options.hkoclassifier, magicNumber = magicNumber)
        if mloFilenames:
            objectDictMLO = getRBValues(mloFilenames, options.mloclassifier, magicNumber = magicNumber)
        if sthFilenames:
            objectDictSTH = getRBValues(sthFilenames, options.sthclassifier, magicNumber = magicNumber)
        if chlFilenames:
            objectDictCHL = getRBValues(chlFilenames, options.chlclassifier, magicNumber = magicNumber)

        # Now we have two dictionaries. Combine them.

        objectScores = defaultdict(dict)

        if hkoFilenames:
            for k, v in list(objectDictHKO.items()):
                objectScores[k]['hko'] = np.array(v)
        if mloFilenames:
            for k, v in list(objectDictMLO.items()):
                objectScores[k]['mlo'] = np.array(v)
        if sthFilenames:
            for k, v in list(objectDictSTH.items()):
                objectScores[k]['sth'] = np.array(v)
        if chlFilenames:
            for k, v in list(objectDictCHL.items()):
                objectScores[k]['chl'] = np.array(v)

        # Some objects will have data from two telescopes, some only one.
        # If we have data from two telescopes, choose the median value of the longest length list.

        finalScores = {}

        objects = list(objectScores.keys())
        for object in objects:
            objectKeys = objectScores[object].keys()
            lengths = {}
            for key in objectKeys:
                lengths[key] = len(objectScores[object][key])
            finalScores[object] = np.median(objectScores[object][max(lengths, key=lambda key: lengths[key])])


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
            pid = os.getpid()
            processSuffix = '_%d_%03d' % (pid, processNumber)

        # Generate the insert statements
        with open('%s%s%s' % (prefix, processSuffix, suffix), 'w') as f:
            for k, v in list(finalScoresSorted.items()):
                print(k, finalScoresSorted[k])
                f.write('%s,%f\n' % (k, finalScoresSorted[k]))

    scores = list(finalScoresSorted.items())

    if options.update and processNumber is None:
        # Only allow database updates in single threaded mode. Otherwise multithreaded code
        # does the updates at the end of processing. (Minimises table locks.)
        for row in scores:
            updateTransientRBValue(conn, row[0], row[1], ps1Data = ps1Data)

    conn.commit()
    conn.close()

    return scores


def main():
    opts = docopt(__doc__, version='0.1')
    opts = cleanOptions(opts)

    # Use utils.Struct to convert the dict into an object for compatibility with old optparse code.
    options = Struct(**opts)
    objectsForUpdate = runKerasTensorflowClassifier(options)

if __name__=='__main__':
    main()
