#!/usr/bin/env python
"""Run the Keras/Tensorflow classifier.

Usage:
  %s <configFile> [--hkoclassifier=<hkoclassifier>] [--mloclassifier=<mloclassifier>] [--ps1classifier=<ps1classifier>] [--outputcsv=<outputcsv>] [--listid=<listid>] [--imageroot=<imageroot>] [--update] [--tablename=<tablename>] [--columnname=<columnname>]
  %s (-h | --help)
  %s --version

Options:
  -h --help                          Show this screen.
  --version                          Show version.
  --listid=<listid>                  List ID [default: 4].
  --hkoclassifier=<hkoclassifier>    HKO Classifier file.
  --mloclassifier=<mloclassifier>    MLO Classifier file.
  --ps1classifier=<mloclassifier>    PS1 Classifier file. This option will cause the HKO and MLO classifiers to be ignored.
  --outputcsv=<outputcsv>            Output file [default: /tmp/update_eyeball_scores.csv].
  --imageroot=<imageroot>            Root location of the actual images [default: /db4/images/].
  --update                           Update the database.
  --tablename=<tablename>            Database table name to update [default: atlas_diff_objects].
  --columnname=<columnname>          Database column name to update [default: zooniverse_score].


"""
import sys
__doc__ = __doc__ % (sys.argv[0], sys.argv[0], sys.argv[0])
from docopt import docopt
from gkutils.commonutils import Struct, cleanOptions, readGenericDataFile, dbConnect
import sys, csv, os
from TargetImage import *
import numpy as np
from kerasTensorflowClassifier import create_model, load_data
from collections import defaultdict, OrderedDict

def getImageDataToCheck(conn, dbName, listId = 4, imageRoot='/db4/images/', ps1Data = False):
    # First get the candidates
    import MySQLdb
    try:
        cursor = conn.cursor (MySQLdb.cursors.DictCursor)

        if ps1Data:
            imageRoot = '/psdb2/images/'
            cursor.execute ("""
                select id
                  from tcs_transient_objects
                 where detection_list_id = %s
                   and confidence_factor is not null
              order by followup_id desc
                 limit 100
            """, (listId,))
        else:
            cursor.execute ("""
            --    select o.id
            --      from atlas_diff_objects o, tcs_object_groups g
            --     where g.transient_object_id = o.id
            --       and g.object_group_id = s
                   -- and o.zooniverse_score is null
                select id
                  from atlas_diff_objects
                 where detection_list_id = %s
                  -- and zooniverse_score is null
            """, (listId,))
        resultSet = cursor.fetchall ()

    except MySQLdb.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    images = []
    # Now, for each candidate, get the image
    for row in resultSet:
        try:
            cursor = conn.cursor (MySQLdb.cursors.DictCursor)
            cursor.execute ("""
            select concat(%s ,%s,'/',truncate(mjd_obs,0), '/', image_filename,'.fits') filename from tcs_postage_stamp_images
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


def getRBValues(imageFilenames, classifier, extension = 0):
    num_classes = 2
    image_dim = 20
    numImages = len(imageFilenames)
    images = np.zeros((numImages, image_dim,image_dim,1))
    #print images
    # loop through and fill the above matrix, remembering to correctly scale the
    # raw pixels for the specified sparse filter.
    for j,imageFilename in enumerate(imageFilenames):
        vector = np.nan_to_num(TargetImage(imageFilename, extension=extension).signPreserveNorm())
        #print vector
        #print vector.shape
        images[j,:,:,0] += np.reshape(vector, (image_dim,image_dim), order="F")

    #print images.shape


    model = create_model(num_classes, image_dim)
    model.load_weights(classifier)

    pred = model.predict(images, verbose=0)
    print(pred)
    # Collect the predictions from all the files, but aggregate into objects
    objectDict = defaultdict(list)
    for i in range(len(pred[:,1])):
        candidate = os.path.basename(imageFilenames[i]).split('_')[0]
        # Each candidate will end up with a list of predictions.
        objectDict[candidate].append(pred[i,1])

        #print "%s,%.3lf"%(imageFilenames[i], pred[i,1])

    return objectDict


def runKerasTensorflowClassifier(opts):

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
    if options.ps1classifier:
        ps1Data = True

    imageFilenames = getImageDataToCheck(conn, database, listId = int(options.listid), imageRoot=options.imageroot, ps1Data = ps1Data)
    if len(imageFilenames) == 0:
        print("No data to check!")
        return 0

    if ps1Data:
        objectDictPS1 = getRBValues([f['filename'] for f in imageFilenames], options.ps1classifier, extension = 1)
        objectScores = defaultdict(dict)
        for k, v in list(objectDictPS1.items()):
            objectScores[k]['ps1'] = np.array(v)
        finalScores = {}

        objects = list(objectScores.keys())
        for object in objects:
            finalScores[object] = np.median(objectScores[object]['ps1'])
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

        #filename = 'hko_57966_20x20_skew3_signpreserve_f77475b232425.mat'
        #train_data, test_data, image_dim = load_data(filename)
        #x_test = test_data[0]

        #hkoClassifier = '/home/kws/keras/hko_57966_20x20_skew3_signpreserve_f77475b232425.model.best.hdf5'
        #mloClassifier = '/home/kws/keras/atlas_mlo_57925_20x20_skew3_signpreserve_f331184b993662.model.best.hdf5'

        objectDictHKO = getRBValues(hkoFilenames, options.hkoclassifier)
        objectDictMLO = getRBValues(mloFilenames, options.mloclassifier)

        # Now we have two dictionaries. Combine them.

        objectScores = defaultdict(dict)

        for k, v in list(objectDictHKO.items()):
            objectScores[k]['hko'] = np.array(v)
        for k, v in list(objectDictMLO.items()):
            objectScores[k]['mlo'] = np.array(v)

        # Some objects will have data from two telescopes, some only one.
        # If we have data from two telescopes, choose the median value of the longest length list.

        finalScores = {}

        objects = list(objectScores.keys())
        for object in objects:
            if len(objectScores[object]) > 1:
                hkoLen = len(objectScores[object]['hko'])
                mloLen = len(objectScores[object]['mlo'])
                if mloLen > hkoLen:
                    finalScores[object] = np.median(objectScores[object]['mlo'])
                else:
                    # Only if MLO is larger than HKO, use MLO. Otherise use HKO
                    finalScores[object] = np.median(objectScores[object]['hko'])

            else:
                try:
                    finalScores[object] = np.median(objectScores[object]['hko'])
                except KeyError as e:
                    finalScores[object] = np.median(objectScores[object]['mlo'])

    finalScoresSorted = OrderedDict(sorted(list(finalScores.items()), key=lambda t: t[1]))

    # Generate a csv file
    with open(options.outputcsv, 'w') as f:
        for k, v in list(finalScoresSorted.items()):
            print(1,k, finalScoresSorted[k])
            f.write(str(k)+','+'1'+','+str(finalScoresSorted[k])+'\n')
            if options.update:
                updateObjectRBFactors(conn, int(k), float(finalScoresSorted[k]), options.tablename, options.columnname)
                

    conn.commit()

    conn.close()
   # with  open(options.outputcsv,"w") as csvFile:
   #     writer = csv.writer(csvFile, delimiter=',')
   #     for i in list(finalScoresSorted.items()):
   #         print(str(i)+' '+str(1)+' ',str(finalScoresSorted[i]))
   #         writer.writerow(str(i),str(1),str(finalScoresSorted[i]))
    return 0


def main():
    opts = docopt(__doc__, version='0.1')
    opts = cleanOptions(opts)

    # Use utils.Struct to convert the dict into an object for compatibility with old optparse code.
    options = Struct(**opts)
    runKerasTensorflowClassifier(options)


if __name__=='__main__':
    main()
