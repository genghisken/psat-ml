#!/usr/bin/env python
"""Run the Keras/Tensorflow classifier on Pan-STARRS and ATLAS images.

Usage:
  %s <configFile> [<candidate>...] [--hkoclassifier=<hkoclassifier>] [--mloclassifier=<mloclassifier>] [--sthclassifier=<sthclassifier>] [--chlclassifier=<chlclassifier>] [--ps1classifier=<ps1classifier>] [--ps2classifier=<ps2classifier>] [--outputcsv=<outputcsv>] [--listid=<listid>] [--imageroot=<imageroot>] [--update] [--tablename=<tablename>] [--columnname=<columnname>] [--loglocation=<loglocation>] [--logprefix=<logprefix>] [--candidatesinfiles]
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
  --outputcsv=<outputcsv>            Output file.
  --imageroot=<imageroot>            Root location of the actual images [default: /psdb3/images/].
  --tablename=<tablename>            Database table name to update [default: atlas_diff_objects].
  --columnname=<columnname>          Database column name to update [default: zooniverse_score].
  --loglocation=<loglocation>        Log file location [default: /tmp/]
  --logprefix=<logprefix>            Log prefix [default: ml_keras_]
  --update                           Update the database.
  --candidatesinfiles                Interpret the inline candidate IDs as a files containing candidates.

Example:
  python %s ~/config.pso3.gw.warp.yaml --ps1classifier=/data/db4data1/scratch/kws/training/ps1/20190115/ps1_20190115_400000_1200000.best.hdf5 --listid=4 --outputcsv=/tmp/pso3_list_4.csv
  python %s ../ps13pi/config/config.yaml --ps1classifier=/data/db4data1/scratch/kws/training/ps1/20190115/ps1_20190115_400000_1200000.best.hdf5 --listid=4 --outputcsv=/tmp/ps13pi_list_4.csv
  python %s /usr/local/ps1code/gitrelease/atlas/config/config4_db1_readonly.yaml /tmp/candidates.txt --hkoclassifier=/usr/local/ps1code/gitrelease/tf_trained_classifiers/02a_asteroids_good330000_bad990000_s3_20230405_classifier.h5 --mloclassifier=/usr/local/ps1code/gitrelease/tf_trained_classifiers/asteroids136521_good13479_bad450000_20x20_skew3_signpreserve_20200819mlo_classifier.h5 --sthclassifier=/usr/local/ps1code/gitrelease/tf_trained_classifiers/03a_asteroids_good320000_bad960000_s3_20230303_classifier.h5 --chlclassifier=/usr/local/ps1code/gitrelease/tf_trained_classifiers/04a_asteroids_good380000_bad1140000_s3_20230213_classifier.h5 --outputcsv=/db4/tc_logs/atlas4/ml_tf_keras_20230502_1017.csv --update --candidatesinfiles
  python %s ../../ps13pi/config/config.yaml --ps1classifier=/data/db4data1/scratch/kws/training/ps1/20190115/ps1_20190115_400000_1200000.best.hdf5 --ps2classifier=/data/db0jbod05/training/ps2/ps2_good95824_bad287472_s3_20230707_20x20_classifier.h5 --listid=4 --imageroot=/db0/images/ --loglocation=/db0/tc_logs/ps13pi/ --update

"""
import sys
__doc__ = __doc__ % (sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0])
from docopt import docopt
from gkutils.commonutils import Struct, cleanOptions, readGenericDataFile, dbConnect, splitList, parallelProcess
import sys, csv, os, datetime
from runKerasTensorflowClassifierOnPSATImages import getObjectsByList, runKerasTensorflowClassifier, updateTransientRBValue


def worker(num, db, objectListFragment, dateAndTime, firstPass, miscParameters, q):
    """thread worker function"""
    # Redefine the output to be a log file.
    options = miscParameters[0]
    sys.stdout = open('%s%s_%s_%d.log' % (options.loglocation, options.logprefix, dateAndTime, num), "w")

    # Override the full candidate list with a sublist of candidates.
    options.candidate = [str(x['id']) for x in objectListFragment]
    # If we've read the candidates from a file, don't try to open the file again!
    options.candidatesinfiles = None

    objectsForUpdate = runKerasTensorflowClassifier(options, processNumber = num)

    # Write the objects for update onto a Queue object
    print ("Adding %d objects onto the queue." % len(objectsForUpdate))

    q.put(objectsForUpdate)

    print ("Process complete.")
    print ("DB Connection Closed - exiting")

    return 0



def runKerasTensorflowClassifierMultiprocess(opts):

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

    db = []

    conn = dbConnect(hostname, username, password, database)
    if not conn:
        print("Cannot connect to the database")
        return 1

    # 2023-03-25 KWS MySQLdb disables autocommit by default. Switch it on globally.
    conn.autocommit(True)

    # If the list isn't specified assume it's the Eyeball List.
    if options.listid is not None:
        try:
            detectionList = int(options.listid)
            if detectionList < 0 or detectionList > 8:
                print ("Detection list must be between 0 and 8")
                return 1
        except ValueError as e:
            sys.exit("Detection list must be an integer")

    # 2018-07-31 KWS We have PS1 data. Don't bother with the HKO/MLO ATLAS data.
    ps1Data = False
    if options.ps1classifier:
        ps1Data = True

    objectList = []
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
        objectList = getObjectsByList(conn, database, listId = int(options.listid), ps1Data = ps1Data)


    # 2019-06-07 KWS For reasons not entirely clear, Tensorflow seems to exhaust every last
    #                bit of CPU and memory.  So let's divide the list by 10 if the list is
    #                larger than 10000 in size.

    if len(objectList) > 100:
        bin, subLists = splitList(objectList, bins=16)
    else:
        subLists = [objectList]

    for l in subLists:
        currentDate = datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S")
        (year, month, day, hour, min, sec) = currentDate.split(':')
        dateAndTime = "%s%s%s_%s%s%s" % (year, month, day, hour, min, sec)

        objectsForUpdate = []

        if len(objectList) > 0:
            # 2019-08-24 KWS Hard-wire the number of workers.
            nProcessors, listChunks = splitList(l, bins=28)

            print ("%s Parallel Processing..." % (datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S")))
            objectsForUpdate = parallelProcess(db, dateAndTime, nProcessors, listChunks, worker, miscParameters = [options])
            print ("%s Done Parallel Processing" % (datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S")))

            print ("TOTAL OBJECTS TO UPDATE = %d" % len(objectsForUpdate))

    #    if len(objectsForUpdate) > 0 and options.update:
    #        updateObjects(conn, objectsForUpdate)

        # Sort the combined list.
        objectsForUpdate = sorted(objectsForUpdate, key = lambda x: x[1])

        if options.outputcsv is not None:
            with open(options.outputcsv, 'w') as f:
                for row in objectsForUpdate:
                    print(row[0], row[1])
                    f.write('%s,%f\n' % (row[0], row[1]))

        if options.update:
            for row in objectsForUpdate:
                updateTransientRBValue(conn, row[0], row[1], ps1Data = ps1Data)

    conn.close()



def main():
    opts = docopt(__doc__, version='0.1')
    opts = cleanOptions(opts)

    # Use utils.Struct to convert the dict into an object for compatibility with old optparse code.
    options = Struct(**opts)
    runKerasTensorflowClassifierMultiprocess(options)


if __name__=='__main__':
    main()
