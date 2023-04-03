#!/usr/bin/env python
"""Use "stampstorm04 to generate "good" and "bad" training set data for Machine Learning.
Note that this code will generate "good" data from known asteroids, and "bad" data from
everything else that has pvr and ptr = 0. This code only works on the database created from
ddc files. Additionally, it currently does NOT attempt to download any missing exposures.
It assumes that all the required exposures are already downloaded.

Usage:
  %s <configFile> [<mjds>...] [--stampSize=<n>] [--stampLocation=<location>] [--test] [--downloadthreads=<threads>] [--stampThreads=<threads>] [--camera=<camera>] [--goodAsReal] [--stampsToGenerate=<stampsToGenerate>]
  %s (-h | --help)
  %s --version

Options:
  -h --help                               Show this screen.
  --version                               Show version.
  --test                                  Just do a quick test.
  --stampSize=<n>                         Size of the postage stamps if requested [default: 40].
  --stampLocation=<location>              Default place to store the stamps. [default: /tmp].
  --camera=<camera>                       Which camera [default: 02a].
  --downloadthreads=<threads>             The number of threads (processes) to use [default: 5].
  --stampThreads=<threads>                The number of threads (processes) to use [default: 28].
  --goodAsReal                            Get the good list, NOT the asteroids as the "real" data.
  --stampsToGenerate=<stampsToGenerate>   Which stamps do we want to generate (real | bogus | all) [default: all].

Example:
  %s ~/config4_readonly.yaml 58362 --stampLocation=/export/raid/db4data1/scratch/kws/training/atlas/hko_58362o
  %s ../atlas/config/config4_db1_readonly.yaml 59909 --stampLocation=/export/raid/db4data1/scratch/atls/training/atlas/sth_59909 --camera=03a --stampsToGenerate=real
  %s ../atlas/config/config4_db1_readonly.yaml 59901 59902 59903 59904 59905 59906 59907 59908 59909 59910 --stampLocation=/export/raid/db4data1/scratch/atls/training/atlas/sth_59909 --camera=03a --stampsToGenerate=bogus
"""
import sys
__doc__ = __doc__ % (sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0])
from docopt import docopt
import os, MySQLdb, shutil, re, csv, subprocess
from gkutils.commonutils import Struct, cleanOptions, dbConnect, splitList, parallelProcess
# 2023-04-03 KWS We need to have makeATLASStamps on the PYTHONPATH.
from makeATLASStamps import doRsync
import datetime
from collections import defaultdict

STAMPSTORM04 = "/atlas/bin/stampstorm04"
LOG_FILE_LOCATION = '/' + os.uname()[1].split('.')[0] + '/tc_logs/'
LOG_PREFIX_EXPOSURES = 'background_exposure_downloads'

def getKnownAsteroids(conn, camera, mjd, pkn = 900):
    """
    Get the asteroids.
    """
    import MySQLdb

    try:
        cursor = conn.cursor (MySQLdb.cursors.DictCursor)

        cursor.execute ("""
            select distinct m.obs, d.x, d.y, d.mag, d.dmag, d.ra, d.dec
              from atlas_detectionsddc d, atlas_metadataddc m
             where m.id = d.atlas_metadata_id
               and m.obs like concat(%s, '%%')
               and m.mjd > %s
               and m.mjd < %s
               and d.pkn > %s
               and d.det = 0
               and d.mag > 0.0
          order by m.obs
        """, (camera, mjd, mjd + 1, pkn))
        resultSet = cursor.fetchall ()

        cursor.close ()

    except MySQLdb.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    return resultSet


#def getJunk(conn, camera, mjd):
#    """
#    Get the garbage.
#    """
#    import MySQLdb
#
#    try:
#        cursor = conn.cursor (MySQLdb.cursors.DictCursor)
#
#        cursor.execute ("""
#            select distinct m.obs, d.x, d.y, d.mag, d.dmag, d.ra, d.dec
#              from atlas_detectionsddc d, atlas_metadataddc m
#             where m.id = d.atlas_metadata_id
#               and m.obs like concat(%s, '%%')
#               and m.mjd > %s
#               and m.mjd < %s
#               and d.pvr = 0
#               and d.ptr = 0
#               and d.pkn = 0
#               and d.det = 0
#               and d.mag > 0.0
#          order by m.obs
#        """, (camera, mjd, mjd+1))
#        resultSet = cursor.fetchall ()
#
#        cursor.close ()
#
#    except MySQLdb.Error as e:
#        print("Error %d: %s" % (e.args[0], e.args[1]))
#
#    return resultSet


# 2019-11-25 KWS Complete rewrite of how we acquire the junk data. Detections with pvr & ptr = 0 are NEVER
#                promoted anyway, so let's stop using them.
# 2020-08-21 KWS Force the indexes to use and switch to JOIN syntax. Seems to help.
def getJunk(conn, camera, mjd):
    """
    Get the garbage.
    """
    import MySQLdb

    try:
        cursor = conn.cursor (MySQLdb.cursors.DictCursor)

        cursor.execute ("""
            select distinct m.obs, d.x, d.y, d.mag, d.dmag, d.ra, d.dec, o.detection_list_id
              from atlas_diff_objects o
              join atlas_detectionsddc d force index (idx_atlas_metadata_id) on o.id = d.atlas_object_id
              join atlas_metadataddc m force index (idx_mjd) on m.id = d.atlas_metadata_id
              join tcs_latest_object_stats s on s.id = o.id
             where m.obs like concat(%s, '%%')
               and m.mjd > %s
               and m.mjd < %s
               and o.realbogus_factor < 0.01
           --    and o.zooniverse_score < 0.2
           --    and o.zooniverse_score < 0.05
               and d.pkn = 0
               and d.det = 0
               and d.mag > 0.0
               and d.image_group_id is not null
               and o.realbogus_factor is not null
               and d.image_group_id is not null
               and o.detection_list_id = 0
               and s.external_crossmatches is null
          order by m.obs
        """, (camera, mjd, mjd+1))
        resultSet = cursor.fetchall ()

        cursor.close ()

    except MySQLdb.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    return resultSet


def getGood(conn, camera, mjd):
    """
    Get good objects.
    """
    import MySQLdb

    try:
        cursor = conn.cursor (MySQLdb.cursors.DictCursor)

        cursor.execute ("""
            select distinct m.obs, d.x, d.y, d.mag, d.dmag, d.ra, d.dec
              from atlas_detectionsddc d, atlas_metadataddc m, atlas_diff_objects o
             where m.id = d.atlas_metadata_id
               and o.id = d.atlas_object_id
               and m.obs like concat(%s, '%%')
               and m.mjd > %s
               and m.mjd < %s
               and d.image_group_id is not null
               and o.detection_list_id = 2
          order by m.obs

        """, (camera, mjd, mjd+1))
        resultSet = cursor.fetchall ()

        cursor.close ()

    except MySQLdb.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    return resultSet


def stampStormWrapper(exposureList, stampSize, stampLocation, objectType='good'):

    for exp in exposureList:
        camera = exp[0:3]
        mjd = exp[3:8]
        imageName = '/atlas/diff/' + camera + '/' + mjd + '/' + exp + '.diff.fz'
        #inFile = stampLocation + '/' + objectType + exp + '.txt'
        # The code changes directory to the one below where the text files have been generated.
        # stampstorm04 can't deal with filenames longer than 80 characters, so use a relative
        # directory filename instead
        inFile = '../' + objectType + exp + '.txt'

        if objectType == 'good':
            goodDir = stampLocation+'/good'
            if not os.path.exists(goodDir):
                print("creating"+goodDir)
                try:
                    os.makedirs(goodDir)
                except FileExistsError as e:
                    pass
            os.chdir(goodDir)

        else:
            badDir = stampLocation+'/bad'
            if not os.path.exists(badDir):
                try:
                    os.makedirs(badDir)   
                except FileExistsError as e:
                    pass
            os.chdir(badDir)

        p = subprocess.Popen([STAMPSTORM04, inFile, imageName, objectType, str(stampSize/2)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, errors = p.communicate()

        if output.strip():
            print(output)
        if errors.strip():
            print(errors)

    return


def workerImageDownloader(num, db, listFragment, dateAndTime, firstPass, miscParameters):
    """thread worker function"""
    # Redefine the output to be a log file.
    sys.stdout = open('%s%s_%s_%d.log' % (LOG_FILE_LOCATION, LOG_PREFIX_EXPOSURES, dateAndTime, num), "w")

    # Call the postage stamp downloader
    objectsForUpdate = doRsync(listFragment, miscParameters[0])
    print ("Process complete.")
    return 0


def workerStampStorm(num, db, listFragment, dateAndTime, firstPass, miscParameters):
    """thread worker function"""
    # Redefine the output to be a log file.
    sys.stdout = open('%s%s_%s_%d.log' % (LOG_FILE_LOCATION, LOG_PREFIX_EXPOSURES, dateAndTime, num), "w")

    stampStormWrapper(listFragment, miscParameters[0], miscParameters[1], objectType = miscParameters[2])    

    print("Process complete.")
    return 0 


# 2018-09-04 KWS Make sure the tiles files do not get written into the good.txt and bad.txt files.
def getGoodBadFiles(path):
    if os.path.exists(path+'/good'):   
        with open(path+'/good.txt', 'w') as good:
            for file in os.listdir(path+'/good'):
                if 'tiles' not in file:
                    good.write(file+'\n')

    if os.path.exists(path+'/bad'):   
        with open(path+'/bad.txt', 'w') as bad:
            for file in os.listdir(path+'/bad'):
                if 'tiles' not in file:
                    bad.write(file+'\n')
    print("Generated good and bad files")


def getATLASTrainingSetCutouts(opts):
    if type(opts) is dict:
        options = Struct(**opts)
    else:
        options = opts

    import yaml
    with open(options.configFile) as yaml_file:
        config = yaml.load(yaml_file)

    stampSize = int(options.stampSize)
    mjds = options.mjds
    if not mjds:
        print ("No MJDs specified")
        return 1

    downloadThreads = int(options.downloadthreads)
    stampThreads = int(options.stampThreads)
    stampLocation = options.stampLocation
    if not os.path.exists(stampLocation):
        os.makedirs(stampLocation)
    username = config['databases']['local']['username']
    password = config['databases']['local']['password']
    database = config['databases']['local']['database']
    hostname = config['databases']['local']['hostname']

    conn = dbConnect(hostname, username, password, database)
    if not conn:
        print("Cannot connect to the database")
        return 1

    currentDate = datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S")
    (year, month, day, hour, min, sec) = currentDate.split(':')
    dateAndTime = "%s%s%s_%s%s%s" % (year, month, day, hour, min, sec)
    header="x,y,mag,dmag,ra,dec,obs".split(',')

    # **** Generate the "real" stamps. ****
    if options.stampsToGenerate == 'real' or options.stampsToGenerate == 'all':
        asteroidExpsDict = defaultdict(list)
        for mjd in mjds:
            if options.goodAsReal:
                asteroidExps = getGood(conn, options.camera, int(mjd))
            else:
                asteroidExps = getKnownAsteroids(conn, options.camera, int(mjd), pkn = 900)
            for exp in asteroidExps:
                asteroidExpsDict[exp['obs']].append(exp)
    
        # Now create the files.  We need to have x, y as the first two items.

        if asteroidExpsDict:
            #m.obs, d.x, d.y, d.mag, d.dmag, d.ra, d.dec

            exposureList = []
            for k,v in asteroidExpsDict.items():
                exposureList.append(k)
                with open(stampLocation + '/' + 'good' + k + '.txt', 'w') as csvfile:
                    w = csv.DictWriter(csvfile, fieldnames=header, delimiter=' ')
                    #w.writeheader()
                    for row in v:
                        w.writerow(row)
            # So now let stampstorm do its stuff

            if len(exposureList) > 0:
                nProcessors, listChunks = splitList(exposureList, bins = stampThreads)

                print("%s Parallel Processing Good objects..." % (datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S")))
                parallelProcess([], dateAndTime, nProcessors, listChunks, workerStampStorm, miscParameters = [stampSize, stampLocation, 'good'], drainQueues = False)
                print("%s Done Parallel Processing" % (datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S")))


    # **** Generate the "bogus" stamps. ****

    if options.stampsToGenerate == 'bogus' or options.stampsToGenerate == 'all':
        junkExpsDict = defaultdict(list)
        for mjd in mjds:
            junkExps = getJunk(conn, options.camera, int(mjd))
            for exp in junkExps:
                # Only append the object if it's in the junk.
                if exp['detection_list_id'] == 0:
                    # Remove the detection_list_id
                    del exp['detection_list_id']
                    junkExpsDict[exp['obs']].append(exp)

        if junkExpsDict:
            exposureList = []
            for k,v in junkExpsDict.items():
                exposureList.append(k)
                with open(stampLocation + '/' + 'bad' + k + '.txt', 'w') as csvfile:
                    w = csv.DictWriter(csvfile, fieldnames=header, delimiter=' ')
                    #w.writeheader()
                    for row in v:
                        w.writerow(row)

            if len(exposureList) > 0:
                nProcessors, listChunks = splitList(exposureList, bins = stampThreads)

                print("%s Parallel Processing Bad objects..." % (datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S")))
                parallelProcess([], dateAndTime, nProcessors, listChunks, workerStampStorm, miscParameters = [stampSize, stampLocation, 'bad'], drainQueues = False)
                print("%s Done Parallel Processing" % (datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S")))
    
    conn.close()
    getGoodBadFiles(stampLocation)


def main():
    opts = docopt(__doc__, version='0.1')
    opts = cleanOptions(opts)

    # Use utils.Struct to convert the dict into an object for compatibility with old optparse code.
    options = Struct(**opts)
    getATLASTrainingSetCutouts(options)


if __name__=='__main__':
    main()
    
