#!/usr/bin/env python
"""Create symlinks to the set of Pan-STARRS training set images.
The cutouts are already done.

Usage:
  %s <configFile> [--stampLocation=<location>] [--test] [--imageRoot=<imageRoot>] [--badrblower=<badrblower>] [--badrbupper=<badrbupper>] [--flagdate=<flagdate>] [--goodlist=<goodlist>] [--badlist=<badlist>] [--imagetype=<imagetype>] [--badaugment=<badaugment>] --camera=<camera>
  %s (-h | --help)
  %s --version

Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --test                       Just do a quick test.
  --stampLocation=<location>   Default place to store the stamps [default: /tmp].
  --imageRoot=<imageHome>      Location of the input images [default: /db0/images].
  --badrblower=<badrblower>    Set the bad RB threshold lower limit [default: 0.01].
  --badrbupper=<badrbupper>    Set the bad RB threshold upper limit [default: 0.3].
  --flagdate=<flagdate>        Flag date before which we will not request images, e.g. because optics have changed [default: 20100101].
  --goodlist=<goodlist>        Good list number - could be 2 (good) or 5 (attic) or 6 (movers) [default: 2].
  --badlist=<badlist>          Bad list number [default: 0].
  --badaugment=<badaugment>    Bad curated custom list number (used to augment the bad list to improve detection of artefacts).
  --imagetype=<imagetype>      Image type (good | bad | all) [default: all].
  --camera=<camera>            Which detector are the images coming from [default: gpc1]?

E.g.:
  %s ../../ps13pi/config/config_readonly.yaml --imageRoot=/db0/images --badrblower=0.01 --badrbupper=0.3 --flagdate=20240101 --stampLocation=/export/dbjbod5/db0jbod05/training/ps2 --camera=gpc2 --badaugment=4

"""
import sys
__doc__ = __doc__ % (sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0])
from docopt import docopt
import os, MySQLdb, shutil, re, csv, subprocess
from gkutils.commonutils import Struct, cleanOptions, dbConnect, parallelProcess, splitList
from datetime import datetime
from datetime import timedelta
from collections import defaultdict




# Get the objects we've collected into the attic over the years
# which are labelled as movers by the ephemeric check software

# 2023-07-08 KWS Set a light RB threshold to clip out obviously bad movers.
def getGoodPS1Objects(conn, listId, flagDate = '2010-01-01', rbThreshold = 0.05):
    """
    Get "good" objects
    """
    import MySQLdb

    try:
        cursor = conn.cursor (MySQLdb.cursors.DictCursor)

        cursor.execute ("""
            select distinct o.id
              from tcs_transient_objects o, tcs_object_comments c
             where o.id = c.transient_object_id
               and detection_list_id = %s
               and observation_status = 'mover'
               and confidence_factor is not null
               and (comment like 'EPH:%%' or comment like 'MPC:%%')
               and comment REGEXP '\\\\([0-5]\.[0-9]\+ arcsec\\\\)'
               and followup_flag_date > %s
               and confidence_factor > %s
             union
            select id from tcs_transient_objects
             where detection_list_id = 2
               and followup_flag_date > %s
        """, (listId, flagDate, rbThreshold, flagDate,))
        resultSet = cursor.fetchall ()

        cursor.close ()

    except MySQLdb.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    return resultSet


def getBadPS1Objects(conn, listId, rbThresholdLower = 0.05, rbThresholdUpper = 0.3, flagDate = '2010-01-01', augmentedList = None):
    """
    Get "bad" objects
    """
    import MySQLdb

    try:
        cursor = conn.cursor (MySQLdb.cursors.DictCursor)

        if augmentedList is not None:
            # Add curated list of a particular type of junk to the garbage list
            cursor.execute ("""
                select o.id
                  from tcs_transient_objects o
                 where confidence_factor >= %s 
                   and confidence_factor < %s
                   and detection_list_id = %s
                   and sherlockClassification is not null
                   and followup_flag_date > %s
                 union
                select g.transient_object_id as id
                  from tcs_object_groups g
                 where g.object_group_id = %s
            """, (rbThresholdLower, rbThresholdUpper, listId, flagDate, int(augmentedList)))

        else:
            cursor.execute ("""
                select distinct o.id
                  from tcs_transient_objects o
                 where confidence_factor >= %s 
                   and confidence_factor < %s
                   and detection_list_id = %s
                   and sherlockClassification is not null
                   and followup_flag_date > %s
            """, (rbThresholdLower, rbThresholdUpper, listId, flagDate,))

        resultSet = cursor.fetchall ()

        cursor.close ()

    except MySQLdb.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    return resultSet



def getImagesForObject(conn, objectId, camera = 'gpc1'):
    """
    Get images for an object.
    """
    import MySQLdb

    try:
        cursor = conn.cursor (MySQLdb.cursors.DictCursor)

        if camera == 'gpc2':
            cursor.execute ("""
                select i.image_filename
                from tcs_postage_stamp_images i
                where i.image_filename like concat(%s, '%%')
                and i.image_filename not like concat(%s, '%%4300000000%%')
                and pss_error_code = 0
                and i.image_type = 'diff'
                and i.filter like '%%00002'
            """, (objectId, objectId))
        else:
            # It can only be gpc1!
            cursor.execute ("""
                select i.image_filename
                from tcs_postage_stamp_images i
                where i.image_filename like concat(%s, '%%')
                and i.image_filename not like concat(%s, '%%4300000000%%')
                and pss_error_code = 0
                and i.image_type = 'diff'
                and i.filter like '%%00000'
            """, (objectId, objectId))

        resultSet = cursor.fetchall ()

        cursor.close ()

    except MySQLdb.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    return resultSet


def getTrainingSetImages(conn, options, database):

    if options.flagdate is not None:
        try:
            dateThreshold = '%s-%s-%s' % (options.flagdate[0:4], options.flagdate[4:6], options.flagdate[6:8])
        except:
            dateThreshold = '2010-01-01'


    class ImageSet:
        pass

    imgs = ImageSet()

    goodImages = []
    badImages = []

    if options.imagetype in ['all', 'good']:
        goodObjects = getGoodPS1Objects(conn, listId = int(options.goodlist), flagDate = dateThreshold)
        print("Number of good objects = ", len(goodObjects))

        for candidate in goodObjects:
            images = getImagesForObject(conn, candidate['id'], camera = options.camera)

            for image in images:
                mjd = image['image_filename'].split('_')[1].split('.')[0]
                imageName = options.imageRoot + '/' + database + '/' + mjd + '/' + image['image_filename']+'.fits'
                goodImages.append(imageName)

        # 2018-07-27 KWS Sort the images in reverse order (most recent at the top).
        #                This should make reading the data from disk quicker.
        goodImages.sort(reverse=True)

    imgs.good = goodImages

    
    if options.imagetype in ['all', 'bad']:
        badObjects = getBadPS1Objects(conn, listId = int(options.badlist), rbThresholdLower = float(options.badrblower), rbThresholdUpper = float(options.badrbupper), flagDate = dateThreshold, augmentedList = options.badaugment)
        print("Number of bad objects = ", len(badObjects))

        for candidate in badObjects:
            images = getImagesForObject(conn, candidate['id'], camera = options.camera)

            for image in images:
                mjd = image['image_filename'].split('_')[1].split('.')[0]
                imageName = options.imageRoot + '/' + database + '/' + mjd + '/' + image['image_filename']+'.fits'
                badImages.append(imageName)

        badImages.sort(reverse=True)

    imgs.bad = badImages

    return imgs


def getGoodBadFiles(path):
       
    with open(path+'/good.txt', 'a') as good:
            for file in os.listdir(path+'/good'):
                    good.write(file+'\n')

    with open(path+'/bad.txt', 'a') as bad:
            for file in os.listdir(path+'/bad'):
                    bad.write(file+'\n')
    print("Generated good and bad files")

def writePS1GoodBadFiles(path, images):

    if images.good:
        with open(path+'/good.txt', 'w') as good:
            for file in images.good:
                good.write(file+'\n')

    if images.bad:
        with open(path+'/bad.txt', 'w') as bad:
            for file in images.bad:
                bad.write(file+'\n')

    print("Generated good and bad files")



def getPS1TrainingSetCutouts(opts):

    if type(opts) is dict:
        options = Struct(**opts)
    else:
        options = opts

    import yaml
    with open(options.configFile) as yaml_file:
        config = yaml.load(yaml_file)

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


    images = getTrainingSetImages(conn, options, database)

    writePS1GoodBadFiles(options.stampLocation, images)

    conn.close()


def main():
    opts = docopt(__doc__, version='0.1')
    opts = cleanOptions(opts)
    options = Struct(**opts)

    getPS1TrainingSetCutouts(options)


if __name__=='__main__':
    main()
    
