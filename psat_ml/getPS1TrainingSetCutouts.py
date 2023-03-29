#!/usr/bin/env python
"""Create symlinks to the set of Pan-STARRS training set images.
The cutouts are already done.

Usage:
  %s <configFile> [--stampLocation=<location>] [--test]
  %s (-h | --help)
  %s --version

Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --test                       Just do a quick test.
  --stampLocation=<location>   Default place to store the stamps. [default: /tmp]

"""
import sys
__doc__ = __doc__ % (sys.argv[0], sys.argv[0], sys.argv[0])
from docopt import docopt
import os, MySQLdb, shutil, re, csv, subprocess
from gkutils.commonutils import Struct, cleanOptions, dbConnect, doRsync
from datetime import datetime
from datetime import timedelta
from collections import defaultdict
from gkmultiprocessingUtils import *




# Get the objects we've collected into the attic over the years
# which are labelled as movers by the ephemeric check software

def getGoodPS1Objects(conn, listId):
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
             union
            select id from tcs_transient_objects
             where detection_list_id = 2
        """, (listId,))
        resultSet = cursor.fetchall ()

        cursor.close ()

    except MySQLdb.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    return resultSet


def getBadPS1Objects(conn, listId, rbThreshold = 0.1):
    """
    Get "bad" objects
    """
    import MySQLdb

    try:
        cursor = conn.cursor (MySQLdb.cursors.DictCursor)

        cursor.execute ("""
            select distinct o.id
              from tcs_transient_objects o
             where confidence_factor < %s 
               and detection_list_id = %s
               and sherlockClassification is not null
        """, (rbThreshold, listId,))
        resultSet = cursor.fetchall ()

        cursor.close ()

    except MySQLdb.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    return resultSet



def getImagesForObject(conn, objectId):
    """
    Get images for an object.
    """
    import MySQLdb

    try:
        cursor = conn.cursor (MySQLdb.cursors.DictCursor)

        cursor.execute ("""
            select i.image_filename
            from tcs_postage_stamp_images i, tcs_transient_objects o
            where i.image_filename like concat(o.id, '%%')
            and i.image_filename not like concat(o.id, '%%4300000000%%')
            and pss_error_code = 0
            and i.image_type = 'diff'
            and o.id = %s
        """, (objectId,))
        resultSet = cursor.fetchall ()

        cursor.close ()

    except MySQLdb.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))

    return resultSet


def getTrainingSetImages(conn, imageHome = '/psdb2/images/ps13pi/'):
    goodObjects = getGoodPS1Objects(conn, listId = 5)

    print("Number of good objects = ", len(goodObjects))
    class ImageSet:
        pass

    imgs = ImageSet()

    goodImages = []
    for candidate in goodObjects:
        images = getImagesForObject(conn, candidate['id'])

        for image in images:
            mjd = image['image_filename'].split('_')[1].split('.')[0]
            imageName = imageHome+mjd+'/' + image['image_filename']+'.fits'
            goodImages.append(imageName)

    # 2018-07-27 KWS Sort the images in reverse order (most recent at the top).
    #                This should make reading the data from disk quicker.
    goodImages.sort(reverse=True)
    imgs.good = goodImages

    badObjects = getBadPS1Objects(conn, listId = 0)
    print("Number of bad objects = ", len(badObjects))
    
    badImages = []
    for candidate in badObjects:
        images = getImagesForObject(conn, candidate['id'])

        for image in images:
            mjd = image['image_filename'].split('_')[1].split('.')[0]
            imageName = imageHome+mjd+'/' + image['image_filename']+'.fits'
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
    with open(path+'/good.txt', 'w') as good:
        for file in images.good:
            good.write(file+'\n')
        

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


    images = getTrainingSetImages(conn)

    writePS1GoodBadFiles(options.stampLocation, images)

    conn.close()


def main():
    opts = docopt(__doc__, version='0.1')
    opts = cleanOptions(opts)
    options = Struct(**opts)

    getPS1TrainingSetCutouts(options)


if __name__=='__main__':
    main()
    
