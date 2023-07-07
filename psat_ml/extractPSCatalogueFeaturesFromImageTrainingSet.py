#!/usr/bin/env python
"""Read the selected image training set filenames and extract the associated catalogue features

Usage:
  %s <configfile> <good> <bad> [--output=<outputfile>]
  %s (-h | --help)
  %s --version

Options:
  -h --help                     Show this screen.
  --version                     Show version.
  --output=<outputfile>         Output file [default: /tmp/catalogue_training_set.mat].

  Example:
    %s ../../../../config/config.yaml good.txt bad.txt
"""
import sys
__doc__ = __doc__ % (sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0])
from docopt import docopt
import os, MySQLdb, shutil, re
from gkutils.commonutils import find, Struct, cleanOptions, dbConnect, readGenericDataFile
import MySQLdb
import subprocess
import io
import h5py
import scipy.io as sio
import numpy as np

class EmptyClass:
    """EmptyClass.
    """

    pass


features = ["sky", \
#            "sky_sigma", \
#            "psf_chisq", \
#            "cr_nsigma", \
#            "ext_nsigma", \
            "psf_major", \
            "psf_minor", \
            "psf_theta", \
            "psf_qf", \
#            "psf_ndof", \
            "psf_npix", \
            "moments_xx", \
            "moments_xy", \
            "moments_yy", \
#            "flags", \
#            "n_frames", \
#            "padding", \
            "diff_npos", \
            "diff_fratio", \
            "diff_nratio_bad", \
            "diff_nratio_mask", \
            "diff_nratio_all", \
#            "diff_r_m", \
#            "diff_r_p", \
#            "diff_sn_m", \
#            "diff_sn_p", \
#            "flags2", \
#            "moments_r1", \
#            "moments_rh", \
            "psf_qf_perfect"]

# Need to intelligently acquire the object images
def getCatalogueData(conn, features, candidate):
    """getCatalogueData.

    Args:
        conn:
        features:
        candidate:
    """

    try:
        cursor = conn.cursor(MySQLdb.cursors.DictCursor)
        query = """ select %s
                      from tcs_transient_reobservations r, tcs_cmf_metadata m
                     where m.id = r.tcs_cmf_metadata_id
                       and transient_object_id = %s
                       and truncate(mjd_obs, 3) = %s
                       and imageid = %s
                       and ipp_idet = %s
                """ % (",".join(features), candidate['id'], candidate['mjd'], candidate['imageid'], candidate['ipp_idet'])
        cursor.execute(query)

        resultSet = cursor.fetchall ()
        cursor.close ()

        if len(resultSet) != 1:
            # Maybe the row is in the tcs_transient_objects table
            cursor = conn.cursor(MySQLdb.cursors.DictCursor)
            query = """ select %s
                          from tcs_transient_objects r, tcs_cmf_metadata m
                         where m.id = r.tcs_cmf_metadata_id
                           and r.id = %s
                           and truncate(mjd_obs, 3) = %s
                           and imageid = %s
                           and ipp_idet = %s
                """ % (",".join(features), candidate['id'], candidate['mjd'], candidate['imageid'], candidate['ipp_idet'])
            cursor.execute(query)

            resultSet = cursor.fetchall ()
            cursor.close ()

        if len(resultSet) != 1:
            print("No data for this object.")


    except MySQLdb.Error as e:
        print("Error %d: %s" % (e.args[0], e.args[1]))
        sys.exit (1)

    return resultSet


def getCatalogueFeaturesProperWay(conn, filename, good = True):
    """getCatalogueFeaturesProperWay.

    Args:
        conn:
        filename:
        good:
    """
    trainFiles = []
    n = len(features)
    files = open(filename).readlines()

    counter = 0


    data = []
    for file in files:
        filename = file.strip().split('/')[-1]
        id, mjd, imageid, ipp_idet, type = filename.split('_')
        object = {'id':id, 'mjd':mjd, 'imageid':imageid, 'ipp_idet': ipp_idet}
        dataRow = getCatalogueData(conn, features, object)
        if len(dataRow) == 1:
            trainFiles.append(filename)
            row = []
            for j,feature in enumerate(features):
                row.append(dataRow[0][feature])
            data.append(row)


    m = len(data)
    if good:
        y = np.ones((m,))
    else:
        y = np.zeros((m,))

    X = np.array(data)

    train_files = np.array(trainFiles)

    arrays = EmptyClass()
    arrays.X = X[:int(.75*m)]
    arrays.testX = X[int(.75*m):]
    arrays.y = y[:int(.75*m)]
    arrays.testy = y[int(.75*m):]
    arrays.train_files = train_files[:int(.75*m)]
    arrays.test_files = train_files[int(.75*m):]

    return arrays

def getCatalogueFeatures(conn, options):
    """getCatalogueFeatures.

    Args:
        conn:
        options:
    """
    good = getCatalogueFeaturesProperWay(conn, options.good)
    bad = getCatalogueFeaturesProperWay(conn, options.bad, good=False)

    # So - now column stack them.

    X = np.concatenate((good.X, bad.X))
    print("X")
    print(X)
    testX = np.concatenate((good.testX, bad.testX))
    print("testX")
    print(testX)
    y = np.concatenate((good.y, bad.y))
    print("y")
    print(y)
    testy = np.concatenate((good.testy, bad.testy))
    print("testy")
    print(testy)
    train_files = np.concatenate((good.train_files, bad.train_files))
    print("train_files")
    print(train_files)
    test_files = np.concatenate((good.test_files, bad.test_files))
    print("test_files")
    print(test_files)

    # Now randomize the data:
    m = len(X)
    order = np.random.permutation(m)
    X = X[order]
    y = y[order]
    train_files = train_files[order]

    n = len(testX)
    order = np.random.permutation(n)
    testX = testX[order]
    testy = testy[order]
    test_files = test_files[order]

    # Write the file.
    sio.savemat(options.output,{"X":X,"y":y,\
                "testX":testX, "testy":testy, "train_files":train_files, \
                "test_files":test_files, "features":features})

def main(argv = None):
    """main.

    Args:
        argv:
    """
    opts = docopt(__doc__, version='0.1')
    opts = cleanOptions(opts)

    # Use utils.Struct to convert the dict into an object for compatibility with old optparse code.
    options = Struct(**opts)

    import yaml
    with open(options.configfile) as yaml_file:
        config = yaml.load(yaml_file)

    username = config['databases']['local']['username']
    password = config['databases']['local']['password']
    database = config['databases']['local']['database']
    hostname = config['databases']['local']['hostname']

    conn = dbConnect(hostname, username, password, database, quitOnError = True)

    catalogueFeatures = getCatalogueFeatures(conn, options)

    conn.commit()


    return





if __name__ == '__main__':
    main()
