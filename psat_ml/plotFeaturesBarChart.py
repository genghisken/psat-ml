#!/usr/bin/env python
"""Plot bar chart of feature importance for RF classifier.

Usage:
  %s <classifierFile> [--outputFile=<file>] [--title=<title>]
  %s (-h | --help)
  %s --version

Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --outputFile=<file>          Output file. If not defined, show plot.
  --title=<title>              Plot title

"""
import sys
__doc__ = __doc__ % (sys.argv[0], sys.argv[0], sys.argv[0])
from docopt import docopt
import os, MySQLdb, shutil, re, csv, subprocess
from gkutils.commonutils import Struct, cleanOptions, readGenericDataFile
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as n

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 25
TINY_SIZE = 12
plt.rc('font', size=SMALL_SIZE)                   # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)            # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)           # fontsize of the x and y labels
plt.rc('xtick', labelsize=TINY_SIZE)            # fontsize of the tick labels
plt.rc('ytick', labelsize=TINY_SIZE)            # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE - 1)               # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title
plt.rcParams["font.family"] = "serif"
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

def plotFeaturesBarChart(data, outputFile = None, title = None):

    lenData = len(data)
    fig = plt.figure()

    ax1 = fig.add_subplot(111)


    #totalImportance = n.sum(n.array([float(xe['importanceerror']) for xe in data]))
    totalImportance = n.array([float(xe['importanceerror']) for xe in data])

    print(totalImportance)

    ax1.bar(range(lenData), n.array([float(x['importance']) for x in data]), color="r", edgecolor='black', linewidth=0.5,
            yerr=n.array([float(xe['importanceerror']) for xe in data]), align="center", capsize=2, ecolor='b')
    ax1.set_xticks(range(lenData))
    ax1.set_xticklabels(n.array([f['feature'] for f in data]),rotation=90)
    ax1.set_xlim(-1, lenData)
    ax1.set_ylabel('Importance')
    ax1.set_xlabel('Feature Name')
    if title is not None:
        ax1.set_title(title)
    plt.tight_layout()

    if outputFile is not None:
        plt.savefig(outputFile)
    else:
        plt.show()


def doPlots(options):
    data = []
    dataRows = readGenericDataFile(options.classifierFile, fieldnames = ['importance','importanceerror','feature'], delimiter=',')
    for row in dataRows:
        data.append(row)

    plotFeaturesBarChart(data, outputFile = options.outputFile, title = options.title)


def main():
    opts = docopt(__doc__, version='0.1')
    opts = cleanOptions(opts)
    options = Struct(**opts)

    doPlots(options)


if __name__=='__main__':
    main()


