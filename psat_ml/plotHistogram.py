#!/usr/bin/env python
"""Plot histogram to show performance of the specified trained classifier.

Usage:
  %s <classifierFile> [--outputFile=<file>] [--threshold=<threshold>] [--panellabel=<panellabel>] [--log] [--title=<title>]
  %s (-h | --help)
  %s --version

Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --outputFile=<file>          Output file. If not defined, show plot.
  --threshold=<threshold>      Threshold at which the classifier is in use. Plots a dotted line on the histogram.
  --panellabel=<panellabel>    Plot label (e.g. 'a)' ) [default: ]
  --log                        Plot log(y) instead of y.
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

def plotHistogram(dataSeries, threshold = None, outputFile = None, logY = False, panellabel='', title=None):

    lenSeries0 = len(dataSeries[0])
    lenSeries1 = len(dataSeries[1])
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    bins = n.linspace(0.0,1.0,41)

    ml = MultipleLocator(0.1)
    ax1.xaxis.set_major_locator(ml)

    ax1 = fig.add_subplot(111)
    ax1.hist(n.array(dataSeries[0]), bins=bins, color = 'r', label = "bogus", edgecolor='black', linewidth=0.5, alpha = 0.5)
    ax1.set_ylabel('Number of Objects')
    for tl in ax1.get_yticklabels():
        tl.set_color('k')

    ax1.hist(dataSeries[1], bins=bins, color='g', label = "real", edgecolor='black', linewidth=0.5, alpha = 0.8)

    ax1.set_xlabel('Realbogus Factor')
    if title:
        ax1.set_title(title)
    ax1.legend(loc=1)
    ml = MultipleLocator(0.2)
    ax1.xaxis.set_minor_locator(ml)
    ax1.get_xaxis().set_tick_params(which='both', direction='out')
    ax1.set_xlim(0.0, 1.0)
    ax1.text(0.1, 0.95, panellabel, transform=ax1.transAxes, va='top', size=MEDIUM_SIZE, weight='bold')

    if logY:
        ax1.set_yscale('log')

    if threshold is not None:
        ax1.axvline(x=float(threshold),color='k',linestyle='--')

    plt.tight_layout()

    if outputFile is not None:
        plt.savefig(outputFile)
    else:
        plt.show()


def doPlots(options):
    dataRows = readGenericDataFile(options.classifierFile, fieldnames = ['file','label','score'], delimiter=',')

    goods = []
    bads = []
    for row in dataRows:
        if row['label'] == '1':
            goods.append(float(row['score']))
        elif row['label'] == '0':
            bads.append(float(row['score']))

    plotHistogram([bads, goods], threshold = options.threshold, outputFile = options.outputFile, logY = options.log, panellabel = options.panellabel, title = options.title)


def main():
    opts = docopt(__doc__, version='0.1')
    opts = cleanOptions(opts)
    options = Struct(**opts)

    doPlots(options)


if __name__=='__main__':
    main()


