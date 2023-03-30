#!/usr/bin/env python
"""Plot histogram to show performance of the specified trained classifier.

Usage:
  %s <inputFile>... [--delimiter=<delimiter>] [--column=<column>] [--outputFile=<file>] [--binwidth=<binwidth>] [--threshold=<threshold>] [--log] [--xlabel=<xlabel>] [--ylabel=<ylabel>] [--binlower=<binlower>] [--binupper=<binupper>] [--majorticks=<majorticks>] [--minorticks=<minorticks>] [--plotlabel=<plotlabel>] [--panellabel=<panellabel>] [--ylimit=<ylimit>] [--alpha=<alpha>] [--colour=<colour>] [--leglabels=<leglabels>]
  %s (-h | --help)
  %s --version

Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --delimiter=<delimiter>      Delimiter to use [default: ,]
  --column=<column>            Column to plot. Use comma separation, no spaces if more than one [default: disc_mag]
  --outputFile=<file>          Output file. If not defined, show plot.
  --threshold=<threshold>      Threshold at which the classifier is in use. Plots a dotted line on the histogram.
  --xlabel=<xlabel>            x label [default: ]
  --ylabel=<ylabel>            y label [default: ]
  --plotlabel=<plotlabel>      Plot label (e.g. MLO ) [default: ]
  --panellabel=<panellabel>    Plot label (e.g. 'a)' ) [default: ]
  --binwidth=<binwidth>        Witdth of the bins [default: 0.2]
  --binlower=<binlower>        lower limit of the bin [default: 0]
  --binupper=<binupper>        upper limit of the bin [default: 1]
  --majorticks=<majorticks>    major ticks [default: 1.0]
  --minorticks=<minorticks>    minor ticks [default: 0.1]
  --ylimit=<ylimit>            hard wired y limit
  --colour=<colour>            Specify colour or more than one colour separated commas with no spaces. [default: orange,cyan]
  --alpha=<alpha>              transparency setting - comma separated no spaces if more than one alpha [default: 0.5]
  --log                        Plot log(y) instead of y.
  --leglabels=<leglabels>      Legend labels (alternative to using the columns).
"""
import sys
__doc__ = __doc__ % (sys.argv[0], sys.argv[0], sys.argv[0])
from docopt import docopt
import os, shutil, re, csv, subprocess
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

def plotHistogram(data, options):

    colours = options.colour.split(',')
    alphas = options.alpha.split(',')
    columns = options.column.split(',')
    leglabels = None
    if options.leglabels:
        leglabels = options.leglabels.split(',')
    #fig = plt.figure(figsize=(6,3))
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    #bins = n.linspace(round(float(options.binlower)), round(float(options.binupper)), int((float(options.binupper) - float(options.binlower))/float(options.binwidth))+1)
    bins = n.linspace(float(options.binlower), float(options.binupper), int((float(options.binupper) - float(options.binlower))/float(options.binwidth))+1)

    ml = MultipleLocator(float(options.majorticks))
    ax1.xaxis.set_major_locator(ml)

    ax1 = fig.add_subplot(111)
    # May have more than one histogram to plot
    i = 0
    for d in data:
        if len(colours) == 1:
            colour = colours[0]
        else:
            colour = colours[i]

        if len(alphas) == 1:
            alpha = alphas[0]
        else:
            alpha = alphas[i]

        ax1.hist(n.array(d), bins=bins, color = colour, edgecolor='black', linewidth=0.5, alpha = float(alpha))
        i += 1

    ax1.set_ylabel(options.ylabel)
    for tl in ax1.get_yticklabels():
        tl.set_color('k')

    ax1.set_xlabel(options.xlabel)
    #ax1.set_title('Classifier performance.')
    if len(columns) > 1:
        ax1.legend(columns, loc=1, frameon=False)
    elif leglabels is not None:
        ax1.legend(leglabels, loc=1, frameon=False)

    ax1.text(0.8, 0.95, options.plotlabel, transform=ax1.transAxes, va='top', size=MEDIUM_SIZE)
    ax1.text(0.1, 0.95, options.panellabel, transform=ax1.transAxes, va='top', size=MEDIUM_SIZE, weight='bold')

    ml = MultipleLocator(float(options.minorticks))
    ax1.xaxis.set_minor_locator(ml)
    ax1.get_xaxis().set_tick_params(which='both', direction='out')
    ax1.set_xlim(float(options.binlower), float(options.binupper))

    ax1.set_ylim(ymin=0)
    if options.ylimit:
        ax1.set_ylim(ymax=float(options.ylimit))

    if options.log:
        ax1.set_yscale('log')

    if options.threshold is not None:
        ax1.axvline(x=float(options.threshold),color='k',linestyle='--')

    plt.tight_layout()

    if options.outputFile is not None:
        plt.savefig(options.outputFile)
    else:
        plt.show()


def doPlots(options):
    # There may be more than one inputFile
    allData = []
    columns = options.column.split(',')
    i = 0
    for datafile in options.inputFile:
        data = []
        dataRows = readGenericDataFile(datafile, delimiter=options.delimiter)

        for row in dataRows:
            if len(columns) == len(options.inputFile):
                datum = float(row[columns[i]])
            else:
                datum = float(row[options.column])
            if datum > 0:
                data.append(datum)
        allData.append(data)
        i += 1

    plotHistogram(allData, options)


def main():
    opts = docopt(__doc__, version='0.1')
    opts = cleanOptions(opts)
    options = Struct(**opts)

    doPlots(options)


if __name__=='__main__':
    main()


