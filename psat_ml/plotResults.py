#!/usr/bin/env python
"""
Usage:
  %s <csvfile>... [--outputFile=<outputFile>] [--xlim=<xlimit>] [--ylim=<ylimit>] [--mdr=<mdr>] [--telescope=<telescope>] [--panellabel=<panellabel>] [--title=<title>]
  %s (-h | --help)
  %s --version

Options:
  -h --help                    Show this screen.
  --version                    Show version.
  --outputFile=<outputFile>    Place to store the outputfile.
  --xlim=<xlimit>              Plot x limit [default: 1.0]
  --ylim=<ylimit>              Set y limit [default: 1.0]
  --mdr=<mdr>                  Missed detection rate [default: 0.04]
  --panellabel=<panellabel>    Plot label (e.g. 'a)' ) [default: ]
  --telescope=<telescope>      Telescope
  --title=<title>              Plot title

Example:
  %s output_results.csv
"""
import sys
__doc__ = __doc__ % (sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0])
from docopt import docopt
from gkutils import Struct, cleanOptions
import numpy as np
import pandas as pd
#from sklearn.metrics import roc_curve, auc
from sklearn.metrics import auc
from plotROC import roc_curve
import optparse
import matplotlib.pyplot as plt

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 25
TINY_SIZE = 12
plt.rc('font', size=SMALL_SIZE)              # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)         # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=TINY_SIZE)        # fontsize of the tick labels
plt.rc('ytick', labelsize=TINY_SIZE)        # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE - 1)        # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)      # fontsize of the figure title
plt.rcParams["font.family"] = "serif"
plt.rcParams['mathtext.fontset'] = 'dejavuserif'


def plot_roc(fpr, tpr,roc_auc,roc):
    roc.plot(fpr,tpr,lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    roc.set_xlabel('False Positive Rate')
    roc.set_ylabel('True Positive Rate')
    roc.set_title('ROC curve')
    roc.legend(loc="lower right")

def plot_tradeoff(mdr, fpr, tradeoff, intercept=[], xlim = 1.0, ylim = 1.0, title = None, panellabel=''):
    tradeoff.plot(mdr,fpr,lw=2)
    if len(intercept) > 0:
        tradeoff.plot([0,intercept[0],intercept[0],intercept[0]], [intercept[1],intercept[1],intercept[1],0], color='black', linestyle='--')
    tradeoff.set_xlabel('Missed detection rate')
    tradeoff.set_ylabel('False positive rate')
    if title:
        tradeoff.set_title(title)
    #tradeoff.set_xlim(0,0.25)
    #tradeoff.set_ylim(0,0.2)
    tradeoff.set_xlim(0,xlim)
    tradeoff.set_ylim(0,ylim)
    tradeoff.text(0.1, 0.95, panellabel, transform=tradeoff.transAxes, va='top', size=MEDIUM_SIZE, weight='bold')


def plotResults(files, outputfile, options = None):
    #fig, (roc, tradeoff) = plt.subplots(1,2,sharey=False)
    fig, (tradeoff) = plt.subplots(1,1,sharey=False)

    for file in files:
        plotTitle = options.title 
        if options.telescope is not None:
            plotTitle += ' (' + options.telescope + ')'
        data = pd.read_csv(file, names=['file', 'tag', 'prediction'])
        y = data['tag']
        scores = data['prediction']
        #fpr,tpr,thresholds = roc_curve(y, scores, pos_label=1)
        fpr,tpr,thresholds = roc_curve(np.array(y), np.array(scores))

        mdrSet = float(options.mdr)
        fpr_at_mdrSet = (fpr[np.where(1-tpr<=mdrSet)[0]][-1])

        print("[+]%.3lf%% mdr gives " % (mdrSet*100) + str(fpr[np.where(1-tpr<=mdrSet)[0]][-1]*100) + "% fpr")
        print("   [+] threshold : %.3lf"%(thresholds[np.where(1-tpr<=mdrSet)[0]][-1]))
        roc_auc = auc(fpr, tpr)    
        mdr = 1-tpr
        if options is not None:
            plot_tradeoff(mdr, fpr, tradeoff, intercept=[mdrSet,fpr_at_mdrSet], xlim = float(options.xlim), ylim = float(options.ylim), title = plotTitle, panellabel=options.panellabel)
        else:
            plot_tradeoff(mdr, fpr, tradeoff, intercept=[mdrSet,fpr_at_mdrSet], xlim = float(options.xlim), ylim = float(options.ylim), title = plotTitle)
        #plot_roc(fpr,tpr,roc_auc,roc)

    plt.tight_layout()
    if options is not None and options.outputFile is not None:
        plt.savefig(outputfile)
    else:
        plt.show()


def main():
    opts = docopt(__doc__, version='0.1')
    opts = cleanOptions(opts)

    # Use utils.Struct to convert the dict into an object for compatibility with old optparse code.
    options = Struct(**opts)
    print (options.csvfile)
    plotResults(options.csvfile, options.outputFile, options = options)


if __name__=='__main__':
    main()
    
