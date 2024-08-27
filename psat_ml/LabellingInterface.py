#!/usr/bin/env python
"""Interface to facilitate manual real/bogus training set labelling.

   Ken W. Smith
   Converted to Python 3, to read HDF5 files, and added configurable stamp numbers (nside).

   Code originally written by Darryl E. Wright.

Usage:
  %s <dataFile> [--nside=<nside>]
  %s (-h | --help)
  %s --version

Options:
  -h --help          Show this screen.
  --version          Show version.
  --nside=<nside>    How many images (squared) do we want to eyeball at once? Max 100 (x 100). [default: 10].

E.g.:
  %s ./relabelled_2024-08-25_00:35:42_ps2_good150000_bad450000_20x20_20240712.h5

"""
import sys
import importlib
__doc__ = __doc__ % (sys.argv[0], sys.argv[0], sys.argv[0], sys.argv[0])
from docopt import docopt
import os
from gkutils.commonutils import Struct, cleanOptions

import wx, sys, optparse, datetime
# The recommended way to use wx with mpl is with the WXAgg
# backend.
#
import wx.lib.agw.hyperlink as hl
import matplotlib
matplotlib.use('WXAgg')

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

#import scipy.io as sio
import h5py
import numpy as np

#from sklearn import preprocessing
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from matplotlib.backends.backend_wxagg import \
     FigureCanvasWxAgg as FigCanvas, \
     NavigationToolbar2WxAgg as NavigationToolbar

class mainFrame(wx.Frame):
    """ The main frame of the application """
    title = 'Main Console'
    def __init__(self, dataFile, nside = 10):
        # 10x10 inches, 100 dots-per-inch, so set size to (1000,1000)
        wx.Frame.__init__(self, None, -1, self.title, size=(1000,1000),pos=(50,50))
        
        self.nside = nside
        self.nstamps = nside * nside

        self.dataFile = dataFile
        #data = sio.loadmat(self.dataFile)
        data = h5py.File(dataFile,'r')

        self.X = np.concatenate((data["X"], data["testX"]))
        self.y = np.concatenate((np.squeeze(data["y"]), np.squeeze(data["testy"])))
        try:
            files = data["files"]
        except KeyError:
            files = data["train_files"]
        self.files = np.concatenate((np.squeeze(files), np.squeeze(data["test_files"])))

        self.real_X  = self.X[np.where(self.y == 1)]
        self.real_y  = self.y[np.where(self.y == 1)]
        self.real_files  = self.files[np.where(self.y == 1)]
        
        
        self.bogus_X = self.X[np.where(self.y == 0)]
        self.bogus_y = self.y[np.where(self.y == 0)]
        self.bogus_files  = self.files[np.where(self.y == 0)]
        
        m, n = np.shape(self.X)
        self.m = m
        self.n = n
        self.plotDim = np.sqrt(self.n)
        self.start = 0
        # 2024-08-24 KWS End set to 1024
        self.end = self.nside * self.nside
        #self.end = 100
        self.to_plot = self.X[self.start:self.end,:]
        self.files_to_plot = self.files[self.start:self.end]
        self.max_index = int(np.ceil(np.shape(self.X)[0]/self.nstamps)*self.nstamps)
        
        self.new_real_files = []
        self.new_bogus_files = []
        
        # Create the mpl Figure and FigCanvas objects.
        # 10x10 inches, 100 dots-per-inch
        
        self.dpi = 100
        self.fig = Figure((8.0, 8.0), dpi=self.dpi)
        self.fig.subplots_adjust(left=0.1, bottom=0.01, right=0.9, top=0.99, wspace=0.05, hspace=0.05)
        
        self.AXES = []
        #for i in range(100):
        #    ax = self.fig.add_subplot(10,10,i+1)
        #    self.AXES.append(ax)
        
        # 2024-08-24 KWS Try 32x32 on a page
        for i in range(self.nstamps):
            ax = self.fig.add_subplot(self.nside,self.nside,i+1)
            self.AXES.append(ax)
         
        self.create_main_panel(nside=self.nside)
        self.navigation_control.previous_button.Disable()
        self.reset_button.Disable()

    def create_main_panel(self, nside):
    
        self.nside = nside

        self.panel = wx.Panel(self)
        self.set_text = wx.StaticText(self.panel, -1, label="Showing : All (%d examples)" % self.m)
        self.set_text.SetBackgroundColour(wx.WHITE)
        self.set_text.SetForegroundColour("#0F0F0F")
        font = wx.Font(20, wx.MODERN, wx.NORMAL, wx.BOLD)
        self.set_text.SetFont(font)
        
        self.hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox1.Add(self.set_text, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        
        self.draw_fig(True)
        self.canvas = FigCanvas(self.panel, -1, self.fig)
        
        # Bind the 'click' event for clicking on one of the axes
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.navigation_control = NavigationControlBox(self.panel, self, -1, "navigation control", nside = self.nside)
        #self.label_key_box = LabelKeyBox(self.panel,-1)
        self.data_set_control = DataSetControlBox(self.panel,self,-1, nside = self.nside)
        
        self.build_button = wx.Button(self.panel, -1, label="Build")
        self.build_button.Bind(wx.EVT_BUTTON, self.on_build)
        self.reset_button = wx.Button(self.panel, -1, label="Reset")
        self.reset_button.Bind(wx.EVT_BUTTON, self.on_reset)
        self.exit_button = wx.Button(self.panel, -1, label="Exit")
        self.exit_button.Bind(wx.EVT_BUTTON, self.on_exit)
        
        self.vbox1 = wx.BoxSizer(wx.VERTICAL)
        self.vbox1.Add(self.build_button, 0, flag=wx.CENTER | wx.BOTTOM)
        self.vbox1.Add(self.reset_button, 0, flag=wx.CENTER | wx.BOTTOM)
        self.vbox1.Add(self.exit_button, 0, flag=wx.CENTER | wx.BOTTOM)
        
        self.hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        #self.hbox2.Add(self.label_key_box, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self.hbox2.Add(self.data_set_control, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self.hbox2.Add(self.navigation_control, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        self.hbox2.Add(self.vbox1, border=5, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL)
        
        self.vbox2 = wx.BoxSizer(wx.VERTICAL)
        self.vbox2.Add(self.hbox1, 0, flag=wx.CENTER | wx.TOP)
        self.vbox2.Add(self.canvas, 1, flag=wx.CENTER | wx.CENTER | wx.GROW)
        self.vbox2.Add(self.hbox2, 0, flag=wx.LEFT | wx.TOP)
        
        self.panel.SetSizer(self.vbox2)
        self.vbox2.Fit(self)
    
    def draw_fig(self, init=False):
        
        for i,ax in enumerate(self.AXES):
            cmap="hot"
            if init:
                try:
                    image = np.reshape(self.to_plot[i,:], (int(self.plotDim), int(self.plotDim)), order="F")
                    image = np.flipud(image)
                    ax.imshow(image, interpolation="nearest", cmap=cmap)
                    ax.axis("off")
                except IndexError:
                    ax.clear()
                    image = np.reshape(np.zeros((self.n,)), (int(self.plotDim), int(self.plotDim)), order="F")
                    image = np.flipud(image)
                    ax.imshow(image, interpolation="nearest", cmap=cmap)
                    ax.axis("off")
            try:
                if str(self.files_to_plot[i]).rstrip().split("/")[-1] in set(self.new_real_files):
                    ax.clear()
                    cmap="cool"
                    image = np.reshape(self.to_plot[i,:], (int(self.plotDim), int(self.plotDim)), order="F")
                    image = np.flipud(image)
                    ax.imshow(image, interpolation="nearest", cmap=cmap)
                    ax.axis("off")
                elif str(self.files_to_plot[i]).rstrip().split("/")[-1] in set(self.new_bogus_files):
                    ax.clear()
                    cmap = "PRGn"
                    image = np.reshape(self.to_plot[i,:], (int(self.plotDim), int(self.plotDim)), order="F")
                    image = np.flipud(image)
                    ax.imshow(image, interpolation="nearest", cmap=cmap)
                    ax.axis("off")
            except IndexError:
                ax.clear()
                image = np.reshape(np.zeros((self.n,)), (int(self.plotDim), int(self.plotDim)), order="F")
                image = np.flipud(image)
                ax.imshow(image, interpolation="nearest", cmap=cmap)
                ax.axis("off")

    def on_click(self, event):
        """
            Enlarge or restore the selected axis.
        """
        # The event received here is of the type
        # matplotlib.backend_bases.PickEvent
        #
        # It carries lots of information, of which we're using
        # only a small amount here.
        #
        
        if hasattr(self, "inspectorFrame"):
             return
    
        self.axes = event.inaxes
        if self.axes is None:
            return

        self.canvas.draw()
        self.canvas.Refresh()
        
        file = self.files_to_plot[self.AXES.index(self.axes)]
        
        print(event)
        if event.button == 1: # Left click
            label = self.y[self.start+self.AXES.index(self.axes)]
            if label == 0:
                self.new_real_files.append(str(file).split("/")[-1])
                # Don't add the same object more than once!
                self.new_real_files = list(set(self.new_real_files))
                print("1")
                self.draw_fig()
                self.canvas.draw()
                self.canvas.Refresh()
            elif label == 1:
                self.new_bogus_files.append(str(file).split("/")[-1])
                # Don't add the same object more than once!
                self.new_bogus_files = list(set(self.new_bogus_files))
                print("2")
                self.draw_fig()
                self.canvas.draw()
                self.canvas.Refresh()
        elif event.button == 3 and str(file).split("/")[-1] in set(self.new_real_files):
                self.new_real_files.remove(str(file).split("/")[-1])
                print("3")
                self.draw_fig(init=True)
                self.canvas.draw()
                self.canvas.Refresh()
        elif event.button == 3 and str(file).split("/")[-1] in set(self.new_bogus_files):
                self.new_bogus_files.remove(str(file).split("/")[-1])
                print("3")
                self.draw_fig(init=True)
                self.canvas.draw()
                self.canvas.Refresh()
        else:
            pass
        #self.draw_fig(init=True)


    def on_build(self, event):
        #data = sio.loadmat(self.dataFile)
        data = h5py.File(self.dataFile,'r')

        X = np.concatenate((data["X"], data["testX"]))
        y = np.concatenate((np.squeeze(data["y"]), np.squeeze(data["testy"])))
        try:
            files = data["files"]
        except KeyError:
            files = data["train_files"]
        files = np.concatenate((np.squeeze(files), np.squeeze(data["test_files"])))

        for i,file in enumerate(files):
            print(str(file).rstrip().split("/")[-1], y[i], end=' ')
            if str(file).rstrip().split("/")[-1] in set(self.new_bogus_files):
                if y[i] == 1:
                    y[i] = 0
            if str(file).rstrip().split("/")[-1] in set(self.new_real_files):
                if y[i] == 0:
                    y[i] = 1
            print(y[i])
        #outputFile = raw_input("Specify output file : ")
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        
        if "relabelled" in self.dataFile:
            saveFile = str(self.dataFile).split("/")[-1][31:]
            print(saveFile)
            #sio.savemat("relabelled_%s_" % current_time + saveFile, \
            #           {"X":X[:.75*self.m], "y":y[:.75*self.m], "train_files": files[:.75*self.m], \
            #            "testX":X[.75*self.m:], "testy":y[.75*self.m:], "test_files":files[.75*self.m:]})

            hf=h5py.File("relabelled_%s_" % current_time + saveFile,'w')
            hf.create_dataset('X',data=X[:int(.75*self.m)])
            hf.create_dataset('y',data=y[:int(.75*self.m)])
            hf.create_dataset('train_files',data=files[:int(.75*self.m)])
            hf.create_dataset('testX',data=X[int(.75*self.m):])
            hf.create_dataset('testy',data=y[int(.75*self.m):])
            hf.create_dataset('test_files',data=files[int(.75*self.m):])
            hf.close()

        else:
            #sio.savemat("relabelled_%s_" % current_time + self.dataFile.split("/")[-1], \
            #           {"X":X[:.75*self.m], "y":y[:.75*self.m], "train_files": files[:.75*self.m], \
            #            "testX":X[.75*self.m:], "testy":y[.75*self.m:], "test_files":files[.75*self.m:]})

            hf=h5py.File("relabelled_%s_" % current_time + str(self.dataFile).split("/")[-1],'w')
            hf.create_dataset('X',data=X[:int(.75*self.m)])
            hf.create_dataset('y',data=y[:int(.75*self.m)])
            hf.create_dataset('train_files',data=files[:int(.75*self.m)])
            hf.create_dataset('testX',data=X[int(.75*self.m):])
            hf.create_dataset('testy',data=y[int(.75*self.m):])
            hf.create_dataset('test_files',data=files[int(.75*self.m):])
            hf.close()

                    
        print("[+] Processing complete.")
            
    def on_reset(self, event):
        
        self.new_real_files = []
        self.new_bogus_files = []
        self.new_ghost_files = []
    
    def on_exit(self, event):
        self.Destroy()
        exit(0)

class NavigationControlBox(wx.Panel):
    def __init__(self, parent, frame, ID, label, nside = 10):
        wx.Panel.__init__(self, parent, ID)
        
        self.nstamps = nside*nside
        self.frame = frame
        box = wx.StaticBox(self, -1, label)
        sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
        
        self.next_button = wx.Button(self, -1, label="Next %d" % self.nstamps)
        self.next_button.Bind(wx.EVT_BUTTON, self.on_next)
        self.previous_button = wx.Button(self, -1, label="Previous %d" % self.nstamps)
        self.previous_button.Bind(wx.EVT_BUTTON, self.on_previous)
        
        manual_box = wx.BoxSizer(wx.HORIZONTAL)
        manual_box.Add(self.previous_button, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.AddSpacer(10)
        manual_box.Add(self.next_button, flag=wx.ALIGN_CENTER_VERTICAL)
        
        sizer.Add(manual_box, 0, wx.ALL, 10)
        
        self.SetSizer(sizer)
        sizer.Fit(self)
    
    def on_next(self, event):

        self.frame.start += self.nstamps
        self.frame.end += self.nstamps
        if self.frame.start > 0:
            self.frame.navigation_control.previous_button.Enable()
        if self.frame.end == self.frame.max_index:
            self.frame.navigation_control.next_button.Disable()
        self.frame.to_plot = self.frame.X[self.frame.start:self.frame.end,:]
        self.frame.files_to_plot = self.frame.files[self.frame.start:self.frame.end]
        self.frame.draw_fig(True)
        self.frame.canvas.draw()
        print("next: %d-%d" % (self.frame.start+1,self.frame.end))
    
    def on_previous(self, event):

        if self.frame.start <=0:
            self.frame.navigation_control.previous_button.Disable()
        else:
            self.frame.start -= self.nstamps
            self.frame.end -= self.nstamps
            print(self.frame.start, self.frame.end)
            if self.frame.start ==0:
                self.frame.navigation_control.previous_button.Disable()
            if self.frame.start < self.frame.max_index:
                self.frame.navigation_control.next_button.Enable()
        self.frame.to_plot = self.frame.X[self.frame.start:self.frame.end,:]
        self.frame.files_to_plot = self.frame.files[self.frame.start:self.frame.end]
        self.frame.draw_fig(True)
        self.frame.canvas.draw()
        print("previous: %d-%d" % (self.frame.start+1,self.frame.end))

class LabelKeyBox(wx.Panel):
    def __init__(self, parent, ID):
        wx.Panel.__init__(self, parent, ID)
        
        
        box = wx.StaticBox(self, -1, "relabelling key")
        sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
        
        self.real = wx.StaticText(self, label="REAL")
        self.real.SetBackgroundColour(wx.WHITE)
        self.real.SetForegroundColour("#3366FF")
        
        self.bogus = wx.StaticText(self, label="BOGUS")
        self.bogus.SetBackgroundColour(wx.WHITE)
        self.bogus.SetForegroundColour("#FF0066")
        
        self.ghost = wx.StaticText(self, label="GHOST")
        self.ghost.SetBackgroundColour(wx.WHITE)
        self.ghost.SetForegroundColour("#9933FF")
        
        manual_box = wx.BoxSizer(wx.HORIZONTAL)
        manual_box.Add(self.real, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.Add(self.bogus, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.Add(self.ghost, flag=wx.ALIGN_CENTER_VERTICAL)
        
        sizer.Add(manual_box, 0, wx.ALL, 10)
        
        self.SetSizer(sizer)
        sizer.Fit(self)

class DataSetControlBox(wx.Panel):
    def __init__(self, parent, frame, ID, nside = 10):
        wx.Panel.__init__(self, parent, ID)
        
        self.nstamps = nside * nside
        self.frame = frame
        box = wx.StaticBox(self, -1, "data set control")
        sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
        
        """
        self.fp_button = wx.Button(self, -1, label="FP")
        self.fp_button.Bind(wx.EVT_BUTTON, self.on_fp)
        self.tp_button = wx.Button(self, -1, label="TP")
        self.tp_button.Bind(wx.EVT_BUTTON, self.on_tp)
        self.fn_button = wx.Button(self, -1, label="FN")
        self.fn_button.Bind(wx.EVT_BUTTON, self.on_fn)
        self.tn_button = wx.Button(self, -1, label="TN")
        self.tn_button.Bind(wx.EVT_BUTTON, self.on_tn)
        """
        self.real_button = wx.Button(self, -1, label="Real")
        self.real_button.Bind(wx.EVT_BUTTON, self.on_real)
        self.bogus_button = wx.Button(self, -1, label="Bogus")
        self.bogus_button.Bind(wx.EVT_BUTTON, self.on_bogus)
        
        manual_box = wx.BoxSizer(wx.HORIZONTAL)
        """
        manual_box.Add(self.tp_button, border=2, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.Add(self.fp_button, border=2, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.Add(self.tn_button, border=2, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.Add(self.fn_button, border=2, flag=wx.ALIGN_CENTER_VERTICAL)
        """
        manual_box.Add(self.real_button, border=2, flag=wx.ALIGN_CENTER_VERTICAL)
        manual_box.Add(self.bogus_button, border=2, flag=wx.ALIGN_CENTER_VERTICAL)
        
        sizer.Add(manual_box, 0, wx.ALL, 10)
        
        self.SetSizer(sizer)
        sizer.Fit(self)
    
    def on_tp(self, event):
        print("true positives")
        self.frame.X = self.frame.dataSetDict["true_pos_X"]
        self.frame.max_index = int(np.ceil(np.shape(self.frame.X)[0]/self.nstamps)*self.nstamps)
        self.frame.y = self.frame.dataSetDict["true_pos_y"]
        self.frame.files = self.frame.dataSetDict["true_pos_files"]
        self.frame.start = 0
        # 2024-08-24 KWS Frame end set to 1024
        self.frame.end = self.nstamps
        #self.frame.end = 100
        self.frame.to_plot = self.frame.X[self.frame.start:self.frame.end,:]
        self.frame.files_to_plot = self.frame.files[self.frame.start:self.frame.end]
        self.frame.navigation_control.previous_button.Disable()
        self.frame.navigation_control.next_button.Enable()
        m,n = np.shape(self.frame.X)
        self.frame.draw_fig(True)
        self.frame.canvas.draw()
        self.frame.set_text.SetLabel("Showing : True Positives (%d examples)" % m)
        self.frame.data_set_control.tp_button.Disable()
        self.frame.data_set_control.fp_button.Enable()
        self.frame.data_set_control.tn_button.Enable()
        self.frame.data_set_control.fn_button.Enable()
        self.frame.data_set_control.real_button.Enable()
        self.frame.data_set_control.bogus_button.Enable()
    
    def on_fp(self, event):
        print("false positives")
        self.frame.X = self.frame.dataSetDict["false_pos_X"]
        self.frame.max_index = int(np.ceil(np.shape(self.frame.X)[0]/self.nstamps)*self.nstamps)
        self.frame.y = self.frame.dataSetDict["false_pos_y"]
        self.frame.files = self.frame.dataSetDict["false_pos_files"]
        self.frame.start = 0
        # 2024-08-24 KWS Frame end set to 1024
        self.frame.end = self.nstamps
        #self.frame.end = 100
        self.frame.to_plot = self.frame.X[self.frame.start:self.frame.end,:]
        self.frame.files_to_plot = self.frame.files[self.frame.start:self.frame.end]
        self.frame.navigation_control.previous_button.Disable()
        self.frame.navigation_control.next_button.Enable()
        m,n = np.shape(self.frame.X)
        self.frame.draw_fig(True)
        self.frame.canvas.draw()
        self.frame.set_text.SetLabel("Showing : False Positives (%d examples)" % m)
        self.frame.data_set_control.tp_button.Enable()
        self.frame.data_set_control.fp_button.Disable()
        self.frame.data_set_control.tn_button.Enable()
        self.frame.data_set_control.fn_button.Enable()
        self.frame.data_set_control.real_button.Enable()
        self.frame.data_set_control.bogus_button.Enable()
    
    def on_tn(self, event):
        print("true negatives")
        self.frame.X = self.frame.dataSetDict["true_neg_X"]
        self.frame.max_index = int(np.ceil(np.shape(self.frame.X)[0]/self.nstamps)*self.nstamps)
        self.frame.y = self.frame.dataSetDict["true_neg_y"]
        self.frame.files = self.frame.dataSetDict["true_neg_files"]
        self.frame.start = 0
        # 2024-08-24 KWS Frame end set to 1024
        self.frame.end = self.nstamps
        #self.frame.end = 100
        self.frame.to_plot = self.frame.X[self.frame.start:self.frame.end,:]
        self.frame.files_to_plot = self.frame.files[self.frame.start:self.frame.end]
        self.frame.navigation_control.next_button.Enable()
        self.frame.navigation_control.previous_button.Disable()
        m,n = np.shape(self.frame.X)
        self.frame.draw_fig(True)
        self.frame.canvas.draw()
        self.frame.set_text.SetLabel("Showing : True Negatives (%d examples)" % m)
        self.frame.data_set_control.tp_button.Enable()
        self.frame.data_set_control.fp_button.Enable()
        self.frame.data_set_control.tn_button.Disable()
        self.frame.data_set_control.fn_button.Enable()
        self.frame.data_set_control.real_button.Enable()
        self.frame.data_set_control.bogus_button.Enable()
    
    def on_fn(self, event):
        print("false negatives")
        self.frame.X = self.frame.dataSetDict["false_neg_X"]
        self.frame.max_index = int(np.ceil(np.shape(self.frame.X)[0]/self.nstamps)*self.nstamps)
        self.frame.y = self.frame.dataSetDict["false_neg_y"]
        self.frame.files = self.frame.dataSetDict["false_neg_files"]
        self.frame.start = 0
        # 2024-08-24 KWS Frame end set to 1024
        self.frame.end = self.nstamps
        #self.frame.end = 100
        self.frame.to_plot = self.frame.X[self.frame.start:self.frame.end,:]
        self.frame.files_to_plot = self.frame.files[self.frame.start:self.frame.end]
        self.frame.navigation_control.next_button.Enable()
        self.frame.navigation_control.previous_button.Disable()
        m,n = np.shape(self.frame.X)
        self.frame.draw_fig(True)
        self.frame.canvas.draw()
        self.frame.set_text.SetLabel("Showing : False Negative (%d examples)" % m)
        self.frame.data_set_control.tp_button.Enable()
        self.frame.data_set_control.fp_button.Enable()
        self.frame.data_set_control.tn_button.Enable()
        self.frame.data_set_control.fn_button.Disable()
        self.frame.data_set_control.real_button.Enable()
        self.frame.data_set_control.bogus_button.Enable()
    
    def on_real(self, event):
        print("show real")
        self.frame.X = self.frame.real_X
        self.frame.max_index = int(np.ceil(np.shape(self.frame.X)[0]/self.nstamps)*self.nstamps)
        self.frame.y = self.frame.real_y
        self.frame.files = self.frame.real_files
        self.frame.start = 0
        # 2024-08-24 KWS Frame end set to 1024
        self.frame.end = self.nstamps
        #self.frame.end = 100
        self.frame.to_plot = self.frame.X[self.frame.start:self.frame.end,:]
        self.frame.files_to_plot = self.frame.files[self.frame.start:self.frame.end]
        self.frame.navigation_control.next_button.Enable()
        self.frame.navigation_control.previous_button.Disable()
        m,n = np.shape(self.frame.X)
        self.frame.draw_fig(True)
        self.frame.canvas.draw()
        self.frame.set_text.SetLabel("Showing : Real (%d examples)" % m)
        self.frame.data_set_control.real_button.Disable()
        self.frame.data_set_control.bogus_button.Enable()
    
    def on_bogus(self, event):
        print("show bogus")
        self.frame.X = self.frame.bogus_X
        self.frame.max_index = int(np.ceil(np.shape(self.frame.X)[0]/self.nstamps)*self.nstamps)
        self.frame.y = self.frame.bogus_y
        self.frame.files = self.frame.bogus_files
        self.frame.start = 0
        # 2024-08-24 KWS Frame end set to 1024
        self.frame.end = self.nstamps
        #self.frame.end = 100
        self.frame.to_plot = self.frame.X[self.frame.start:self.frame.end,:]
        self.frame.files_to_plot = self.frame.files[self.frame.start:self.frame.end]
        self.frame.navigation_control.next_button.Enable()
        self.frame.navigation_control.previous_button.Disable()
        m,n = np.shape(self.frame.X)
        self.frame.draw_fig(True)
        self.frame.canvas.draw()
        self.frame.set_text.SetLabel("Showing : Bogus (%d examples)" % m)
        self.frame.data_set_control.bogus_button.Disable()
        self.frame.data_set_control.real_button.Enable()


def main():
    opts = docopt(__doc__, version='0.1')
    opts = cleanOptions(opts)
    
    # Use utils.Struct to convert the dict into an object for compatibility with old optparse code.
    options = Struct(**opts)

    dataFile = options.dataFile
    nside = 10

    try:
        nside = int(options.nside)
        if nside > 100:
            nside = 100
    except ValueError as e:
        nside = 10

    app = wx.App(False)
    app.frame = mainFrame(dataFile, nside = nside)
    app.frame.Show()
    app.MainLoop()

if __name__ == "__main__":
    main()
