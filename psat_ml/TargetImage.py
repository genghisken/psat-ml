#!/usr/bin/python

from astropy.io import fits as pyfits
#import pyfits
import pylab
import random
import numpy as np

class TargetImage(object):

    def __init__(self, fitsFile, extent=10, extension=1):
        """
            fitsFile: name of File from which to extract the image of the object

            The image should be already centred on the object of interest, so can
            find its position by accesing the centre of the array of pixel data. 
            The image must also be fpacked as recieved from the postage stamp 
            server which means the data is in hdulist extension 1. If the image is
            funpacked the hdulist extension is 0.
            
            returns: a 20x20 np.array of pixel data centreed on the object
        """
        self.fitsFile = fitsFile.split("/")[-1]
        self.objectID = self.fitsFile.split("_")[0]
        self.extent = extent
        pathAndFitsFile = fitsFile 
        try:
            hdulist = pyfits.open(pathAndFitsFile)
        except OSError as e:
            print("Problem opening %s" % pathAndFitsFile)

        data = hdulist[extension].data # think this reads in x and y opposite to ds9 see docs
        maxX = np.shape(data[0])
        maxY = np.shape(data[1])
        imageCentre = (maxX[0]/2.0, maxY[0]/2.0) # changed to float division, shouldn't make a difference
        self.object = data[int(imageCentre[0]-extent): int(imageCentre[0]+extent),int(imageCentre[0]-extent): int(imageCentre[0]+extent)]

    def getObject(self):
        return self.object
    
    def getObjectID(self):
        return self.objectID
    
    def getObjectFile(self):
        return self.fitsFile

    def visualiseObject(self, cmap="hot"):
        pylab.ion()
        #pylab.set_cmap("gray")
        pylab.gray()
        pylab.title("image: %s" % self.fitsFile)
        pylab.imshow(self.getObject(), interpolation="nearest", cmap=cmap)
        pylab.colorbar()
        pylab.ylim(-1, 2*self.extent)
        pylab.xlim(-1, 2*self.extent)
        pylab.xlabel("Pixels")
        pylab.ylabel("Pixels")
        pylab.show()

    def unravelObject(self):
        return np.ravel(self.getObject(), order="F")

    def rescale(self):
        """
            Rescale such that the data lie in the range [0,1]
            or [-1,1].
            
            See:
                http://deeplearning.stanford.edu/wiki/index.php/Data_Preprocessing
        """
        Vec = np.nan_to_num(self.unravelObject())
        Vec = Vec / (np.max(np.abs(Vec)))
        return Vec
    
    def meanSubtract(self):
        """
            Simple preprocessing to be perfomred before 
            whitening and PCA.
            
            See: 
                http://deeplearning.stanford.edu/wiki/index.php/Data_Preprocessing
            
            Subtract off the mean of the vector.
        """
        Vec = np.nan_to_num(self.unravelObject())
        mean = np.mean(Vec)
        return Vec - mean
    
    def featureStandardisation(self):
        """
            Subtract the mean then divide by the standard deviation.
            
            See:
                http://deeplearning.stanford.edu/wiki/index.php/Data_Preprocessing
            
            Result is zero mean unit variance vector.
        """    
        meanSubVec = self.meanSubtract()
        # rescale meanSubVec to range [-1,1]
        rescaleVec = meanSubVec / (np.max(np.abs(meanSubVec)))
        std = np.std(rescaleVec)
        return (rescaleVec / float(std))
    
    def signPreserveNorm(self):
        """
            This is a sign preserving nomalisation used in Eye.
            Similar to that used by Romano et al. in SVM paper
            except they use log(1+|x|) i.e. don't divide by sigma.
            nomalizes the unraveled image
            
            vectorized on 24/07/13
        """
        #shape = np.shape(self.getObject())
        Vec = np.nan_to_num(self.unravelObject())
        #normVec = np.zeros((np.shape(Vec)))
        std = np.std(Vec)
        #for i in range(len(Vec)):
        #    # log1p returns the natural log of (1+x)x
        #    normVec[i] += ((Vec[i])/ np.abs(Vec[i]))*(np.log1p(np.abs(Vec[i])/std))
        #    #print normVec[i]
        normVec = ((Vec)/ np.abs(Vec))*(np.log1p(np.abs(Vec)/std))
        return normVec

    
    def visualiseNormObject(self):
        shape = (2*self.extent, 2*self.extent)
        pylab.ion()
        pylab.clf()
        #pylab.set_cmap("bone")
        pylab.hot()
        pylab.title("image: %s" % self.fitsFile)
        pylab.imshow(np.reshape(self.signPreserveNorm(), shape, order="F"), interpolation="nearest")
        pylab.plot(np.arange(0,2*self.extent), self.extent*np.ones((2*self.extent,)), "r--")
        pylab.plot(self.extent*np.ones((2*self.extent,)), np.arange(0,2*self.extent), "r--")
        pylab.colorbar()
        pylab.ylim(-1, 2*self.extent)
        pylab.xlim(-1, 2*self.extent)
        pylab.xlabel("Pixels")
        pylab.ylabel("Pixels")
        pylab.show()
