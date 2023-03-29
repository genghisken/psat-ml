import luigi
from buildMLDataSet import buildMLDataSet
from getATLASTrainingSetCutouts import getATLASTrainingSetCutouts
from kerasTensorflowClassifier import kerasTensorflowClassifier
from plotResults import plotResults

"""
Example command:

python atlasClassificationPipeline.py PlotResults --local-scheduler \
    --inputFiles '["/export/raid/db4data1/scratch/kws/training/atlas/pipeline_test/output_hko_58310.csv"]' \
    --outputFile /export/raid/db4data1/scratch/kws/training/atlas/pipeline_test/output_hko_58310.png \
    --GetCutOuts-mjds '[58310]' \
    --GetCutOuts-configFile ~/config4_readonly.yaml \
    --GetCutOuts-stampLocation /export/raid/db4data1/scratch/kws/training/atlas/pipeline_test \
    --BuildMLDataSet-good /export/raid/db4data1/scratch/kws/training/atlas/pipeline_test/good.txt \
    --BuildMLDataSet-bad /export/raid/db4data1/scratch/kws/training/atlas/pipeline_test/bad.txt \
    --BuildMLDataSet-outputFile /export/raid/db4data1/scratch/kws/training/atlas/pipeline_test/58310_data.h5 \
    --KerasTensorflowClassifier-classifierfile /export/raid/db4data1/scratch/kws/training/atlas/pipeline_test/58310_data_classifier.h5 \
    --KerasTensorflowClassifier-outputcsv /export/raid/db4data1/scratch/kws/training/atlas/pipeline_test/output_hko_58310.csv
"""

defaultDir = '/export/raid/db4data1/scratch/amanda/hko'

class GetCutOuts(luigi.Task):
    global defaultDir
    configFile = luigi.Parameter(default='~/config4_readonly.yaml') 
    mjds = luigi.ListParameter(default=[58226])
    stampSize = luigi.IntParameter(default=40)
    stampLocation = luigi.Parameter(default=defaultDir)
    camera = luigi.Parameter(default='02a')
    downloadthreads=luigi.IntParameter(default=5)
    stampThreads = luigi.IntParameter(default=28)
    
    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget(self.stampLocation+'/good.txt') 

    def run(self):
        print(defaultDir)
        options = {'configFile':self.configFile,
            'mjds':self.mjds,
            'stampSize':self.stampSize,
            'stampLocation':self.stampLocation,
            'camera': self.camera,
            'downloadthreads': self.downloadthreads,
            'stampThreads': self.stampThreads}
             
        getATLASTrainingSetCutouts(options) 

class BuildMLDataSet(luigi.Task):
    global defaultDir
    good = luigi.Parameter(default=defaultDir+'/good.txt')
    bad = luigi.Parameter(default=defaultDir+'/bad.txt')
    outputFile = luigi.Parameter(defaultDir+'/dataset.h5')
    e = luigi.IntParameter(default=10)
    E = luigi.IntParameter(default=0)
    s = luigi.IntParameter(default=3)
    r = luigi.Parameter(default=None)
    N = luigi.Parameter(default='signPreserveNorm') 
    def requires(self):
        return [GetCutOuts()]

    def output(self):
        return luigi.LocalTarget(self.outputFile)

    def run(self):
        print(defaultDir)
        options = {
        'posFile':self.good,
        'negFile':self.bad,
        'outputFile':self.outputFile,
        'extent':self.e,
        'extension':self.E,
        'skewFactor':self.s,
        'rotate':self.r,
        'norm':self.N}
        buildMLDataSet(options)
       
class KerasTensorflowClassifier(luigi.Task):
    outputcsv = luigi.Parameter(default=defaultDir+'/output.csv')
    trainingset = luigi.Parameter(default=defaultDir+'/dataset.h5')
    classifierfile = luigi.Parameter(default=defaultDir+'/model.h5')

    def requires(self):
        return [BuildMLDataSet()]

    def output(self):
        return luigi.LocalTarget(self.outputcsv)

    def run(self):
        print(defaultDir)
        options = {
        'outputcsv':self.outputcsv,
        'trainingset':self.trainingset,
        'classifierfile':self.classifierfile}
        kerasTensorflowClassifier(options)
        
class PlotResults(luigi.Task):
    inputFiles = luigi.ListParameter(default=[defaultDir+'/output.csv'])
    outputFile = luigi.Parameter(default=defaultDir+'/plots.png')

    def requires(self):
        return [KerasTensorflowClassifier()]

    def output(self):
        return luigi.LocalTarget(self.outputFile)

    def run(self):
        plotResults(self.inputFiles, self.outputFile)

if __name__ == '__main__':
    luigi.run()
