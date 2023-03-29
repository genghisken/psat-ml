# psat-ml
Automatic classification of Pan-STARRS and ATLAS images. Based on the code originally written by Amanda Ibsen and Ken W. Smith. Documentation originally written by Amanda Ibsen. 

## In a Nutshell: 

This repo contains a pipeline to connect to the [ATLAS](http://atlas.fallingstar.com/home.php) (or PS1) database, get cutouts of difference images, build a data set, train a classifier to differentiate between real and bogus images, and plot the results. 

## How does it work?

![alt text](/imgs/classification_pipeline.png)

### GetCutOuts
#### Input options
    configFile : .yaml with database credentials
    mjds : list of nights 
    stampSize : size of cutouts
    stampLocation : where to store cutouts
    camera : '02a' for Haleakala, '01a' for Maunaloa
    downloadthreads : number of threads
    stampThreads : number of threads
    
#### Explanation    
-**getATLASTrainingSetCutouts.py**: It takes as input a config file, a list of dates (in MJD) and a directory to store the output in. It connects to the ATLAS database using the credentials in the config file and gets all exposures for the given time frame. For each exposure it creates a .txt file containing all x,y positions for the objects in the images and a 40x40 pixels cutout image for each object. It also creates a "good.txt" and a "bad.txt" file, containing the x,y positions for the real and bogus objects, respectively.

-**getPS1TrainingSetCutouts.py**: Same as the above file, but it connects to the PS1 data base instead.

### BuildMLDataset
#### Input options
    good : file with x,y pixel positions for real objects
    bad : file with x,y pixel positions for bogus objects
    outputFile : .h5 output file
    e : extent (default=10)
    E : Extension (default=0)
    s : skew, how many bogus objects per real ones(default=3)
    r : rotation (default=None)
    N : normalization function (default='signPreserveNorm') 

#### Explanation
-**buildMLDataset.py**: It takes as input the good.txt and bad.txt files with all x,y positions for real and bogus objects. From those, it builds an .h5 file containing the features (20x20 pixels of the image) and targets (real or bogus label) to be used later as training set.

### KerasTensorflowClassifier
#### Input options
    outputcsv : output csv file
    trainingset : .h5 input dataset 
    classifierfile : .h5 file to store model (classifier)

#### Explanation
-**kerasTensorflowClassifier.py**: It takes as input an .h5 file with the training set and a path to store a classifier as an .h5 file. If the model doesn't exist yet, it creates it, trains it and classifies a test set. It returns a .csv file containing  the targets and scores for all images. The classifier used is a [CNN] (http://cs231n.github.io/convolutional-networks/) with the following architecture:

![alt text](/imgs/model.png)

### PlotResults
#### Input options
    inputFiles : csv files to be plotted, with both target and score for each object
    outputFile : output .png file with the plots

#### Explanation
-**plotResults.py**: It takes as input a csv file with the scores and targets for all images and plots the [ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) and the [Detection error tradeoff graph](https://en.wikipedia.org/wiki/Detection_error_tradeoff) for the data set.

## Some results

### ROC curve and trade-off plots for ATLAS test data-set
![alt text](/imgs/roc_tradeoff.png)
### Recall for 'confirmed' and 'good' transients
![alt text](/imgs/recall_confirmed_and_good.png)
## How to run the pipeline?

When trying to run one task, the pipeline will search for the necessary resources to complete it and try to run it. If it doesn't find them, it'll run the task that's needed to produce those resources and will keep doing this recursively until it can run the task.

### To run a task:
```
python atlasClassificationPipeline.py Name_of_Task --local-scheduler --name_of_oition1 option1 ... --name_of_optionN optionN
```
### Examples:
-To run the **PlotResults** task
```
python atlasClassificationPipeline.py PlotResults --local-scheduler --inputfiles [file1.csv,...,filen.csv] --outputFile output.png## How to run the pipeline?
```
For more information on how to run a pipeline, go check the [luigi documentation](http://luigi.readthedocs.io/en/stable/running_luigi.html#)

## To set up:

- create virtual environment with python 3.6 and activate it
- pip install -r requirements.txt
