import optparse, time
import numpy as np
import h5py
import scipy.io as sio
from TargetImage import *
from scipy.ndimage import gaussian_filter
from skimage import img_as_float
from skimage.morphology import reconstruction

np.seterr(all="ignore")

def imageFile_to_list(imageFile):
    print(imageFile)
    counter = 0
    imageList = []
    for line in open(imageFile,"r").readlines():
        imageList.append(line.rstrip())
        counter += 1
    return imageList
    
def group_images(imageList):

    imageList.sort()
    tti_pairs = {}
    for item in imageList:
        id = item.split("_")[0]
        #mjd = item.split("_")[1].split(".")[0]
        try:
            tti_pairs[id].append(item)
        except KeyError:
            tti_pairs[id] = []
            tti_pairs[id].append(item)
    return tti_pairs

def noNorm(imageFile, path,  extent, extension, magic = None):
    a=np.nan_to_num(TargetImage(path+imageFile, extent, extension).unravelObject())
    if magic is not None:
        # Replace all the magic numbers with zeros
        a[a==magic] = 0
    return a

def signPreserveNorm(imageFile, path, extent, extension, magic = None):
    a=np.nan_to_num(TargetImage(path+imageFile, extent, extension).signPreserveNorm())
    if magic is not None:
        # Replace all the magic numbers with zeros
        a[a==magic] = 0
    return a

def bg_sub_signPreserveNorm(imageFile, path, extent, extension, magic = None):
    vec = signPreserveNorm(imageFile, path, extent, extension, magic = magic)
    image = np.reshape(vec, (20,20), order="F")

    image = gaussian_filter(image, 1)
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image

    dilated = reconstruction(seed, mask, method='dilation')
    
    return np.ravel(image - dilated, order="F")
    
def generate_vectors(imageList, path, extent, normFunc, extension, magic = None):
    print("PATH = ", path)
    m = len(imageList)
    X = np.ones((m, 4*extent*extent))

    for i,imageFile in enumerate(imageList):
        try:
            if '/' in imageFile:
                vector = normFunc(imageFile, "", extent, extension, magic = magic)
            else:
                vector = normFunc(imageFile, path+"good/", extent, extension, magic = magic)
        except IOError:
            try:
                vector = normFunc(imageFile, path+"bad/", extent, extension, magic = magic)
            except IOError:
                try:
                    vector = normFunc(imageFile, path+"4_20160706/", extent, extension, magic = magic)
                except IOError:
                    print("[!] Exiting: Could not find %s" % imageFile)
                    exit(0)
        X[i,:] = X[i,:] * vector
    return X

def generate_key(file):
    id = file.split("_")[0]
    #mjd = file.split("_")[1].split(".")[0]
    return id
    
def process_examples(list, path, label, extent, normFunc, extension, trainingFraction=.75, magic = None):

    m = len(list) # number of training examples
    np.random.seed(0)
    
    X = generate_vectors(list, path, extent, normFunc, extension, magic = magic)
    grouped_X = np.ones((np.shape(X)))
    grouped_dict = group_images(list[:])
    
    # randomly shuffle keys
    # tti_keys = list(grouped_dict.keys()) this does not work on python3
    tti_keys = []
    for k in grouped_dict.keys():
        tti_keys.append(k)
    np.random.shuffle(tti_keys)
    
    i = 0
    grouped_list = []
    # for all tti groups
    for tti in tti_keys:
        # for each image in the tti group
        for image in grouped_dict[tti]:
            # add its vector to X
            grouped_X[i,:] = grouped_X[i,:] * X[list.index(image),:]
            # add its file to the file list
            grouped_list.append(image)
            i+=1
    # create label vector 
    y = np.ones((m,))*label
    # define the index that separates training and test sets
    boundary_index = int(np.floor(trainingFraction*m))
    # check the next index is not member of the same tti group
    while True:
        boundary_file = grouped_list[boundary_index]
        key = generate_key(boundary_file)
        boundary_file_plus = grouped_list[boundary_index+1]
        key_plus = generate_key(boundary_file_plus)
        # if tti group key is the same they are in the same group
        if key == key_plus:
            # so increase boundary_index to include this example
            # and repeat the process
            boundary_index += 1
        else:
            # if not the same break and continue with original index
            break
    
    # divide up the data according to the boundary index
    train_x = grouped_X[:boundary_index,:]
    train_y = y[:boundary_index]
    train_files = grouped_list[:boundary_index]
    
    test_x = grouped_X[boundary_index:,:]
    test_y = y[boundary_index:]
    test_files = grouped_list[boundary_index:]
    
    return train_x, train_y, train_files, test_x, test_y, test_files

def build_data_set(pos_data, neg_data):
    
    X = np.concatenate((pos_data[0], neg_data[0]))
    m, n = np.shape(X)
    y = np.concatenate((pos_data[1], neg_data[1]))
    files = np.concatenate((pos_data[2], neg_data[2]))
    print(np.shape(files))
    order = np.random.permutation(m)
    X = X[order,:]
    y = y[order]
    files = files[order]
    return X, y, files

def rotate_examples(X, y, files, extent, k=3):
    m,n = np.shape(X)
    augmentedX = np.ones(((k+1)*m,n))
    augmentedy = np.squeeze(np.ones(((k+1)*m,)))
    augmented_files = []
    for i in range(m):
        #print y[i]
        print((k+1)*i)
        augmentedX[(k+1)*i,:] *= X[i,:]
        augmentedy[(k+1)*i] *= y[i]
        #print augmentedy[(k+1)*i]
        augmented_files.append(files[i])
        for j in range(1,k+1):
            print(((k+1)*i)+j)
            rotatedX = np.rot90(np.reshape(X[i,:], (2*extent,2*extent), order="F"), j)
            augmentedX[((k+1)*i)+j,:] *= np.ravel(rotatedX, order="F")
            augmentedy[((k+1)*i)+j] *= y[i]
            augmented_files.append(files[i])
            #print augmentedX[:16,:2]
    #print np.shape(augmentedX)
    #print len(augmented_files)
    return augmentedX, augmentedy, augmented_files

def save_to_hdf5(X,y,train_files,testX,testy,test_files,outputFile):
    ascii_train_files = [n.encode("ascii", "ignore") for n in train_files]
    ascii_test_files = [n.encode("ascii", "ignore") for n in test_files]
    hf=h5py.File(outputFile,'w')
    hf.create_dataset('X',data=X)
    hf.create_dataset('y',data=y)
    hf.create_dataset('train_files',data=ascii_train_files)
    hf.create_dataset('testX',data=testX)
    hf.create_dataset('testy',data=testy)
    hf.create_dataset('test_files',data=ascii_test_files)
    hf.close()


def buildMLDataSet(opts):
    from gkutils.commonutils import Struct
    if type(opts) is dict:
        options = Struct(**opts)
    else:
        options = opts

    startTime = time.time()
    
    posFile = options.posFile
    negFile = options.negFile
    outputFile = options.outputFile
    extent = options.extent
    extension = options.extension
    skewFactor = options.skewFactor
    rotate = options.rotate
    print(rotate)
    norm = options.norm
    magic = options.magic
    
    if posFile == None or outputFile == None:
       # print(parser.usage)
        print("missing good or bad .txt")
        exit(0)
        
    if extent == None:
        extent = 10
   
    if extension == None:
        extension = 1
        
    if skewFactor == None:
        skewFactor = 1

    if norm == None:
        norm = "spn"

    if norm == "signPreserveNorm" or norm == "spn":
        normFunc = signPreserveNorm
    elif norm == "bg_sub_signPreserveNorm" or norm == "bg_sub_spn":
        normFunc = bg_sub_signPreserveNorm
    elif norm == "noNorm":
        normFunc = noNorm
        
    if negFile == None:
        print("[*] No negative example data file specified.")
        print("    [+] Building unlabelled data set.")
        imageList = imageFile_to_list(posFile)
        path = posFile.strip(posFile.split("/")[-1])
        print(path)
        X = generate_vectors(imageList, path, extent, normFunc, extension, magic = magic)
        #sio.savemat(outputFile, {"X": X, "images": imageList})
        hf = h5py.File(outputFile,'w')
        hf.create_dataset('X', data=X)
        hf.create_dataset('images',data=imageList)
        hf.close()
        exit(0)

    # process positive examples
    print("[+] Processing positve examples.")
    pos_list = imageFile_to_list(posFile)
    m_pos = len(pos_list)
    path = posFile.strip(posFile.split("/")[-1])
    print(path)
    pos_data = process_examples(pos_list, path, 1, extent, normFunc, extension, magic = magic)
    print("[+] %d positive examples processed." % m_pos)
    
    # process positive examples
    print("[+] Processing negative examples.")
    neg_list = imageFile_to_list(negFile)
    # account for skewFactor
    neg_list = neg_list[:skewFactor*m_pos]
    m_neg = len(neg_list)
    path = negFile.strip(negFile.split("/")[-1])
    print(path)
    neg_data = process_examples(neg_list, path, 0, extent, normFunc, extension, magic = magic)
    print("[+] %d negative examples processed." % m_neg)

    print("[+] Building training set.")
    X, y, train_files = build_data_set(pos_data[:3], neg_data[:3])

    if rotate:
        print("[+] Augmenting training set with rotated examples.")
        X, y, train_files = rotate_examples(X, y, train_files, extent)
    
    print("[+] Building test set.")
    testX, testy, test_files = build_data_set(pos_data[3:], neg_data[3:])

    print("[+] Saving data sets.")
    # sio.savemat(outputFile, {"X": X, "y":y, "train_files": train_files, \
    #                         "testX":testX, "testy":testy, "test_files":test_files})
    save_to_hdf5(X,y,train_files,testX,testy,test_files,outputFile)
    print("[+] Processing complete.")
    print("[*] Run time: %d minutes." % ((time.time() - startTime) / 60))
    
def main():
    parser = optparse.OptionParser("[!] usage: python build_data_set.py\n"+\
                                   " -p <positive data file>\n"+\
                                   " -o <output file>\n"+\
                                   " -n <negative data file [optional]>\n"+\
                                   " -e <extent [default=10]>\n"+\
                                   " -E <extension [default=1]>\n"+\
                                   " -s <skew factor [default=1]>\n"+\
                                   " -r <augment training data with rotation [optional]>\n"
                                   " -N <normalisation function [default=signPreserveNorm]>\n"
                                   " -m <integer mask magic number")

    parser.add_option("-p", dest="posFile", type="string", \
                      help="specify file listing positive examples")
    parser.add_option("-n", dest="negFile", type="string", \
                      help="specify file listing bogus examples [optional]")
    parser.add_option("-o", dest="outputFile", type="string", \
                      help="specify output file name")
    parser.add_option("-e", dest="extent", type="int", \
                      help="specify image size [default=10]")
    parser.add_option("-E", dest="extension", type="int", \
                      help="specify image extension [default=1]")
    parser.add_option("-s", dest="skewFactor", type="int", \
                      help="specify skew to negative examples [default=1]")
    parser.add_option("-r", dest="rotate", action="store_true", \
                      help="specify whether to augment training set with roatated examples [optional]")
    parser.add_option("-N", dest="norm", type="string", \
                      help="specify normalisation function to apply to data [default=signPreserveNorm]")
    parser.add_option("-m", dest="magic", type="int", \
                      help="specify an integer magic number mask (e.g. -31415)")

    (options, args) = parser.parse_args()

    buildMLDataSet(options)

if __name__ == "__main__":
    main()
