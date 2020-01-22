# coding: utf-8

import os
from glob import glob
import numpy as np
from shutil import copyfile,move
import time
import sys
# from PIL import Image
# import cv2
from errno import EEXIST

# import keras
# import keras.backend as K
# from keras.preprocessing import image, sequence
# from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import to_categorical
# from keras.utils.data_utils import get_file
# from keras.utils.np_utils import get_file
    
def working_directory(in_path):
        '''
        Set a path to current active working directory.
        Note:- This is important for creating and splitting datasets. 
        '''
        in_path = in_path
        # Change current working directory
        curr_dir = os.chdir(in_path)
        curr_dir = os.getcwd()
        print("The current working directory is: ",curr_dir)

def count(in_path):
    file_list = os.listdir(in_path)
    sizes = len(file_list)

    return sizes
        
        
def rename_files(in_path, newname):
        
        '''
        Labels the data. Uses loop to achieve the task.
        index_contd_val is the value that the renaming should be carried on from previous
        index in the target folder.
        Note:- It is necessary to label data in dataset for recognizing target variables.
        '''
        
    # List all files in the directory
        file_list = os.listdir(in_path)

        working_directory(in_path)
    
    # Rename all files in that directory
        for index, filename in enumerate(file_list):
            extension = os.path.splitext(filename)[1]
            if extension == extension:
                new_file = newname+'_{}.bmp'.format(index+1)
                
                if os.path.exists(new_file):
                    raise Exception("Cannot Rename %s to %s as it is already present." %(filename, new_file))

                print("Renaming %s to %s." %(filename, new_file))
                os.rename(filename, new_file)
                
                
def move_files(in_path, out_path, n_samples=20):
    '''
    Splits the data into train, test, validation set. 
    For each set this function needs to be executed separately.
    '''
    # List all files in the directory
    file_list = os.listdir(in_path)
    #current_dir = os.getcwd()     
    working_directory(in_path)
    shuf = np.random.permutation(file_list)
    print("Moving to %r.Please wait..  " % out_path)
    start = time.time()
    for i in range(n_samples):
            move(shuf[i], out_path + shuf[i])
    end = time.time()
    print("Op took %r secs." %(end - start))
    print("All {} files moved.".format(n_samples))
        

def copy_files(in_path, out_path, n_samples = None, shuffle = False):
    out_path = out_path+"/"
    counter = 0
    filelist = os.listdir(in_path)
    # working_directory(in_path)

    if shuffle:
        shuf = np.random.permutation(filelist)
        print("Copying to %r. Please wait... " % out_path)

        start = time.time()
        print("Shuffle:True")
        for i in range(n_samples):
            copyfile(shuf[i], out_path+shuf[i])
            counter+=1
            sys.stdout.write("\rimages copied: %r" % counter)
            sys.stdout.flush()   

        end = time.time()
        print("\nOp took %r secs." %(end - start))
        print("All {} files moved.".format(n_samples))
    else:
        start = time.time()
        print("Shuffle: False")
        for i in range(n_samples):
            copyfile(filelist[i], out_path+filelist[i])
            counter+=1
            sys.stdout.write("\rTotal images: %r" % counter)
            sys.stdout.flush()   

        end = time.time()

        print("Op took %r secs." %(end - start))
        print("All {} files moved.".format(n_samples))


def onehot_encoding(x, num_classes = None, dir = False):
        '''
        Gives a value for each classes or target variables.
        If data is extracted from directory instead of csv, pickle, excel etc then set dir to True.
        Note:- Uses Keras abstraction. 
        '''
        if dir:
            return to_categorical(x)
        else:
            return to_categorical(x, num_classes)
    
        
def normalize_by_pixels_range(data):
        '''
        Normalizes the images by pixels within [0 - 255]
        '''
        data = data
        x = data.astype(np.float32)
        x /= 255
        
        return x
def normalize_by_mean_std(data):
        '''
        Normalizes the images by subtracting the image from its mean from the standard deviation.
        '''
        data = data
        pixel_mean = np.mean(data).astype(np.float32)
        pixel_std = np.std(data).astype(np.float32)
        
        return (data - pixel_mean) / pixel_std

def calculate_mean(img_list, img_height = 256, img_width=256 ):
    images = []
    for img_path in img_list:
        img = read_image(path=img_path, img_height=img_height, img_width=img_width)
        images.append(img)
    
    img_mean = np.mean(images)
    print("Total image count: ", len(images))
    print("Mean of the given imageset is: ", img_mean)


def save_array(file, arr):
    import bcolz
    c = bcolz.carray(arr, rootdir = file, mode = 'w')
    c.flush()

def load_array(file):
    import bcolz
    return bcolz.open(file)[:]

def read_image(path, img_height = 256, img_width=256):
    # Load image. By default it takes BGR format.
    img = cv2.imread(path,0)
    print(img)
    resized = cv2.resize(img, (img_width,img_height))
    resized = resized.astype(np.float32)
    return resized

def get_files_from_disk(path, pattern="*.bmp", verbose=False):

    files = []

    for dirpath, _, _ in os.walk(path):
        files.extend(glob(os.path.join(dirpath, pattern)))
    if verbose:
        print("Size of returned list 'files' is: ", len(files))

    return files

def create_directory(directory):
    '''
    Creates new directory with a provided input path.
    '''
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

    except OSError as e:
        if e.errno != EEXIST:
            raise("Race condition. Directory occurs between 'os.path.exists() and os.makedirs()'")


def load_images_X_y(img_list, output_class_list, img_height = 256, img_width = 256):
    X_test = []
    y_test = []
    for img_path in img_list:
        img = cv2.imread(img_path,0)
        resized = cv2.resize(img, (img_height,img_width))
        X_test.append(resized)

        for l in range(len(output_class_list)):
            if output_class_list[l] in img_path:
                labels = output_class_list.index(output_class_list[l])
                #print("%s %s" % (img_path,labels))

        y_test.append(labels)
    print(X_test[1])

    print("X :data and y: labels 'list' created.")
    print("Total size of X_test: ",len(X_test))
    print("Total size of y_test: ", len(y_test))
    
    return X_test, y_test
   


    






