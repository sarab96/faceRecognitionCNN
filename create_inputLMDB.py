import os
import glob
import random
import numpy as np
import re

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

# Size of images
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    # Histogram Equalization
    # Histogram equalization is a technique for adjusting the contrast of images.
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


def make_datum(img, img_label):
    # image is numpy.ndarray format. BGR instead of RGB
    # With the new cv2 interface images loaded are now numpy arrays automatically.
    # But openCV cv2.imread() loads images as BGR while numpy.imread() loads them as RGB.
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=img_label,
        data=np.rollaxis(img, 2).tostring())


train_lmdb = '/home/ubuntu/faceIdentityDataset/train_lmdb'
validation_lmdb = '/home/ubuntu/faceIdentityDataset/test_lmdb'

fDebug = open('outputDebug.txt', 'a')
fDebug.write('=============START===============\n')
fDebug.write('Deleting old images in DB lmdb\n')

# Delete old lmdb, as creating new one below
os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)

train_data = [img for img in glob.glob("images/*jpg")]
# test_data = [img for img in glob.glob("*_test*jpg")]

# Shuffle train_data
random.shuffle(train_data)

fDebug.write('Creating train_lmdb\n')

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        fDebug.write('Training File processed: ' + '{:0>5d}'.format(in_idx) + ':' + img_path + '\n')

        # Keep every 6th image as part of validation set
        # For now let us keep training and validation sets separate
        # if in_idx % 6 == 0:
        #     continue
        # image is numpy.ndarray format. BGR instead of RGB
        # With the new cv2 interface images loaded are now numpy arrays automatically.
        # But openCV cv2.imread() loads images as BGR while numpy.imread() loads them as RGB.

        # Use below way to convert BGR to RGB
        # srcBGR = cv2.imread("sample.png")
        # destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)

        # puts the channel as the first dimention
        # transformer.set_transpose('data', (2,0,1))
        # img = img.transpose(2,1,0)

        # (2,1,0) maps RGB to BGR for example
        # transformer.set_channel_swap('data', (2,1,0))

        # Sarab: since we are generating our own training model, so let us keep everything in RGB way. But make sure
        # mean image is also generated in RGB way.. compute_image_mean.cpp relies on OpenCV, which uses BGR as
        # default channels. So looks like it is better to stick with BGR in Caffee.. Scaling the input data to the
        # range of [0..1] or [0..255] is entirely up to you. Some models works in [0..1] range, others in [0..255]
        # and it is completely unrelated to the choice of input method (LMDB/HDF5) Caffe blobs are always 4-D:-
        # batch-channel-width-height

        # Recently, models subtract mean per channel, rather than mean per pixel. It is more convenient, especially
        # if you change the input size of your net. When processing the HDF5 data you can compute the mean image and
        # save it into a binaryproto. See an example here. http://stackoverflow.com/a/27645934/1714410

        # You may notice that recent models (e.g., googlenet) do not use a mean file the same size as the input
        # image, but rather a 3-vector representing a mean value per image channel. These values are quite "immune"
        # to the specific dataset used (as long as it is large enough and contains "natural images"). So, as long as
        # you are working with natural images you may use the same values as e.g., GoogLenet is using: B=104, G=117,
        # R=123.

        # If you are using the caffe.io.load_image to read image, you need to do the following conversion. Because
        # caffe.io.load_image uses skimage, which reads the color channels in order of R G B: img =
        # caffe.io.load_image(path/to/image) img = img[:,:,(2,1,0)]

        # set_transpose of an input of the size (227,227,3) with parameters (2,0,1) will be (3,227,227).
        # Applying set_channel_swap will preserve the order ((3,227,227)) but change it for example, from RGB to BGR
        # https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

        matchObj = re.match('images/face(.+)_.*jpg', img_path, re.M | re.I)
        matchObjTest = re.match('images/face(.+)_test.*jpg', img_path, re.M | re.I)
        if matchObjTest:
            fDebug.write('Skipping test file: ' + img_path + '\n')
            continue
        elif matchObj:
            # print "matchObj.group() : ", matchObj.group()
            # print "matchObj.group(1) : ", matchObj.group(1)
            # print "matchObj.group(2) : ", matchObj.group(2)
            label = int(matchObj.group(1))
        else:
            label = 0
            fDebug.write('ERROR: No match for face file number!. File with unexpected name n lable: ' + img_path + '\n')

        datum = make_datum(img, label)
        # Left pad the fileNum with zeros to 5 characters..So max number of files in training set is 99999
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        fDebug.write('Training File processed: ' + '{:0>5d}'.format(in_idx) + ':' + img_path + ', label: ' + str(label) + '\n')

in_db.close()

fDebug.write('\nCreating validation_lmdb\n')

in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    # Sarab: For now just use some images from training set..
    i = 0;
    for in_idx, img_path in enumerate(train_data):
        if in_idx % 3 != 0:
            continue

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

        matchObj = re.match('images/face(.+)_.*jpg', img_path, re.M | re.I)
        matchObjTest = re.match('images/face(.+)_test.*jpg', img_path, re.M | re.I)
        if matchObjTest:
            fDebug.write('Skipping test file: ' + img_path + '\n')
            continue
        elif matchObj:
            # print "matchObj.group() : ", matchObj.group()
            # print "matchObj.group(1) : ", matchObj.group(1)
            # print "matchObj.group(2) : ", matchObj.group(2)
            label = int(matchObj.group(1))
        else:
            label = 0
            fDebug.write('ERROR: No match for face file number!. File with unexpected name n lable: ' + img_path + '\n')

        datum = make_datum(img, label)
        # in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        in_txn.put('{:0>5d}'.format(i), datum.SerializeToString())
        fDebug.write('Validation File processed: ' + '{:0>5d}'.format(i) + ':' + img_path + ', label: ' + str(label) + '\n')
        i += 1

in_db.close()

fDebug.write('\nFinished processing all images\n')

# Use the function cv2.imwrite() to save an image.
# First argument is the file name, second argument is the image you want to save.
# cv2.imwrite('messigray.png',img)
# This will save the image in PNG format in the working directory.

fDebug.close()
