import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend when there is no DISPLAY or GUI
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from caffe.proto import caffe_pb2

caffe.set_mode_gpu()

# Size of images
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

'''
Image processing helper function
'''


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    # Histogram Equalization
    # Histogram equalization is a technique for adjusting the contrast of images.
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    # Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    return img


'''
Reading mean image, caffe model and its weights 
'''
# Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('trainImageMean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))
mean_array_forImg = np.array(caffe.io.blobproto_to_array(mean_blob))[0,:,:,:].mean(0)


# Read model architecture and trained model's weights
net = caffe.Net('NIN_mavenlin_deploy.prototxt',
                caffe.TEST, weights='nin_face_iter_6000.caffemodel')

# Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2, 0, 1))

plt.ioff() # turn of interactive plotting mode
# plt.rcParams['figure.figsize'] = (32, 32)
plt.rcParams['image.interpolation'] = 'nearest'

# set size & DPI=100 for 64x64 figure; dpi=100 is selected to support most screen resolutions
# plt.figure(figsize=(0.064, 0.064), dpi=100)  # size is in inches!
# save image in dpi=1000 to get image of size 64x64 pixels
# plt.savefig('myfig.png', dpi=1000)

# plt.title("Training mean image")
# plt.plot(mean_array_forImg.flat)
# plt.axis('off')
# plt.savefig('trainingMeanImage.png')
# plt.close()

'''
Making predicitions
'''

fPredict = open('outputPredictions.txt', 'a')
fPredict.write('=============START===============\n')

# Making predictions
test_ids = []
preds = []
img_path = 'face1_43.jpg'
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

net.blobs['data'].data[...] = transformer.preprocess('data', img)
out = net.forward()
# pred_probas = out['prob']
fPredict.write(img_path + ' == ' + str(out) + '\n')
# `out` is now a dictionary of {output name: ndarray of outputs}. So for instance
# if your softmax classifier output layer is called "prob", `out['prob'][0]` is
# the prediction vector for your first test input.
pred_probas = out['loss']

# test_ids = test_ids + [img_path.split('/')[-1][:-4]]
# preds = preds + [pred_probas.argmax()]

fPredict.write(img_path + ' == ' + str(pred_probas.argmax()) + '\n')
fPredict.write(img_path + ' == ' + str(pred_probas.argmax()) + '; Accuracy is == ' + str(pred_probas[0][pred_probas.argmax()] * 100) + ' %\n')

# print img_path
# print pred_probas.argmax()
# print '-------'


fPredict.close()


# VISUALIZE DATA DEFINITION 
# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    #data -= data.min()
    #data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)

    # the parameters are a list of [weights, biases]


# VISUALIZE THE WEIGHTS OF THE 1ST CONV LAYER
# plt.rcParams['figure.figsize'] = (25.0, 20.0)
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))
plt.axis('off')
plt.savefig('conv1_weightMaps.jpg', bbox_inches='tight', pad_inches=0.0)
plt.close()


# VISUALIZE THE FEATURE MAPS AFTER 1ST CONV LAYER
feat = net.blobs['conv1'].data[0,:192] # 192 conv1 outputs
vis_square(feat, padval=1)
#net.blobs['conv1'].data.shape
plt.axis('off')
plt.savefig('conv1_featureMaps.jpg', bbox_inches='tight', pad_inches=0.0)
plt.close()

# VISUALIZE THE FEATURE MAPS AFTER 2ND CONV LAYER
feat = net.blobs['conv2'].data[0]
vis_square(feat, padval=1)
#net.blobs['conv1'].data.shape
plt.axis('off')
plt.savefig('conv2_featureMaps.jpg', bbox_inches='tight', pad_inches=0.0)
plt.close()

# VISUALIZE THE FEATURE MAPS AFTER 1ST CCCP LAYER
feat = net.blobs['cccp1'].data[0]
vis_square(feat, padval=1)
#net.blobs['conv1'].data.shape
plt.axis('off')
plt.savefig('cccp1_featureMaps.jpg', bbox_inches='tight', pad_inches=0.0)
plt.close()





