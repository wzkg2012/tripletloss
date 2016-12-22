#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 22:19:30 2016

@author: XFZ
"""

import mxnet as mx
import cPickle
import numpy as np
from random import shuffle
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pdb
import sys
from inception import  get_inception

#from magnetLoss import *
def unpickle(file):	
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict
def provideBatch(data,batchSize=128):
    nRound = len(data)/batchSize
    nSample = len(data)
    for start in range(nRound+1):
        batch = np.zeros((batchSize,3,32,32))
        for i in range(batchSize) :
            ii=start*batchSize+i
            if ii<nSample:
                img = data[ii]
                img = img.reshape(32,32,3,order='F')
                #img = cv2.resize(img,(224,224))

                img=img.swapaxes(2,0)
                batch[i]=img
        yield batch
        
def KNN_test(data,label,featureExector,splitRatio,n_neighbors,hash_len):
	#extrat feature by provide featureEx
    batchSize = 128
    nSample = len(data)
    featureSize=hash_len
    remains = nSample%batchSize
    features = np.zeros((nSample,featureSize))
    i=0
    print 'extract features'
    for batch in provideBatch(data,batchSize):
        f = featureExector.predict(batch)
        f = np.squeeze(f)
        if (i+1)*batchSize <= nSample :
            features[i*batchSize:(i+1)*batchSize]=f
        else:
            features[i*batchSize:nSample-1]=f[0:(remains-1)]
        i+=1
        print i
    data_label = zip(features,label)
    shuffle(data_label)
    splitPoint = int(len(data_label)*splitRatio)
    test = data_label[0:splitPoint]
    train = data_label[splitPoint:]
    test_data = [x for (x,y) in test]
    test_label = [y for (x, y) in test]
    train_data = [x for (x, y) in train]
    train_label = [y for (x, y) in train]
    print('knn test')
    neigh = KNeighborsClassifier(metric='euclidean',\
    		n_neighbors=n_neighbors)
    neigh.fit(train_data, train_label)
    score = neigh.score(test_data,test_label)
    return score

def get_test_net(hash_len):
    anchor = mx.sym.Variable('data')

    cdata= get_inception(concat, hash_len)
    return cdata
  
def main():
    prefix = '' 
    hash_len = 32
    test_network = get_test_net(hash_len)
    model = mx.model.FeedForward.load(prefix,int(sys.argv[1]), ctx=mx.gpu(3), numpy_batch_size=128)
    feature_extractor = mx.model.FeedForward(ctx=mx.gpu(3), symbol=test_network, numpy_batch_size=128,
                                             arg_params=model.arg_params, aux_params=model.aux_params,
                                             allow_extra_params=True)

    dic=unpickle('../cifar10/test_batch')
    data = dic['data']
    label = dic['labels']
    score = KNN_test(data,label,feature_extractor,0.2,1,hash_len)
    print score
if __name__ == '__main__':
	main()
