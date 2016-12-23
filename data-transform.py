import os
import cPickle
import numpy as np
import logging
import pdb
import cv2

logger = logging.getLogger(__name__)
def unpickle( baseName):
    logger.info('loading file %s' % baseName)
    fo = open(baseName, 'rb')
    data = cPickle.load(fo)
    fo.close()
    return data

LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog','horse', 'ship', 'truck']

def futz(X):
    return X.reshape((10000, 3, 32, 32))

if __name__ == '__main__':
	fnames = ['data_batch_%i'%i for i in range(1,6)]
	#fnames.append('test_batch')
        size = 224
	pixel = np.zeros((50000, 3,224,224), dtype='uint8')
	labels = np.zeros(50000, dtype='int32')
	tmpData = np.zeros((10000,3,32,32), dtype='uint8')
	tmpLabel = np.zeros(10000, dtype='int32')
	startPtr = [0,5000,10000,15000,20000,25000,30000,35000,40000,45000]

	for i, fname in enumerate(fnames):
		data = unpickle(fname)
		assert data['data'].dtype == np.uint8
		tmpData = futz(data['data'])
		tmpLabel = data['labels']
		for j in range(len(tmpLabel)):
			index = startPtr[tmpLabel[j]]
			pixel[index] = cv2.resize(tmpData[j][0,:,:], dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
			startPtr[tmpLabel[j]] = index+1
				

	for i in range(len(LABELS)):
		np.save(LABELS[i]+'.npy',pixel[i*5000:5000*(i+1)])
		





