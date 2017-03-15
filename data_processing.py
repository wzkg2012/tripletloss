import cPickle,gzip
import numpy as np
import pdb

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

label = train_set[1]
pixel = np.zeros((50000,1,28,28))
startPtr = [0,5000,10000,15000,20000,25000,30000,35000,40000,45000]
#for i in range(
for i in range(len(label)):
    index = startPtr[label[i]]
    pixel[index][0] = np.reshape(train_set[0][i],(28,-1))
    startPtr[label[i]] = index+1

LABELS = ['0','1','2','3','4','5','6','7','8','9']
for i in range(len(LABELS)):
    np.save(LABELS[i]+'.npy',pixel[i*5000:5000*(i+1)])
    
    
