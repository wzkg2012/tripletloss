# pylint: disable=superfluous-parens, no-member, invalid-name
import sys, random, os
import mxnet as mx
import numpy as np
import cv2
from operator import itemgetter
from get_symbol import *

class Batch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]

class DataIter(mx.io.DataIter):
    def __init__(self, names, batch_size):
        super(DataIter, self).__init__()
        self.cache = []
        size = 28
        for name in names:
            self.cache.append(np.load(name))
        print 'load data ok'
        self.batch_size = batch_size
        self.provide_data = [('same', (batch_size,1,  size, size)), \
                             ('diff', (batch_size, 1, size, size)), \
                             ('anchor', (batch_size,1,  size, size))]
        self.provide_label = [('one', (batch_size,))]
        
    def generate_batch(self, n):
        n1, n2 = random.sample(range(len(self.cache)), 2)
        d1 = self.cache[n1]
        d2 = self.cache[n2]
        ret = []
        while len(ret) < n:
            k1 = random.randint(0, len(d1) - 1)
            k2 = random.randint(0, len(d1) - 1)
            k3 = random.randint(0, len(d2) - 1)
            if k1 == k2:
                continue
            ret.append((d1[k1], d1[k2], d2[k3]))
        return ret

    def __iter__(self):
        print 'begin'
        count = 100000 / self.batch_size
        for i in range(count):
            batch = self.generate_batch(self.batch_size)
            batch_anchor = [x[0] for x in batch]
            batch_same = [x[1] for x in batch]
            batch_diff = [x[2] for x in batch]
            batch_one = np.ones(self.batch_size)
                        
            data_all = [mx.nd.array(batch_same), mx.nd.array(batch_diff), \
                        mx.nd.array(batch_anchor)]
            label_all = [mx.nd.array(batch_one)]
            data_names = ['same', 'diff', 'anchor']
            label_names = ['one']
            
            data_batch = Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass



def get_net(batch_size=128,hash_len=32):
    same = mx.sym.Variable('same')
    diff = mx.sym.Variable('diff')
    anchor = mx.sym.Variable('anchor')
    one = mx.sym.Variable('one')
    one = mx.sym.Reshape(data = one, shape = (-1, 1))
    concat = mx.symbol.Concat(*[same, diff, anchor], dim=0, name='concat')
    output = get_symbol(concat,batch_size, hash_len)
    #output = mx.sym.L2Normalization(data=output,name = 'bn_fc')
    
    '''
    fs = get_conv(same, conv_weight, conv_bias, fc_weight, fc_bias)
    fd = get_conv(diff, conv_weight, conv_bias, fc_weight, fc_bias)
    '''
    fs = mx.symbol.slice_axis(output, axis=0, begin=0, end=batch_size)
    fd = mx.symbol.slice_axis(output, axis=0, begin=batch_size, end=2*batch_size)
    fa = mx.symbol.slice_axis(output, axis=0, begin=2*batch_size, end=3*batch_size)
    fs = fa - fs
    fd = fa - fd
    fs = fs * fs
    fd = fd * fd
    fs = mx.sym.sum(fs, axis = 1, keepdims = 1)
    fd = mx.sym.sum(fd, axis = 1, keepdims = 1)
    loss = fd - fs
    loss = one - loss
    loss = mx.sym.Activation(data = loss, act_type = 'relu')
    return mx.sym.MakeLoss(loss)

class Auc(mx.metric.EvalMetric):
    def __init__(self):
        super(Auc, self).__init__('auc')

    def update(self, labels, preds):
        pred = preds[0].asnumpy().reshape(-1)
        self.sum_metric += np.sum(pred)
        self.num_inst += len(pred)

if __name__ == '__main__':
    batch_size = 32
    hashing_len = 10
    network = get_net(batch_size,hashing_len)
    devs = [mx.gpu(0)]
    #pre_trained = mx.model.FeedForward.load('Inception-BN',126)
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 200,
                                 learning_rate = 0.001,
                                 wd = 0.00001,
                                 initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
                                 momentum = 0.0)
    names = []
    root = sys.argv[1]
    for fn in os.listdir(root):
        if fn.endswith('.npy'):
            names.append(root + '/' + fn)
    print len(names)
    data_train = DataIter(names, batch_size)
    
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    
    metric = Auc()
    model.fit(X = data_train,
              eval_metric = metric, 
              kvstore = 'local_allreduce_device',
              batch_end_callback=mx.callback.Speedometer(batch_size, 50),)
    
    model.save(sys.argv[2])
