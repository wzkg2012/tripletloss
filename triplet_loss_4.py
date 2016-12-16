
# pylint: disable=superfluous-parens, no-member, invalid-name
import sys, random, os
import mxnet as mx
import numpy as np
from operator import itemgetter
from inception import  get_inception

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
        for name in names:
            self.cache.append(np.load(name))
        print 'load data ok'
        self.batch_size = batch_size
        self.provide_data = [('same', (batch_size, 3, 32, 32)), \
                             ('diff', (batch_size, 3, 32, 32))]
        self.provide_label = [('anchor', (batch_size, 3, 32, 32))]
        
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
                        mx.nd.array(batch_one)]
            label_all = [mx.nd.array(batch_anchor)]
            data_names = ['same', 'diff']
            label_names = ['anchor']
            
            data_batch = Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass




def get_net(batch_size=128,hash_len=100,scale=100,alpha=5):
    same = mx.sym.Variable('same')
    diff = mx.sym.Variable('diff')
    anchor = mx.sym.Variable('anchor')
    concat = mx.symbol.Concat(*[same, diff, anchor], dim=0, name='concat')
    
    output = get_inception(concat, hash_len)
    #triplet loss
    fc1_fs =  mx.symbol.slice_axis(output, axis=0, begin=0, end=batch_size,name='fc1_fs')
    fc1_fd =  mx.symbol.slice_axis(output, axis=0, begin=batch_size, end=2*batch_size,name='fc1_fd')
    fc1_fa =  mx.symbol.slice_axis(output, axis=0, begin=2*batch_size, end=3*batch_size,name='fc1_fa')
    '''
    theta_as = mx.symbol.dot(lhs=sigmoid_fa.transpose(0,1),rhs=sigmoid_fs)
    theta_ad = mx.symbol.dot(lhs=sigmoid_fa.transpose(0,1),rhs=sigmoid_fd)
   '''

    theta_as = fc1_fa*fc1_fs
    theta_ad = fc1_fa*fc1_fd

    tripletLoss = -(theta_as-theta_ad-alpha-mx.symbol.log(1+mx.symbol.exp(theta_as-theta_ad-alpha)))
    tripletLoss = mx.symbol.sum(data=tripletLoss,axis=1)
    tripletLoss = mx.symbol.MakeLoss(data = tripletLoss,name="tripelt_hash_loss")
    totalLoss = mx.sym.Group([tripletLoss])
    return totalLoss


class Auc(mx.metric.EvalMetric):
    def __init__(self):
        super(Auc, self).__init__('auc')

    def update(self, labels, preds):
        pred = preds[0].asnumpy().reshape(-1)
        self.sum_metric += np.sum(pred)
        self.num_inst += len(pred)

if __name__ == '__main__':
    batch_size = 128
    hashing_len = 1024
    network = get_net(batch_size,hashing_len)
    devs = [mx.gpu(3)]
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 100,
                                 learning_rate = 1e-2,
                                 wd = 0.00001,
                                 initializer = mx.init.Normal(sigma=0.1),
                                 momentum = 0.9)#.Xavier(factor_type="in", magnitude=2.34)
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
