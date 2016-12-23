import sys, random, os
import mxnet as mx
import numpy as np
import pdb
from operator import itemgetter
from inception import get_inception_symbol


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
        self.provide_data = [('same', (batch_size, 3, 224, 224)), \
                             ('diff', (batch_size, 3, 224, 224)), \
                             ('anchor', (batch_size, 3, 224, 224))
                             ]
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

            data_all = [mx.nd.array(batch_same), mx.nd.array(batch_diff), mx.nd.array(batch_anchor)]
            label_all = [mx.nd.array(batch_one)]
            data_names = ['same', 'diff', 'anchor']
            label_names = ['one']

            data_batch = Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass


def get_net(batch_size=128, hash_len=100, scale=0.1, alpha=1):
    same = mx.sym.Variable('same')
    diff = mx.sym.Variable('diff')
    anchor = mx.sym.Variable('anchor')
    one = mx.sym.Variable('one')
    one = mx.sym.Reshape(data=one, shape=(-1, 1)) * alpha
    concat = mx.symbol.Concat(*[same, diff, anchor], dim=0, name='concat')

    output = get_inception_symbol(concat, hash_len)
    output = mx.symbol.FullyConnected(data=output, \
                                  num_hidden=feature_len, name='fc')
    output = mx.sym.L2Normalization(data=output,name = 'bn_fc')
    # triplet loss
    fc1_fs = mx.symbol.slice_axis(output, axis=0, begin=0, end=batch_size, name='fc1_fs')
    fc1_fd = mx.symbol.slice_axis(output, axis=0, begin=batch_size, end=2 * batch_size, name='fc1_fd')
    fc1_fa = mx.symbol.slice_axis(output, axis=0, begin=2 * batch_size, end=3 * batch_size, name='fc1_fa')
    '''
    theta_as = mx.symbol.dot(lhs=sigmoid_fa.transpose(0,1),rhs=sigmoid_fs)
    theta_ad = mx.symbol.dot(lhs=sigmoid_fa.transpose(0,1),rhs=sigmoid_fd)
   '''
    quantLoss = mx.symbol.sign(data=output, name='sign') - output
    quantLoss = mx.symbol.square(data=quantLoss, name='square')
    quantLoss1 = mx.symbol.sum(data=quantLoss, axis=1, name='qsum')
    # quantLoss =mx.symbol.sqrt(data = quantLoss,name = 'quantLoss_')
    quantLoss = quantLoss1 * scale
    quantLoss = mx.symbol.MakeLoss(data=quantLoss, name='quantLoss')

    theta_as = 0.5 * fc1_fa * fc1_fs
    theta_ad = 0.5 * fc1_fa * fc1_fd
    theta_as = mx.sym.sum(data=theta_as, axis=(1), keepdims=1)
    theta_ad = mx.sym.sum(data=theta_ad, axis=(1), keepdims=1)
    # tripletLoss1 = theta_ad-theta_as-one
    # tripletLoss1 = mx.symbol.MakeLoss(data=tripletLoss1, name="tripelt_hash_loss")


    tripletLoss = -(theta_as - theta_ad - one - mx.symbol.log(1 + mx.symbol.exp(theta_as - theta_ad - one)))
    tripletLoss = mx.symbol.MakeLoss(data=tripletLoss, name="tripelt_hash_loss")

    totalLoss = mx.sym.Group([tripletLoss, quantLoss])
    return totalLoss


class tripletLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(tripletLossMetric, self).__init__('triplet_hash_loss')

    def update(self, labels, preds):
        pred = preds[0].asnumpy().reshape(-1)
        self.sum_metric += np.sum(pred)
        self.num_inst += len(pred)




class Auc(mx.metric.EvalMetric):
    def __init__(self):
        super(Auc, self).__init__('quantLoss')

    def update(self, labels, preds):
        pred = preds[1].asnumpy().reshape(-1)
        self.sum_metric += np.sum(pred)
        self.num_inst += len(pred)


if __name__ == '__main__':
    batch_size = 32
    hashing_len = 32
    train_flag = True
    network = get_net(batch_size, hashing_len)
    devs = [mx.gpu(3)]
    model = mx.model.FeedForward(ctx=devs,
                                     symbol=network,
                                     num_epoch=50,
                                     learning_rate=1e-3,
                                     wd=0.00001,
                                     initializer=mx.init.Normal(sigma=1),
                                     momentum=0.9)  # .Xavier(factor_type="in", magnitude=2.34)
    eval_metrics = mx.metric.CompositeEvalMetric()
    eval_metrics.add(Auc())
    eval_metrics.add(tripletLossMetric())
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
    model.fit(X=data_train,
                  eval_metric=eval_metrics)
    model.save(sys.argv[2])
'''
    else:
	prefix = "."
        pre_trained = mx.model.FeedForward.load(prefix, 1, ctx=devs)
        feature_map(pre_trained,network,data_train)
'''
