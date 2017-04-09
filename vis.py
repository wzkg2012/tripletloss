import os
import mxnet as mx
import numpy as np
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import cPickle,gzip
import pdb
from get_symbol import *

# load MNIST data 
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)


def visual_feature_space(features,labels, num_classes, name_dict):
    num = len(labels)

    # draw
    palette = np.array(sns.color_palette("hls", num_classes))
    
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features[:,0], features[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    ax.axis('off')
    ax.axis('tight')
    
    # We add the labels for each digit.
    txts = []
    for i in range(num_classes):
        # Position of each label.
        xtext, ytext = np.median(features[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, name_dict[i],fontsize=50)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.tight_layout()
    plt.savefig('triplet.eps',format='eps',dpi=500)
    plt.show()
    return f, ax, sc, txts

def main():
    # load model, get embedding layer
    batchSize = 1
    data = mx.sym.Variable('data')
    hash_len = 10
    network =  get_symbol(data,batchSize, hash_len)
    model = mx.model.FeedForward.load('.', 200, ctx=mx.cpu(0), numpy_batch_size=batchSize)
    internals = network.get_internals()
    embedding_layer = internals['embedding_output']
    feature_extractor = mx.model.FeedForward(ctx=mx.cpu(0), symbol=embedding_layer, numpy_batch_size=1,\
            arg_params = model.arg_params, aux_params=model.aux_params, allow_extra_params=True)
    print 'feature_extractor loaded'

           
    # extract feature 
    print 'extracting feature'
    embeds = []
    labels = []
    tmparray = np.zeros((1,1,28,28))
    for i in range(len(valid_set[1])):   
        tmparray[0][0] = np.reshape(valid_set[0][i],(28,28))
        preds = feature_extractor.predict(tmparray)
        embeds.append( preds )
        labels.append( valid_set[1][i])
    
    embeds = np.vstack(embeds)
    labels = np.hstack(labels)
    
    print 'embeds shape is ', embeds.shape
    print 'labels shape is ', labels.shape
    
    np.save('embeds.npy',embeds)
    np.save('labels.npy',labels)
    
    # prepare dict for display
    namedict = dict()
    for i in range(10):
        namedict[i]=str(i)

    visual_feature_space(embeds, labels, 10, namedict)

if __name__ == "__main__":
    main()
