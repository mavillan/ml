import os
import sys
import time
import numpy as np
import scipy as sp
import cv2 as cv
from sknn.mlp import Classifier, Convolution, Layer
from sknn.platform import cpu32, threading


def load_data(path, stacked=False):
    #total number of classes
    M = len(os.listdir(path))
    #dimensions of each image
    N = 200*180
    #samples per class
    spc = int(path.strip().split('-')[1][:-2])
    #matrix with features
    if stacked:
        data = np.empty((M*spc,200,180), dtype=np.float32)
    else: 
        data = np.empty((M*spc,N), dtype=np.float32)
    labels = np.empty(M*spc, dtype=np.uint8)
    #index of data matrix
    m = 0
    for i in xrange(1,M+1):
        tgt = path+str(i)+'/'
        pics = os.listdir(tgt)
        for pic in pics:
            if stacked:
                #store each image, as a bidimensional array in data matrix
                data[m,:,:] = cv.imread(tgt+pic, cv.IMREAD_GRAYSCALE)
            else:
                #store each flattened image, as a row in data matrix
                data[m,:] = cv.imread(tgt+pic, cv.IMREAD_GRAYSCALE).ravel()
            labels[m] = i
            m += 1
    return (data, labels)


def solve_cnn(dataset, spc, verbose=False):
    #samples per class on training set
    spc_tr = spc
    spc_ts = 20-spc_tr
    #training and testing paths
    tr_path = './db/train'+dataset[-2:]+'/tr-{0}pc-{1}/'
    ts_path = './db/test'+dataset[-2:]+'/ts-{0}pc-{1}/'
    #errors through all datasets
    tr_err = list()
    ts_err = list()
    #iterating through datasets
    for set_num in xrange(20):
        #loading training and testing set
        X_tr,y_tr = load_data(tr_path.format(spc_tr,set_num), stacked=True)
        X_ts,y_ts = load_data(ts_path.format(spc_ts,set_num), stacked=True)
        
        #creating CNN object and fitting the training data
        init_learning_rate=0.0001
        while True:
            try:
                clf = Classifier(
                	layers=[
                		Convolution('Rectifier', channels=12, kernel_shape=(5,5), pool_type='max', pool_shape=(2,2)),
        		        Convolution('Rectifier', channels=8, kernel_shape=(4,4), pool_type='max', pool_shape=(2,2)),
        		        Layer('Rectifier', units=128, dropout=0.25),
        		        Layer('Softmax')],
            		learning_rule='nesterov',
            		learning_rate=init_learning_rate,
            		n_iter=1000,
            		batch_size=50,
            		verbose=True)
                #fitting and timing
                t0 = time.clock()
                clf.fit(X_tr, y_tr)
                elapsed = time.clock()-t0
                break
            except:
                #if diverges, try with this new learning rate
                init_learning_rate /= 10.

        #computing training error
        tr_err.append(1.-clf.score(X_tr,y_tr))
        #computing testing error
        ts_err.append(1.-clf.score(X_ts,y_ts))
        if verbose:
            print "#####################################################################################"
            print "{0}: {1} samples per class (dataset {2})".format(dataset, spc, set_num)
            print "Training error rate: {0}".format(tr_err[-1])
            print "Testing error rate: {0}".format(ts_err[-1])
            print "Fitting time: {0}".format(elapsed)
        #releasing memory of big objects
        del X_tr, X_ts, clf
    return np.array(tr_err),np.array(ts_err)


if __name__=='__main__':
	if len(sys.argv)!=3: 
		sys.exit('bad number of arguments!')

	#dataset name and samples per class
	dataset = sys.argv[1]
	spc = int(sys.argv[2])

	#training on all datasets
	tr_err, ts_err = solve_cnn(dataset, spc, verbose=True)

	#storing results
	np.save('tr_err_{0}_{1}spc_cnn'.format(dataset,spc), tr_err)
	np.save('ts_err_{0}_{1}spc_cnn'.format(dataset,spc), ts_err)

	#done!
	sys.exit('done!')
