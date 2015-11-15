import os
import random
import sys
import shutil

"""
Script Usage

> execute it in ml/project/ directory as: python2 scripts/generating_datasets.py n_train
> n_train is the number for samples per class in training set
> The remaining samples will be copied in testing set
"""


#path were original files are
path = './faces/'

#verify parameters
#n_train is the number of training samples per class
if len(sys.argv)>2:
	sys.exit('too much parameters!')
elif len(sys.argv)==2:
	n_train = int(sys.argv[1])
else:
	n_train = 5

#create target directories
if os.path.exists('./training_set/'):
	sys.exit('training_set directory already exists!')
else:
	os.makedirs('./training_set/')

if os.path.exists('./testing_set/'):
	sys.exit('training_set directory already exists!')
else:
	os.makedirs('./testing_set/')

#iterate through all files in ./faces directory
for i in range(1,154):
	tgt = path+str(i)+'/'
	files = os.listdir(tgt)
	#shuffle the files to make it fair
	random.shuffle(files)

	#create target directories
	os.makedirs('./training_set/'+str(i))
	os.makedirs('./testing_set/'+str(i))

	#copy the first 5 photos in training and the remaining
	#15 in testing directories
	for j in range(n_train):
		shutil.copy(tgt+files[j], './training_set/'+str(i))
	for j in range(n_train,20):
		shutil.copy(tgt+files[j], './testing_set/'+str(i))


	
		


