import os
import random
import sys
import shutil

"""
Script Usage

> execute it in ml/project/ directory as: 
  python2 scripts/generating_datasets.py <inp_db>  <n_train>
> <inp_db> input database: faces94, faces95 or faces96 as string
> <n_train> is the number for samples per class in training set
> The remaining samples will be copied in testing_set
"""


#verify parameters are valid
if len(sys.argv)>3:
	sys.exit('too much parameters!')
elif len(sys.argv)<3:
	sys.exit('missing parameters!')
else:
	inp_db = sys.argv[1]
	if inp_db not in ['faces94','faces95','faces96']:
		sys.exit('database not valid!')
	n_train = int(sys.argv[2])
	if n_train<1 or n_train>5:
		sys.exit('wrong number of samples per class!')
	
#set some variables according to input parameters
path = './db/'+inp_db+'/'                                    #source path
tr_path = './db/train'+inp_db[-2:]+'/'+str(n_train)+'pc/'    #training path
ts_path = './db/test'+inp_db[-2:]+'/'+str(20-n_train)+'pc/'  #testing path
N = len(os.listdir(path))                                    #total number classes (people)


if os.path.exists(tr_path):
	sys.exit(tr_path+' directory already exists!')
else:
	os.makedirs(tr_path)

if os.path.exists(ts_path):
	sys.exit(ts_path+' directory already exists!')
else:
	os.makedirs(ts_path)

#iterate through all files in ./faces directory
for i in range(1,N):
	tgt = path+str(i)+'/'
	files = os.listdir(tgt)
	#shuffle the filenames to make it fair
	random.shuffle(files)

	#create target directories
	os.makedirs(tr_path+str(i))
	os.makedirs(ts_path+str(i))

	#copy the first n_train photos in train and the remaining
	#20-n_train in test directories
	for j in range(n_train):
		shutil.copy(tgt+files[j], tr_path+str(i))
	for j in range(n_train,20):
		shutil.copy(tgt+files[j], ts_path+str(i))
print "Done!"
