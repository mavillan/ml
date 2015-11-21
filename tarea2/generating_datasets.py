import numpy as np
import sys

#dataname - string - dataset name
#ns - integer - number of sets to generate
#otype - string  - type of output sets (binary or text files)

def generating_datasets(dataname, ns=20, otype='binary'):
	if dataname == 'credit':
		orig_name = 'credit.data'
		#load original dataset as numpy array
		A = np.loadtxt(orig_name, dtype=float, delimiter=' ')
	elif dataname == 'diabetes':
		orig_name = 'diabetes.data'
		#load original dataset as numpy array
		A = np.loadtxt(orig_name, dtype=float, delimiter=',')
	else:
		 sys.error('Error: Requested data is not in this database')

	#rows and columns
	r,c = A.shape

	for i in range(ns):
		#shuffle rows of data matrix
		B = A.copy()
		np.random.shuffle(B)
		#names for training and testing files
		name_tr = dataname+'-tr-'+str(i)
		name_ts = dataname+'-ts-'+str(i)
		if otype=='binary':
			#save sets as numpy binary files
			np.save(name_tr, B[0:np.round(0.75*r)])
			np.save(name_ts, B[np.round(0.75*r):r])
		elif otype=='text':
			#save sets as text files
			np.savetxt(name_tr, B[0:np.round(0.75*r)], delimiter=' ')
			np.savetxt(name_ts, B[np.round(0.75*r):r], delimiter=' ')
	return 1
