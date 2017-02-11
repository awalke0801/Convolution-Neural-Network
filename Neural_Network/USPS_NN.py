import cPickle
import gzip
import numpy as np 
from os import listdir
from os.path import isfile, join
import cv2
import scipy
from scipy import misc


filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = cPickle.load(f)
f.close()

#Loading training features and labels
x = np.matrix(np.asarray(training_data[0:-1][:]))
x_y = (np.matrix(training_data[-1][:])).T

#Loading Validation features and labels
v = np.matrix(np.asarray(validation_data[0:-1][:]))
v_y = (np.matrix(validation_data[-1][:])).T

#Loading Testing features and labels
t = np.matrix(np.asarray(test_data[0:-1][:]))
t_y = (np.matrix(test_data[-1][:])).T

o = (np.matrix(np.ones(len(x_y)))).transpose()
x = np.concatenate((o,x), axis = 1)
o = (np.matrix(np.ones(len(v_y)))).transpose()
v = np.concatenate((o,v), axis = 1)
o = (np.matrix(np.ones(len(t_y)))).transpose()
t = np.concatenate((o,t), axis = 1)


print 'start'
for i in range(10):
	mypath = '/home/aditya/Documents/ML/MLProject3/Numerals/%d'%i
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	img = np.empty(len(onlyfiles), dtype = object)
	for n in range(0, len(onlyfiles)):
		img[n] = cv2.imread(join(mypath, onlyfiles[n]),0)
	if i == 0:
		images = img
	else:
		images = np.concatenate((images, img))
maxsize = (28,28)
print 'images read'
for i in range(len(images)):
	images[i] = scipy.misc.imresize(images[i], maxsize)
	images[i] = (images[i]-images[i].max())	
	images[i] = np.divide(images[i],images[i].max(), dtype= float)
	images[i] = np.matrix(images[i].flatten())
	if i == 0:
		uspsx = images[i]
	else:
		uspsx = np.concatenate((uspsx,images[i]))
	if float(i)%1000 == float(0):
		print i 
print 'images scaled'
uspsy = np.empty(2000)
uspsy.fill(0)
j = 1
for i in range(2000,len(images),2000):
	emp = np.empty(2000)
	emp.fill(j)
	uspsy = np.concatenate((uspsy, emp))
	j += 1
uspsy = (np.matrix(uspsy.astype(int))).T
print 'y created'
o = (np.matrix(np.ones(len(uspsy)))).transpose()
uspsx = np.concatenate((o,uspsx), axis = 1)
print 'Classification starts'

# 1-0f-K coding
def OneOfK(ty):
	s = (len(ty),10)
	tk = np.zeros(s)
	for i in range(0,len(ty)):
		tk[i,ty[i]] = 1
	return tk

#Calculating the error
def err(w1, w2, x, tk):
	count = 0
	z1 = x*w1
	a1 = np.tanh(z1)
	z2 = a1*w2
	s = (len(tk),10)
	ypp = np.zeros(s)
	for i in range(len(tk)):
		ypp[i,np.argmax(z2[i])] = 1
		if not np.array_equal(tk[i],ypp[i]):
			count = count +1
	print 'Wrong Classifications:',count 
	Err = 1-(float(count)/float(len(tk)))
	print 'Accuracy:',Err
	

def NeuralNetwork(x, num_z, num_y, tk):
	s = (x.shape[1],num_z)
	w1 = np.random.randn(x.shape[1], num_z)
	b1 = np.matrix(np.random.randn(num_z)).T
	s = (num_z, num_y)
	w2 = np.random.randn(num_z,num_y)
	b2 = np.matrix(np.random.randn(num_y)).T
	for steps in range(20):
		print 'iteration number:',steps
		for i in range(len(x)):
			z1 = w1.T*x[i].T + b1
			a1 = np.tanh(z1)
			z2 = w2.T*a1 + b2
			exp = np.exp(z2)
			prob_y = exp/np.sum(exp, axis = 0)
			dk = prob_y - np.matrix(tk[i]).T
			dj = np.multiply((1 - np.power(a1, 2)),(w2*dk))
			delE_w1 = dj*x[i]
			delE_w2 = dk*a1.T
			w1 = w1 - 0.01*(delE_w1.T)
			w2 = w2 - 0.01*(delE_w2.T)
		err(w1, w2, x, tk)
	print 'Validation Error'
	tk = OneOfK(v_y)
	err(w1, w2, v, tk)
	print 'Testing Error'
	tk = OneOfK(t_y)
	err(w1, w2, t, tk)	
	print 'USPS data Error'
	uspsk = OneOfK(uspsy)
	err(w1, w2, uspsx, uspsk)
	
	
tk = OneOfK(x_y)
NeuralNetwork(x, 100, 10, tk)
