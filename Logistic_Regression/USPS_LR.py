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

a1 = (np.matrix(np.ones(len(x_y)))).transpose()
x = np.concatenate((a1,x), axis = 1)
a1 = (np.matrix(np.ones(len(v_y)))).transpose()
v = np.concatenate((a1,v), axis = 1)
a1 = (np.matrix(np.ones(len(t_y)))).transpose()
t = np.concatenate((a1,t), axis = 1)


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
def err(w, t, tk):
	count = 0
	yp = w.T*t.T
	yp = yp.T
	s = (len(tk),10)
	ypp = np.zeros(s)
	for i in range(len(tk)):
		ypp[i,np.argmax(yp[i])] = 1
		if not np.array_equal(tk[i],ypp[i]):
			count = count +1
	print 'Wrong Classifications:',count 
	Err = 1-(float(count)/float(len(tk)))
	print 'Accuracy:',Err

#initializing weight matrix & bias vector
s = (785, 10)
w = np.zeros((s))
B = np.matrix(np.zeros(10)).transpose()
tk = OneOfK(x_y)

#Training of the weight matrix
for step in range(10):
	print 'Step:',step
	for i in range(0, len(x_y)):
		A = w.T*x[i,:].T + B
 		
		yk = np.exp(A)/np.sum(np.exp(A), axis = 0)

		E = -(tk[i,:]*np.log(yk))
		dE = x[i,:].T*(yk.T - tk[i,:])
		w = w - 0.001*dE
	err(w, x, tk)

vk = OneOfK(v_y)
print 'Validation Error'
err(w, v, vk)
ttk = OneOfK(t_y)
print 'Test Error'
err(w, t, ttk)
uspsk = OneOfK(uspsy)
print 'USPS data Error'
err(w, uspsx, uspsk)
