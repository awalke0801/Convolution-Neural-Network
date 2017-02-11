import cPickle
import gzip
import numpy as np 

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
		print 'Step:',steps
		for i in range(len(x)):
			print 'n',i
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

	
	
tk = OneOfK(x_y)
NeuralNetwork(x, 100, 10, tk)
