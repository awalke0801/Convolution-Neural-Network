import cPickle
import gzip
import numpy as np 
import Neural_Network

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
		print i 
		A = w.T*x[i,:].T + B
 		
		yk = np.exp(A)/np.sum(np.exp(A), axis = 0)

		E = -(tk[i,:]*np.log(yk))
		dE = x[i,:].T*(yk.T - tk[i,:])
		w = w - 0.001*dE
	err(w, x, tk)
#Finding the Error on Validation and test sets
vk = OneOfK(v_y)
print 'Validation Error'
err(w, v, vk)
ttk = OneOfK(t_y)
print 'Test Error'
err(w, t, ttk)


