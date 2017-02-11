from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import scipy
from scipy import misc
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)


x = tf.placeholder(tf.float32,[None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


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

# 1-0f-K coding
def OneOfK(ty):
	s = (len(ty),10)
	tk = np.zeros(s)
	for i in range(0,len(ty)):
		tk[i,ty[i]] = 1
	return tk


uspsk = OneOfK(uspsy)

# o = (np.matrix(np.ones(len(uspsy)))).transpose()
# uspsx = np.concatenate((o,uspsx), axis = 1)
print uspsx.shape, uspsk.shape


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
j = 0
y_arr =[]
step_arr = []
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
for i in range(20000):
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict = { x: batch[0], y_:batch[1], keep_prob: 1.0})
		print "step %d, training accuracy %g" %(i, train_accuracy)
		if i%1000 == 0:
			y_arr.append(train_accuracy)
			step_arr.append(j)
			j += 1
	train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g"%accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})

print "USPS accuracy %g"%accuracy.eval(feed_dict = {x: uspsx, y_:uspsk, keep_prob: 1.0})
print y_arr
print step_arr