

import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize
from PIL import Image

from os import listdir
from os.path import isfile, join

import fnmatch
import os
import random


def read_svhn():


	## class
	class SVHN10Record(object):
    		pass

	## scprit for loading images
	
  	result = SVHN10Record()	

	
  	result.height = 32
  	result.width  = 32
  	result.depth  = 3

        directories = []
	directories.append('../data/10/')
  	directories.append('../data/1/')
	directories.append('../data/2/')
	directories.append('../data/3/')
	directories.append('../data/4/')
	directories.append('../data/5/')
	directories.append('../data/6/')
	directories.append('../data/7/')
	directories.append('../data/8/')
	directories.append('../data/9/')
	
	train_images = []
	train_labels = []
	print directories
	for i in xrange(len(directories)):
		for root, dirs, filenames in os.walk(top = directories[i], topdown=True):
			for filename in fnmatch.filter(filenames, '*.jpg'):
				print root, filenames
				train_images.append(os.path.join(root, filename))
				train_labels.append(i)
	
			
	result.images = train_images
	result.labels = train_labels
           
	result.n_images = 100

 	return result
 

class cnn_inception:
    def __init__(self, imgs, labels, flag=True, sess=None):

	print "in __init__"
        self.imgs = imgs
        self.labels = labels
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc2)
	self.logits = self.fc2	
        self.cross_entropy_mean = self.loss()
	self.train_op = self.train()
	
	self.train = flag

    def convlayers(self):

	print "in convlayer"

	self.map1 = 32
	self.map2 = 64
	self.num_fc1 = 700 
	self.num_fc2 = 10
	self.reduce1x1 = 16

        self.parameters = []

        # zero-mean input
        ##with tf.name_scope('preprocess') as scope:
        ##    mean = tf.constant([125], dtype=tf.float32, shape=[1, 1, 1, 1], name='img_mean')
        ##    images = self.imgs - mean

	images = self.imgs

        # conv1
        with tf.name_scope('conv1_1x1_1') as scope:
            	kernel = tf.Variable(tf.truncated_normal([1, 1, 1, self.map1], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            	conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            	biases = tf.Variable(tf.constant(0.0, shape=[self.map1], dtype=tf.float32),
                                 trainable=True, name='biases')
            	out = tf.nn.bias_add(conv, biases)
		self.conv1_1x1_1 = out
	    	self.parameters += [kernel, biases]
	    
	with tf.name_scope('conv1_1x1_2') as scope:

	    	kernel = tf.Variable(tf.truncated_normal([1, 1, 1, self.reduce1x1], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            	conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            	biases = tf.Variable(tf.constant(0.0, shape=[self.reduce1x1], dtype=tf.float32),
                                 trainable=True, name='biases')
	
		out = tf.nn.bias_add(conv, biases)
            	self.conv1_1x1_2 = tf.nn.relu(out, name=scope)
            	self.parameters += [kernel, biases]

	with tf.name_scope('conv1_1x1_3') as scope:

	    	kernel = tf.Variable(tf.truncated_normal([1, 1, 1, self.reduce1x1], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            	conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            	biases = tf.Variable(tf.constant(0.0, shape=[self.reduce1x1], dtype=tf.float32),
                                 trainable=True, name='biases')
	
		out = tf.nn.bias_add(conv, biases)
            	self.conv1_1x1_3 = tf.nn.relu(out, name=scope)
            	self.parameters += [kernel, biases]


	with tf.name_scope('conv1_3x3') as scope:

	    	kernel = tf.Variable(tf.truncated_normal([3, 3, self.reduce1x1, self.map1], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            	conv = tf.nn.conv2d(self.conv1_1x1_2, kernel, [1, 1, 1, 1], padding='SAME')
            	biases = tf.Variable(tf.constant(0.0, shape=[self.map1], dtype=tf.float32),
                                 trainable=True, name='biases')
	
		out = tf.nn.bias_add(conv, biases)
            	self.conv1_3x3 = out
            	self.parameters += [kernel, biases]

	with tf.name_scope('conv1_5x5') as scope:

	    	kernel = tf.Variable(tf.truncated_normal([5, 5, self.reduce1x1, self.map1], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            	conv = tf.nn.conv2d(self.conv1_1x1_3, kernel, [1, 1, 1, 1], padding='SAME')
            	biases = tf.Variable(tf.constant(0.0, shape=[self.map1], dtype=tf.float32),
                                 trainable=True, name='biases')
	
		out = tf.nn.bias_add(conv, biases)
            	self.conv1_5x5= out
            	self.parameters += [kernel, biases]

	self.maxpool1 = tf.nn.max_pool(images,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='maxpool1')

	with tf.name_scope('conv1_1x1_4') as scope:

	    	kernel = tf.Variable(tf.truncated_normal([1, 1, 1, self.map1], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            	conv = tf.nn.conv2d(self.maxpool1, kernel, [1, 1, 1, 1], padding='SAME')
            	biases = tf.Variable(tf.constant(0.0, shape=[self.map1], dtype=tf.float32),
                                 trainable=True, name='biases')
	
		out = tf.nn.bias_add(conv, biases)
            	self.conv1_1x1_4= out
            	self.parameters += [kernel, biases]

	## concat( values, axis, name='concat')
	self.inception1 = tf.nn.relu(tf.concat(axis=3, values = [self.conv1_1x1_1, self.conv1_3x3, self.conv1_5x5, self.conv1_1x1_4]))

	
        # conv2
        	
	with tf.name_scope('conv2_1x1_1') as scope:
            	kernel = tf.Variable(tf.truncated_normal([1, 1, 4*self.map1, self.map2], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            	conv = tf.nn.conv2d(self.inception1, kernel, [1, 1, 1, 1], padding='SAME')
            	biases = tf.Variable(tf.constant(0.0, shape=[self.map2], dtype=tf.float32),
                                 trainable=True, name='biases')
            	out = tf.nn.bias_add(conv, biases)
		self.conv2_1x1_1 = out
	    	self.parameters += [kernel, biases]
	    
	with tf.name_scope('conv2_1x1_2') as scope:

	    	kernel = tf.Variable(tf.truncated_normal([1, 1, 4*self.map1, self.reduce1x1], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            	conv = tf.nn.conv2d(self.inception1, kernel, [1, 1, 1, 1], padding='SAME')
            	biases = tf.Variable(tf.constant(0.0, shape=[self.reduce1x1], dtype=tf.float32),
                                 trainable=True, name='biases')
	
		out = tf.nn.bias_add(conv, biases)
            	self.conv2_1x1_2 = tf.nn.relu(out, name=scope)
            	self.parameters += [kernel, biases]

	with tf.name_scope('conv2_1x1_3') as scope:

	    	kernel = tf.Variable(tf.truncated_normal([1, 1, 4*self.map1, self.reduce1x1], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            	conv = tf.nn.conv2d(self.inception1, kernel, [1, 1, 1, 1], padding='SAME')
            	biases = tf.Variable(tf.constant(0.0, shape=[self.reduce1x1], dtype=tf.float32),
                                 trainable=True, name='biases')
	
		out = tf.nn.bias_add(conv, biases)
            	self.conv2_1x1_3 = tf.nn.relu(out, name=scope)
            	self.parameters += [kernel, biases]


	with tf.name_scope('conv2_3x3') as scope:

	    	kernel = tf.Variable(tf.truncated_normal([3, 3, self.reduce1x1, self.map2], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            	conv = tf.nn.conv2d(self.conv2_1x1_2, kernel, [1, 1, 1, 1], padding='SAME')
            	biases = tf.Variable(tf.constant(0.0, shape=[self.map2], dtype=tf.float32),
                                 trainable=True, name='biases')
	
		out = tf.nn.bias_add(conv, biases)
            	self.conv2_3x3 = out
            	self.parameters += [kernel, biases]

	with tf.name_scope('conv2_5x5') as scope:

	    	kernel = tf.Variable(tf.truncated_normal([5, 5, self.reduce1x1, self.map2], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            	conv = tf.nn.conv2d(self.conv2_1x1_3, kernel, [1, 1, 1, 1], padding='SAME')
            	biases = tf.Variable(tf.constant(0.0, shape=[self.map2], dtype=tf.float32),
                                 trainable=True, name='biases')
	
		out = tf.nn.bias_add(conv, biases)
            	self.conv2_5x5= out
            	self.parameters += [kernel, biases]

	self.maxpool2 = tf.nn.max_pool(self.inception1,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='maxpool1')

	with tf.name_scope('conv2_1x1_4') as scope:

	    	kernel = tf.Variable(tf.truncated_normal([1, 1, 4*self.map1, self.map2], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            	conv = tf.nn.conv2d(self.maxpool2, kernel, [1, 1, 1, 1], padding='SAME')
            	biases = tf.Variable(tf.constant(0.0, shape=[self.map2], dtype=tf.float32),
                                 trainable=True, name='biases')
	
		out = tf.nn.bias_add(conv, biases)
            	self.conv2_1x1_4= out
            	self.parameters += [kernel, biases]
	
	self.inception2 = tf.nn.relu(tf.concat(axis=3, values = [self.conv2_1x1_1, self.conv2_3x3, self.conv2_5x5, self.conv2_1x1_4]))	


        
    def fc_layers(self):

	self.dropout=0.5
	
	#flatten features for fully connected layer
        self.inception2_flat = tf.reshape(self.inception2,[-1,32*32*4*self.map2])
		
        #Fully connected layers
	
        fc1w = tf.Variable(tf.truncated_normal([32*32*(4*self.map2),self.num_fc1],
                                                         dtype=tf.float32,
                                                         stddev=0.01), name='weights')
        fc1b = tf.Variable(tf.constant(1.0, shape=[self.num_fc1], dtype=tf.float32),
                                 trainable=True, name='biases')
        
        self.fc1 = tf.nn.bias_add(tf.matmul(self.inception2_flat, fc1w), fc1b)
	
        if self.train:
            self.h_fc1 =tf.nn.dropout(tf.nn.relu( self.fc1 ), self.dropout)
        else:
            self.h_fc1 = tf.nn.relu(self.fc1)

	
 	fc2w = tf.Variable(tf.truncated_normal([self.num_fc1, self.num_fc2],
                                                         dtype=tf.float32,
                                                         stddev=0.01), name='weights')
        fc2b = tf.Variable(tf.constant(1.0, shape=[self.num_fc2], dtype=tf.float32),
                                 trainable=True, name='biases')

	self.fc2 = tf.nn.bias_add(tf.matmul(self.h_fc1, fc2w), fc2b)
	
        return self.fc2
	

    def loss(self):
		
		print "in loss -- without one-host"
		## for one-shot, use tf.nn.softmax_cross_entropy_with_logits
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      					labels=self.labels, logits=self.logits, name='cross_entropy_per_example')
  		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		return cross_entropy_mean

    def train(self):

		print "in train"
		train_op = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy_mean)
		
		return train_op
    	

if __name__ == '__main__':

    batch_size = 2
    
   
    ## Launch the graph in a session.
    with tf.Session() as sess:
	
    	## placeholder( dtype, shape=None, name=None)
    	## Its value must be fed using the feed_dict 
    	images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 32,32,1))
    	labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
        flag = True
    	cnn = cnn_inception(images_placeholder, labels_placeholder, flag, sess)

	saver = tf.train.Saver()
    	init = tf.global_variables_initializer()
    	sess.run(init)

    	result = read_svhn()
    	imgs_info = result.images
    	labels = result.labels

	combined = list(zip(imgs_info, labels))
    	random.shuffle(combined)

    	imgs_info[:], labels[:] = zip(*combined)

	##for i in xrange(len(labels)):
	##	print imgs_info[i], " -- > ",labels[i]

	for epoch in xrange(2):
	
		counter = 0
		for i in xrange(int((len(labels)-batch_size)/(batch_size))):
			batch_images = imgs_info[batch_size*i:batch_size*(i+1)]
			batch_labels = labels[batch_size*i:batch_size*(i+1)]

			print batch_images, " -- ", batch_labels
			batch_list = []
		
			for j in xrange(len(batch_labels)):
				x=Image.open(batch_images[j],'r')
				x=x.convert('L') ## color to gray

				x_ac = np.array(x)
				x_ac = x_ac - 125
				##print np.shape(x_ac)
				x_new = np.reshape(x_ac,(32, 32, 1))
				print np.shape(x_new)

				batch_list.append(x_new)
			

			batch_array = np.asarray(batch_list)
			label_array = np.asarray(batch_labels)
			print np.shape(batch_array)
			print np.shape(label_array)

			_, loss_val = sess.run( [cnn.train_op, cnn.cross_entropy_mean], feed_dict={cnn.imgs: batch_array, cnn.labels: label_array })
			print ".. loss = ", loss_val
	
		saver.save(sess, "my_cnn_model.ckpt")
