
##
##  Tensorflow Inception with (tiny) SVHN on Docker
##


NOTICE 1: the (tiny) svhn dataset is from the Street View House Numbers (SVHN) Dataset.

NOTICE 2: the source code is based on
 
		[1] https://hacktilldawn.com/2016/09/25/inception-modules-explained-and-implemented/
	
		[2] Davi Frossard, 2016,   VGG16 implementation in Tensorflow , http://www.cs.toronto.edu/~frossard/post/vgg16/   

NOTICE 3: the performance is not evaluated due to the tiny dataset.


##

[1] download (or git clone ) this source code folder.

[2] cd downloaded-source-code-folder

[3] sudo make BIND_DIR=. shell

[4] wait... wait ... then a bash shell (root@1d1515666a6f:/#) will be ready.

[5]  root@b0530bb93ecf:/# cd /home/deeplearning/

[6]  root@b0530bb93ecf:/home/deeplearning# ldd --version

		ldd (Ubuntu GLIBC 2.23-0ubuntu5) 2.23


[7]  root@b0530bb93ecf:/home/deeplearning# git clone https://github.com/tensorflow/tensorflow.git --branch r1.1

[8]  root@b0530bb93ecf:/home/deeplearning# cd tensorflow/

[9]  root@b0530bb93ecf:/home/deeplearning/tensorflow# ./configure
	
	
	Please specify the location of python. [Default is /usr/bin/python]: 
	Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 
	Do you wish to use jemalloc as the malloc implementation? [Y/n] y
	jemalloc enabled
	Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] n
	No Google Cloud Platform support will be enabled for TensorFlow
	Do you wish to build TensorFlow with Hadoop File System support? [y/N] n
	No Hadoop File System support will be enabled for TensorFlow
	Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] n
	No XLA JIT support will be enabled for TensorFlow
	Do you wish to build TensorFlow with VERBS support? [y/N] y
	VERBS support will be enabled for TensorFlow
	Found possible Python library paths:
	  /usr/local/lib/python2.7/dist-packages
	  /usr/lib/python2.7/dist-packages
	Please input the desired Python library path to use.  Default is [/usr/local/lib/python2.7/dist-packages]
	
	Using python library path: /usr/local/lib/python2.7/dist-packages
	Do you wish to build TensorFlow with OpenCL support? [y/N] n
	No OpenCL support will be enabled for TensorFlow
	Do you wish to build TensorFlow with CUDA support? [y/N] n
	No CUDA support will be enabled for TensorFlow
	Extracting Bazel installation...
	...........
	INFO: Starting clean (this may take a while). Consider using --async if the clean takes more than several minutes.
	Configuration finished
	
	
[10]  root@b0530bb93ecf:/home/deeplearning/tensorflow# gcc -v

	
	gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.4) 
	

[11]  root@b0530bb93ecf:/home/deeplearning/tensorflow# bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package 

	Target //tensorflow/tools/pip_package:build_pip_package up-to-date:
  	bazel-bin/tensorflow/tools/pip_package/build_pip_package
	INFO: Elapsed time: 433.238s, Critical Path: 432.46s

[12] root@b0530bb93ecf:/home/deeplearning/tensorflow# bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

	Output wheel file is in: /tmp/tensorflow_pkg


[13] root@b0530bb93ecf:/home/deeplearning/tensorflow# cd /tmp/tensorflow_pkg/
[14] root@b0530bb93ecf:/tmp/tensorflow_pkg# ls (copy the .whl file name)

		tensorflow-1.1.0-cp27-cp27mu-linux_x86_64.whl

[15] root@b0530bb93ecf:/tmp/tensorflow_pkg# pip install ./tensorflow-1.1.0-cp27-cp27mu-linux_x86_64.whl 

[16] root@b0530bb93ecf:/tmp/tensorflow_pkg# cd /home/deeplearning/cnn

[17] root@b0530bb93ecf:/home/deeplearning/cnn# python -c 'import tensorflow as tf; print(tf.__version__)'

	1.1.0

[18] root@b0530bb93ecf:/home/deeplearning/cnn# python ./cnn_inception_train.py

	The output looks somthing like this:

	...
	...
	['../data/6/svhn_6_352.jpg', '../data/10/svhn_10_109.jpg']  --  [6, 0]
	(32, 32, 1)
	(32, 32, 1)
	(2, 32, 32, 1)
	(2,)
	.. loss =  2.39313
	['../data/10/svhn_10_318.jpg', '../data/2/svhn_2_174.jpg']  --  [0, 2]
	(32, 32, 1)
	(32, 32, 1)
	(2, 32, 32, 1)
	(2,)
	.. loss =  2.23676
	['../data/1/svhn_1_416.jpg', '../data/9/svhn_9_745.jpg']  --  [1, 9]
	(32, 32, 1)
	(32, 32, 1)
	(2, 32, 32, 1)
	(2,)
	.. loss =  2.15391
	
	

[19] root@b0530bb93ecf:/home/deeplearning/cnn# python ./cnn_inception_prediction.py

	The output may look like this:

	
	['../data/8/svhn_8_254.jpg']  --  [8]
	(32, 32, 1)
	(1, 32, 32, 1)
	(1,)
	.. probs_val =  [ 0.11383528  0.10478989  0.11900914  0.07403846  0.09969129  0.07817058  0.08090097  0.10057139  0.08684327  0.14214972]


	9 0.14215
	2 0.119009
	0 0.113835
	1 0.10479
	7 0.100571
	4 0.0996913
	8 0.0868433
	6 0.080901
	5 0.0781706
	3 0.0740385
	...
	...
	

[20 cleanup] root@b0530bb93ecf:/home/deeplearning/cnn# rm ./my_cnn_model.ckpt.*
[21 cleanup] root@b0530bb93ecf:/home/deeplearning/cnn# rm ./checkpoint
[22 cleanup] root@b0530bb93ecf:/home/deeplearning/cnn# cd ..
[23 cleanup] root@b0530bb93ecf:/home/deeplearning# rm -rf ./tensorflow/

