import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import imageio
import scipy.misc as imageio
import scipy.stats
from scipy.signal import savgol_filter
from scipy.signal import fftconvolve
import os
import sys
import pickle
from functools import partial
import functools
import time
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
import tarfile
import json

from ops import *

def main(a):
    #Create / enter working directory
    if not os.path.isdir(a.output_folder): os.mkdir(a.output_folder)
    else: raise ValueError('Invalid directory')

    #Save configuration settings to output folder
    with open(a.output_folder + '/args.json', 'w') as _: json.dump(vars(a), _)

    #Load data / prepare batch suppliers
    if a.dataset == 'cifar':
        N_SAMPLES = 50000
        IMAGE_SHAPE = [32,32,3]
        def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
            with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
                batch = pickle.load(file)
            features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
            labels = batch['labels']
            return features
        cifar10_images = np.vstack( [load_cfar10_batch('data/cifar-10', i) for i in xrange(1,6)] )
        def get_batch(i):
            batch_index = i % ((cifar10_images.shape[0] / a.batch_size) - 1)
            if batch_index == 0: np.random.shuffle(cifar10_images)
            batch = cifar10_images[batch_index*a.batch_size:(batch_index+1)*a.batch_size,:,:,:]
            batch = batch / 128.0 - 1
            return batch
    elif a.dataset == 'mnist':
        N_SAMPLES = 60000
        IMAGE_SHAPE = [32,32,1]
        from tensorflow.examples.tutorials.mnist import input_data
        (mnist_images, _),(_,_) = tf.keras.datasets.mnist.load_data()
        mnist_images = np.reshape(mnist_images, [60000, 28, 28, 1])
        mnist_images = np.pad(mnist_images, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        mnist_images = mnist_images.astype(np.float32)
        mnist_images = (2.0 * mnist_images / 255.0) - 1
        def get_batch(i):
            batch_index = i % ((N_SAMPLES / a.batch_size) - 1)
            if batch_index == 0: np.random.shuffle(mnist_images)
            batch = mnist_images[batch_index*a.batch_size:(batch_index+1)*a.batch_size,:,:,:]
            return batch
    else: raise ValueError('Invalid dataset')
    
    IMAGE_H,IMAGE_W,IMAGE_C= IMAGE_SHAPE

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=a.gpus
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    DEVICES = ['/gpu:{}'.format(i) for i in xrange(len(a.gpus.split(',')))] #GPU-naming

    #DEFINE TENSORFLOW GRAPH#

    ##NETWORKS##
    _choice = [tf.nn.relu, None],[tf.nn.selu, tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')]
    d_act, d_ki = _choice[a.d_selu]
    g_act, g_ki = _choice[a.g_selu]

    def ConvDisc(x):
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):
            x = tf.layers.conv2d(x, 1*a.m_dim, 3, 2, padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 2*a.m_dim, 3, 2, padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 4*a.m_dim, 3, 2, padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 8*a.m_dim, 3, 2, padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 1, 2, 2, padding='valid')
            x = tf.reshape(x, [-1, 1])
            return x
    def ConvGen(z):
        with tf.variable_scope('Gen', reuse=tf.AUTO_REUSE):
            x = tf.layers.dense(z, 4*4*4*a.m_dim, activation=tf.nn.relu)
            x = tf.reshape(x, [-1,4,4,4*a.m_dim])
            x = tf.layers.conv2d_transpose(x, 8*a.m_dim, 3, (2,2), padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d_transpose(x, 4*a.m_dim, 3, (2,2), padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d_transpose(x, 2*a.m_dim, 3, (2,2), padding='same', activation=tf.nn.relu)
            x = tf.layers.conv2d_transpose(x, IMAGE_C, 1, (1,1), activation = tf.nn.tanh)
            return x

    def DenseDisc(x):
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):
            x = tf.reshape(x, [a.batch_size, np.prod(IMAGE_SHAPE)])
            for _ in xrange(a.d_layers-1):
                x = tf.layers.dense(x, a.m_dim, activation=d_act, kernel_initializer=d_ki)
            x = tf.layers.dense(x, 1)
            return x
    def DenseGen(z):
        with tf.variable_scope('Gen', reuse=tf.AUTO_REUSE):
            for _ in xrange(a.g_layers-1):
                z = tf.layers.dense(z, a.m_dim, activation=g_act, kernel_initializer=g_ki)
            z = tf.layers.dense(z, IMAGE_H * IMAGE_W * IMAGE_C, activation = tf.nn.tanh)
            return tf.reshape(z, [-1, IMAGE_H, IMAGE_W, IMAGE_C])

    #GENERALIZED NETWORKS WITH SUPPORT FOR SPECTRAL NORMALIZATION
    def SNDenseDisc(x):
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):
            x = tf.reshape(x, [a.batch_size, np.prod(IMAGE_SHAPE)])
            for _ in xrange(a.d_layers-1):
                x = d_act(fully_connected(x, a.m_dim, sn=a.d_sn, kernel_initializer=d_ki, scope='fc_'+str(_)))
            x = fully_connected(x, 1, sn=False, scope='fc_final')
            return x
    def SNDenseGen(z):
        with tf.variable_scope('Gen', reuse=tf.AUTO_REUSE):
            for _ in xrange(a.g_layers-1):
                z = g_act(fully_connected(z, a.m_dim, sn=a.g_sn, kernel_initializer=g_ki, scope='fc_'+str(_)))
            z = tf.nn.tanh(fully_connected(z, IMAGE_H * IMAGE_W * IMAGE_C, sn=False, scope='fc_final'))
            return tf.reshape(z, [-1, IMAGE_H, IMAGE_W, IMAGE_C])
    def SNConvDisc(x):
        with tf.variable_scope('Disc', reuse=tf.AUTO_REUSE):
            for i in xrange((a.d_layers-1)/4): x = tf.nn.relu(conv(x, 1*a.m_dim, kernel=3, stride=1, sn=a.d_sn, scope='down_conv_1'+'_'+str(i)))
            x = tf.nn.relu(conv(x, 1*a.m_dim, kernel=3, stride=2, sn=a.d_sn, scope='down_conv_1'))
            for i in xrange((a.d_layers-2)/4): x = tf.nn.relu(conv(x, 2*a.m_dim, kernel=3, stride=1, sn=a.d_sn, scope='down_conv_2'+'_'+str(i)))
            x = tf.nn.relu(conv(x, 2*a.m_dim, kernel=3, stride=2, sn=a.d_sn, scope='down_conv_2'))
            for i in xrange((a.d_layers-3)/4): x = tf.nn.relu(conv(x, 4*a.m_dim, kernel=3, stride=1, sn=a.d_sn, scope='down_conv_3'+'_'+str(i)))
            x = tf.nn.relu(conv(x, 4*a.m_dim, kernel=3, stride=2, sn=a.d_sn, scope='down_conv_3'))
            for i in xrange((a.d_layers-4)/4): x = tf.nn.relu(conv(x, 8*a.m_dim, kernel=3, stride=1, sn=a.d_sn, scope='down_conv_4'+'_'+str(i)))
            x = tf.nn.relu(conv(x, 8*a.m_dim, kernel=3, stride=2, sn=a.d_sn, scope='down_conv_4'))
            x =           (conv(x, 1        , kernel=2, stride=2, sn=False, scope='down_conv_5'))
            return x
    def SNConvGen(z):
        with tf.variable_scope('Gen', reuse=tf.AUTO_REUSE):
            x = tf.nn.relu(fully_connected(z, 4*4*4*a.m_dim))
            x = tf.reshape(x, [-1, 4, 4, 4*a.m_dim])
            for i in xrange((a.g_layers-1)/4): x = tf.nn.relu(deconv(x, 8*a.m_dim, kernel=3, stride=1, sn=a.g_sn, scope='up_conv_1'+'_' + str(i)))
            x = tf.nn.relu(deconv(x, 8*a.m_dim, kernel=3, stride=2, sn=a.g_sn, scope='up_conv_1'))
            for i in xrange((a.g_layers-2)/4): x = tf.nn.relu(deconv(x, 4*a.m_dim, kernel=3, stride=1, sn=a.g_sn, scope='up_conv_2'+'_' + str(i)))
            x = tf.nn.relu(deconv(x, 4*a.m_dim, kernel=3, stride=2, sn=a.g_sn, scope='up_conv_2'))
            for i in xrange((a.g_layers-3)/4): x = tf.nn.relu(deconv(x, 2*a.m_dim, kernel=3, stride=1, sn=a.g_sn, scope='up_conv_3'+'_' + str(i)))
            x = tf.nn.relu(deconv(x, 2*a.m_dim, kernel=3, stride=2, sn=a.g_sn, scope='up_conv_3'))
            for i in xrange((a.g_layers-4)/4): x = tf.nn.relu(deconv(x, 1*a.m_dim, kernel=3, stride=1, sn=a.g_sn, scope='ch_conv'+'_' + str(i)))
            x = tf.nn.tanh(deconv(x, IMAGE_C , kernel=1, stride=1, sn=False, scope='channel_conv'))
            return x
    ##END NETWORKS##

    dnet_dict = {'dense':DenseDisc,'conv':ConvDisc,'sndense':SNDenseDisc,'snconv':SNConvDisc}
    gnet_dict = {'dense':DenseGen,'conv':ConvGen,'sndense':SNDenseGen,'snconv':SNConvGen}
    D,G = dnet_dict[a.d_net], gnet_dict[a.g_net]

    z_feed = tf.placeholder(tf.float32, [a.batch_size, a.z_dim])
    x_feed = tf.placeholder(tf.float32, [a.batch_size, IMAGE_H, IMAGE_W, IMAGE_C])

    #GPU parallelization
    x_gen, d_feed, d_gen = [],[],[]
    for device_index, (device, x_feed_, z_feed_) in enumerate(zip(DEVICES, tf.split(x_feed, len(DEVICES)), tf.split(z_feed, len(DEVICES)))):
        with tf.device(device), tf.name_scope('device_index'):
            x_gen_ = G(z_feed_)
            x_gen.append(x_gen_)
            d_feed.append(D(x_feed_))
            d_gen.append(D(x_gen_))
    x_gen, d_feed, d_gen = tf.concat(x_gen, axis=0),tf.concat(d_feed,axis=0),tf.concat(d_gen,axis=0)
    sig_d_feed, sig_d_gen = tf.sigmoid(d_feed), tf.sigmoid(d_gen)
    
    def batch_diversity(x0,x1):
        meansquare_distance = tf.reduce_mean(tf.square(x0 - x1[::-1]), axis=[1,2,3])
        return tf.reduce_mean(tf.sqrt(meansquare_distance))
    x_gen_div = batch_diversity(x_gen,x_gen)
    x_feed_div = batch_diversity(x_feed,x_feed)
    x_mix_div = batch_diversity(x_gen,x_feed)

    R = (1.0 + 1e-18 - tf.reduce_mean(sig_d_gen)) / (tf.reduce_mean(sig_d_gen) + 1e-18)

    current_training_iteration = tf.Variable(0.0, trainable=False)
    inc_op = current_training_iteration.assign_add(1.0)
    
    ##COST FUNCTIONS##
    msce_real_1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_feed), logits=d_feed)
    msce_real_0 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_feed), logits=d_feed)
    msce_gen_0 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_gen), logits=d_gen)
    msce_gen_1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_gen), logits=d_gen)
    
    def d_cost_ns(df,dg): return msce_real_1 + msce_gen_0 
    def d_cost_ns_simplegp(df,dg, r1_gamma=0.1):
        loss = msce_real_1 + msce_gen_0
        real_loss = tf.reduce_sum(df)
        real_grads = tf.gradients(real_loss, [x_feed])[0]
        r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3])
        loss += r1_penalty * (r1_gamma * 0.5)
        return loss
    def d_cost_ns_softplus(df,dg): return tf.nn.softplus(dg) + tf.nn.softplus(-df)

    d_cost_dict = {'ns':d_cost_ns,'mm':d_cost_ns,'ns_softplus':d_cost_ns_softplus}
    disc_loss = d_cost_dict[a.d_cost](d_feed,d_gen)

    def g_cost_mm(d): return -msce_gen_0
    def g_cost_ns(d): return msce_gen_1
    def g_cost_mmstab(d): return -(tf.maximum(d, 0) + tf.log(1 + tf.exp(-tf.abs(d))))
    def g_cost_nsstab(d): return tf.maximum(d, 0) - d + tf.log(1 + tf.exp(-tf.abs(d)))
    def g_cost_mm_softplus(d): return -tf.nn.softplus(d_gen)
    def g_cost_ns_softplus(d): return tf.nn.softplus(-d_gen)
    def g_cost_js(d):
        R = (1.0 + 1e-18 - tf.reduce_mean(sig_d_gen)) / (tf.reduce_mean(sig_d_gen) + 1e-18)
        return a.g_cost_parameter * g_cost_mm(d) *tf.stop_gradient(R) + (1.0 - a.g_cost_parameter) * g_cost_ns(d)
    def g_cost_js_simple(d):
        return a.g_cost_parameter * g_cost_mm(d) + (1.0 - a.g_cost_parameter) * g_cost_ns(d)
    
    g_cost_dict = {'mm':g_cost_mm,'ns':g_cost_ns,'mms':g_cost_mm_softplus,'nss':g_cost_ns_softplus,'js':g_cost_js,'js_simple':g_cost_js_simple}
    gen_loss = g_cost_dict[a.g_cost](d_gen)
    
    gen_vars, disc_vars = tf.trainable_variables(scope='Gen'), tf.trainable_variables(scope='Disc')
    gen_grad, disc_grad = tf.gradients(gen_loss, gen_vars, colocate_gradients_with_ops=True), tf.gradients(disc_loss, disc_vars, colocate_gradients_with_ops=True)
    gen_grad_norm, disc_grad_norm = tf.global_norm(gen_grad), tf.global_norm(disc_grad)
                                                                           
    ##RENORMING OF GRADIENT##
    norm_epsilon = 1e-18
    def norm_frac(): return (1 + norm_epsilon - tf.reduce_mean(sig_d_gen)) / (tf.reduce_mean(sig_d_gen) + norm_epsilon)
    def norm_nsat(): return tf.global_norm(tf.gradients(msce_gen_1, gen_vars)) / (gen_grad_norm + norm_epsilon)
    def norm_unit(): return 1. / (gen_grad_norm + norm_epsilon) * np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables('Gen')])

    norm_dict = {'frac':norm_frac,'nsat':norm_nsat,'unit':norm_unit}
    if a.g_renorm != 'none': renorm = norm_dict[a.g_renorm]()
    else: renorm = tf.constant(1.0)
    gen_grad = map(lambda x: x * renorm, gen_grad)

    ##CHOICE OF OPTIMIZER##
    if a.opt == 'adam':
        d_opt = tf.train.AdamOptimizer(learning_rate=a.d_lr, beta1=a.d_beta1, beta2=a.d_beta2, epsilon=a.d_adameps)
        g_opt = tf.train.AdamOptimizer(learning_rate=a.g_lr, beta1=a.g_beta1, beta2=a.g_beta2, epsilon=a.g_adameps)
    elif a.opt == 'sgd':
        d_opt = tf.train.GradientDescentOptimizer(learning_rate=a.d_lr)
        g_opt = tf.train.GradientDescentOptimizer(learning_rate=a.g_lr)
    else: raise ValueError('Invalid optimizer')

    disc_train_op = d_opt.apply_gradients(zip(disc_grad, disc_vars))
    gen_train_op = g_opt.apply_gradients(zip(gen_grad, gen_vars))
    gen_opt_mom_reset_op = tf.group( [tf.assign(v, tf.zeros_like(v)) for v in g_opt.variables() if 'Gen' in v.name] )
    #gen_opt_beta_reset_op = tf.group([tf.assign(v, tf.zeros_like(v) for v in g_opt.variables() if 'Gen' in v.name])
    #for v in g_opt.variables(): print v

    ##FRECHET INCEPTION DISTANCE##
    if a.metrics:
	    def get_graph_def_custom(filename='classify_image_graph_def.pb',tar_filename='data/inception-2015-12-05.tgz'):
		with tarfile.open(tar_filename, 'r:gz') as tar:
		    proto_str = tar.extractfile(filename).read()
		return graph_pb2.GraphDef.FromString(proto_str)

	    inception_images = tf.placeholder(tf.float32, [None, 3, None, None])
	    activations1 = tf.placeholder(tf.float32, [None, None], name = 'activations1')
	    activations2 = tf.placeholder(tf.float32, [None, None], name = 'activations2')
	    fcd = tf.contrib.gan.eval.frechet_classifier_distance_from_activations(activations1, activations2)

	    def inception_activations(images=inception_images, num_splits = 1):
		images = tf.transpose(images, [0, 2, 3, 1])
		size = 299
		images = tf.image.resize_bilinear(images, [size, size])
		generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
		activations = functional_ops.map_fn(
		    fn = functools.partial(tf.contrib.gan.eval.run_inception,
		        default_graph_def_fn=get_graph_def_custom,
		        output_tensor = 'pool_3:0'),
		    elems = array_ops.stack(generated_images_list),
		    parallel_iterations = 1,
		    back_prop = False,
		    swap_memory = True,
		    name = 'RunClassifier')
                activations = array_ops.concat(array_ops.unstack(activations), 0)
		return activations
	    activations=inception_activations()

	    def get_inception_activations(inps):
		n_batches = inps.shape[0]//a.batch_size
		act = np.zeros([n_batches * a.batch_size, 2048], dtype = np.float32)
		for i in range(n_batches):
		    inp = inps[i * a.batch_size:(i + 1) * a.batch_size] / 255. * 2 - 1
		    act[i * a.batch_size:(i + 1) * a.batch_size] = activations.eval(feed_dict = {inception_images: inp})
		return act

	    def activations2distance(act1, act2):
		 return fcd.eval(feed_dict = {activations1: act1, activations2: act2})
		    
	    def get_fid(images1, images2):
		assert(type(images1) == np.ndarray)
		assert(len(images1.shape) == 4)
		assert(images1.shape[1] == 3)
		assert(np.min(images1[0]) >= 0 and np.max(images1[0]) > 10), 'Image values should be in the range [0, 255]'
		assert(type(images2) == np.ndarray)
		assert(len(images2.shape) == 4)
		assert(images2.shape[1] == 3)
		#assert(np.min(images2[0]) >= 0 and np.max(images2[0]) > 10), 'Image values should be in the range [0, 255]'
		#Generated images sometimes all black for terrible generators
		assert(images1.shape == images2.shape), 'The two numpy arrays must have the same shape'
		start_time = time.time()
		act1 = get_inception_activations(images1)
		act2 = get_inception_activations(images2)
		fid = activations2distance(act1, act2)
		print('FID calculation time: %f s' % (time.time() - start_time))
		return fid

	    def cast_inception(im):
		#IF SINGLE CHANNEL MNIST IMAGE; TRIPLICATE CHANNEL TO FAKE RGB
		im = np.array(im, copy=True)
		im = ((im+1)*(256/2)).astype(np.uint8)
		if a.dataset == 'mnist': im = np.repeat(im,3,axis=3)
		elif a.dataset == 'cifar': pass
		elif a.dataset == 'mnist1k': pass
		elif a.dataset == 'cats': pass
		elif a.dataset == 'dock': pass
		else: raise ValueError('FID not implemented for dataset')
		im = np.transpose(im, [0,3,1,2])
		return im
    ##END FRECHET DISTANCE##

    ##END DEFINE TENSORFLOW GRAPH##

    #HELPER FUNCTIONS
    def convolve_filter(size):
        f = np.concatenate( (np.arange(size+1), np.zeros(size)) )
        f /= np.sum(f)
        return f
    def conv_smooth(a,f): return fftconvolve(np.pad(a, (0,len(f)-1), 'edge'), f, 'valid')
    def confusion(real,fake): return np.mean( (np.sort(real) - np.sort(fake)[::-1]) <= 0 )
    def epoch_from_iter(i): return (i * a.batch_size * 1.0) / N_SAMPLES
    def smooth(arr):
        i = ITERATIONS / 1000
        i = max(5, i + (1*i%2==0))
        return savgol_filter(arr, i, 3)
    def mosaic_batch(images, w, d):
        mosaic = np.zeros([w*d, w*d, 3])
        for u in xrange(d):
            for v in xrange(d):
                mosaic[w*u:w*(u+1), w*v:w*(v+1), :] = images[d*u+v,:,:,:]
        return mosaic

    #CLASS DISTRIBUTION DIVERGENCE
    def initialize_class_distribution():
        if a.dataset == 'mnist' or a.dataset == 'cifar': class_counts_gen = np.zeros(10)
        elif a.dataset == 'mnist1k': class_counts_gen = np.zeros(1000)
        else: raise ValueError('Classifier not implemented for dataset')
        return class_counts_gen

    def update_class_distribution(x_gen, class_counts, classifier):
        x_gen_onehots = classifier.predict(x_gen, batch_size=x_gen.shape[0])
        x_gen_classes = np.argmax(x_gen_onehots, axis=1)
        for cls in x_gen_classes: class_counts[cls] += 1
        return class_counts

    def print_class_distribution(class_freqs, js):
        print 'Fake classes: %.4f' % js
        print ''

    #Jensen-Shannon distance between class distributions of fake and real data (bounded [0,1])
    #Real data defaults to even class balance
    #Somewhat awkward implementation to give correct beavior for an exact 0 class frequency
    def get_jensen_shannon(f, r=None):
        if a.dataset == 'mnist' or a.dataset == 'cifar':
            if r == None: r = np.zeros(10)+0.1
            m = 0.5*(f+r)
            s = np.sum([fi*np.log(fi/mi) for fi,mi in zip(f,m) if fi>0]) + np.sum([ri*np.log(ri/mi) for ri,mi in zip(r,m) if ri>0])
            return 0.5 * s / np.log(2)
        else: raise ValueError('Jensen-Shannon not implemented for dataset')


    z_fixed = np.random.normal(size=[a.batch_size,a.z_dim])
    ITERATIONS = (a.epochs * N_SAMPLES / a.batch_size) + 1
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        session.run(tf.global_variables_initializer())
        #session.run(tf.local_variables_initializer())

        ##CLASSIFIERS FOR CLASS DISTRIBUTIONS##
        if a.metrics:
            if a.dataset == 'mnist' or a.dataset == 'mnist1k': clf = tf.keras.models.load_model('data/keras_mnist_classifier.h5')
            elif a.dataset == 'cifar': clf = tf.keras.models.load_model('data/keras_resnet_cifar10_classifier.h5')
            else: raise ValueError('Classifier not implemented for dataset')

        gen_vars_num, disc_vars_num = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables('Gen')]), np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables('Disc')])
        #print 'Generator:     ', gen_vars_num
        #print 'Discriminator: ', disc_vars_num

        #LOGGING FOR PLOTTING
        log_confusion = np.zeros(ITERATIONS)
        log_d_real = np.zeros(ITERATIONS)
        log_d_gen = np.zeros(ITERATIONS)
        log_xvals = np.arange(ITERATIONS)
        log_lr_ratio = np.zeros(ITERATIONS)
        log_renorm = np.zeros(ITERATIONS)
        log_gen_grad_norm = np.zeros(ITERATIONS)
        log_disc_grad_norm = np.zeros(ITERATIONS)
        log_d_names = ['dmax','gmax','dmin','gmin','dmean','gmean']
        log_d = [ [] for _ in xrange(len(log_d_names))]
        log_d_colors = ['darkred', 'darkblue', 'red', 'blue', 'orangered', 'blueviolet']
        log_xg_div = np.zeros(ITERATIONS)
        log_xr_div = np.zeros(ITERATIONS)
        log_xm_div = np.zeros(ITERATIONS)
        c_size = np.maximum(a.epochs*N_SAMPLES/a.batch_size,2500)
        c_filter = np.ones(ITERATIONS/c_size)*((0.0+c_size)/ITERATIONS)

        fid_batches = (np.ceil(float(a.fid_n)/a.batch_size)).astype(int)
        fid_images_real = np.zeros(shape=[fid_batches*a.batch_size, 3, IMAGE_W, IMAGE_H], dtype=np.uint8)
        fid_images_gen  = np.zeros(shape=[fid_batches*a.batch_size, 3, IMAGE_W, IMAGE_H], dtype=np.uint8)

        for i in xrange(ITERATIONS):
            #IF ENABLED, RESET G ADAM OPTIMIZER AT INTERVALS
            if a.g_adamreset != 0:
                if (i % a.g_adamreset) == 0: _ = session.run(gen_opt_mom_reset_op)

            #EVALUATION - AVOID EXCESSIVE OVERHEAD FOR EVERY TRAINING ITERATION
            if ((i % (ITERATIONS/a.eval_n)) == 0 or (i == (ITERATIONS-1))):
                _d_real, _d_gen, _x_gen_div, _x_real_div = session.run(
                    (sig_d_feed, sig_d_gen, x_gen_div, x_feed_div),
                    feed_dict={x_feed : get_batch(i), z_feed : np.random.normal(size=[a.batch_size,a.z_dim])})
                _x_gen = session.run( x_gen, feed_dict={z_feed : z_fixed} )
        
                print epoch_from_iter(i), a.g_cost, a.g_renorm, a.g_cost_parameter, confusion(_d_real, _d_gen), _x_gen_div / _x_real_div
                w = IMAGE_SHAPE[0]
                d = int(np.sqrt(a.batch_size))
                imageio.imsave(a.output_folder + '/gen_{}.png'.format(i), mosaic_batch(_x_gen, w, d))
                
                #FID and Class frequencies
                if (a.metrics and not a.eval_skip):
                    class_counts_gen = initialize_class_distribution()
                    for j in xrange(fid_batches):
                        _x_real = get_batch(j)
                        fid_images_real[j*a.batch_size:(j+1)*a.batch_size,:,:,:] = cast_inception(_x_real)
                        _x_gen = session.run(x_gen,feed_dict={z_feed:np.random.normal(size=[a.batch_size,a.z_dim])})
                        class_counts_gen = update_class_distribution(_x_gen, class_counts_gen, clf)
                        fid_images_gen[j*a.batch_size:(j+1)*a.batch_size,:,:,:] = cast_inception(_x_gen)
                    class_frequencies_gen = class_counts_gen*(1.0/np.sum(class_counts_gen))
                    jensen_shannon_gen = get_jensen_shannon(class_frequencies_gen)
                    print_class_distribution(class_frequencies_gen, jensen_shannon_gen)
                    fid_value = get_fid(fid_images_real[:a.fid_n,:,:,:], fid_images_gen[:a.fid_n,:,:,:])                    
                    print 'FID value: ', fid_value, '\n'
                    with open(a.output_folder + '/fid_'+str(int(epoch_from_iter(i)))+'.txt','w') as f: f.write(str(fid_value))
                    with open(a.output_folder + '/cls_'+str(int(epoch_from_iter(i)))+'.txt','w') as f: f.write(str(class_frequencies_gen) + '\n' + str(jensen_shannon_gen))

                #PLOTTING
                if True:
                    fig, ax_log = plt.subplots()
                    ax_log.set_yscale('log')
                    ax_lin = ax_log.twinx()
                    for arr,lab,ax,c in zip(
                        [log_confusion, log_d_real - log_d_gen, log_lr_ratio, log_gen_grad_norm/gen_vars_num, log_disc_grad_norm/disc_vars_num, log_xg_div/(log_xr_div+1e-12)],
                        ['$D(x)$:$D(G(z))$ overlap', '$D(x)$:$D(G(z))$ distance', '$R$ scaling factor', '$|\\nabla J_{G}|$ / #$\\theta$', '$|\\nabla J_{D}|$ / #$\phi$', '$G$ diversity'],
                        [ax_lin, ax_log, ax_log, ax_log, ax_log, ax_lin],
                        ['b','g','r','c','m','y']
                        ):
                        if ax==ax_lin:
                            ax.plot(epoch_from_iter(log_xvals), conv_smooth(np.clip(arr,0.0,1.0),c_filter), label=lab, color=c, alpha=0.5)
                        else:
                            ax.plot(epoch_from_iter(log_xvals), np.exp(conv_smooth(np.log(np.clip(arr,1e-9,1e6)),c_filter)), label=lab, color=c, alpha=0.5)
                    ax_log.set_ylim((1e-9*0.80,1e6*1.25))
                    ax_lin.set_ylim((-0.25,1.25))
                    ax_log.set_xlim((0,a.epochs))
                    ax_log.set_xlabel('Training epochs (batch size=' + str(a.batch_size) + ', #samples=' + str(N_SAMPLES) + ')')
                    ax_lin.legend(loc='upper right',fancybox=True, framealpha=0.5)
                    leg = ax_log.legend(loc='upper left',fancybox=True, framealpha=0.5)
                    leg.remove()
                    ax_lin.add_artist(leg)
                    plt.title('Training diagnostics (clipped and smoothed)')
                    plt.savefig(a.output_folder + '/all_pretty.jpg')
                    plt.close()

            #TRAINING
            _, _, _, _x_gen, _d_real,_d_gen, _sig_d_real, _sig_d_gen, _x_gen_div, _x_real_div, _x_mix_div, _renorm, _gen_grad_norm, _disc_grad_norm = session.run(
                (disc_train_op, gen_train_op, inc_op, x_gen, d_feed, d_gen, sig_d_feed, sig_d_gen, x_gen_div, x_feed_div, x_mix_div, renorm, gen_grad_norm, disc_grad_norm),
                feed_dict={x_feed : get_batch(i), z_feed : np.random.normal(size=[a.batch_size,a.z_dim])})

            #LOG FROM TRAINING
            log_confusion[i] = confusion(_d_real, _d_gen)
            log_renorm[i] = _renorm
            log_d_real[i] = np.mean(_d_real)
            log_d_gen[i] = np.mean(_d_gen)
            log_lr_ratio[i] = (1 + 1e-8 - np.mean(_sig_d_gen))/(np.mean(_sig_d_gen) + 1e-8)
            log_gen_grad_norm[i] = _gen_grad_norm
            log_disc_grad_norm[i] = _disc_grad_norm
            log_xg_div[i] = _x_gen_div
            log_xr_div[i] = _x_real_div
            log_xm_div[i] = _x_mix_div

        #FID AND CLASS FREQUENCIES AFTER TRAINING
        if a.metrics:
		print 'End evaluation', a.g_cost, a.g_renorm, a.g_cost_parameter, confusion(_d_real, _d_gen), _x_gen_div / _x_real_div
		class_counts_gen = initialize_class_distribution()
		for j in xrange(fid_batches):
		    fid_images_real[j*a.batch_size:(j+1)*a.batch_size,:,:,:] = cast_inception(get_batch(j))
		    _x_gen = session.run(x_gen,feed_dict={z_feed:np.random.normal(size=[a.batch_size,a.z_dim])})
		    class_counts_gen = update_class_distribution(_x_gen, class_counts_gen, clf)
		    fid_images_gen[j*a.batch_size:(j+1)*a.batch_size,:,:,:] = cast_inception(_x_gen)
		class_frequencies_gen = class_counts_gen*(1.0/np.sum(class_counts_gen))
		jensen_shannon_gen = get_jensen_shannon(class_frequencies_gen)
		print_class_distribution(class_frequencies_gen, jensen_shannon_gen)
		fid_value = get_fid(fid_images_real[:a.fid_n,:,:,:], fid_images_gen[:a.fid_n,:,:,:])                    
		print 'FID value: ', fid_value, '\n'
		with open(a.output_folder + '/cls.txt','w') as f: f.write(str(class_frequencies_gen) + '\n' + str(jensen_shannon_gen))
		with open(a.output_folder + '/fid.txt','w') as f: f.write(str(fid_value))

    tf.reset_default_graph()
    tf.keras.backend.clear_session()
