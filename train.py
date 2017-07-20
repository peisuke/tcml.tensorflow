import tqdm
import numpy as np

import tensorflow as tf

def make_input(batch_y, nb_batch, nb_class):    
    batch_p = (np.arange(nb_class) == batch_y[:,:-1,None]).astype(int)
    dummy = np.zeros((nb_batch, 1, nb_class), dtype=np.float32)
    return np.concatenate((dummy, batch_p), axis=1)

def get_minibatch(data, perm, nb_episode):
    nb_batch = len(perm)
    dshape = data.shape[2:]
    
    batch_x = np.zeros((nb_batch, nb_episode) + dshape + (1,))
    batch_y = np.zeros((nb_batch, nb_episode), dtype=np.int)
    batch_mask = np.ones((nb_batch, nb_episode), dtype=np.bool)
    
    for i in range(nb_batch):
        sample = np.random.randint(0, nb_class, nb_episode)
        batch_y[i] = sample
        
        _, first = np.unique(sample, return_index=True)
        mask = np.ones(nb_episode, np.bool)
        mask[first] = False
        batch_mask[i] = mask
        
        for j in range(nb_class):
            idx = (sample == j)
            eidx = np.random.choice(data.shape[1], np.sum(sample == j), False)
            imgs = data[perm[i, j], eidx]
            
            batch_x[i, idx, :, :, 0] = np.rot90(imgs, np.random.randint(4), axes=(1,2))

    # generate previous label
    batch_p = make_input(batch_y, nb_batch, nb_class)
    
    return batch_x, batch_p, batch_y, batch_mask

def embd_net(inp, scope, reuse=False, stop_grad=False):
    nb_episode = int(inp.shape[1])
    
    with tf.variable_scope(scope) as varscope:
        if reuse: 
            varscope.reuse_variables()

        _inp = tf.reshape(inp, [-1, 28, 28, 1])
        cur_input = _inp
        cur_filters = 1
        
        for i in range(4):
            with tf.variable_scope('conv'+str(i)):
                W = tf.get_variable('W', [3, 3, cur_filters, 64])
                beta = tf.get_variable('beta', [64], initializer=tf.constant_initializer(0.0))
                gamma = tf.get_variable('gamma', [64], initializer=tf.constant_initializer(1.0))

                cur_filters = 64
                pre_norm = tf.nn.conv2d(cur_input, W, strides=[1,1,1,1], padding='SAME')
                mean, variance = tf.nn.moments(pre_norm, [0, 1, 2])
                post_norm = tf.nn.batch_normalization(pre_norm, mean, variance, beta, gamma, variance_epsilon = 1e-10)
                conv = tf.nn.relu(post_norm)
                cur_input = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding = 'VALID')

        if stop_grad:
            squeezed = tf.squeeze(cur_input, [1,2])
            output = tf.stop_gradient(tf.reshape(squeezed, [-1, nb_episode, 64]))
        else:
            squeezed = tf.squeeze(cur_input, [1,2])
            output = tf.reshape(squeezed, [-1, nb_episode, 64])
            
    return output

def calual_conv_with_activation(inp, nb_input, nb_output, dilation_rate):
        Wf = tf.get_variable('W_filter', [2, nb_input, nb_output])
        bf = tf.get_variable('b_filter', [nb_output])    
        Wg = tf.get_variable('W_gate', [2, nb_input, nb_output])
        bg = tf.get_variable('b_gate', [nb_output])            
        
        x = tf.pad(inp, [[0, 0], [dilation_rate, 0], [0, 0]])
        
        xf = tf.nn.convolution(x, Wf, strides=[1,], dilation_rate=[dilation_rate,], padding='VALID')
        xf = tf.nn.bias_add(xf, bf)
        
        xg = tf.nn.convolution(x, Wg, strides=[1,], dilation_rate=[dilation_rate,], padding='VALID')
        xg = tf.nn.bias_add(xg, bg)
        
        out = tf.tanh(xf) * tf.sigmoid(xg)
        
        return out
    
def res_block(inp, nb_dim, dilation_rate, scope):
    with tf.variable_scope(scope):
        x = calual_conv_with_activation(inp, nb_dim, nb_dim, dilation_rate)
        x = x + inp
    return x

def dense_block(inp, nb_dim, dilation_rate, scope):
    with tf.variable_scope(scope):
        x = calual_conv_with_activation(inp, nb_dim, 128, dilation_rate)
        x = res_block(x, 128, dilation_rate, 'res_01')
        x = res_block(x, 128, dilation_rate, 'res_02')
        
        x = tf.concat((inp, x), axis=2)
        
        return x
    
def build_tcml(inp, label, nb_class, scope, reuse=False, stop_grad=False):
    with tf.variable_scope(scope):
        with tf.variable_scope('preprocess'):
            x = tf.concat((inp, label), axis=2)

        nb_channel = int(x.shape[2])
        x = dense_block(x, nb_channel+0*128, 1, 'dense_01')
        x = dense_block(x, nb_channel+1*128, 2, 'dense_02')
        x = dense_block(x, nb_channel+2*128, 4, 'dense_03')
        x = dense_block(x, nb_channel+3*128, 8, 'dense_04')
        x = dense_block(x, nb_channel+4*128, 16, 'dense_05')
        x = dense_block(x, nb_channel+5*128, 1, 'dense_06',)
        x = dense_block(x, nb_channel+6*128, 2, 'dense_07',)
        x = dense_block(x, nb_channel+7*128, 4, 'dense_08')
        x = dense_block(x, nb_channel+8*128, 8, 'dense_09')
        x = dense_block(x, nb_channel+9*128, 16, 'dense_10')
        
        with tf.variable_scope('postprocess'):
            W1 = tf.get_variable('W1', [1, nb_channel+10*128, 512])
            b1 = tf.get_variable('b1', [512])
            W2 = tf.get_variable('W2', [1, 512, nb_class])
            b2 = tf.get_variable('b2', [nb_class])

            x = tf.nn.conv1d(x, W1, stride=1, padding='SAME')
            x = tf.nn.bias_add(x, b1)
            x = tf.nn.relu(x)
            
            x = tf.nn.conv1d(x, W2, stride=1, padding='SAME')            
            x = tf.nn.bias_add(x, b2)

    return x

def build(img, prev_label, nb_class):
    feature = embd_net(img, 'embd')
    tcml = build_tcml(feature, prev_label, nb_class, 'TCML')
    
    return tcml

data = np.load('data.npy')
data = np.reshape(data,[-1,20,28,28])
train_data = data[:1200,:,:,:]
test_data = data[1200:,:,:,:]

nb_episode = 32
nb_class = 5

img = tf.placeholder(tf.float32, shape=[None, nb_episode, 28, 28, 1])
prev_label = tf.placeholder(tf.float32, shape=[None, nb_episode, 5])
train_label = tf.placeholder(tf.int32, shape=[None, nb_episode])
valid_label = tf.placeholder(tf.bool, shape=[None, nb_episode])

net = build(img, prev_label, nb_class)

with tf.variable_scope('loss'):
    y = tf.boolean_mask(net, valid_label)
    t = tf.to_int32(tf.boolean_mask(train_label, valid_label))

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t, logits=y)
    loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    
    train_step = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()
   
with tf.Session() as sess:
    sess.run(init)
    
    summary_writer = tf.summary.FileWriter('./log', graph=sess.graph)

    nb_class_total = train_data.shape[0]
    prev_perm = None
    nb_epoch = 1000
    nb_batch = 8
    display_step = 1

    for epoch in range(nb_epoch):
        # random chice
        perm = np.random.permutation(nb_class_total)

        # concat previous rest classes and current classes
        if prev_perm is not None and len(prev_perm) > 0:
            perm = np.concatenate((prev_perm, perm))

        # the current number of classes
        nb_iter = len(perm) // (nb_batch * nb_class)
        nb_total = nb_iter * nb_batch * nb_class

        curr_perm = perm[:nb_total]

        # rest classes
        prev_perm = perm[nb_total:]

        avg_cost = 0.

        for i in range(0, nb_iter):
            idx = i * nb_batch * nb_class
            _perm = perm[idx:idx+nb_batch * nb_class]
            _perm = _perm.reshape(nb_batch, nb_class)

            batch_x, batch_p, batch_y, batch_m = get_minibatch(train_data, _perm, nb_episode)

            feed_dict = {
                img: batch_x,
                prev_label: batch_p,
                train_label: batch_y,
                valid_label: batch_m
            }

            _, ret = sess.run([train_step, loss], feed_dict=feed_dict)

            avg_cost += ret / nb_iter

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))