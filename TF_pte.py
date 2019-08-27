#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.5.3 

"""

import os
import sys
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import numpy as np
import tensorflow as tf
import utils
import network

logger = logging.getLogger("HNE")



class Edge_sampler(object):
    def __init__(self, net, batch_size, distortion_power = 0, neg_sampled = 5):
        self._net = net
        self._edges_type_dict = {}
        for s_type, t_type in net.edge_types:
            self._edges_type_dict[(s_type, t_type)] = []
            if not net.isdirected:
                self._edges_type_dict[(t_type, s_type)] = []
        for start_node, target_node in net.edges:
            start_type = net.get_node_type(start_node)
            target_type = net.get_node_type(target_node)
            self._edges_type_dict[(start_type, target_type)].append((start_node, target_node))
            if not net.isdirected:
                self._edges_type_dict[(target_type, start_type)].append((target_node, start_node))
        for k,v in self._edges_type_dict.items():
            if len(v) == 0:
                self._edges_type_dict.pop(k)

        self._edges_type_size = len(self._edges_type_dict)
        self._edges_size = 0
        for v in self._edges_type_dict.values():
            if len(v) > self._edges_size:
                self._edges_size = len(v)
        self._batch_size = batch_size
        self._distortion_power = distortion_power
        self._neg_sampled = neg_sampled

        # preprocess_nodesdegrees:
        self._nodes_type_dict = {}
        for node_type in net.node_types:
            self._nodes_type_dict[node_type] = [[], None] # [nodes, nodes_degrees]
        for node in net.nodes:
            node_type = net.get_node_type(node)
            self._nodes_type_dict[node_type][0].append(node)
            # self._nodes_type_dict[node_type][1].append(net.get_degrees(node))
        self.preprocess_nodesdegrees()

    @property
    def edges_type_size(self):
        return self._edges_type_size
    @property
    def edges_size(self):
        return self._edges_size

    def preprocess_nodesdegrees(self):
        time_start = time.time()
        logger.info("preprocessing nodesdegrees with distortion_power = {} ...".format(self._distortion_power))
        if self._distortion_power == 0:
            pass
        else:
            for node_type in self._nodes_type_dict.keys():
                nodes_degrees = self._net.get_degrees(self._nodes_type_dict[node_type][0])
                normed_degrees = np.power(nodes_degrees, self._distortion_power)
                normed_degrees = normed_degrees / np.sum(normed_degrees)
                if np.sum(normed_degrees) != 1.:
                    normed_degrees = normed_degrees / np.sum(normed_degrees)
                if np.sum(normed_degrees) != 1.:
                    normed_degrees = normed_degrees / np.sum(normed_degrees)
                alias_nodesdegrees = utils.alias_setup(normed_degrees)  # J, q
                self._nodes_type_dict[node_type][1] = alias_nodesdegrees
        logger.info('nodesdegrees processed in {}s'.format(time.time() - time_start))

    def next_batch(self):
        data_list = []
        labels_list = []
        neg_labels_list = []
        for edge_type, edge_list in self._edges_type_dict.items():
            edges = random.sample(edge_list, k=self._batch_size)
            data = []
            labels = []
            neg_labels = []
            for source, target in edges:
                target_type = edge_type[1]
                neg_nodes = utils.neg_sample(self._nodes_type_dict[target_type][0], {source, target},
                                             num=self._neg_sampled,
                                             alias_table=self._nodes_type_dict[target_type][1])
                data.append(source)
                labels.append(target)
                neg_labels.append(neg_nodes)
            data_list.append(data)
            labels_list.append(labels)
            neg_labels_list.append(neg_labels)
        return np.asarray(data_list), np.asarray(labels_list), np.asarray(neg_labels_list)


## model
class SGNS(object):
    """A customized version of Word2Vec(SkipGram-NegativeSampling) model."""

    def __init__(self, vocab_size, embedding_size=128, order = 1):
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._order = order

        init_width = 0.5 / self._embedding_size
        self._embedding = tf.Variable(
            tf.random_uniform([self._vocab_size, self._embedding_size], -init_width, init_width),
            name='embedding')
        if self._order == 2:
            self._nce_weight = tf.Variable(
                tf.truncated_normal([self._vocab_size, self._embedding_size],
                                    stddev=1.0 / np.sqrt(self._embedding_size)),
                name='nce_weight')
            self._nce_biases = tf.Variable(tf.zeros([self._vocab_size]),
                                           name='nce_biases')
    @property
    def vectors(self):
        return self._embedding
    @property
    def context_weights(self):
        return self._nce_weight
    @property
    def context_biases(self):
        return self._nce_biases

    def inference(self, batch_input, batch_labels, batch_neg_labels):
        """
        construct the model
        :param input: 1D tensor of idx in vocabulary
        :param batch_labels: 1D tensor if index, positive samples
        :return:
        """
        embed = tf.nn.embedding_lookup(self._embedding, batch_input)

        if self._order == 2:
            true_w = tf.nn.embedding_lookup(self._nce_weight, batch_labels)
            true_b = tf.nn.embedding_lookup(self._nce_biases, batch_labels)
            sampled_w = tf.nn.embedding_lookup(self._nce_weight, batch_neg_labels)
            sampled_b = tf.nn.embedding_lookup(self._nce_biases, batch_neg_labels)
            true_logits = tf.reduce_sum(tf.multiply(embed, true_w), 2) + true_b
            embed = tf.expand_dims(embed, -1)
            sampled_logits = tf.squeeze(tf.matmul(sampled_w, embed), -1) + sampled_b
        else:
            true_w = tf.nn.embedding_lookup(self._embedding, batch_labels)
            sampled_w = tf.nn.embedding_lookup(self._embedding, batch_neg_labels)
            true_logits = tf.reduce_sum(tf.multiply(embed, true_w), 2)
            embed = tf.expand_dims(embed, -1)
            sampled_logits = tf.squeeze(tf.matmul(sampled_w, embed), -1)
        return true_logits, sampled_logits

    def nce_loss(self, true_logits, sampled_logits):
        """Build the graph for the NCE loss."""

        # cross-entropy(logits, labels)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)
        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

        # NCE-loss is the sum of the true and noise (sampled words)
        # contributions, averaged over the batch.
        nce_loss_tensor = tf.reduce_sum(tf.reduce_mean(true_xent, axis=1) + tf.reduce_mean(tf.reduce_sum(sampled_xent, axis=2), axis=1))
        tf.summary.scalar("NCE loss", nce_loss_tensor)
        self._loss = nce_loss_tensor
        return nce_loss_tensor

    def optimize(self, loss, global_step, lr):
        """Build the graph to optimize the loss function."""
        tf.summary.scalar('learning_rate', lr)

        # Compute gradients
        # opt = tf.train.MomentumOptimizer(lr, option.MOMENTUM)
        # opt = tf.train.AdamOptimizer(lr)
        # opt = tf.train.RMSPropOptimizer(lr)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = optimizer.minimize(loss,
                                      global_step=global_step,
                                      gate_gradients=optimizer.GATE_NONE)
        return train_op

    def train(self, batch_input, batch_labels, batch_neg_labels, global_step, learning_rate):
        true_logits, sampled_logits = self.inference(batch_input, batch_labels, batch_neg_labels)
        loss = self.nce_loss(true_logits, sampled_logits)
        train_op = self.optimize(loss, global_step, learning_rate)
        return train_op, loss


## train
def train(net, vectors_path, lr_file,
          ckpt_dir, checkpoint, embedding_size, neg_sampled, order, distortion_power,
          iter_epochs, batch_size, initial_learning_rate, decay_epochs, decay_interval, decay_rate,
          allow_soft_placement, log_device_placement, gpu_memory_fraction, using_gpu, allow_growth,
          loss_interval, summary_steps, summary_interval, ckpt_epochs, ckpt_interval):
    edge_sampler = Edge_sampler(net, batch_size, distortion_power = distortion_power, neg_sampled = neg_sampled)
    edges_size = edge_sampler.edges_size
    edges_type_size = edge_sampler.edges_type_size
    nodes_size = net.get_nodes_size()
    types_size = net.get_node_types_size()
    num_steps_per_epoch = int(edges_size / batch_size)
    iter_steps = round(iter_epochs * num_steps_per_epoch)  # iter_epochs should be big enough to converge.
    decay_steps = round(decay_epochs * num_steps_per_epoch)
    ckpt_steps = round(ckpt_epochs * num_steps_per_epoch)

    LR = utils.LearningRateGenerator(initial_learning_rate=initial_learning_rate, initial_steps=0,
                                     decay_rate=decay_rate, decay_steps=decay_steps, iter_steps=iter_steps)

    with tf.Graph().as_default(), tf.device('/gpu:0' if using_gpu else '/cpu:0'):

        inputs = tf.placeholder(tf.int32, shape=[edges_type_size, batch_size], name='inputs')
        labels = tf.placeholder(tf.int32, shape=[edges_type_size, batch_size], name='labels')
        neg_labels = tf.placeholder(tf.int32, shape=[edges_type_size, batch_size, neg_sampled], name='neg_labels')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        model_list = []
        trains_list = []
        if order == "1":
            with tf.name_scope("1st_order"):
                model = SGNS(vocab_size=nodes_size, embedding_size=embedding_size, order=1)
                global_step = tf.Variable(0, trainable=False, name="global_step")
            train_op, loss = model.train(inputs, labels, neg_labels, global_step, learning_rate)
            model_list.append(model)
            trains_list.append((train_op, loss, global_step))
        elif order == "2":
            with tf.name_scope("2st_order"):
                model = SGNS(vocab_size=nodes_size, embedding_size=embedding_size, order=2)
                global_step = tf.Variable(0, trainable=False, name="global_step")
            train_op, loss = model.train(inputs, labels, neg_labels, global_step, learning_rate)
            model_list.append(model)
            trains_list.append((train_op, loss, global_step))
        elif order == "3":
            with tf.name_scope("1st_order"):
                model = SGNS(vocab_size=nodes_size, embedding_size=embedding_size//2, order=1)
                global_step = tf.Variable(0, trainable=False, name="global_step")
            train_op, loss = model.train(inputs, labels, neg_labels, global_step, learning_rate)
            model_list.append(model)
            trains_list.append((train_op, loss, global_step))
            with tf.name_scope("2st_order"):
                model = SGNS(vocab_size=nodes_size, embedding_size=embedding_size//2, order=2)
                global_step = tf.Variable(0, trainable=False, name="global_step")
            train_op, loss = model.train(inputs, labels, neg_labels, global_step, learning_rate)
            model_list.append(model)
            trains_list.append((train_op, loss, global_step))
        else:
            logger.error("unvalid order in LINE: '%s'. " % order)
            sys.exit()

        # Create a saver.
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5)

        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init_op = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU implementations.
        config = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        config.gpu_options.allow_growth = allow_growth
        # config.gpu_options.visible_device_list = visible_device_list

        with tf.Session(config=config) as sess:
            # first_step = 0
            if checkpoint == '0':  # new train
                sess.run(init_op)
            elif checkpoint == '-1':  # choose the latest one
                ckpt = tf.train.get_checkpoint_state(ckpt_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    # new_saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
                    # Restores from checkpoint
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # global_step_for_restore = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # first_step = int(global_step_for_restore) + 1
                else:
                    logger.warning('No checkpoint file found')
                    return
            else:
                if os.path.exists(os.path.join(ckpt_dir, 'model.ckpt-' + checkpoint + '.index')):
                    # new_saver = tf.train.import_meta_graph(
                    #     os.path.join(ckpt_dir, 'model.ckpt-' + checkpoint + '.meta'))
                    saver.restore(sess,
                                  os.path.join(ckpt_dir, 'model.ckpt-' + checkpoint))
                    # first_step = int(checkpoint) + 1
                else:
                    logger.warning('checkpoint {} not found'.format(checkpoint))
                    return

            summary_writer = tf.summary.FileWriter(ckpt_dir, sess.graph)

            last_loss_time = time.time() - loss_interval
            last_summary_time = time.time() - summary_interval
            last_decay_time = last_checkpoint_time = time.time()
            last_decay_step = last_summary_step = last_checkpoint_step = 0
            while True:
                start_time = time.time()
                batch_data, batch_labels, batch_neg_labels = edge_sampler.next_batch()
                feed_dict = {inputs: batch_data, labels: batch_labels, neg_labels: batch_neg_labels,
                             learning_rate: LR.learning_rate}
                loss_value_list = []
                for train_op, loss, global_step in trains_list:
                    _, loss_value, cur_step = sess.run([train_op, loss, global_step], feed_dict=feed_dict)
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    loss_value_list.append(loss_value)
                now = time.time()

                epoch, epoch_step = divmod(cur_step, num_steps_per_epoch)

                if now - last_loss_time >= loss_interval:
                    if len(loss_value_list)==1: loss_str = "%.6f" % loss_value_list[0]
                    else: loss_str = "[%.6f, %.6f]" % (loss_value_list[0],loss_value_list[1])
                    format_str = '%s: step=%d(%d/%d), lr=%.6f, loss=%s, duration/step=%.4fs'
                    logger.info(format_str % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                                              cur_step, epoch_step, epoch, LR.learning_rate, loss_str,
                                              now - start_time))
                    last_loss_time = time.time()
                if now - last_summary_time >= summary_interval or cur_step - last_summary_step >= summary_steps or cur_step >= iter_steps:
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, cur_step)
                    last_summary_time = time.time()
                    last_summary_step = cur_step
                ckpted = False
                # Save the model checkpoint periodically. (named 'model.ckpt-global_step.meta')
                if now - last_checkpoint_time >= ckpt_interval or cur_step - last_checkpoint_step >= ckpt_steps or cur_step >= iter_steps:
                    # embedding_vectors = sess.run(model.vectors, feed_dict=feed_dict)
                    vecs_list = []
                    for model in model_list:
                        vecs = sess.run(model.vectors, feed_dict=feed_dict)
                        vecs_list.append(vecs)
                    vecs = np.concatenate(vecs_list,axis=1)
                    checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')
                    utils.save_word2vec_format_and_ckpt(vectors_path, vecs, checkpoint_path, sess, saver, cur_step, types_size)
                    last_checkpoint_time = time.time()
                    last_checkpoint_step = cur_step
                    ckpted = True
                # update learning rate
                if ckpted or now - last_decay_time >= decay_interval or (
                        decay_steps > 0 and cur_step - last_decay_step >= decay_steps):
                    lr_info = np.loadtxt(lr_file, dtype=float)
                    if np.abs(lr_info[1] - decay_epochs) > 1e-6:
                        decay_epochs = lr_info[1]
                        decay_steps = round(decay_epochs * num_steps_per_epoch)
                    if np.abs(lr_info[2] - decay_rate) > 1e-6:
                        decay_rate = lr_info[2]
                    if np.abs(lr_info[3] - iter_epochs) > 1e-6:
                        iter_epochs = lr_info[3]
                        iter_steps = round(iter_epochs * num_steps_per_epoch)
                    if np.abs(lr_info[0] - initial_learning_rate) > 1e-6:
                        initial_learning_rate = lr_info[0]
                        LR.reset(initial_learning_rate=initial_learning_rate, initial_steps=cur_step,
                                 decay_rate=decay_rate, decay_steps=decay_steps, iter_steps=iter_steps)
                    else:
                        LR.exponential_decay(cur_step,
                                             decay_rate=decay_rate, decay_steps=decay_steps, iter_steps=iter_steps)
                    last_decay_time = time.time()
                    last_decay_step = cur_step
                if cur_step >= LR.iter_steps:
                    break

            summary_writer.close()

def train_vectors(options):
    # check vectors and ckpt
    checkpoint = '0'
    train_vec_dir = os.path.split(options.vectors_path)[0]
    ckpt_dir = os.path.join(train_vec_dir, 'ckpt')
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        cur_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        logger.info("model and vectors already exists, checkpoint step = {}".format(cur_step))
        checkpoint = input(
            "please input 0 to start a new train, or input a choosed ckpt to restore (-1 for latest ckpt)")
    if checkpoint == '0':
        if ckpt:
            tf.gfile.DeleteRecursively(ckpt_dir)
        logger.info('start a new embedding train using tensorflow ...')
    elif checkpoint == '-1':
        logger.info('restore a embedding train using tensorflow from latest ckpt...')
    else:
        logger.info('restore a embedding train using tensorflow from ckpt-%s...' % checkpoint)
    if not os.path.exists(train_vec_dir):
        os.makedirs(train_vec_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # construct network
    net = network.construct_network(options, isHIN=True)

    lr_file = os.path.join(train_vec_dir, "lr.info")
    np.savetxt(lr_file,
               np.asarray([options.learning_rate, options.decay_epochs, options.decay_rate, options.iter_epoches],
                          dtype=np.float32),
               fmt="%.6f")

    assert options.order == "2", "PTE model only utilize the 2nd order proximity of LINE to HIN."

    # train info
    logger.info('Train info:')
    logger.info('\t data_dir = {}'.format(options.data_dir))
    logger.info('\t data_name = {}'.format(options.data_name))
    logger.info('\t isdirected = {}\n'.format(options.isdirected))
    logger.info('\t train_model = {}'.format(options.model))
    logger.info('\t order = {}'.format(options.order))
    logger.info('\t total embedding nodes = {}'.format(net.get_nodes_size()))
    logger.info('\t total edges = {}'.format(net.get_edges_size()))
    logger.info('\t embedding size = {}'.format(options.embedding_size))
    logger.info('\t negative = {}'.format(options.negative))
    logger.info('\t distortion_power = {}\n'.format(options.distortion_power))
    logger.info('\t batch_size = {}'.format(options.batch_size))
    logger.info('\t iter_epoches = {}'.format(options.iter_epoches))
    logger.info('\t init_learning_rate = {}'.format(options.learning_rate))
    logger.info('\t decay_epochs = {}'.format(options.decay_epochs))
    logger.info('\t decay_interval = {}'.format(options.decay_interval))
    logger.info('\t decay_rate = {}'.format(options.decay_rate))
    logger.info('\t loss_interval = {}s'.format(options.loss_interval))
    logger.info('\t summary_steps = {}'.format(options.summary_steps))
    logger.info('\t summary_interval = {}s'.format(options.summary_interval))
    logger.info('\t ckpt_epochs = {}'.format(options.ckpt_epochs))
    logger.info('\t ckpt_interval = {}s\n'.format(options.ckpt_interval))
    logger.info('\t using_gpu = {}'.format(options.using_gpu))
    logger.info('\t visible_device_list = {}'.format(options.visible_device_list))
    logger.info('\t log_device_placement = {}'.format(options.log_device_placement))
    logger.info('\t allow_soft_placement = {}'.format(options.allow_soft_placement))
    logger.info('\t gpu_memory_fraction = {}'.format(options.gpu_memory_fraction))
    logger.info('\t gpu_memory_allow_growth = {}'.format(options.allow_growth))
    logger.info('\t train_workers = {}\n'.format(options.train_workers))

    logger.info('\t ckpt_dir = {}'.format(ckpt_dir))
    logger.info('\t vectors_path = {}'.format(options.vectors_path))
    logger.info('\t learning_rate_path = {}'.format(lr_file))

    fr_vec = open(os.path.join(train_vec_dir, 'embedding.info'), 'w')
    fr_vec.write('embedding info:\n')
    fr_vec.write('\t data_dir = {}\n'.format(options.data_dir))
    fr_vec.write('\t data_name = {}\n'.format(options.data_name))
    fr_vec.write('\t isdirected = {}\n\n'.format(options.isdirected))
    fr_vec.write('\t train_model = {}\n'.format(options.model))
    fr_vec.write('\t order = {}\n'.format(options.order))
    fr_vec.write('\t total embedding nodes = {}\n'.format(net.get_nodes_size()))
    fr_vec.write('\t total edges = {}\n'.format(net.get_edges_size()))
    fr_vec.write('\t embedding size = {}\n'.format(options.embedding_size))
    fr_vec.write('\t negative = {}\n'.format(options.negative))
    fr_vec.write('\t distortion_power = {}\n\n'.format(options.distortion_power))
    fr_vec.write('\t batch_size = {}\n'.format(options.batch_size))
    fr_vec.write('\t iter_epoches = {}\n'.format(options.iter_epoches))
    fr_vec.write('\t init_learning_rate = {}\n'.format(options.learning_rate))
    fr_vec.write('\t decay_epochs = {}\n'.format(options.decay_epochs))
    fr_vec.write('\t decay_interval = {}\n'.format(options.decay_interval))
    fr_vec.write('\t decay_rate = {}\n'.format(options.decay_rate))
    fr_vec.write('\t loss_interval = {}s\n'.format(options.loss_interval))
    fr_vec.write('\t summary_steps = {}\n'.format(options.summary_steps))
    fr_vec.write('\t summary_interval = {}s\n'.format(options.summary_interval))
    fr_vec.write('\t ckpt_epochs = {}\n'.format(options.ckpt_epochs))
    fr_vec.write('\t ckpt_interval = {}s\n\n'.format(options.ckpt_interval))
    fr_vec.write('\t using_gpu = {}\n'.format(options.using_gpu))
    fr_vec.write('\t visible_device_list = {}\n'.format(options.visible_device_list))
    fr_vec.write('\t log_device_placement = {}\n'.format(options.log_device_placement))
    fr_vec.write('\t allow_soft_placement = {}\n'.format(options.allow_soft_placement))
    fr_vec.write('\t gpu_memory_fraction = {}\n'.format(options.gpu_memory_fraction))
    fr_vec.write('\t gpu_memory_allow_growth = {}\n'.format(options.allow_growth))
    fr_vec.write('\t train_workers = {}\n\n'.format(options.train_workers))

    fr_vec.write('\t ckpt_dir = {}\n'.format(ckpt_dir))
    fr_vec.write('\t vectors_path = {}\n'.format(options.vectors_path))
    fr_vec.write('\t learning_rate_path = {}\n'.format(lr_file))

    fr_vec.close()

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if options.using_gpu:
        visible_devices = str(options.visible_device_list[0])
        for dev in options.visible_device_list[1:]:
            visible_devices = visible_devices + ',%s' % dev
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


    # set log_level for gpu:
    console_log_level = options.log.upper()
    if console_log_level == "CRITICAL":
        gpu_log = '3'
    elif console_log_level == "ERROR":
        gpu_log = '2'
    elif console_log_level == "WARNING":
        gpu_log = '1'
    else:
        gpu_log = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = gpu_log


    # train
    logger.info('training...')
    time_start = time.time()
    train(net=net, vectors_path=options.vectors_path, lr_file=lr_file,
          ckpt_dir=ckpt_dir, checkpoint=checkpoint, order = options.order,
          embedding_size=options.embedding_size, neg_sampled=options.negative,
          batch_size=options.batch_size, distortion_power=options.distortion_power,
          initial_learning_rate=options.learning_rate, decay_epochs=options.decay_epochs,
          decay_rate=options.decay_rate, iter_epochs=options.iter_epoches,
          allow_soft_placement=options.allow_soft_placement, log_device_placement=options.log_device_placement,
          gpu_memory_fraction=options.gpu_memory_fraction, using_gpu=options.using_gpu,
          allow_growth=options.allow_growth, loss_interval=options.loss_interval, summary_steps=options.summary_steps,
          ckpt_interval=options.ckpt_interval, ckpt_epochs=options.ckpt_epochs,
          summary_interval=options.summary_interval, decay_interval=options.decay_interval)
    logger.info('train completed in {}s'.format(time.time() - time_start))
    return









