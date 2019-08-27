#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 
Note：
    Don’t forget to install Cython (pip install cython).
"""

import os
import time
import random
import logging
import shutil
from multiprocessing import cpu_count
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors

import walker
import utils
import numpy as np

logger = logging.getLogger("HNE")
# word2vec.logger = logging.getLogger("HINwalk")



class Skipgram(object):
    """
    A subclass of Word2Vec to allow more customization.
    """

    def __init__(self, **kwargs):
        # parameter used for word2vec...
        self._word2vec_para = {}
        self._word2vec_para["sentences"] = kwargs.get("sentences", None) # corpus
        self._word2vec_para["size"] = kwargs.get("embedding_size", 100) # embedding size, default: 100
        self._word2vec_para["alpha"] = kwargs.get("learning_rate", 0.025) # learning rate, default: 0.025
        self._word2vec_para["window"] = kwargs.get("window_size", 5)  # maximum distance to search pair, default: 5
        self._word2vec_para["min_count"] = kwargs.get("min_count", 0) # frequency floor, default: 5
        self._word2vec_para["sample"] = kwargs.get("downsample", 1e-3)  # downsampling for higher-frequency words. default: 1e-3
        self._word2vec_para["sg"] = kwargs.get("sg", 1) # (sg=0)CBOW, (sg=1)skip-gram
        self._word2vec_para["hs"] = kwargs.get("hs", 0) # (hs=1)hierarchical softmax, (hs=0)negative sampling
        self._word2vec_para["negative"] = kwargs.get("negative", 5) # negative sampling, it should be a multiple of metapath size.
        # self._word2vec_para["iter"] = kwargs.get("iter", 5) # iterations (epochs) over the corpus, default: 5
        self._word2vec_para["seed"] = kwargs.get("seed", random.randint(0, 100)) # seed, default: 1
        self._word2vec_para["workers"] = kwargs.get("workers", cpu_count()) # multithreading, default: 3
        self._word2vec_para["compute_loss"] = kwargs.get("compute_loss", True) # default: False
        # self._word2vec_para["batch_words"] = kwargs.get("batch_words", 10000) # default: 10000


        self._ckpt_dir = kwargs.get("ckpt_dir", "./ckpt")
        self._ckpt_path = os.path.join(self._ckpt_dir ,'ckpt')
        self._ckpt_info = os.path.join(self._ckpt_dir , 'ckpt.info')
        self._ckpt = kwargs.get("ckpt_epoch", 1)
        self._iter = kwargs.get("iter", 5)
        self._num_ecophes = 0
        self._model = None
        self._sentences = kwargs.get("sentences", None)

    def _check_ckpt(self):
        # delete all the file that in ckptfile.
        ckpt_linelist = open(self._ckpt_info).readlines()
        # Read lines from ckpts information
        if len(ckpt_linelist) > 2:
            # If the length of the ckpt_linelist is more than 2 lines,then:
            for line in ckpt_linelist[0:-2]:
                ecoph = line.strip().split(',')[0].strip().split('=')[1].strip()
                model_name = 'ckpt-{}'.format(ecoph)
                # model_name = ckpt-ecoph
                for file in os.listdir(self._ckpt_dir):
                    if file.startswith(model_name):
                        os.remove(os.path.join(self._ckpt_dir, file))
            ckpt_linelist = ckpt_linelist[-2:]
            with open(self._ckpt_info, 'w') as fr:
                for line in ckpt_linelist:
                    fr.write(line)

    def _save_ckpt(self):
        with open(self._ckpt_info, 'a') as fr:
            fr.write('ecoph={}, loss={}\n'.format(self._num_ecophes, self._model.get_latest_training_loss()))
        self._model.save(self._ckpt_path + '-{}'.format(self._num_ecophes) + '.model')
        self._model.wv.save_word2vec_format(self._ckpt_path + '-{}'.format(self._num_ecophes) + '.vectors', binary=False)
        self._check_ckpt()

    def train(self):
        if os.path.exists(self._ckpt_info):
            # if there is already checkpoint, it means that a train is processing. Wating to be reloaded.
            epoch, _ = open(os.path.join(self._ckpt_dir, 'ckpt.info')).readlines()[-1].strip().split(',')
            self._num_ecophes = int(epoch.strip().split('=')[1].strip())
            logger.info('loading from ckpt {}...'.format(self._num_ecophes))
            self._model = word2vec.Word2Vec.load(self._ckpt_path + '-{}'.format(self._num_ecophes) + '.model')
        else:
            self._num_ecophes = min(self._ckpt,  self._iter )
            self._word2vec_para["iter"] = self._num_ecophes
            logger.info('training ecophs {}/{}...'.format(self._num_ecophes, self._iter))
            self._model = word2vec.Word2Vec(**self._word2vec_para)
            self._save_ckpt()

        while self._num_ecophes < self._iter:
            batch_ecophs = min(self._ckpt,  self._iter-self._num_ecophes)
            logger.info('training ecophs {}/{}...'.format(self._num_ecophes+batch_ecophs, self._iter))
            self._model.train(self._sentences,total_examples=self._model.corpus_count,epochs=batch_ecophs)
            self._num_ecophes += batch_ecophs
            self._save_ckpt()

    def save_vectors(self, vector_path):
        shutil.copy(src=self._ckpt_path + '-{}'.format(self._num_ecophes) + '.vectors',
                    dst=vector_path)
        # Copy the final file to the predefination place.
    def get_latest_training_loss(self):
        return self._model.get_latest_training_loss()

    def get_word2vec(self):
        return self._model.wv



# train vectors
def train_vectors(options, sens = None):
    """
    train vectors, load from files, if not, load from memory.
    :param options:
    :param sens: corpus
    :return:
    """
    train_vec_dir = os.path.split(options.vectors_path)[0]
    train_ckpt_dir = os.path.join(train_vec_dir,'ckpt')
    if os.path.exists(os.path.join(train_ckpt_dir,'ckpt.info')):
        epoches = open(os.path.join(train_ckpt_dir,'ckpt.info')).readlines()[-1].strip().split(',')[0].strip().split('=')[1].strip()
        logger.warning('iter epoches = {} in ckpt, enter yes to continue train or enter others to new train:(y/others)'.format(epoches))
        str_input = input()
        if str_input in ['y', 'Y', 'yes', 'YES']:
            logger.warning('continue training from ckpt {}...'.format(epoches))
        else:
            logger.warning('will remove ckpt files after 10 seconds, and start a new train...')
            time.sleep(10)
            shutil.rmtree(train_ckpt_dir)
    if not os.path.exists(train_vec_dir):
        os.makedirs(train_vec_dir)
    if not os.path.exists(train_ckpt_dir):
        os.makedirs(train_ckpt_dir)

    if sens is None:
        # load corpus from files
        filebase = options.corpus_store_path
        if (not os.path.exists(filebase)) or (not os.path.isfile(filebase)):
            logger.warning('corpus file \'%s\' not exits, walk mode automaticly starts...'
                           '(Continue running after 10 seconds...)' % filebase)
            time.sleep(10)
            logger.warning('Continue running...')
            sens = walker.build_walk_corpus(options)
        else:
            files_list = []
            # check index file
            with open(filebase,'r') as f:
                headline = f.readline().strip()
                if headline == options.headflag_of_index_file:
                    for line in f:
                        line = line.strip()
                        if line[0:5] == 'FILE:':
                            if os.path.exists(line[6:]):
                                logger.info('corpus file: {}'.format(line[6:]))
                                files_list.append(line[6:])
                            else:
                                logger.warning('cannot find corpus file: {}, skiped.'.format(line[6:]))
                else:
                    files_list.append(filebase)
            logger.info('load corpus files: #{}#\n{}'.format(len(files_list), files_list))
            if options.load_from_memory:
                sens = walker.load_walks_corpus(files_list)
            else:
                sens = walker.WalksCorpus(files_list)

    min_count = 0
    iter_epoches = int(options.iter_epoches)
    ckpt_epochs = int(options.ckpt_epochs)
    logger.info('Train vectors: train info:')
    logger.info('\t data_dir = {}'.format(options.data_dir))
    logger.info('\t data_name = {}'.format(options.data_name))
    logger.info('\t isdirected = {}\n'.format(options.isdirected))
    logger.info('\t corpus_store_path = {}'.format(options.corpus_store_path))
    logger.info('\t close train log = {}'.format(str(options.close_train_log)))
    logger.info('\t load from memory = {}'.format(str(options.load_from_memory)))
    logger.info('\t train_model = {}'.format(options.model))
    logger.info('\t embedding size = {}'.format(options.embedding_size))
    logger.info('\t window size = {}'.format(options.window_size))
    logger.info('\t negative = {}'.format(options.negative))
    # logger.info('\t downsample = {}'.format(options.downsample))
    logger.info('\t min_count = {}'.format(min_count))
    logger.info('\t iter_epoches = {}'.format(iter_epoches))
    logger.info('\t ckpt_epochs = {}'.format(ckpt_epochs))
    # logger.info('\t learning_rate = {}'.format(options.learning_rate))
    logger.info('\t train workers = {}'.format(options.train_workers))
    logger.info('\t vectors_path = {}'.format(options.vectors_path))
    logger.info('\t model_path = {}'.format(options.model_path))

    fr_vec = open(os.path.join(os.path.split(options.vectors_path)[0], 'embedding.info'), 'w')
    fr_vec.write('embedding info:\n')
    fr_vec.write('\t data_dir = {}\n'.format(options.data_dir))
    fr_vec.write('\t data_name = {}\n'.format(options.data_name))
    fr_vec.write('\t isdirected = {}\n\n'.format(options.isdirected))
    fr_vec.write('\t corpus_store_path = {}\n'.format(options.corpus_store_path))
    fr_vec.write('\t train_model = {}\n'.format(options.model))
    fr_vec.write('\t embedding size = {}\n'.format(options.embedding_size))
    fr_vec.write('\t window size = {}\n'.format(options.window_size))
    fr_vec.write('\t negative = {}\n'.format(options.negative))
    # fr_vec.write('\t downsample = {}\n'.format(options.downsample))
    fr_vec.write('\t min_count = {}\n'.format(min_count))
    fr_vec.write('\t iter_epoches: {}\n'.format(iter_epoches))
    fr_vec.write('\t ckpt_epochs: {}\n'.format(ckpt_epochs))
    # fr_vec.write('\t learning_rate: {}\n'.format(options.learning_rate))
    fr_vec.write('\t vectors_path = {}\n'.format(options.vectors_path))
    fr_vec.write('\t model_path = {}\n'.format(options.model_path))
    fr_vec.close()

    logger.info('Train vectors: training...')
    if not options.close_train_log:
        word2vec.logger = logging.getLogger("HNE")  # change the logger of called module to merge log info.
    time_start = time.time()
    # train word2vec model using sens.
    model = Skipgram(ckpt_dir = train_ckpt_dir,
                     sentences=sens,  # corpus
                     embedding_size=options.embedding_size,  # embedding size, default: 100
                     window_size=options.window_size,  # maximum distance to search pair, default: 5
                     negative=options.negative,  #
                     # downsample=options.downsample,  #
                     iter=iter_epoches,  # iterations (epochs) over the corpus, default: 5
                     ckpt_epoch=ckpt_epochs,
                     workers=options.train_workers,  # multithreading, default: 3
                     # learning_rate = options.learning_rate,
                     )
    model.train()
    logger.info('Train vectors: train completed in {}s'.format(time.time() - time_start))
    training_loss = model.get_latest_training_loss()
    logger.info('final training loss: {}'.format(training_loss))
    model.save_vectors(options.vectors_path)

    # if len(os.path.split(options.model_path)[1]) > 0 :
    #     if utils.check_rebuild(options.model_path, descrip='HIN model', always_rebuild=options.always_rebuild):
    #         model.save(options.model_path)
    # if len(os.path.split(options.vectors_path)[1]) > 0 :
    #     if utils.check_rebuild(options.vectors_path, descrip='HIN vectors', always_rebuild=options.always_rebuild):
    #         model.wv.save_word2vec_format(options.vectors_path, binary=False)

    return






