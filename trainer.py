#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 
 
"""
import sys
import os
import time
import logging

import skipgram
import TF_line
import TF_pte
import TF_spaceywalk
import utils


logger = logging.getLogger("HNE")


# train vectors
def train_vectors(options, sens = None):
    if not utils.check_rebuild(options.vectors_path, descrip='embedding vectors', always_rebuild=options.always_rebuild):
        return

    if options.model == 'DeepWalk':
        skipgram.train_vectors(options, sens= sens)
    elif options.model == 'LINE':
        TF_line.train_vectors(options)
    elif options.model == 'PTE':
        TF_pte.train_vectors(options)
    elif options.model == "SpaceyWalk":
        TF_spaceywalk.train_vectors(options)
    else:
        logger.error("Unknown model for embedding: '%s'. "% options.model+
                     "Valid models: 'DeepWalk', 'LINE', 'PTE', 'SpaceyWalk'.")
        sys.exit()







