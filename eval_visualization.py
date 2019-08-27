#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
__author__ = "He Yu"
Version: 1.0.0
ref:

"""
import tensorflow as tf
import shutil
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import logging
import time
import os

import utils
import eval_utils
import network

logger = logging.getLogger("HNE")

def plot_embedding_in_2D(Markersize, features_matrix, labels_matrix, label_size, figure_path):
    node_num, embedding_dimension = features_matrix.shape
    node_colors = eval_utils.get_node_color(labels_matrix, label_size)

    # use t-SNE to reduce the dimension to 2
    if(embedding_dimension > 2):
        time_start = time.time()
        logger.info('Embedding dimension greater than 2, use t-SNE to reduce it to 2 ...')
        model = TSNE(n_components=2)
        node_pos = model.fit_transform(features_matrix)
        logger.info('t-SNE reduce completed in {}s'.format(time.time() - time_start))
    else:
        node_pos = features_matrix

    plt.switch_backend('agg') # 指定不需要GUI图形界面，以便ssh服务器运行

    # plot scatter
    plt.scatter(node_pos[:, 0], node_pos[:, 1], c=node_colors, s=Markersize)

    # # set the range of xlim and ylim 设置坐标轴范围，防止数据离散的太厉害，这样可以只显示散点图的主要聚集位置
    # plt.xlim((-1.5, 1.5))
    # plt.ylim((-1.5, 1.5))

    # # 关闭坐标轴
    # plt.axis('off')

    # hide the ticks 隐藏坐标轴上的刻度，这样展示的时候会很有用
    plt.xticks(())
    plt.yticks(())


    plt.savefig(figure_path + '.pdf', format='pdf', bbox_inches='tight', dpi=500)
    plt.savefig(figure_path + '.png', format='png', bbox_inches='tight', dpi=500)
    # plt.figure()
    CCD = eval_utils.clustering_center_distance(node_pos=node_pos,node_label=labels_matrix)
    return CCD


def eval_once(options):
    # visual_dir, visual_file = os.path.split(options.visualization_path)
    if not utils.check_rebuild(options.visualization_path, descrip='visualization', always_rebuild=options.always_rebuild):
        return
    # print logger
    logger.info('eval case: visualization...')
    logger.info('\t data_dir = {}'.format(options.data_dir))
    logger.info('\t data_name = {}'.format(options.data_name))
    logger.info('\t isdirected = {}'.format(options.isdirected))
    logger.info('\t label_path = {}'.format(options.label_path))
    logger.info('\t label_size = {}'.format(options.label_size))
    logger.info('\t eval_node_type: {}'.format(options.eval_node_type))
    logger.info('\t save_path: {}\n'.format(options.visualization_path))
    logger.info('\t method: t-SNE')
    logger.info('\t multilabel_rule: {}'.format(options.multilabel_rule))
    logger.info('\t marker_size: {}'.format(options.marker_size))
    logger.info('\t eval_online: {}'.format(options.eval_online))


    # get embedding vectors and markersize
    logger.info('\t reading labeled data from file {}'.format(options.label_path))
    time_start = time.time()
    id_list, labels_list = utils.get_labeled_data(options.label_path, type=options.eval_node_type,
                                                  multilabel_rule=options.multilabel_rule,
                                                  type_filepath=os.path.join(options.data_dir,
                                                                             options.data_name + ".nodes"))
    id_list, features_matrix, labels_list = utils.get_vectors(utils.get_KeyedVectors(options.vectors_path), id_list, labels_list)
    labels_matrix = np.array([item[0] for item in labels_list])
    logger.info('\t reading labeled data completed in {}s'.format(time.time() - time_start))
    logger.info('\t total labeled data size: {}'.format(np.size(features_matrix,axis=0)))
    logger.info('\t the labels data embedding_dimension: {}'.format(np.size(features_matrix,axis=1)))
    logger.info('\t total labels size: {}'.format(options.label_size))
    for i in range(options.label_size):
        logger.info('\t\t label {}: {}'.format(i, np.sum(labels_matrix == i)))

    fr = open(options.visualization_path, 'w')
    fr.write('eval case: visualization...\n')
    fr.write('\t data_dir = {}\n'.format(options.data_dir))
    fr.write('\t data_name = {}\n'.format(options.data_name))
    fr.write('\t isdirected = {}\n'.format(options.isdirected))
    fr.write('\t label_path = {}\n'.format(options.label_path))
    fr.write('\t label_size = {}\n'.format(options.label_size))
    fr.write('\t eval_node_type: {}\n'.format(options.eval_node_type))
    fr.write('\t save_path: {}\n\n'.format(options.visualization_path))
    fr.write('\t method: t-SNE\n')
    fr.write('\t multilabel_rule: {}\n'.format(options.multilabel_rule))
    fr.write('\t marker_size: {}\n'.format(options.marker_size))
    fr.write('\t eval_online: {}\n'.format(options.eval_online))
    fr.write('\t total labeled data size: {}\n'.format(np.size(features_matrix, axis=0)))
    fr.write('\t the labels data embedding_dimension: {}\n'.format(np.size(features_matrix, axis=1)))
    fr.write('\t total labels size: {}\n'.format(options.label_size))
    for i in range(options.label_size):
        fr.write('\t\t label {}: {}\n'.format(i, np.sum(labels_matrix==i)))

    figure_name = "visualization_" + str(np.size(features_matrix, axis=1))
    figure_path = os.path.join(os.path.split(options.visualization_path)[0],figure_name)
    CCD = plot_embedding_in_2D(Markersize=options.marker_size,
                               features_matrix=features_matrix,
                               labels_matrix=labels_matrix,
                               label_size=options.label_size,
                               figure_path = figure_path)

    fr.write('\n figure_path: {}\n'.format(figure_path))
    fr.write(' clustering_center_distance_sim: {}\n'.format(CCD))
    fr.write('\neval case: visualization completed in {}s\n ======================'.format(time.time() - time_start))
    fr.close()
    logger.info('eval case: visualization completed in {}s\n ======================'.format(time.time() - time_start))

def eval_online(options):
    visual_dir = os.path.split(options.visualization_path)[0]
    if not utils.check_rebuild(visual_dir, descrip='visualization', always_rebuild=options.always_rebuild):
        return
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    # print logger
    logger.info('eval case: visualization...')
    logger.info('\t data_dir = {}'.format(options.data_dir))
    logger.info('\t data_name = {}'.format(options.data_name))
    logger.info('\t isdirected = {}'.format(options.isdirected))
    logger.info('\t label_path = {}'.format(options.label_path))
    logger.info('\t label_size = {}'.format(options.label_size))
    logger.info('\t eval_node_type: {}'.format(options.eval_node_type))
    logger.info('\t save_dir: {}\n'.format(visual_dir))
    logger.info('\t method: t-SNE')
    logger.info('\t multilabel_rule: {}'.format(options.multilabel_rule))
    logger.info('\t marker_size: {}'.format(options.marker_size))
    logger.info('\t eval_online: {}'.format(options.eval_online))
    logger.info('\t eval_interval: {}s'.format(options.eval_interval))


    logger.info('\t reading labeled data from file {}'.format(options.label_path))
    # get embedding vectors and markersize
    time_start = time.time()
    id_list_totoal, labels_list_totoal = utils.get_labeled_data(options.label_path, type=options.eval_node_type,
                                                                multilabel_rule=options.multilabel_rule,
                                                                type_filepath=os.path.join(options.data_dir,
                                                                                           options.data_name + ".nodes"))
    logger.info('\t reading labeled data completed in {}s'.format(time.time() - time_start))

    logger.info('\t total labeled data size: {}'.format(len(id_list_totoal)))
    logger.info('\t total labels size: {}'.format(options.label_size))


    fr_total = open(options.visualization_path, 'w')
    fr_total.write('eval case: visualization...\n')
    fr_total.write('\t data_dir = {}\n'.format(options.data_dir))
    fr_total.write('\t data_name = {}\n'.format(options.data_name))
    fr_total.write('\t isdirected = {}\n'.format(options.isdirected))
    fr_total.write('\t label_path = {}\n'.format(options.label_path))
    fr_total.write('\t label_size = {}\n'.format(options.label_size))
    fr_total.write('\t eval_node_type: {}\n'.format(options.eval_node_type))
    fr_total.write('\t save_dir: {}\n\n'.format(visual_dir))
    fr_total.write('\t method: t-SNE\n')
    fr_total.write('\t multilabel_rule: {}\n'.format(options.multilabel_rule))
    fr_total.write('\t marker_size: {}\n'.format(options.marker_size))
    fr_total.write('\t eval_online: {}\n'.format(options.eval_online))
    fr_total.write('\t eval_interval: {}s\n'.format(options.eval_interval))
    fr_total.write('\t total labeled data size: {}\n'.format(len(id_list_totoal)))
    fr_total.write('\t total labels size: {}\n'.format(options.label_size))
    fr_total.write('\t results(CCD-clustering_center_distance_sim):\n'
                   '=============================================================\n')
    fr_total.write('finish_time\tckpt\tCCD\n')


    last_step = 0
    summary_writer = tf.summary.FileWriter(visual_dir, tf.Graph())
    summary = tf.Summary()
    summary.value.add(tag='CCD', simple_value=0.)
    summary_writer.add_summary(summary, last_step)

    best_CCD = 0

    ckpt_dir = os.path.join(os.path.split(options.vectors_path)[0], 'ckpt')
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    while (not (ckpt and ckpt.model_checkpoint_path)):
        logger.info("model and vectors not exist, waiting...")
        time.sleep(options.eval_interval)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)

    reading = options.vectors_path + ".reading_visualization_{}".format(options.eval_node_type)
    writing = options.vectors_path + ".writing"

    while (options.eval_online):
        while True:
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            cur_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            if cur_step <= last_step or (not os.path.exists(options.vectors_path)) or os.path.exists(writing):
                if os.path.exists(os.path.join(os.path.split(options.vectors_path)[0], "RUN_SUCCESS")):
                    return
                time.sleep(options.eval_interval)
                continue
            # ready for reading
            logger.info("\t declare for reading ...")
            open(reading, "w")  # declare
            time.sleep(30)
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            cur_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            if cur_step <= last_step or (not os.path.exists(options.vectors_path)) or os.path.exists(writing):
                os.remove(reading)  # undeclare
                logger.info("\t confliction! undeclare and waiting ...")
                time.sleep(options.eval_interval)
                continue
            break
        logger.info("\t eval ckpt-{}.......".format(cur_step))
        time_start = time.time()
        logger.info('\t reading embedding vectors from file {}'.format(options.vectors_path))
        id_list, features_matrix, labels_list = utils.get_vectors(utils.get_KeyedVectors(options.vectors_path),
                                                         id_list_totoal, labels_list_totoal)
        os.remove(reading)  # synchrolock for multi-process
        logger.info("\t done for reading ...")
        labels_matrix = np.array([item[0] for item in labels_list])
        logger.info('\t reading labeled data completed in {}s'.format(time.time() - time_start))
        logger.info('\t total labeled data size: {}'.format(np.size(features_matrix, axis=0)))
        logger.info('\t total labels size: {}'.format(options.label_size))
        for i in range(options.label_size):
            logger.info('\t\t label {}: {}'.format(i, np.sum(labels_matrix == i)))

        # visualization
        fr = open(options.visualization_path + '.{}'.format(cur_step), 'w')
        fr.write('eval case: visualization...\n')
        fr.write('\t data_dir = {}\n'.format(options.data_dir))
        fr.write('\t data_name = {}\n'.format(options.data_name))
        fr.write('\t isdirected = {}\n'.format(options.isdirected))
        fr.write('\t label_path = {}\n'.format(options.label_path))
        fr.write('\t label_size = {}\n'.format(options.label_size))
        fr.write('\t eval_node_type: {}\n'.format(options.eval_node_type))
        fr.write('\t method: t-SNE\n')
        fr.write('\t multilabel_rule: {}\n'.format(options.multilabel_rule))
        fr.write('\t marker_size: {}\n'.format(options.marker_size))
        fr.write('\t eval_online: {}\n'.format(options.eval_online))
        fr.write('\t eval_interval: {}s\n'.format(options.eval_interval))
        fr.write('\t total labeled data size: {}\n'.format(np.size(features_matrix, axis=0)))
        fr.write('\t total labels size: {}\n'.format(options.label_size))
        for i in range(options.label_size):
            fr.write('\t\t label {}: {}\n'.format(i, np.sum(labels_matrix == i)))

        fr_total.write('%s ckpt-%-9d: ' % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), cur_step))
        summary = tf.Summary()

        figure_name = "visualization_" + str(np.size(features_matrix, axis=1)) + '.{}'.format(cur_step)
        figure_path = os.path.join(visual_dir, figure_name)
        CCD = plot_embedding_in_2D(Markersize=options.marker_size,
                                   features_matrix=features_matrix,
                                   labels_matrix=labels_matrix,
                                   label_size=options.label_size,
                                   figure_path=figure_path)

        fr.write('\n figure_path: {}\n'.format(figure_path))
        fr.write(' clustering_center_distance_sim:{}\n'.format(CCD))
        fr.write('\neval case: visualization completed in {}s\n ======================'.format(time.time() - time_start))
        fr.close()

        fr_total.write('%.4f\n' % CCD)
        fr_total.flush()
        summary.value.add(tag='CCD', simple_value=CCD)
        summary_writer.add_summary(summary, cur_step)
        summary_writer.flush()
        logger.info('visualization completed in {}s\n================================='.format(time.time() - time_start))

        # copy ckpt-files according to last mean_Micro_F1 (0.9 ratio).
        if CCD > best_CCD:
            best_CCD = CCD

            ckptIsExists = os.path.exists(os.path.join(ckpt_dir, 'model.ckpt-%d.index' % cur_step))
            if ckptIsExists:
                fr_best = open(os.path.join(visual_dir, 'best_ckpt.info'), 'w')
            else:
                fr_best = open(os.path.join(visual_dir, 'best_ckpt.info'), 'a')
                fr_best.write("Note:the model.ckpt-best is the remainings of last best_ckpt!\n"
                              "the current best_ckpt model is loss, but the result is:\n")
            fr_best.write("best_CCD: {}\n".format(best_CCD))
            fr_best.write("best_ckpt: ckpt-{}\n".format(cur_step))
            fr_best.close()

            if ckptIsExists:
                sourceFile = os.path.join(ckpt_dir, 'model.ckpt-%d.data-00000-of-00001' % cur_step)
                targetFile = os.path.join(visual_dir, 'model.ckpt-best.data-00000-of-00001')
                if os.path.exists(targetFile):
                    os.remove(targetFile)
                shutil.copy(sourceFile, targetFile)
                sourceFile = os.path.join(ckpt_dir, 'model.ckpt-%d.index' % cur_step)
                targetFile = os.path.join(visual_dir, 'model.ckpt-best.index')
                if os.path.exists(targetFile):
                    os.remove(targetFile)
                shutil.copy(sourceFile, targetFile)
                sourceFile = os.path.join(ckpt_dir, 'model.ckpt-%d.meta' % cur_step)
                targetFile = os.path.join(visual_dir, 'model.ckpt-best.meta')
                if os.path.exists(targetFile):
                    os.remove(targetFile)
                shutil.copy(sourceFile, targetFile)
        last_step = cur_step

    fr_total.close()
    summary_writer.close()
    return

def eval(options):
    if "all" not in options.metapath_path:
        metatree = network.construct_meta_tree(metapaths_filename=options.metapath_path)
        flag = False
        for each in metatree.nodes():
            if options.eval_node_type == metatree.nodes[each]["type"]:
                flag = True
                break
        if not flag:
            return flag
    flag = True

    if options.eval_online:
        eval_online(options)
    else:
        eval_once(options)

    return flag
