#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
__author__ = "He Yu"
Version: 1.5.4
ref:
    http://blog.csdn.net/lilianforever/article/details/53780613
    http://blog.csdn.net/EleanorWiser/article/details/70226704
"""
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import random
import shutil
import time
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
import utils
import network


# global sharing variable
logger = logging.getLogger("HNE")
features_matrix = None
labels_matrix = None
LABEL_SIZE = None

def evalute_NMI(y_true, y_pred):
    """
    The larger the value, the better the cluster result.
    :param y_true:
    :param y_pred:
    :return:
    """
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    # logger.info('cluster NMI: %.4f'%nmi)
    return nmi


def kmeans(n_clusters, X, y):

    km = KMeans(n_clusters=n_clusters)
    km.fit(X) # clustering
    y_pred = km.labels_  # get clustering labels
    centroids = km.cluster_centers_  # get clustering center
    inertia = km.inertia_  # get the final value of the clustering criteria (evalute the cluster number whether suitable)

    # evalute:
    return y_pred, centroids, inertia


def _cluster_thread_body(repeated_times):
    nmi_list = []
    X = features_matrix
    y = labels_matrix
    for _ in range(repeated_times):
        X, y = shuffle(X, y, random_state = utils.get_random_seed())
        # clr
        clr = KMeans(n_clusters=LABEL_SIZE)
        clr.fit(X)  # clustering
        y_pred = clr.labels_  # get clustering labels
        nmi_list.append(evalute_NMI(y, y_pred))
    return nmi_list



def eval_once(options):
    global features_matrix, labels_matrix, LABEL_SIZE
    if not utils.check_rebuild(options.cluster_path, descrip='cluster', always_rebuild=options.always_rebuild):
        return
    logger.info('eval case: cluster...')
    logger.info('\t data_dir = {}'.format(options.data_dir))
    logger.info('\t data_name = {}'.format(options.data_name))
    logger.info('\t isdirected = {}'.format(options.isdirected))
    logger.info('\t label_path = {}'.format(options.label_path))
    logger.info('\t label_size = {}'.format(options.label_size))
    logger.info('\t eval_node_type: {}'.format(options.eval_node_type))
    logger.info('\t save_path: {}\n'.format(options.cluster_path))
    logger.info('\t cluster: kmeans')
    logger.info('\t multilabel_rule: {}'.format(options.multilabel_rule))
    logger.info('\t eval_online: {}'.format(options.eval_online))
    logger.info('\t eval_workers: {}'.format(options.eval_workers))
    logger.info('\t repeat {} times'.format(options.repeated_times))

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
    true_label_size = 0
    for i in range(options.label_size):
        i_count = np.sum(labels_matrix == i)
        logger.info('\t\t label {}: {}'.format(i, i_count))
        if i_count > 0:
            true_label_size +=1
    logger.info('\t true labels size: {}'.format(true_label_size))
    LABEL_SIZE = true_label_size

    # cluster
    fr = open(options.cluster_path, 'w')
    fr.write('eval case: cluster...\n')
    fr.write('\t data_dir = {}\n'.format(options.data_dir))
    fr.write('\t data_name = {}\n'.format(options.data_name))
    fr.write('\t isdirected = {}\n'.format(options.isdirected))
    fr.write('\t label_path = {}\n'.format(options.label_path))
    fr.write('\t label_size = {}\n'.format(options.label_size))
    fr.write('\t eval_node_type: {}\n'.format(options.eval_node_type))
    fr.write('\t save_path: {}\n\n'.format(options.cluster_path))
    fr.write('\t cluster: kmeans\n')
    fr.write('\t multilabel_rule: {}\n'.format(options.multilabel_rule))
    fr.write('\t eval_online: {}\n'.format(options.eval_online))
    fr.write('\t eval_workers: {}\n'.format(options.eval_workers))
    fr.write('\t repeat {} times\n'.format(options.repeated_times))
    fr.write('\t total labeled data size: {}\n'.format(np.size(features_matrix, axis=0)))
    fr.write('\t true labels size: {}\n'.format(true_label_size))
    for i in range(options.label_size):
        fr.write('\t\t label {}: {}\n'.format(i, np.sum(labels_matrix==i)))


    if options.eval_workers > 1 and options.repeated_times > 1:
        # speed up by using multi-process
        logger.info("\t allocating repeat_times to workers ...")
        if options.repeated_times <= options.eval_workers:
            times_per_worker = [1 for _ in range(options.repeated_times)]
        else:
            div, mod = divmod(options.repeated_times, options.eval_workers)
            times_per_worker = [div for _ in range(options.eval_workers)]
            for idx in range(mod):
                times_per_worker[idx] = times_per_worker[idx] + 1
        assert sum(times_per_worker) == options.repeated_times, 'workers allocating failed: %d != %d' % (
            sum(times_per_worker), options.repeated_times)

        logger.info("\t using {} processes for evaling:".format(len(times_per_worker)))
        for idx, rep_times in enumerate(times_per_worker):
            logger.info("\t process-{}: repeat {} times".format(idx, rep_times))

        try:
            nmi_list = [] # (train_ratio, macro, micro)
            with ProcessPoolExecutor(max_workers=options.eval_workers) as executor:
                for ret in executor.map(_cluster_thread_body, times_per_worker):
                    nmi_list.extend(ret)
        except:
            logger.warning("concurrent.futures.process failed, retry...")
            time.sleep(10)
            nmi_list = []  # (train_ratio, macro, micro)
            with ProcessPoolExecutor(max_workers=options.eval_workers) as executor:
                for ret in executor.map(_cluster_thread_body, times_per_worker):
                    nmi_list.extend(ret)

        if len(nmi_list) != options.repeated_times:
            logger.warning(
                "warning: eval unmatched repeated_times: {} != {}".format(len(nmi_list), options.repeated_times))
    else:
        try:
            nmi_list = _cluster_thread_body(options.repeated_times)
        except:
            nmi_list = _cluster_thread_body(options.repeated_times)


    mean_nmi = sum(nmi_list) / float(len(nmi_list))
    fr.write('expected repeated_times: {}, actual repeated_times: {}, mean results as follows:\n'.format(options.repeated_times, len(nmi_list)))
    fr.write('\t\t NMI = {}\n'.format(mean_nmi))
    fr.write('details:\n')
    for repeat in range(len(nmi_list)):
        fr.write('\t repeated {}/{}: NMI = {}\n'.format(repeat+1, len(nmi_list), nmi_list[repeat]))
    fr.write('\neval case: cluster completed in {}s.'.format(time.time() - time_start))
    fr.close()
    logger.info('eval case: cluster completed in {}s.'.format(time.time() - time_start))

    return


def eval_online(options):
    global features_matrix, labels_matrix, LABEL_SIZE
    cluster_dir = os.path.split(options.cluster_path)[0]
    if not utils.check_rebuild(cluster_dir, descrip='cluster', always_rebuild=options.always_rebuild):
        return
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)

    logger.info('eval case: cluster...')
    logger.info('\t data_dir = {}'.format(options.data_dir))
    logger.info('\t data_name = {}'.format(options.data_name))
    logger.info('\t isdirected = {}'.format(options.isdirected))
    logger.info('\t label_path = {}'.format(options.label_path))
    logger.info('\t label_size = {}'.format(options.label_size))
    logger.info('\t eval_node_type: {}'.format(options.eval_node_type))
    logger.info('\t save_dir: {}\n'.format(cluster_dir))
    logger.info('\t cluster: kmeans')
    logger.info('\t multilabel_rule: {}'.format(options.multilabel_rule))
    logger.info('\t eval_online: {}'.format(options.eval_online))
    logger.info('\t eval_interval: {}s'.format(options.eval_interval))
    logger.info('\t eval_workers: {}'.format(options.eval_workers))
    logger.info('\t repeat {} times'.format(options.repeated_times))

    logger.info('\t reading labeled data from file {}'.format(options.label_path))
    time_start = time.time()
    id_list_totoal, labels_list_totoal = utils.get_labeled_data(options.label_path, type=options.eval_node_type,
                                                  multilabel_rule=options.multilabel_rule,
                                                  type_filepath=os.path.join(options.data_dir,
                                                                             options.data_name + ".nodes"))
    logger.info('\t reading labeled data completed in {}s'.format(time.time() - time_start))

    logger.info('\t total labeled data size: {}'.format(len(id_list_totoal)))
    logger.info('\t total labels size: {}'.format(options.label_size))


    if options.eval_workers > 1 and options.repeated_times > 1:
        # speed up by using multi-process
        logger.info("\t allocating repeat_times to workers ...")
        if options.repeated_times <= options.eval_workers:
            times_per_worker = [1 for _ in range(options.repeated_times)]
        else:
            div, mod = divmod(options.repeated_times, options.eval_workers)
            times_per_worker = [div for _ in range(options.eval_workers)]
            for idx in range(mod):
                times_per_worker[idx] = times_per_worker[idx] + 1
        assert sum(times_per_worker) == options.repeated_times, 'workers allocating failed: %d != %d' % (
            sum(times_per_worker), options.repeated_times)

        logger.info("\t using {} processes for evaling:".format(len(times_per_worker)))
        for idx, rep_times in enumerate(times_per_worker):
            logger.info("\t process-{}: repeat {} times".format(idx, rep_times))


    fr_total = open(options.cluster_path, 'w')
    fr_total.write('eval case: cluster...\n')
    fr_total.write('\t data_dir = {}\n'.format(options.data_dir))
    fr_total.write('\t data_name = {}\n'.format(options.data_name))
    fr_total.write('\t isdirected = {}\n'.format(options.isdirected))
    fr_total.write('\t label_path = {}\n'.format(options.label_path))
    fr_total.write('\t label_size = {}\n'.format(options.label_size))
    fr_total.write('\t eval_node_type: {}\n'.format(options.eval_node_type))
    fr_total.write('\t save_dir: {}\n\n'.format(cluster_dir))
    fr_total.write('\t cluster: kmeans\n')
    fr_total.write('\t multilabel_rule: {}\n'.format(options.multilabel_rule))
    fr_total.write('\t eval_online: {}\n'.format(options.eval_online))
    fr_total.write('\t eval_interval: {}s\n'.format(options.eval_interval))
    fr_total.write('\t eval_workers: {}\n'.format(options.eval_workers))
    fr_total.write('\t repeat {} times\n'.format(options.repeated_times))
    fr_total.write('\t total labeled data size: {}\n'.format(len(id_list_totoal)))
    fr_total.write('\t total labels size: {}\n'.format(options.label_size))
    fr_total.write('\t results(NMI):\n=============================================================\n')
    fr_total.write('finish_time\tckpt\tNMI\n')


    last_step = 0
    summary_writer = tf.summary.FileWriter(cluster_dir, tf.Graph())
    summary = tf.Summary()
    summary.value.add(tag='nmi', simple_value=0.)
    summary_writer.add_summary(summary, last_step)

    best_nmi = 0

    ckpt_dir = os.path.join(os.path.split(options.vectors_path)[0], 'ckpt')
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    while (not (ckpt and ckpt.model_checkpoint_path)):
        logger.info("\t model and vectors not exist, waiting ...")
        time.sleep(options.eval_interval)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)

    reading = options.vectors_path + ".reading_cluster_{}".format(options.eval_node_type)
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
            open(reading, "w") # declare
            time.sleep(30)
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            cur_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            if cur_step <= last_step or (not os.path.exists(options.vectors_path)) or os.path.exists(writing):
                os.remove(reading) # undeclare
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
        true_label_size = 0
        for i in range(options.label_size):
            i_count = np.sum(labels_matrix == i)
            logger.info('\t\t label {}: {}'.format(i, i_count))
            if i_count > 0:
                true_label_size += 1
        logger.info('\t true labels size: {}'.format(true_label_size))
        LABEL_SIZE = true_label_size

        # cluster
        fr = open(options.cluster_path + '.{}'.format(cur_step), 'w')
        fr.write('eval case: cluster...\n')
        fr.write('\t data_dir = {}\n'.format(options.data_dir))
        fr.write('\t data_name = {}\n'.format(options.data_name))
        fr.write('\t isdirected = {}\n'.format(options.isdirected))
        fr.write('\t label_path = {}\n'.format(options.label_path))
        fr.write('\t label_size = {}\n'.format(options.label_size))
        fr.write('\t eval_node_type: {}\n\n'.format(options.eval_node_type))
        fr.write('\t cluster: kmeans\n')
        fr.write('\t multilabel_rule: {}\n'.format(options.multilabel_rule))
        fr.write('\t eval_workers: {}\n'.format(options.eval_workers))
        fr.write('\t repeat {} times\n'.format(options.repeated_times))
        fr.write('\t total labeled data size: {}\n'.format(np.size(features_matrix, axis=0)))
        fr.write('\t true labels size: {}\n'.format(true_label_size))
        for i in range(options.label_size):
            fr.write('\t\t label {}: {}\n'.format(i, np.sum(labels_matrix == i)))


        if options.eval_workers > 1 and options.repeated_times > 1:
            # speed up by using multi-process
            fr.write("\t using {} processes for evaling:\n".format(len(times_per_worker)))
            for idx, rep_times in enumerate(times_per_worker):
                fr.write("\t process-{}: repeat {} times\n".format(idx, rep_times))

            try:
                nmi_list = []  # (train_ratio, macro, micro)
                with ProcessPoolExecutor(max_workers=options.eval_workers) as executor:
                    for ret in executor.map(_cluster_thread_body, times_per_worker):
                        nmi_list.extend(ret)
            except:
                logger.warning("concurrent.futures.process failed, retry...")
                time.sleep(10)
                nmi_list = []  # (train_ratio, macro, micro)
                with ProcessPoolExecutor(max_workers=options.eval_workers) as executor:
                    for ret in executor.map(_cluster_thread_body, times_per_worker):
                        nmi_list.extend(ret)

            if len(nmi_list) != options.repeated_times:
                logger.warning("warning: eval unmatched repeated_times: {} != {}".format(len(nmi_list) , options.repeated_times))
        else:
            try:
                nmi_list = _cluster_thread_body(options.repeated_times)
            except:
                nmi_list = _cluster_thread_body(options.repeated_times)

        fr_total.write('%s ckpt-%-9d: ' % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), cur_step))
        summary = tf.Summary()


        mean_nmi = sum(nmi_list) / float(len(nmi_list))
        fr.write('expected repeated_times: {}, actual repeated_times: {}, mean results as follows:\n'.format(options.repeated_times, len(nmi_list)))
        fr.write('\t\t NMI = {}\n'.format(mean_nmi))
        fr.write('details:\n')
        for repeat in range(len(nmi_list)):
            fr.write('\t repeated {}/{}: NMI = {}\n'.format(repeat+1, len(nmi_list), nmi_list[repeat]))
        fr.write('\neval case: cluster completed in {}s\n'.format(time.time() - time_start))
        fr.close()

        # fr_total.write('%.4f\n' % mean_nmi)
        fr_total.write('{}\n'.format(mean_nmi))
        fr_total.flush()
        summary.value.add(tag='nmi', simple_value=mean_nmi)
        summary_writer.add_summary(summary, cur_step)
        summary_writer.flush()
        logger.info('cluster completed in {}s\n================================='.format(time.time() - time_start))

        # copy ckpt-files according to last mean_Micro_F1 (0.9 ratio).
        if mean_nmi > best_nmi:
            best_nmi = mean_nmi

            ckptIsExists = os.path.exists(os.path.join(ckpt_dir, 'model.ckpt-%d.index' % cur_step))
            if ckptIsExists:
                fr_best = open(os.path.join(cluster_dir, 'best_ckpt.info'), 'w')
            else:
                fr_best = open(os.path.join(cluster_dir, 'best_ckpt.info'), 'a')
                fr_best.write("Note:the model.ckpt-best is the remainings of last best_ckpt!\n"
                              "the current best_ckpt model is loss, but the result is:\n")
            fr_best.write("best_nmi: {}\n".format(best_nmi))
            fr_best.write("best_ckpt: ckpt-{}\n".format(cur_step))
            fr_best.close()

            if ckptIsExists:
                sourceFile = os.path.join(ckpt_dir, 'model.ckpt-%d.data-00000-of-00001' % cur_step)
                targetFile = os.path.join(cluster_dir, 'model.ckpt-best.data-00000-of-00001')
                if os.path.exists(targetFile):
                    os.remove(targetFile)
                shutil.copy(sourceFile, targetFile)
                sourceFile = os.path.join(ckpt_dir, 'model.ckpt-%d.index' % cur_step)
                targetFile = os.path.join(cluster_dir, 'model.ckpt-best.index')
                if os.path.exists(targetFile):
                    os.remove(targetFile)
                shutil.copy(sourceFile, targetFile)
                sourceFile = os.path.join(ckpt_dir, 'model.ckpt-%d.meta' % cur_step)
                targetFile = os.path.join(cluster_dir, 'model.ckpt-best.meta')
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



if __name__ == '__main__':
    y_true = [11,11,11,12,12,12,13]
    y_pred = [11111,11111,11111,2222222,2222222,2222222,5435363]
    print(metrics.normalized_mutual_info_score(y_true, y_pred))
