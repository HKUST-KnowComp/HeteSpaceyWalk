#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
__author__ = "He Yu"
Version: 1.5.4
ref:
    https://www.jianshu.com/p/516f009c0875
    http://scikit-learn.org/dev/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
    http://scikit-learn.org/dev/modules/multiclass.html#ovr-classification
    http://sklearn.apachecn.org/cn/0.19.0/modules/linear_model.html#logistic
"""
import random
import os
import time
import logging
import shutil
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
import utils
import eval_utils
import network
import sys


# global sharing variable
logger = logging.getLogger("HNE")
features_dict = None
true_edges_list_by_repeat = None
neg_edges_list_by_repeat = None



def _classify_thread_body(train_ratio_list):
    global features_dict, true_edges_list_by_repeat, neg_edges_list_by_repeat
    ret_list = []
    for repeat, op, train_ratio in train_ratio_list:
        time_start = time.time()
        logger.info('\t repeat={}, train_ratio={}, op={}, evaling ...'.format(repeat, train_ratio, op))

        edges_train, edges_test, labels_train, labels_test = train_test_split(
            true_edges_list_by_repeat[repeat] + neg_edges_list_by_repeat[repeat],
            [1] * len(true_edges_list_by_repeat[repeat]) + [0] * len(neg_edges_list_by_repeat[repeat]),
            test_size=1.0 - train_ratio,
            random_state=utils.get_random_seed(),
            shuffle=True)

        train1 = np.array([features_dict[e[0]] for e in edges_train], dtype=np.float32)
        train2 = np.array([features_dict[e[1]] for e in edges_train], dtype=np.float32)
        test1 = np.array([features_dict[e[0]] for e in edges_test], dtype=np.float32)
        test2 = np.array([features_dict[e[1]] for e in edges_test], dtype=np.float32)

        if op == 'average':
            X_train = (train1 + train2) / 2
            X_test = (test1 + test2) / 2
        elif op == 'hadamard':
            X_train = np.multiply(train1, train2)
            X_test = np.multiply(test1, test2)
        elif op == 'l1':
            X_train = np.absolute(train1 - train2)
            X_test = np.absolute(test1 - test2)
        elif op == 'l2':
            X_train = np.square(train1 - train2)
            X_test = np.square(test1 - test2)
        elif op == 'concat':
            X_train = np.concatenate((train1, train2),axis=1)
            X_test = np.concatenate((test1, test2),axis=1)
        else:
            logger.error("error: invalid feature operator: {}".format(op))

        clf = LogisticRegression()
        clf.fit(X_train, np.asarray(labels_train))
        preds = clf.predict(X_test)
        # preds = clf.predict_proba(X_test)[:,1] # better choice!
        auc = roc_auc_score(np.asarray(labels_test), preds)

        logger.info('\t repeat={}, train_ratio={}, op={}, eval completed in {}s.'.format(repeat, train_ratio, op, time.time() - time_start))
        ret_list.append((train_ratio, op, auc))
    return ret_list

def load_features(options, net):
    global features_dict
    features_dict = {}
    time_start = time.time()
    logger.info('\t loading embedding features...')
    start_nodes = net.get_nodes(node_type=options.eval_edge_type[0])
    target_nodes = net.get_nodes(node_type=options.eval_edge_type[1])
    id_list = start_nodes + target_nodes
    id_list, features_matrix = utils.get_vectors(utils.get_KeyedVectors(options.vectors_path), id_list, missing_rule="random")
    for idx, node_id in enumerate(id_list):
        features_dict[node_id] = features_matrix[idx]
    logger.info('\t loading embedding features completed in {}s'.format(time.time() - time_start))

def load_edges(options, net):
    global true_edges_list_by_repeat, neg_edges_list_by_repeat
    true_edges_list_by_repeat = []
    neg_edges_list_by_repeat = []
    time_start = time.time()
    logger.info('\t loading edges ...')
    start_nodes = net.get_nodes(node_type=options.eval_edge_type[0])
    target_nodes = net.get_nodes(node_type=options.eval_edge_type[1])
    total_true_edges_list = net.get_edges(start_node_type = options.eval_edge_type[0], target_node_type = options.eval_edge_type[1])
    logger.info("original true edges size = {}, sample_size = {}, repeated_times={} ...".format(len(total_true_edges_list),
                                                                             options.sample_size, options.repeated_times))
    total_sampled_true_edges_size = options.repeated_times * options.sample_size
    if total_sampled_true_edges_size > 0 and total_sampled_true_edges_size <= len(total_true_edges_list):
        random.shuffle(total_true_edges_list)
        total_sampled_true_edges = total_true_edges_list[0:total_sampled_true_edges_size]
    else:
        total_sampled_true_edges = total_true_edges_list[0:]
        while len(total_sampled_true_edges) < total_sampled_true_edges_size:
            random.shuffle(total_true_edges_list)
            total_sampled_true_edges.extend(total_true_edges_list[0:min(len(total_true_edges_list), total_sampled_true_edges_size-len(total_sampled_true_edges))])
        assert len(total_sampled_true_edges) == total_sampled_true_edges_size, "{} != {}".format(len(total_sampled_true_edges), total_sampled_true_edges_size)

    total_sampled_neg_edges = []
    total_sampled_neg_edges_set = set()
    while True:
        if len(total_sampled_neg_edges) == total_sampled_true_edges_size:
            break
        s_i = random.choice(start_nodes)
        t_i = random.choice(target_nodes)
        if not net.has_edge(s_i, t_i) and (s_i, t_i) not in total_sampled_neg_edges_set:
            total_sampled_neg_edges_set.add((s_i, t_i))
            total_sampled_neg_edges.append((s_i, t_i))

    for repeat in range(options.repeated_times):
        true_edges_list_by_repeat.append(total_sampled_true_edges[repeat*options.sample_size:(repeat+1)*options.sample_size])
        neg_edges_list_by_repeat.append(total_sampled_neg_edges[repeat*options.sample_size:(repeat+1)*options.sample_size])
    logger.info('\t loading edges completed in {}s'.format(time.time() - time_start))





def eval_once(options, net):
    global features_dict, true_edges_list_by_repeat, neg_edges_list_by_repeat
    if not utils.check_rebuild(options.link_prediction_path, descrip='link_prediction', always_rebuild=options.always_rebuild):
        return
    logger.info('eval case: link_prediction...')
    logger.info('\t data_dir = {}'.format(options.data_dir))
    logger.info('\t data_name = {}'.format(options.data_name))
    logger.info('\t isdirected = {}'.format(options.isdirected))
    logger.info('\t eval_edge_type: {}'.format(options.eval_edge_type))
    logger.info('\t save_path: {}\n'.format(options.link_prediction_path))
    logger.info('\t classifier: LogisticRegression')
    logger.info('\t eval_online: {}'.format(options.eval_online))
    logger.info('\t eval_workers: {}'.format(options.eval_workers))
    logger.info('\t repeated_times: {}'.format(options.repeated_times))
    logger.info('\t feature_operators: {}'.format(options.feature_operators))
    logger.info('\t sample_size: {}'.format(options.sample_size))

    time_start = time.time()

    load_features(options, net)
    load_edges(options, net)

    logger.info('\t total true edges size: {}'.format(len(true_edges_list_by_repeat[0])))
    logger.info('\t total neg edges size: {}'.format(len(neg_edges_list_by_repeat[0])))

    # repeated 10times
    repeated_times = options.repeated_times
    # split ratio
    if options.train_ratio > 0:
        train_ratio_list = [options.train_ratio]
    else:
        train_ratio_list = [0.01, 0.05] + [v / 10.0 for v in range(1, 10)]

    logger.info('\t repeat {} times for each train_ratio in {}'.format(repeated_times, train_ratio_list))


    fr = open(options.link_prediction_path,'w')
    fr.write('eval case: link-prediction ...\n')
    fr.write('\t data_dir = {}\n'.format(options.data_dir))
    fr.write('\t data_name = {}\n'.format(options.data_name))
    fr.write('\t isdirected = {}\n'.format(options.isdirected))
    fr.write('\t eval_edge_type: {}\n'.format(options.eval_edge_type))
    fr.write('\t save_path: {}\n\n'.format(options.link_prediction_path))
    fr.write('\t classifier: LogisticRegression\n')
    fr.write('\t eval_online: {}\n'.format(options.eval_online))
    fr.write('\t eval_workers: {}\n'.format(options.eval_workers))
    fr.write('\t feature_operators: {}\n'.format(options.feature_operators))
    fr.write('\t repeated_times: {}\n'.format(options.repeated_times))
    fr.write('\t sample_size: {}\n'.format(options.sample_size))
    fr.write('\t total true edges size: {}\n'.format(len(true_edges_list_by_repeat[0])))
    fr.write('\t total neg edges size: {}\n'.format(len(neg_edges_list_by_repeat[0])))
    fr.write('\t repeat {} times for each train_ratio in {}\n'.format(repeated_times, train_ratio_list))


    full_train_ratio_info_list = []
    for repeat in range(repeated_times):
        for op in options.feature_operators:
            for train_ratio in train_ratio_list:
                full_train_ratio_info_list.append((repeat, op, train_ratio))

    if options.eval_workers > 1 and len(full_train_ratio_info_list) > 1:
        # speed up by using multi-process
        if len(full_train_ratio_info_list) <= options.eval_workers:
            train_ratios_per_worker = [ [train_ratio_info] for train_ratio_info in full_train_ratio_info_list]
        else:
            div, mod = divmod(len(full_train_ratio_info_list), options.eval_workers)
            train_ratios_per_worker = [full_train_ratio_info_list[div*i:div*(i+1)] for i in range(options.eval_workers)]
            for idx, train_ratio_info in enumerate(full_train_ratio_info_list[div*options.eval_workers:]):
                train_ratios_per_worker[idx].append(train_ratio_info)
        logger.info("\t using {} processes for evaling:".format(len(train_ratios_per_worker)))
        for idx, train_ratios in enumerate(train_ratios_per_worker):
            logger.info("\t process-{}: {}".format(idx, train_ratios))

        try:
            ret_list = []  # (train_ratio, op, auc)
            with ProcessPoolExecutor(max_workers=options.eval_workers) as executor:
                for ret in executor.map(_classify_thread_body, train_ratios_per_worker):
                    ret_list.extend(ret)
        except:
            logger.warning("concurrent.futures.process failed, retry...")
            time.sleep(10)
            ret_list = []  # (train_ratio, op, auc)
            with ProcessPoolExecutor(max_workers=options.eval_workers) as executor:
                for ret in executor.map(_classify_thread_body, train_ratios_per_worker):
                    ret_list.extend(ret)

    else:
        ret_list = _classify_thread_body(full_train_ratio_info_list)


    ret_dict = {}
    for train_ratio, op, auc in ret_list: # ret: (train_ratio, op, auc)
        if (train_ratio, op) in ret_dict:
            ret_dict[(train_ratio, op)].append(auc)
        else:
            ret_dict[(train_ratio, op)] = [auc]

    for train_ratio in train_ratio_list:
        for op in options.feature_operators:
            fr.write('\n' + '-' * 20 + '\n' + 'train_ratio = {}, operator = {}\n'.format(train_ratio, op))
            auc_list = ret_dict[(train_ratio, op)]
            if len(auc_list) != repeated_times:
                logger.warning("warning: train_ratio={},operator={},, eval unmatched repeated_times: {} != {}".format(train_ratio, op, len(auc_list), repeated_times))
            mean_auc = sum(auc_list) / float(len(auc_list))
            fr.write('expected repeated_times: {}, actual repeated_times: {}, mean results as follows:\n'.format(repeated_times, len(auc_list)))
            fr.write('\t\t AUC = {}\n'.format(mean_auc))
            fr.write('details:\n')
            for repeat in range(len(auc_list)):
                fr.write('\t repeated {}/{}: AUC = {}\n'.format( repeat + 1, len(auc_list), auc_list[repeat]))
    fr.write('\neval case: link_prediction completed in {}s'.format(time.time() - time_start))
    fr.close()
    logger.info('eval case: link_prediction completed in {}s'.format(time.time() - time_start))


def eval_online(options, net):
    global features_dict, true_edges_list_by_repeat, neg_edges_list_by_repeat
    link_prediction_dir = os.path.split(options.link_prediction_path)[0]
    if not utils.check_rebuild(link_prediction_dir, descrip='link_prediction', always_rebuild=options.always_rebuild):
        return
    if not os.path.exists(link_prediction_dir):
        os.makedirs(link_prediction_dir)
    logger.info('eval case: link_prediction ...')
    logger.info('\t data_dir = {}'.format(options.data_dir))
    logger.info('\t data_name = {}'.format(options.data_name))
    logger.info('\t isdirected = {}'.format(options.isdirected))
    logger.info('\t eval_edge_type: {}'.format(options.eval_edge_type))
    logger.info('\t save_dir: {}\n'.format(link_prediction_dir))
    logger.info('\t classifier: LogisticRegression')
    logger.info('\t eval_online: {}'.format(options.eval_online))
    logger.info('\t eval_interval: {}s'.format(options.eval_interval))
    logger.info('\t eval_workers: {}'.format(options.eval_workers))
    logger.info('\t repeated_times: {}'.format(options.repeated_times))
    logger.info('\t feature_operators: {}'.format(options.feature_operators))
    logger.info('\t sample_size: {}'.format(options.sample_size))

    time_start = time.time()

    # load_features(options, net)
    load_edges(options, net)

    logger.info('\t total true edges size: {}'.format(len(true_edges_list_by_repeat[0])))
    logger.info('\t total neg edges size: {}'.format(len(neg_edges_list_by_repeat[0])))

    # repeated 10times
    repeated_times = options.repeated_times
    # split ratio
    if options.train_ratio > 0:
        train_ratio_list = [options.train_ratio]
    else:
        train_ratio_list = [0.01, 0.05] + [v / 10.0 for v in range(1,10)]

    logger.info('\t repeat {} times for each train_ratio in {}'.format(repeated_times, train_ratio_list))


    fr_total = open(options.link_prediction_path, 'w')
    fr_total.write('eval case: link_prediction...\n')
    fr_total.write('\t data_dir = {}\n'.format(options.data_dir))
    fr_total.write('\t data_name = {}\n'.format(options.data_name))
    fr_total.write('\t isdirected = {}\n'.format(options.isdirected))
    fr_total.write('\t eval_edge_type: {}\n'.format(options.eval_edge_type))
    fr_total.write('\t save_dir: {}\n\n'.format(link_prediction_dir))
    fr_total.write('\t classifier: LogisticRegression\n')
    fr_total.write('\t eval_online: {}\n'.format(options.eval_online))
    fr_total.write('\t eval_interval: {}s\n'.format(options.eval_interval))
    fr_total.write('\t eval_workers: {}\n'.format(options.eval_workers))
    fr_total.write('\t feature_operators: {}\n'.format(options.feature_operators))
    fr_total.write('\t repeated_times: {}\n'.format(options.repeated_times))
    fr_total.write('\t sample_size: {}\n'.format(options.sample_size))
    fr_total.write('\t total true edges size: {}\n'.format(len(true_edges_list_by_repeat[0])))
    fr_total.write('\t total neg edges size: {}\n'.format(len(neg_edges_list_by_repeat[0])))
    fr_total.write('\t repeat {} times for each train_ratio in {}\n'.format(repeated_times, train_ratio_list))
    fr_total.write('\t results(AUC):\n=============================================================\n')
    tmp_str = ""
    for train_ratio in train_ratio_list:
        for op in options.feature_operators:
            tmp_str = tmp_str+"\t{}({})".format(train_ratio, op)
    fr_total.write('finish_time\tckpt\t'+tmp_str+"\n")


    full_train_ratio_info_list = []
    for repeat in range(repeated_times):
        for op in options.feature_operators:
            for train_ratio in train_ratio_list:
                full_train_ratio_info_list.append((repeat, op, train_ratio))
    if options.eval_workers > 1 and len(full_train_ratio_info_list) > 1:
        # speed up by using multi-process
        if len(full_train_ratio_info_list) <= options.eval_workers:
            train_ratios_per_worker = [[train_ratio_info] for train_ratio_info in full_train_ratio_info_list]
        else:
            div, mod = divmod(len(full_train_ratio_info_list), options.eval_workers)
            train_ratios_per_worker = [full_train_ratio_info_list[div * i:div * (i + 1)] for i in
                                       range(options.eval_workers)]
            for idx, train_ratio_info in enumerate(full_train_ratio_info_list[div * options.eval_workers:]):
                train_ratios_per_worker[idx].append(train_ratio_info)
        logger.info("\t using {} processes for evaling:".format(len(train_ratios_per_worker)))
        for idx, train_ratios in enumerate(train_ratios_per_worker):
            logger.info("\t process-{}: {}".format(idx, train_ratios))

    last_step = 0
    summary_writer = tf.summary.FileWriter(link_prediction_dir, tf.Graph())
    summary = tf.Summary()
    for train_ratio in train_ratio_list:
        for op in options.feature_operators:
            summary.value.add(tag='auc_{}_{}'.format(train_ratio, op), simple_value=0.)
    summary_writer.add_summary(summary, last_step)

    best_auc = 0

    ckpt_dir = os.path.join(os.path.split(options.vectors_path)[0], 'ckpt')
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    while (not (ckpt and ckpt.model_checkpoint_path)):
        logger.info("\t model and vectors not exist, waiting ...")
        time.sleep(options.eval_interval)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)

    reading = options.vectors_path + ".reading_link_prediction_{}_{}".format(options.eval_edge_type[0], options.eval_edge_type[1])
    writing = options.vectors_path + ".writing"
    while(options.eval_online):
        while True:
            ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            cur_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            ## synchrolock for multi-process:
            # while(not(cur_step > last_step and os.path.exists(options.vectors_path) and
            #                       time.time() - os.stat(options.vectors_path).st_mtime > 200)):
            #     time.sleep(options.eval_interval)
            #     ckpt = tf.train.get_checkpoint_state(ckpt_dir)
            #     cur_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            # os.utime(options.vectors_path, None)
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

        # loading features_matrix(already trained)
        load_features(options, net)

        os.remove(reading)  # synchrolock for multi-process
        logger.info("\t done for reading ...")

        logger.info('\t eval_workers: {}'.format(options.eval_workers))
        logger.info('\t repeated_times: {}'.format(options.repeated_times))
        logger.info('\t feature_operators: {}'.format(options.feature_operators))
        logger.info('\t sample_size: {}'.format(options.sample_size))
        logger.info('\t repeat {} times for each train_ratio in {}'.format(repeated_times, train_ratio_list))
        logger.info('\t total true edges size: {}'.format(len(true_edges_list_by_repeat[0])))
        logger.info('\t total neg edges size: {}'.format(len(neg_edges_list_by_repeat[0])))

        fr = open(options.link_prediction_path+'.{}'.format(cur_step),'w')
        fr.write('eval case: link_prediction ...\n')
        fr.write('\t data_dir = {}\n'.format(options.data_dir))
        fr.write('\t data_name = {}\n'.format(options.data_name))
        fr.write('\t isdirected = {}\n'.format(options.isdirected))
        fr.write('\t eval_edge_type: {}\n'.format(options.eval_edge_type))
        fr.write('\t classifier: LogisticRegression\n')
        fr.write('\t eval_online: {}\n'.format(options.eval_online))
        fr.write('\t eval_workers: {}\n'.format(options.eval_workers))
        fr.write('\t feature_operators: {}\n'.format(options.feature_operators))
        fr.write('\t sample_size: {}\n'.format(options.sample_size))
        fr.write('\t repeated_times: {}\n'.format(options.repeated_times))
        fr.write('\t total true edges size: {}\n'.format(len(true_edges_list_by_repeat[0])))
        fr.write('\t total neg edges size: {}\n'.format(len(neg_edges_list_by_repeat[0])))
        fr.write('\t repeat {} times for each train_ratio in {}\n'.format(repeated_times, train_ratio_list))

        if options.eval_workers > 1 and len(full_train_ratio_info_list) > 1:
            fr.write("\t using {} processes for evaling:\n".format(len(train_ratios_per_worker)))
            for idx, train_ratios in enumerate(train_ratios_per_worker):
                fr.write("\t process-{}: {}\n".format(idx, train_ratios))

            try:
                ret_list = []  # (train_ratio, op, auc)
                with ProcessPoolExecutor(max_workers=options.eval_workers) as executor:
                    for ret in executor.map(_classify_thread_body, train_ratios_per_worker):
                        ret_list.extend(ret)
            except:
                logger.warning("concurrent.futures.process failed, retry...")
                time.sleep(10)
                ret_list = []  # (train_ratio, op, auc)
                with ProcessPoolExecutor(max_workers=options.eval_workers) as executor:
                    for ret in executor.map(_classify_thread_body, train_ratios_per_worker):
                        ret_list.extend(ret)

        else:
            ret_list = _classify_thread_body(full_train_ratio_info_list)


        fr_total.write('%s ckpt-%-9d: '%(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), cur_step))
        summary = tf.Summary()

        ret_dict = {}
        for train_ratio, op, auc in ret_list:  # ret: (train_ratio, op, auc)
            if (train_ratio, op) in ret_dict:
                ret_dict[(train_ratio, op)].append(auc)
            else:
                ret_dict[(train_ratio, op)] = [auc]

        for train_ratio in train_ratio_list:
            for op in options.feature_operators:
                fr.write('\n' + '-' * 20 + '\n' + 'train_ratio = {}, operator = {}\n'.format(train_ratio, op))
                auc_list = ret_dict[(train_ratio, op)]
                if len(auc_list) != repeated_times:
                    logger.warning(
                        "warning: train_ratio={},operator={},, eval unmatched repeated_times: {} != {}".format(
                            train_ratio, op, len(auc_list), repeated_times))
                mean_auc = sum(auc_list) / float(len(auc_list))
                fr.write('expected repeated_times: {}, actual repeated_times: {}, mean results as follows:\n'.format(
                    repeated_times, len(auc_list)))
                fr.write('\t\t AUC = {}\n'.format(mean_auc))
                fr.write('details:\n')
                for repeat in range(len(auc_list)):
                    fr.write('\t repeated {}/{}: AUC = {}\n'.format(repeat + 1, len(auc_list), auc_list[repeat]))
                fr_total.write('%.4f    '%(mean_auc))
                summary.value.add(tag='auc_{}_{}'.format(train_ratio, op), simple_value=mean_auc)
        fr.write('\n eval case: link_prediction completed in {}s\n'.format(time.time() - time_start))
        fr.close()
        fr_total.write('\n')
        fr_total.flush()
        summary_writer.add_summary(summary, cur_step)
        summary_writer.flush()
        logger.info('link_prediction completed in {}s\n================================='.format(time.time() - time_start))

        cur_auc = np.mean(ret_dict[(train_ratio_list[-1],options.feature_operators[-1])])
        # copy ckpt-files according to last mean_Micro_F1 (0.9 ratio).
        if cur_auc > best_auc:
            best_auc = cur_auc

            ckptIsExists = os.path.exists(os.path.join(ckpt_dir, 'model.ckpt-%d.index' % cur_step))
            if ckptIsExists:
                fr_best = open(os.path.join(link_prediction_dir, 'best_ckpt.info'), 'w')
            else:
                fr_best = open(os.path.join(link_prediction_dir, 'best_ckpt.info'), 'a')
                fr_best.write("Note:the model.ckpt-best is the remainings of last best_ckpt!\n"
                              "the current best_ckpt model is loss, but the result is:\n")
            fr_best.write("best_auc(for train_ratio {} and operator {}): {}\n".format(train_ratio_list[-1],options.feature_operators[-1], best_auc))
            fr_best.write("best_ckpt: ckpt-{}\n".format(cur_step))
            fr_best.close()

            if ckptIsExists:
                sourceFile = os.path.join(ckpt_dir, 'model.ckpt-%d.data-00000-of-00001' % cur_step)
                targetFile = os.path.join(link_prediction_dir, 'model.ckpt-best.data-00000-of-00001')
                if os.path.exists(targetFile):
                    os.remove(targetFile)
                shutil.copy(sourceFile, targetFile)
                sourceFile = os.path.join(ckpt_dir, 'model.ckpt-%d.index' % cur_step)
                targetFile = os.path.join(link_prediction_dir, 'model.ckpt-best.index')
                if os.path.exists(targetFile):
                    os.remove(targetFile)
                shutil.copy(sourceFile, targetFile)
                sourceFile = os.path.join(ckpt_dir, 'model.ckpt-%d.meta' % cur_step)
                targetFile = os.path.join(link_prediction_dir, 'model.ckpt-best.meta')
                if os.path.exists(targetFile):
                    os.remove(targetFile)
                shutil.copy(sourceFile, targetFile)

        last_step = cur_step

    fr_total.close()
    summary_writer.close()




def eval(options):
    if "all" not in options.metapath_path:
        metatree = network.construct_meta_tree(metapaths_filename=options.metapath_path)
        flag0 = False
        flag1 = False
        for each in metatree.nodes():
            if options.eval_edge_type[0] == metatree.nodes[each]["type"]:
                flag0 = True
            if options.eval_edge_type[1] == metatree.nodes[each]["type"]:
                flag1 = True
        if not (flag0 and flag1):
            return (flag0 and flag1)
    flag0 = True
    flag1 = True

    net = network.construct_network(options, isHIN=True, print_net_info=False)
    if options.eval_online:
        eval_online(options, net)
    else:
        eval_once(options, net)

    return (flag0 and flag1)


if __name__ == '__main__':
    pass
