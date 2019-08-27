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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
import utils
import eval_utils
import network


# global sharing variable
logger = logging.getLogger("HNE")
features_matrix = None
labels_matrix = None



class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            # all_labels.append(labels)
            probs_ = np.zeros(np.shape(probs_),dtype=np.int32)
            probs_[labels] = 1
            all_labels.append(probs_)
        return np.asarray(all_labels)


def _classify_thread_body(train_ratio_list):
    ret_list = []
    for train_ratio in train_ratio_list:
        time_start = time.time()
        logger.info('\t train_ratio = {}, evaling ...'.format(train_ratio))

        X_train, X_test, Y_train, Y_test = train_test_split(features_matrix, labels_matrix,
                                                                test_size= 1.0 - train_ratio,
                                                                random_state = utils.get_random_seed(),
                                                                shuffle=True)
        # find out how many labels should be predicted
        top_k_list = [np.sum(Y_test[i]) for i in range(np.size(Y_test,axis=0))]
        clf = TopKRanker(LogisticRegression())
        clf.fit(X_train, Y_train)
        preds = clf.predict(X_test, top_k_list)
        # averages = ["micro", "macro", "samples", "weighted"]
        # results[average] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)
        # macro = f1_score(Y_test, preds, average="macro")
        # micro = f1_score(Y_test, preds, average="micro")
        macro, micro = eval_utils.f1_scores_multilabel(Y_test, preds)
        logger.info('\t train_ratio = {}, eval completed in {}s'.format(train_ratio, time.time() - time_start))
        ret_list.append((train_ratio, macro, micro))
    return ret_list



def eval_once(options):
    global features_matrix, labels_matrix
    if not utils.check_rebuild(options.classify_path, descrip='classify', always_rebuild=options.always_rebuild):
        return
    logger.info('eval case: classify...')
    logger.info('\t data_dir = {}'.format(options.data_dir))
    logger.info('\t data_name = {}'.format(options.data_name))
    logger.info('\t isdirected = {}'.format(options.isdirected))
    logger.info('\t label_path = {}'.format(options.label_path))
    logger.info('\t label_size = {}'.format(options.label_size))
    logger.info('\t eval_node_type: {}'.format(options.eval_node_type))
    logger.info('\t save_path: {}\n'.format(options.classify_path))
    logger.info('\t classifier: LogisticRegression')
    logger.info('\t eval_online: {}'.format(options.eval_online))
    logger.info('\t eval_workers: {}'.format(options.eval_workers))
    logger.info('\t reading labeled data from file {}'.format(options.label_path))
    time_start = time.time()
    id_list, labels_list = utils.get_labeled_data(options.label_path, type=options.eval_node_type,
                                                  type_filepath = os.path.join(options.data_dir, options.data_name+".nodes"))
    id_list, features_matrix, labels_list = utils.get_vectors(utils.get_KeyedVectors(options.vectors_path), id_list, labels_list)
    # mlb = MultiLabelBinarizer(range(options.label_size))
    mlb = MultiLabelBinarizer()
    labels_matrix = mlb.fit_transform(labels_list)
    logger.info('\t reading labeled data completed in {}s'.format(time.time() - time_start))
    logger.info('\t total labeled data size: {}'.format(np.size(features_matrix,axis=0)))
    logger.info('\t true labels size: {}'.format(np.size(labels_matrix,axis=1)))
    for i in range(np.size(labels_matrix,axis=1)):
        logger.info('\t\t label {}: {}'.format(i, np.sum(labels_matrix[:,i])))
    # repeated 10times
    repeated_times = options.repeated_times
    # split ratio
    if options.train_ratio > 0:
        train_ratio_list = [options.train_ratio]
    else:
        train_ratio_list = [0.01, 0.05] + [v / 10.0 for v in range(1, 10)]

    logger.info('\t repeat {} times for each train_ratio in {}'.format(repeated_times, train_ratio_list))

    train_ratio_fulllist = [train_ratio for train_ratio in train_ratio_list for _ in range(repeated_times)]


    # classify
    fr = open(options.classify_path,'w')
    fr.write('eval case: classify...\n')
    fr.write('\t data_dir = {}\n'.format(options.data_dir))
    fr.write('\t data_name = {}\n'.format(options.data_name))
    fr.write('\t isdirected = {}\n'.format(options.isdirected))
    fr.write('\t label_path = {}\n'.format(options.label_path))
    fr.write('\t label_size = {}\n'.format(options.label_size))
    fr.write('\t eval_node_type: {}\n'.format(options.eval_node_type))
    fr.write('\t save_path: {}\n\n'.format(options.classify_path))
    fr.write('\t classifier: LogisticRegression\n')
    fr.write('\t eval_online: {}\n'.format(options.eval_online))
    fr.write('\t eval_workers: {}\n'.format(options.eval_workers))
    fr.write('\t total labeled data size: {}\n'.format(np.size(features_matrix, axis=0)))
    fr.write('\t true labels size: {}\n'.format(np.size(labels_matrix, axis=1)))
    for i in range(np.size(labels_matrix, axis=1)):
        fr.write('\t\t label {}: {}\n'.format(i, np.sum(labels_matrix[:, i])))
    fr.write('\t repeat {} times for each train_ratio in {}\n'.format(repeated_times, train_ratio_list))

    if options.eval_workers > 1 and len(train_ratio_fulllist) > 1:
        # speed up by using multi-process
        if len(train_ratio_fulllist) <= options.eval_workers:
            train_ratios_per_worker = [ [train_ratio] for train_ratio in train_ratio_fulllist]
        else:
            div, mod = divmod(len(train_ratio_fulllist), options.eval_workers)
            train_ratios_per_worker = [train_ratio_fulllist[div*i:div*(i+1)] for i in range(options.eval_workers)]
            for idx, train_ratio in enumerate(train_ratio_fulllist[div*options.eval_workers:]):
                train_ratios_per_worker[len(train_ratios_per_worker)-1-idx].append(train_ratio)
        logger.info("\t using {} processes for evaling:".format(len(train_ratios_per_worker)))
        for idx, train_ratios in enumerate(train_ratios_per_worker):
            logger.info("\t process-{}: {}".format(idx, train_ratios))

        try:
            ret_list = []  # (train_ratio, macro, micro)
            with ProcessPoolExecutor(max_workers=options.eval_workers) as executor:
                for ret in executor.map(_classify_thread_body, train_ratios_per_worker):
                    ret_list.extend(ret)
        except:
            logger.warning("concurrent.futures.process failed, retry...")
            time.sleep(10)
            ret_list = []  # (train_ratio, macro, micro)
            with ProcessPoolExecutor(max_workers=options.eval_workers) as executor:
                for ret in executor.map(_classify_thread_body, train_ratios_per_worker):
                    ret_list.extend(ret)

    else:
        ret_list = _classify_thread_body(train_ratio_fulllist)


    ret_dict = {}
    for ret in ret_list:
        if ret[0] in ret_dict:
            ret_dict[ret[0]][0].append(ret[1])
            ret_dict[ret[0]][1].append(ret[2])
        else:
            ret_dict[ret[0]] = [[ret[1]], [ret[2]]]

    for train_ratio, macro_micro in sorted(ret_dict.items(), key=lambda item: item[0]):
        fr.write('\n' + '-' * 20 + '\n' + 'train_ratio = {}\n'.format(train_ratio))
        Macro_F1_list = macro_micro[0]
        Micro_F1_list = macro_micro[1]
        if len(Macro_F1_list) != repeated_times:
            logger.warning("warning: train_ratio = {} eval unmatched repeated_times: {} != {}".format(train_ratio, len(Macro_F1_list), repeated_times))
        mean_Macro_F1 = sum(Macro_F1_list) / float(len(Macro_F1_list))
        mean_Micro_F1 = sum(Micro_F1_list) / float(len(Micro_F1_list))
        fr.write('expected repeated_times: {}, actual repeated_times: {}, mean results as follows:\n'.format(repeated_times, len(Macro_F1_list)))
        fr.write('\t\t Macro_F1 = {}\n'.format(mean_Macro_F1))
        fr.write('\t\t Micro_F1 = {}\n'.format(mean_Micro_F1))
        fr.write('details:\n')
        for repeat in range(len(Macro_F1_list)):
            fr.write('\t repeated {}/{}: Macro_F1 = {}, Micro_F1 = {}\n'.format(
                repeat + 1, len(Macro_F1_list), Macro_F1_list[repeat], Micro_F1_list[repeat]))
    fr.write('\neval case: classify completed in {}s'.format(time.time() - time_start))
    fr.close()
    logger.info('eval case: classify completed in {}s'.format(time.time() - time_start))


def eval_online(options):
    global features_matrix, labels_matrix
    classify_dir = os.path.split(options.classify_path)[0]
    if not utils.check_rebuild(classify_dir, descrip='classify', always_rebuild=options.always_rebuild):
        return
    if not os.path.exists(classify_dir):
        os.makedirs(classify_dir)
    logger.info('eval case: classify...')
    logger.info('\t data_dir = {}'.format(options.data_dir))
    logger.info('\t data_name = {}'.format(options.data_name))
    logger.info('\t isdirected = {}'.format(options.isdirected))
    logger.info('\t label_path = {}'.format(options.label_path))
    logger.info('\t label_size = {}'.format(options.label_size))
    logger.info('\t eval_node_type: {}'.format(options.eval_node_type))
    logger.info('\t save_dir: {}\n'.format(classify_dir))
    logger.info('\t classifier: LogisticRegression')
    logger.info('\t eval_online: {}'.format(options.eval_online))
    logger.info('\t eval_interval: {}s'.format(options.eval_interval))
    logger.info('\t eval_workers: {}'.format(options.eval_workers))

    time_start = time.time()
    logger.info('\t reading labeled data from file {}'.format(options.label_path))
    id_list_totoal, labels_list_total = utils.get_labeled_data(options.label_path, type=options.eval_node_type,
                                                               type_filepath=os.path.join(options.data_dir,
                                                                                          options.data_name + ".nodes"))
    logger.info('\t reading labeled data completed in {}s'.format(time.time() - time_start))

    logger.info('\t total labeled data size: {}'.format(len(id_list_totoal)))
    logger.info('\t total labels size: {}'.format(options.label_size))

    # repeated 10times
    repeated_times = options.repeated_times
    # split ratio
    if options.train_ratio > 0:
        train_ratio_list = [options.train_ratio]
    else:
        train_ratio_list = [0.01, 0.05] + [v / 10.0 for v in range(1, 10)]

    logger.info('\t repeat {} times for each train_ratio in {}'.format(repeated_times, train_ratio_list))

    train_ratio_fulllist = [train_ratio for train_ratio in train_ratio_list for _ in range(repeated_times)]
    if options.eval_workers > 1 and len(train_ratio_fulllist) > 1:
        # speed up by using multi-process
        if len(train_ratio_fulllist) <= options.eval_workers:
            train_ratios_per_worker = [ [train_ratio] for train_ratio in train_ratio_fulllist]
        else:
            div, mod = divmod(len(train_ratio_fulllist), options.eval_workers)
            train_ratios_per_worker = [train_ratio_fulllist[div*i:div*(i+1)] for i in range(options.eval_workers)]
            for idx, train_ratio in enumerate(train_ratio_fulllist[div*options.eval_workers:]):
                train_ratios_per_worker[len(train_ratios_per_worker)-1-idx].append(train_ratio)
        logger.info("\t using {} processes for evaling:".format(len(train_ratios_per_worker)))
        for idx, train_ratios in enumerate(train_ratios_per_worker):
            logger.info("\t process-{}: {}".format(idx, train_ratios))


    fr_total = open(options.classify_path, 'w')
    fr_total.write('eval case: classify...\n')
    fr_total.write('\t data_dir = {}\n'.format(options.data_dir))
    fr_total.write('\t data_name = {}\n'.format(options.data_name))
    fr_total.write('\t isdirected = {}\n'.format(options.isdirected))
    fr_total.write('\t label_path = {}\n'.format(options.label_path))
    fr_total.write('\t label_size = {}\n'.format(options.label_size))
    fr_total.write('\t eval_node_type: {}\n'.format(options.eval_node_type))
    fr_total.write('\t save_dir: {}\n\n'.format(classify_dir))
    fr_total.write('\t classifier: LogisticRegression\n')
    fr_total.write('\t eval_online: {}\n'.format(options.eval_online))
    fr_total.write('\t eval_interval: {}s\n'.format(options.eval_interval))
    fr_total.write('\t eval_workers: {}\n'.format(options.eval_workers))
    fr_total.write('\t total labeled data size: {}\n'.format(len(id_list_totoal)))
    fr_total.write('\t total labels size: {}\n'.format(options.label_size))
    fr_total.write('\t repeat {} times for each train_ratio in {}\n'.format(repeated_times, train_ratio_list))
    fr_total.write('\t results(Macro_F1,Micro_F1):\n=============================================================\n')
    tmp_str = ""
    for train_ratio in train_ratio_list:
        tmp_str = tmp_str + "\t{}".format(train_ratio)
    fr_total.write('finish_time\tckpt\t' + tmp_str + "\n")


    last_step = 0
    summary_writer = tf.summary.FileWriter(classify_dir, tf.Graph())
    summary = tf.Summary()
    for train_ratio in train_ratio_list:
        summary.value.add(tag='macro_train_{}'.format(train_ratio), simple_value=0.)
        summary.value.add(tag='micro_train_{}'.format(train_ratio), simple_value=0.)
    summary_writer.add_summary(summary, last_step)

    best_micro = 0

    ckpt_dir = os.path.join(os.path.split(options.vectors_path)[0], 'ckpt')
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    while (not (ckpt and ckpt.model_checkpoint_path)):
        logger.info("\t model and vectors not exist, waiting ...")
        time.sleep(options.eval_interval)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)

    reading = options.vectors_path + ".reading_classify_{}".format(options.eval_node_type)
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
        logger.info('\t reading embedding vectors from file {}'.format(options.vectors_path))
        id_list, features_matrix, labels_list = utils.get_vectors(utils.get_KeyedVectors(options.vectors_path),
                                                         id_list_totoal, labels_list_total)
        os.remove(reading)  # synchrolock for multi-process
        logger.info("\t done for reading ...")
        mlb = MultiLabelBinarizer()
        labels_matrix = mlb.fit_transform(labels_list)
        logger.info('\t reading embedding vectors completed in {}s'.format(time.time() - time_start))
        logger.info('\t total labeled data size: {}'.format(np.size(features_matrix,axis=0)))
        logger.info('\t true labels size: {}'.format(np.size(labels_matrix, axis=1)))
        for i in range(np.size(labels_matrix, axis=1)):
            logger.info('\t\t label {}: {}'.format(i, np.sum(labels_matrix[:, i])))

        # classify
        fr = open(options.classify_path+'.{}'.format(cur_step),'w')
        fr.write('eval case: classify...\n')
        fr.write('\t classifier: LogisticRegression\n')
        fr.write('\t eval_workers: {}\n'.format(options.eval_workers))
        fr.write('\t repeat {} times for each train_ratio in {}\n'.format(repeated_times, train_ratio_list))
        fr.write('\t total labeled data size: {}\n'.format(np.size(features_matrix, axis=0)))
        fr.write('\t true labels size: {}\n'.format(np.size(labels_matrix, axis=1)))
        for i in range(np.size(labels_matrix, axis=1)):
            fr.write('\t\t label {}: {}\n'.format(i, np.sum(labels_matrix[:, i])))

        if options.eval_workers > 1 and len(train_ratio_fulllist) > 1:
            fr.write("\t using {} processes for evaling:\n".format(len(train_ratios_per_worker)))
            for idx, train_ratios in enumerate(train_ratios_per_worker):
                fr.write("\t process-{}: {}\n".format(idx, train_ratios))

            try:
                ret_list = []  # (train_ratio, macro, micro)
                with ProcessPoolExecutor(max_workers=options.eval_workers) as executor:
                    for ret in executor.map(_classify_thread_body, train_ratios_per_worker):
                        ret_list.extend(ret)
            except:
                logger.warning("concurrent.futures.process failed, retry...")
                time.sleep(10)
                ret_list = []  # (train_ratio, macro, micro)
                with ProcessPoolExecutor(max_workers=options.eval_workers) as executor:
                    for ret in executor.map(_classify_thread_body, train_ratios_per_worker):
                        ret_list.extend(ret)

        else:
            ret_list = _classify_thread_body(train_ratio_fulllist)


        fr_total.write('%s ckpt-%-9d: '%(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), cur_step))
        summary = tf.Summary()

        ret_dict = {}
        for ret in ret_list:
            if ret[0] in ret_dict:
                ret_dict[ret[0]][0].append(ret[1])
                ret_dict[ret[0]][1].append(ret[2])
            else:
                ret_dict[ret[0]] = [[ret[1]], [ret[2]]]

        for train_ratio in train_ratio_list:
            macro_micro = ret_dict[train_ratio]
            fr.write('\n' + '-' * 20 + '\n' + 'train_ratio = {}\n'.format(train_ratio))
            Macro_F1_list = macro_micro[0]
            Micro_F1_list = macro_micro[1]
            if len(Macro_F1_list) != repeated_times:
                logger.warning("warning: train_ratio = {} eval unmatched repeated_times: {} != {}".format(train_ratio, len(Macro_F1_list), repeated_times))
            mean_Macro_F1 = sum(Macro_F1_list) / float(len(Macro_F1_list))
            mean_Micro_F1 = sum(Micro_F1_list) / float(len(Micro_F1_list))
            fr.write('expected repeated_times: {}, actual repeated_times: {}, mean results as follows:\n'.format(repeated_times, len(Macro_F1_list)))
            fr.write('\t\t Macro_F1 = {}\n'.format(mean_Macro_F1))
            fr.write('\t\t Micro_F1 = {}\n'.format(mean_Micro_F1))
            fr.write('details:\n')
            for repeat in range(len(Macro_F1_list)):
                fr.write('\t repeated {}/{}: Macro_F1 = {}, Micro_F1 = {}\n'.format(
                    repeat + 1, len(Macro_F1_list), Macro_F1_list[repeat], Micro_F1_list[repeat]))
            fr_total.write('%.4f, %.4f    '%(mean_Macro_F1, mean_Micro_F1))
            summary.value.add(tag='macro_train_{}'.format(train_ratio), simple_value=mean_Macro_F1)
            summary.value.add(tag='micro_train_{}'.format(train_ratio), simple_value=mean_Micro_F1)

        fr.write('\n eval case: classify completed in {}s\n'.format(time.time() - time_start))
        fr.close()
        fr_total.write('\n')
        fr_total.flush()
        summary_writer.add_summary(summary, cur_step)
        summary_writer.flush()
        logger.info('classify completed in {}s\n================================='.format(time.time() - time_start))

        # copy ckpt-files according to last mean_Micro_F1 (0.9 ratio).
        if mean_Micro_F1 > best_micro:
            best_micro = mean_Micro_F1

            ckptIsExists = os.path.exists(os.path.join(ckpt_dir, 'model.ckpt-%d.index' % cur_step))
            if ckptIsExists:
                fr_best = open(os.path.join(classify_dir, 'best_ckpt.info'), 'w')
            else:
                fr_best = open(os.path.join(classify_dir, 'best_ckpt.info'), 'a')
                fr_best.write("Note:the model.ckpt-best is the remainings of last best_ckpt!\n"
                              "the current best_ckpt model is loss, but the result is:\n")
            fr_best.write("best_micro(for train_ratio 0.9): {}\n".format(best_micro))
            fr_best.write("best_ckpt: ckpt-{}\n".format(cur_step))
            fr_best.close()

            if ckptIsExists:
                sourceFile = os.path.join(ckpt_dir, 'model.ckpt-%d.data-00000-of-00001' % cur_step)
                targetFile = os.path.join(classify_dir, 'model.ckpt-best.data-00000-of-00001')
                if os.path.exists(targetFile):
                    os.remove(targetFile)
                shutil.copy(sourceFile, targetFile)
                sourceFile = os.path.join(ckpt_dir, 'model.ckpt-%d.index' % cur_step)
                targetFile = os.path.join(classify_dir, 'model.ckpt-best.index')
                if os.path.exists(targetFile):
                    os.remove(targetFile)
                shutil.copy(sourceFile, targetFile)
                sourceFile = os.path.join(ckpt_dir, 'model.ckpt-%d.meta' % cur_step)
                targetFile = os.path.join(classify_dir, 'model.ckpt-best.meta')
                if os.path.exists(targetFile):
                    os.remove(targetFile)
                shutil.copy(sourceFile, targetFile)

        last_step = cur_step

    fr_total.close()
    summary_writer.close()



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
    mlb = MultiLabelBinarizer()
    labels_matrix = mlb.fit_transform([[1,5,7],[1,10],[2]])
    print(labels_matrix)
