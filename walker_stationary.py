#! /usr/bin/env python
# -*- coding:utf-8 -*-
#====#====#====#====
# __author__ = "He Yu"
# Version: 1.0.0
#====#====#====#====

import logging
import sys
import os
import gc
import time
import random
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import pickle
import scipy.stats


import network
import utils


logger = logging.getLogger("HNE")

# global sharing variable
walker = None


class Walker(object):
    """random walker on the network."""
    def __init__(self, net, random_walker = 'spacey', walk_length = 100, walk_restart = 0, task = "train", alpha = None,
                 distortion_power = 0, neg_sampled = 5, metagraph = None, using_metapath = "metaschema", history_position = "global",
                 walk_times = -1):
        self._net = net
        self._random_walker = random_walker
        self._walk_length = walk_length
        self._walk_restart = walk_restart
        self._distortion_power = distortion_power
        self._neg_sampled = neg_sampled
        self._metagraph = metagraph
        self._history_position = history_position
        self._using_metapath = using_metapath
        self._task = task
        self._alpha = alpha
        self._walk_times = walk_times

        if self._net.isHIN:
            self.node_types_size = self._net.get_node_types_size()
            self.edge_types_size = self._net.get_edge_types_size()

        self.nodes_size = self._net.get_nodes_size()
        self.edges_size = self._net.get_edges_size()

        if self._random_walker == 'uniform': # Deepwalk
            self.random_walk = self._uniform_random_walk
        elif self._random_walker == 'spacey':
            self.preprocess_nodesdegrees()
            self.preprocess_nodesadjs()
            if self._history_position == "global":
                if self._using_metapath == "metatree":
                    self._history = np.ones([len(self._metagraph.nodes())], dtype=np.float64)
                else:
                    self._history = np.ones([self.node_types_size], dtype=np.float64)
            if self._using_metapath == "metatree":
                self._metatree_type_id_dict = {}
                for each_id in self._metagraph.nodes():
                    each_type = self._metagraph.nodes[each_id]["type"]
                    if each_type in self._metatree_type_id_dict:
                        self._metatree_type_id_dict[each_type].append(each_id)
                    else:
                        self._metatree_type_id_dict[each_type] = [each_id]
            if self._task == "train":
                if self._using_metapath == "metagraph":  # spacey random walk using metapath or metagraph
                    self.random_walk = self._spacey_metagraph_random_walk
                elif self._using_metapath == "metatree":  # spacey random walk using metapath or metagraph
                    self.random_walk = self._spacey_metatree_random_walk
                elif self._using_metapath == "metaschema":
                    self.random_walk = self._spacey_metaschema_random_walk
                else:
                    self.random_walk = None
            elif self._task == "walk":
                if self._using_metapath == "metagraph":  # spacey random walk using metapath or metagraph
                    self.random_walk = self._spacey_metagraph_only_random_walk
                elif self._using_metapath == "metatree":  # spacey random walk using metapath or metagraph
                    self.random_walk = self._spacey_metatree_only_random_walk
                elif self._using_metapath == "metaschema":
                    self.random_walk = self._spacey_metaschema_only_random_walk
                else:
                    self.random_walk = None
        elif self._random_walker == 'metatreewalk':
            self.preprocess_nodesdegrees()
            self.preprocess_nodesadjs()
            if self._history_position == "global":
                self._history = np.ones([len(self._metagraph.nodes())], dtype=np.float64)
            self._metatree_type_id_dict = {}
            for each_id in self._metagraph.nodes():
                each_type = self._metagraph.nodes[each_id]["type"]
                if each_type in self._metatree_type_id_dict:
                    self._metatree_type_id_dict[each_type].append(each_id)
                else:
                    self._metatree_type_id_dict[each_type] = [each_id]
            if self._task == "train":
                pass
            elif self._task == "walk":
                self.random_walk = self._metatree_only_random_walk
        else:
            self.random_walk = None

    def _spacey_metaschema_only_random_walk(self, root_node, file_vector, file_dist, window_size):
        root_type = self._net.get_node_type(root_node)
        if self._history_position == "local":
            history = np.ones([self.node_types_size], dtype=np.float64)
        elif self._history_position == "global":
            history = self._history

        occur_count_vector = np.zeros(self.nodes_size, dtype=np.int32)
        occur_count_vector[root_node] = 1
        last_distribution = occur_count_vector / np.sum(occur_count_vector)
        fr_dist = open(file_dist, "w")

        cur_node = root_node
        cur_type = root_type
        for wl_cnt in range(1, self._walk_length + 1):
            next_type_list = list(self._adj_lookupdict[cur_node].keys())
            if random.random() < self._alpha:
                if len(next_type_list) == 0:
                    cur_type = root_type
                    cur_node = root_node
                elif len(next_type_list) == 1:
                    cur_type = next_type_list[0]
                    cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
                else:
                    occupancy = history[next_type_list]
                    cur_type = utils.unigram_sample(population = next_type_list, size=1, weight=occupancy)[0]
                    cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
                history[cur_type] += 1
            else:
                if len(next_type_list) == 0:
                    cur_type = root_type
                    cur_node = root_node
                else:
                    cur_type = random.choice(next_type_list)
                    cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
                history[cur_type] += 1
            occur_count_vector[cur_node] += 1
            if wl_cnt % window_size == 0:
                cur_distribution = occur_count_vector / np.sum(occur_count_vector)
                mix_distribution = (cur_distribution + last_distribution) / 2
                JS_dist = (scipy.stats.entropy(cur_distribution, mix_distribution) + scipy.stats.entropy(last_distribution,
                                                                                                         mix_distribution)) / 2
                EUC_dist = np.sqrt(np.sum(np.square(cur_distribution - last_distribution)))
                fr_dist.write("{} {}\n".format(JS_dist, EUC_dist))
                fr_dist.flush()
                last_distribution = cur_distribution
        fr_dist.close()
        np.savetxt(file_vector, occur_count_vector, fmt="%d")
        return
    def _spacey_metatree_only_random_walk(self, root_node, file_vector, file_dist, window_size): # metapath, multi-metapath, metagraph
        root_type = self._net.get_node_type(root_node)
        if root_type not in self._metatree_type_id_dict:
            return []
        root_id = random.choice(self._metatree_type_id_dict[root_type])
        if self._history_position == "local":
            history = np.ones([len(self._metagraph.nodes())], dtype=np.float64)
        elif self._history_position == "global":
            history = self._history

        occur_count_vector = np.zeros(self.nodes_size, dtype=np.int32)
        occur_count_vector[root_node] = 1
        last_distribution = occur_count_vector / np.sum(occur_count_vector)
        fr_dist = open(file_dist, "w")

        # current node
        cur_node = root_node
        cur_type = root_type
        cur_id = root_id
        for wl_cnt in range(1, self._walk_length + 1):
            if random.random() < self._alpha:
                cur_node_adj_typelist = self._adj_lookupdict[cur_node].keys()
                next_id_list = [v for v in self._metagraph[cur_id] if
                                self._metagraph.nodes[v]["type"] in cur_node_adj_typelist]
                if len(next_id_list) == 0:
                    cur_node = root_node
                    cur_type = root_type
                    cur_id = root_id
                elif len(next_id_list) == 1:
                    cur_id = next_id_list[0]
                    cur_type = self._metagraph.nodes[cur_id]["type"]
                    cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
                else:
                    occupancy = history[next_id_list]
                    cur_id = utils.unigram_sample(population=next_id_list, size=1, weight=occupancy)[0]
                    cur_type = self._metagraph.nodes[cur_id]["type"]
                    cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
                # logger.info('next: %d %d' % (cur_type, cur_id))
                history[cur_id] += 1
                # spacey out
                cur_id = utils.unigram_sample(population=self._metatree_type_id_dict[cur_type], size=1,
                                              weight=history[self._metatree_type_id_dict[cur_type]])[0]
            else:
                cur_node_adj_typelist = self._adj_lookupdict[cur_node].keys()
                next_id_list = [v for v in self._metagraph[cur_id] if
                                self._metagraph.nodes[v]["type"] in cur_node_adj_typelist]
                if len(next_id_list) == 0:
                    cur_node = root_node
                    cur_type = root_type
                    cur_id = root_id
                else:
                    cur_id = random.choice(next_id_list)
                    cur_type = self._metagraph.nodes[cur_id]["type"]
                    cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
                history[cur_id] += 1
            occur_count_vector[cur_node] += 1
            if wl_cnt % window_size == 0:
                cur_distribution = occur_count_vector / np.sum(occur_count_vector)
                mix_distribution = (cur_distribution + last_distribution) / 2
                JS_dist = (scipy.stats.entropy(cur_distribution, mix_distribution) + scipy.stats.entropy(last_distribution,
                                                                                                         mix_distribution)) / 2
                EUC_dist = np.sqrt(np.sum(np.square(cur_distribution - last_distribution)))
                fr_dist.write("{} {}\n".format(JS_dist, EUC_dist))
                fr_dist.flush()
                last_distribution = cur_distribution
        fr_dist.close()
        np.savetxt(file_vector, occur_count_vector, fmt="%d")
        return
    def _metatree_only_random_walk(self, root_node, file_vector, file_dist, window_size):
        root_type = self._net.get_node_type(root_node)
        if root_type not in self._metatree_type_id_dict:
            return []
        root_id = random.choice(self._metatree_type_id_dict[root_type])
        if self._history_position == "local":
            history = np.ones([len(self._metagraph.nodes())], dtype=np.float64)
        elif self._history_position == "global":
            history = self._history

        occur_count_vector = np.zeros(self.nodes_size, dtype=np.int32)
        occur_count_vector[root_node] = 1
        last_distribution = occur_count_vector / np.sum(occur_count_vector)
        fr_dist = open(file_dist, "w")

        # current node
        cur_node = root_node
        cur_type = root_type
        cur_id = root_id
        for wl_cnt in range(1, self._walk_length+1):
            cur_node_adj_typelist = self._adj_lookupdict[cur_node].keys()
            next_id_list = [v for v in self._metagraph[cur_id] if
                            self._metagraph.nodes[v]["type"] in cur_node_adj_typelist]
            if len(next_id_list) == 0:
                cur_node = root_node
                cur_type = root_type
                cur_id = root_id
            else:
                cur_id = random.choice(next_id_list)
                cur_type = self._metagraph.nodes[cur_id]["type"]
                cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
            # logger.info('next: %d %d' % (cur_type, cur_id))
            history[cur_id] += 1
            occur_count_vector[cur_node] += 1
            if wl_cnt % window_size == 0:
                cur_distribution = occur_count_vector / np.sum(occur_count_vector)
                # if np.sum(cur_distribution) != 1.:
                #     cur_distribution = cur_distribution / np.sum(cur_distribution)
                # if np.sum(cur_distribution) != 1.:
                #     cur_distribution = cur_distribution / np.sum(cur_distribution)
                # https://blog.csdn.net/hfut_jf/article/details/71403741
                # http://www.cnblogs.com/zhangchaoyang/articles/7103888.html

                # KL = scipy.stats.entropy(cur_distribution, last_distribution)
                mix_distribution = (cur_distribution + last_distribution) / 2
                JS_dist = (scipy.stats.entropy(cur_distribution, mix_distribution) + scipy.stats.entropy(last_distribution, mix_distribution))/2
                EUC_dist = np.sqrt(np.sum(np.square(cur_distribution-last_distribution)))
                fr_dist.write("{} {}\n".format(JS_dist, EUC_dist))
                fr_dist.flush()
                last_distribution = cur_distribution
        fr_dist.close()
        np.savetxt(file_vector, occur_count_vector, fmt="%d")
        return

    def preprocess_nodesdegrees(self):
        time_start = time.time()
        logger.info("preprocessing nodesdegrees with distortion_power = {} ...".format(self._distortion_power))
        self._nodes_type_dict = {}
        for node_type in self._net.node_types:
            self._nodes_type_dict[node_type] = [[], None]  # [nodes, nodes_degrees]
        for node in self._net.nodes:
            node_type = self._net.get_node_type(node)
            self._nodes_type_dict[node_type][0].append(node)
            # self._nodes_type_dict[node_type][1].append(net.get_degrees(node))
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
    def preprocess_nodesadjs(self):
        time_start = time.time()
        logger.info("preprocessing nodesadjs ...")
        self._adj_lookupdict = {}
        for node in self._net.nodes:
            # node_type = self._net.get_node_type(node)
            self._adj_lookupdict[node] = {}
            for adj in self._net.get_neighbors(node):
                adj_type = self._net.get_node_type(adj)
                if adj_type in self._adj_lookupdict[node]:
                    self._adj_lookupdict[node][adj_type].append(adj)
                else:
                    self._adj_lookupdict[node][adj_type] = [adj]
        logger.info('nodesadjs processed in {}s'.format(time.time() - time_start))






def _construct_walk_corpus_and_write_singprocess(args):
    global walker
    corpus_store_dir, node, wt_from, wt_to, window_size = args
    logger.info('\t new walk process node={}, wt_from={},wt_to={}'.format(node, wt_from, wt_to))
    time_start = time.time()
    for wt in range(wt_from, wt_to+1):
        corpus_store_dir_wt = os.path.join(corpus_store_dir, str(wt))
        if not os.path.exists(corpus_store_dir_wt):
            os.mkdir(corpus_store_dir_wt)
        file_vector = os.path.join(corpus_store_dir_wt, str(node)+".stat")
        file_dist = os.path.join(corpus_store_dir_wt, str(node)+".dist")
        if os.path.exists(file_vector):
            logger.info('\t walk node={}, wt={} already finished, skiped!'.format(node, wt))
            return
        walker.random_walk(node, file_vector, file_dist, window_size)
    logger.info('\t !walk ended, node={}, wt_from={},wt_to={}, using {} seconds'.format(node, wt_from, wt_to, time.time()-time_start))
    return



# walk sentences
def build_walk_corpus(options):
    global walker

    # check walk info  and record
    if not utils.check_rebuild(options.corpus_store_path, descrip='walk corpus',
                              always_rebuild=options.always_rebuild):
        return
    if options.model == "DeepWalk":
        random_walker = "uniform"
        net = network.construct_network(options, isHIN=False)
    elif options.model == "SpaceyWalk":
        random_walker = "spacey"
        net = network.construct_network(options, isHIN=True)
    elif options.model == "MetatreeWalk":
        random_walker = "metatreewalk"
        net = network.construct_network(options, isHIN=True)
    else:
        logger.error("Unknown model or it cann't build walk corpus: '%s'." % options.model)
        sys.exit()

    logger.info('Corpus bulid: walk info:')
    logger.info('\t data_dir = {}'.format(options.data_dir))
    logger.info('\t data_name = {}'.format(options.data_name))
    logger.info('\t isdirected = {}\n'.format(options.isdirected))
    logger.info('\t random_walker = {}'.format(random_walker))
    logger.info('\t walk_times = {}'.format(options.walk_times))
    logger.info('\t walk_length = {}'.format(options.walk_length))
    logger.info('\t max_walk_workers = {}'.format(options.walk_workers))
    logger.info('\t walk_to_memory = {}'.format(options.walk_to_memory))
    logger.info('\t seed = {}'.format(options.seed))
    logger.info('\t alpha = {}'.format(options.alpha))
    logger.info('\t window_size = {}'.format(options.window_size))
    logger.info('\t sample_size = {}'.format(options.sample_size))
    if options.walk_to_memory:
        logger.info('\t donot store corpus = {}'.format(str(options.not_store_corpus)))
        if not options.not_store_corpus:
            logger.info('\t corpus store path = {}'.format(options.corpus_store_path))
    else:
        logger.info('\t corpus store path = {}'.format(options.corpus_store_path))

    fr_walks = open(os.path.join(os.path.split(options.corpus_store_path)[0], 'walks.info'), 'w')
    fr_walks.write('Corpus walk info:\n')
    fr_walks.write('\t data_dir = {}\n'.format(options.data_dir))
    fr_walks.write('\t data_name = {}\n'.format(options.data_name))
    fr_walks.write('\t isdirected = {}\n\n'.format(options.isdirected))
    fr_walks.write('\t random_walker = {}\n'.format(random_walker))
    fr_walks.write('\t walk times = {}\n'.format(options.walk_times))
    fr_walks.write('\t walk length = {}\n'.format(options.walk_length))
    fr_walks.write('\t max walk workers = {}\n'.format(options.walk_workers))
    fr_walks.write('\t seed = {}\n'.format(options.seed))
    fr_walks.write('\t alpha = {}\n'.format(options.alpha))
    fr_walks.write('\t window_size = {}\n'.format(options.window_size))
    fr_walks.write('\t sample_size = {}\n'.format(options.sample_size))
    fr_walks.write('\t walk to memory = {}\n'.format(str(options.walk_to_memory)))
    if options.walk_to_memory:
        fr_walks.write('\t donot store corpus = {}\n'.format(str(options.not_store_corpus)))
        if not options.not_store_corpus:
            fr_walks.write('\t corpus store path = {}\n'.format(options.corpus_store_path))
    else:
        fr_walks.write('\t corpus store path = {}\n'.format(options.corpus_store_path))
    fr_walks.close()


    if options.model == "SpaceyWalk":
        if options.using_metapath == "metagraph":
            metagraph = network.construct_meta_graph(options.metapath_path, isdirected=options.isdirected)
        elif options.using_metapath == "metatree":
            metagraph = network.construct_meta_tree(options.metapath_path, isdirected=True)
        elif options.using_metapath == "metaschema":
            metagraph = None
        else:
            logger.error("Unknown feature : '%s'." % options.using_metapath)
            sys.exit()
        walker = Walker(net, random_walker=random_walker, walk_length=options.walk_length,
                        metagraph=metagraph, using_metapath=options.using_metapath,
                        history_position=options.history_position, task = "walk", alpha=options.alpha)
    elif options.model == "MetatreeWalk":
        metagraph = network.construct_meta_tree(options.metapath_path, isdirected=True)
        walker = Walker(net, random_walker=random_walker, walk_length=options.walk_length,
                        metagraph=metagraph, task = "walk")

    corpus_store_dir = os.path.split(options.corpus_store_path)[0]
    if not os.path.exists(corpus_store_dir):
        os.makedirs(corpus_store_dir)

    logger.info('Corpus bulid: walking and computing (using %d workers for multi-process)...' % options.walk_workers)
    time_start = time.time()


    if options.walk_times <= options.walk_workers:
        times_per_worker = [1 for _ in range(options.walk_times)]
    else:
        div, mod = divmod(options.walk_times, options.walk_workers)
        times_per_worker = [div for _ in range(options.walk_workers)]
        for idx in range(mod):
            times_per_worker[idx] = times_per_worker[idx] + 1
    assert sum(times_per_worker) == options.walk_times, 'workers allocating failed: %d != %d' % (
        sum(times_per_worker), options.walk_times)

    nodes_total = list(range(walker.nodes_size))
    sp_random = random.Random(options.seed)
    sp_random.shuffle(nodes_total)
    nodes_total = nodes_total[0:options.sample_size]
    nodes_total.insert(0, 8407)
    nodes_total.insert(0, 9891)
    nodes_total.insert(0, 8354)
    nodes_total.insert(0, 8798)
    for node in nodes_total:
        args_list = []
        begin = 0
        for cnt in times_per_worker:
            args_list.append((corpus_store_dir, node, begin+1, begin+cnt, options.window_size))
            begin += cnt
        with ProcessPoolExecutor(max_workers=options.walk_workers) as executor:
            executor.map(_construct_walk_corpus_and_write_singprocess, args_list)
    logger.info('Corpus bulid: walk completed in {}s'.format(time.time() - time_start))
    del walker
    gc.collect()
    return
