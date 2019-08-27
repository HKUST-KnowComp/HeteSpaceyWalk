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

import network
import utils


logger = logging.getLogger("HNE")

# global sharing variable
walker = None


class Walker(object):
	"""random walker on the network."""
	def __init__(self, net, random_walker = 'spacey', walk_length = 100, walk_restart = 0, task = "train", alpha = None,
				 distortion_power = 0, neg_sampled = 5, metagraph = None, using_metapath = "metaschema", history_position = "global"):
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

	def _uniform_random_walk(self, start_node = None):
		"""truncated uniform random walk used in deepwalk model."""
		if start_node == None:
			# Sampling is uniform w.r.t V, and not w.r.t E
			start_node = random.choice(range(self.nodes_size))
		path = [start_node]
		while len(path) < self._walk_length:
			#if random.random() < self._walk_restart:
			#    path.append(start_node)
			#    continue
			cur = path[-1]
			adj_list = self._net.get_adj_list(cur)
			if len(adj_list) > 0:
				path.append(random.choice(adj_list)) # Generate a uniform random sample
			else:
				# logger.warning('no type-corresponding node found, walk discontinued, generate a path less than specified length.')
				# break
				# logger.warning('no type-corresponding node found, walk restarted.')
				path.append(start_node)

		return [str(node) for node in path]
	def _spacey_metaschema_only_random_walk(self, root_node):
		root_type = self._net.get_node_type(root_node)
		if self._history_position == "local":
			history = np.ones([self.node_types_size], dtype=np.float64)
		elif self._history_position == "global":
			history = self._history
		# current node
		cur_node = root_node
		cur_type = root_type
		path = [str(cur_node)]
		for __ in range(self._walk_length):
			# logger.info("history={}".format(history))
			# choose next type
			if random.random() < self._walk_restart:
				cur_node = root_node
				cur_type = root_type
			next_type_list = list(self._adj_lookupdict[cur_node].keys())
			if random.random() < self._alpha:
				if len(next_type_list) == 0:
					break
					# cur_type = root_type
					# cur_node = root_node
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
					break
					# cur_type = root_type
					# cur_node = root_node
				else:
					cur_type = random.choice(next_type_list)
					cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
				history[cur_type] += 1
			path.append(str(cur_node))
		return path
	def _spacey_metagraph_only_random_walk(self, root_node): # metapath, multi-metapath, metagraph
		root_type = self._net.get_node_type(root_node)
		if root_type not in self._metagraph:
			# root_type = random.choice(list(self._metagraph.nodes))
			# root_node = random.choice(self._nodes_type_dict[root_type][0])
			return []
		if self._history_position == "local":
			history = np.ones([self.node_types_size], dtype=np.float64)
		elif self._history_position == "global":
			history = self._history
		# current node
		cur_node = root_node
		cur_type = root_type
		path = [str(cur_node)]
		for __ in range(self._walk_length):
			# logger.info("history={}".format(history))
			# choose next type
			if random.random() < self._walk_restart:
				cur_node = root_node
				cur_type = root_type
			if random.random() < self._alpha:
				cur_node_adj_typelist = self._adj_lookupdict[cur_node].keys()
				next_type_list = [v for v in self._metagraph[cur_type] if v in cur_node_adj_typelist]
				if len(next_type_list) == 0:
					# cur_type = root_type
					# cur_node = root_node
					break
				elif len(next_type_list) == 1:
					cur_type = next_type_list[0]
					cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
				else:
					occupancy = history[next_type_list]
					cur_type = utils.unigram_sample(population = next_type_list, size=1, weight=occupancy)[0]
					cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
				history[cur_type] += 1
			else:
				cur_node_adj_typelist = self._adj_lookupdict[cur_node].keys()
				next_type_list = [v for v in self._metagraph[cur_type] if v in cur_node_adj_typelist]
				if len(next_type_list) == 0:
					# cur_type = root_type
					# cur_node = root_node
					break
				else:
					cur_type = random.choice(next_type_list)
					cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
				history[cur_type] += 1
			path.append(str(cur_node))
		return path
	def _spacey_metatree_only_random_walk(self, root_node): # metapath, multi-metapath, metagraph
		root_type = self._net.get_node_type(root_node)
		if root_type not in self._metatree_type_id_dict:
			return []
		root_id = random.choice(self._metatree_type_id_dict[root_type])
		if self._history_position == "local":
			history = np.ones([len(self._metagraph.nodes())], dtype=np.float64)
		elif self._history_position == "global":
			history = self._history
		# current node
		cur_node = root_node
		cur_type = root_type
		cur_id = root_id
		path = [str(cur_node)]
		for __ in range(self._walk_length):
			# logger.info("history={}".format(history))
			# choose next type
			if random.random() < self._walk_restart:
				cur_node = root_node
				cur_type = root_type
				cur_id = root_id
			if random.random() < self._alpha:
				cur_node_adj_typelist = self._adj_lookupdict[cur_node].keys()
				next_id_list = [v for v in self._metagraph[cur_id] if
								self._metagraph.nodes[v]["type"] in cur_node_adj_typelist]
				if len(next_id_list) == 0:
					break
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
					break
				else:
					cur_id = random.choice(next_id_list)
					cur_type = self._metagraph.nodes[cur_id]["type"]
					cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
				history[cur_id] += 1
				# logger.info('next: %d %d' % (cur_type, cur_id))
				# spacey out
			path.append(str(cur_node))
		return path
	def _metatree_only_random_walk(self, root_node):
		root_type = self._net.get_node_type(root_node)
		if root_type not in self._metatree_type_id_dict:
			return []
		root_id = random.choice(self._metatree_type_id_dict[root_type])
		if self._history_position == "local":
			history = np.ones([len(self._metagraph.nodes())], dtype=np.float64)
		elif self._history_position == "global":
			history = self._history
		# current node
		cur_node = root_node
		cur_type = root_type
		cur_id = root_id
		path = [str(cur_node)]
		for __ in range(self._walk_length):
			# logger.info("history={}".format(history))
			# choose next type
			if random.random() < self._walk_restart:
				cur_node = root_node
				cur_type = root_type
				cur_id = root_id
			cur_node_adj_typelist = self._adj_lookupdict[cur_node].keys()
			next_id_list = [v for v in self._metagraph[cur_id] if
							self._metagraph.nodes[v]["type"] in cur_node_adj_typelist]
			if len(next_id_list) == 0:
				break
			else:
				cur_id = random.choice(next_id_list)
				cur_type = self._metagraph.nodes[cur_id]["type"]
				cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
			# logger.info('next: %d %d' % (cur_type, cur_id))
			history[cur_id] += 1
			path.append(str(cur_node))
		return path
	def _spacey_metaschema_random_walk(self, root_node, walk_times = 0):
		root_type = self._net.get_node_type(root_node)
		context_nodes_dict = {} #
		if self._history_position == "local":
			history = np.ones([self.node_types_size], dtype=np.float64)
		elif self._history_position == "global":
			history = self._history
		for _ in range(walk_times):
			if self._history_position == "local_walktime":
				history = np.ones([self.node_types_size], dtype=np.float64)
			# current node
			cur_node = root_node
			cur_type = root_type
			for __ in range(self._walk_length):
				# choose next type
				if random.random() < self._walk_restart:
					cur_node = root_node
					cur_type = root_type
				next_type_list = list(self._adj_lookupdict[cur_node].keys())
				if len(next_type_list) == 0:
					cur_type = root_type
					cur_node = root_node
				elif len(next_type_list) == 1:
					cur_type = next_type_list[0]
					cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
					history[cur_type] += 1
				else:
					occupancy = history[next_type_list]
					cur_type = utils.unigram_sample(population = next_type_list, size=1, weight=occupancy)[0]
					cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
					history[cur_type] += 1
				if cur_type in context_nodes_dict:
					context_nodes_dict[cur_type][0].append(cur_node) # context_list
					context_nodes_dict[cur_type][1].add(cur_node) # except_set
				else:
					context_nodes_dict[cur_type] = [[cur_node], {cur_node, root_node}]
		type_context_nodes_list = []
		type_neg_nodes_list = []
		type_mask_list = []
		for k in range(self.node_types_size):
			if k in context_nodes_dict:
				context_nodes = context_nodes_dict[k][0]
				except_set = context_nodes_dict[k][1]
				type_mask_list.append(1)
				type_context_nodes_list.append(context_nodes)
				type_neg_nodes_list.append(utils.neg_sample(self._nodes_type_dict[k][0], except_set,
															num=self._neg_sampled,
															alias_table=self._nodes_type_dict[k][1]))
			else:
				type_mask_list.append(0)
				type_context_nodes_list.append([0])
				type_neg_nodes_list.append([0])
		return root_node, type_context_nodes_list, type_mask_list, type_neg_nodes_list
	def _spacey_metagraph_random_walk(self, root_node, walk_times = 0): # metapath, multi-metapath, metagraph
		root_type = self._net.get_node_type(root_node)
		if root_type not in self._metagraph:
			root_type = random.choice(list(self._metagraph.nodes))
			root_node = random.choice(self._nodes_type_dict[root_type][0])
		context_nodes_dict = {} #
		if self._history_position == "local":
			history = np.ones([self.node_types_size], dtype=np.float64)
		elif self._history_position == "global":
			history = self._history
		for _ in range(walk_times):
			if self._history_position == "local_walktime":
				history = np.ones([self.node_types_size], dtype=np.float64)
			# current node
			cur_node = root_node
			cur_type = root_type
			for __ in range(self._walk_length):
				# choose next type
				if random.random() < self._walk_restart:
					cur_node = root_node
					cur_type = root_type
				cur_node_adj_typelist = self._adj_lookupdict[cur_node].keys()
				next_type_list = [v for v in self._metagraph[cur_type] if v in cur_node_adj_typelist]
				if len(next_type_list) == 0:
					cur_type = root_type
					cur_node = root_node
				elif len(next_type_list) == 1:
					cur_type = next_type_list[0]
					cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
					history[cur_type] += 1
				else:
					occupancy = history[next_type_list]
					cur_type = utils.unigram_sample(population = next_type_list, size=1, weight=occupancy)[0]
					cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
					history[cur_type] += 1
				if cur_type in context_nodes_dict:
					context_nodes_dict[cur_type][0].append(cur_node) # context_list
					context_nodes_dict[cur_type][1].add(cur_node) # except_set
				else:
					context_nodes_dict[cur_type] = [[cur_node], {cur_node, root_node}]
		type_context_nodes_list = []
		type_neg_nodes_list = []
		type_mask_list = []
		for k in range(self.node_types_size):
			if k in context_nodes_dict:
				context_nodes = context_nodes_dict[k][0]
				except_set = context_nodes_dict[k][1]
				type_mask_list.append(1)
				type_context_nodes_list.append(context_nodes)
				type_neg_nodes_list.append(utils.neg_sample(self._nodes_type_dict[k][0], except_set,
															num=self._neg_sampled,
															alias_table=self._nodes_type_dict[k][1]))
			else:
				type_mask_list.append(0)
				type_context_nodes_list.append([0])
				type_neg_nodes_list.append([0])
		return root_node, type_context_nodes_list, type_mask_list, type_neg_nodes_list
	def _spacey_metatree_random_walk(self, root_node, walk_times = 0): # metapath, multi-metapath, metagraph
		root_type = self._net.get_node_type(root_node)
		if root_type not in self._metatree_type_id_dict:
			root_type = random.choice(list(self._metatree_type_id_dict.keys()))
			root_node = random.choice(self._nodes_type_dict[root_type][0])
		root_id = random.choice(self._metatree_type_id_dict[root_type])
		context_nodes_dict = {} #
		if self._history_position == "local":
			history = np.ones([len(self._metagraph.nodes())], dtype=np.float64)
		elif self._history_position == "global":
			history = self._history
		for _ in range(walk_times):
			if self._history_position == "local_walktime":
				history = np.ones([len(self._metagraph.nodes())], dtype=np.float64)
			# current node
			cur_node = root_node
			cur_type = root_type
			cur_id = root_id
			# logger.info('start: %d %d' % (cur_type, cur_id))
			for __ in range(self._walk_length):
				# choose next type
				if random.random() < self._walk_restart:
					cur_node = root_node
					cur_type = root_type
					cur_id   = root_id
				cur_node_adj_typelist = self._adj_lookupdict[cur_node].keys()
				next_id_list = [v for v in self._metagraph[cur_id] if self._metagraph.nodes[v]["type"] in cur_node_adj_typelist]
				if len(next_id_list) == 0:
					cur_type = root_type
					cur_node = root_node
					cur_id   = root_id
				elif len(next_id_list) == 1:
					cur_id = next_id_list[0]
					cur_type = self._metagraph.nodes[cur_id]["type"]
					cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
					history[cur_id] += 1
				else:
					occupancy = history[next_id_list]
					cur_id = utils.unigram_sample(population = next_id_list, size=1, weight=occupancy)[0]
					cur_type = self._metagraph.nodes[cur_id]["type"]
					cur_node = random.choice(self._adj_lookupdict[cur_node][cur_type])
					history[cur_id] += 1
				cur_id = utils.unigram_sample(population = self._metatree_type_id_dict[cur_type], size=1,
											  weight=history[self._metatree_type_id_dict[cur_type]])[0]
				if cur_type in context_nodes_dict:
					context_nodes_dict[cur_type][0].append(cur_node) # context_list
					context_nodes_dict[cur_type][1].add(cur_node) # except_set
				else:
					context_nodes_dict[cur_type] = [[cur_node], {cur_node, root_node}]
		type_context_nodes_list = []
		type_neg_nodes_list = []
		type_mask_list = []
		for k in range(self.node_types_size):
			if k in context_nodes_dict:
				context_nodes = context_nodes_dict[k][0]
				except_set = context_nodes_dict[k][1]
				type_mask_list.append(1)
				type_context_nodes_list.append(context_nodes)
				type_neg_nodes_list.append(utils.neg_sample(self._nodes_type_dict[k][0], except_set,
															num=self._neg_sampled,
															alias_table=self._nodes_type_dict[k][1]))
			else:
				type_mask_list.append(0)
				type_context_nodes_list.append([0])
				type_neg_nodes_list.append([0])
		return root_node, type_context_nodes_list, type_mask_list, type_neg_nodes_list
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








# walk to memory
def _construct_walk_corpus(walk_times):
	global walker
	logger.info('\t new walk process starts to walk %d times' % walk_times)
	walks = []
	nodes = list(range(walker.nodes_size))
	for cnt in range(walk_times):
		random.shuffle(nodes)
		for node in nodes:
			path_instance = walker.random_walk(node)
			if path_instance is not None and len(path_instance) > 1:  # ???????????????????
				walks.append(path_instance)
	return walks
# def _construct_walk_corpus(walk_times):
#     global walker
#     logger.info('\t new walk process starts to walk %d times' % walk_times)
#     walks = []
#     nodes = list(range(walker.nodes_size))
#     # for cnt in range(10):
#         # random.shuffle(nodes)
#     for node in nodes:
#         for cnt in range(10):
#             path_instance = walker.random_walk(node)
#             if path_instance is not None and len(path_instance) > 1:  # ???????????????????
#                 walks.append(path_instance)
#     return walks

def _construct_walk_corpus_no_multiprocess(walk_times):
	logger.info('Corpus bulid: walking to memory (without using multi-process)...')
	time_start = time.time()
	walks = _construct_walk_corpus(walk_times)
	logger.info('Corpus bulid: walk completed in {}s'.format(time.time() - time_start))
	return walks

def _construct_walk_corpus_multiprocess(walk_times, max_num_workers=cpu_count()):
	""" Use multi-process scheduling"""
	# allocate walk times to workers
	if walk_times <= max_num_workers:
		times_per_worker = [1 for _ in range(walk_times)]
	else:
		div, mod = divmod(walk_times, max_num_workers)
		times_per_worker = [div for _ in range(max_num_workers)]
		for idx in range(mod):
			times_per_worker[idx] = times_per_worker[idx] + 1
	assert sum(times_per_worker) == walk_times, 'workers allocating failed: %d != %d' % (
		sum(times_per_worker), walk_times)

	sens = []
	args_list = []
	for index in range(len(times_per_worker)):
		args_list.append(times_per_worker[index])
	logger.info('Corpus bulid: walking to memory (using %d workers for multi-process)...' % len(times_per_worker))
	time_start = time.time()
	with ProcessPoolExecutor(max_workers=max_num_workers) as executor:
	# # the walker for node2vec is so large that we can not use multi-process, so we use multi-thread instead.
	# with ThreadPoolExecutor(max_workers=max_num_workers) as executor:
		for walks in executor.map(_construct_walk_corpus, args_list):
			sens.extend(walks)
	logger.info('Corpus bulid: walk completed in {}s'.format(time.time() - time_start))
	return sens

def build_walk_corpus_to_memory(walk_times, max_num_workers=cpu_count()):
	if max_num_workers <= 1 or walk_times <= 1:
		if max_num_workers > 1:
			logger.warning('Corpus bulid: walk times too small, using single-process instead...')
		return _construct_walk_corpus_no_multiprocess(walk_times)
	else:
		return _construct_walk_corpus_multiprocess(walk_times, max_num_workers=max_num_workers)


# store corpus
def store_walk_corpus(filebase, walk_sens, always_rebuild = False):
	if not utils.check_rebuild(filebase, descrip='walk corpus', always_rebuild=always_rebuild):
		return
	logger.info('Corpus store: storing...')
	time_start = time.time()
	with open(filebase, 'w') as fout:
		for sen in walk_sens:
			for v in sen:
				fout.write(u"{} ".format(str(v)))
			fout.write('\n')
	logger.info('Corpus store: store completed in {}s'.format(time.time() - time_start))
	return


# walk to files
def _construct_walk_corpus_iter(walk_times, walk_process_id):
	global walker
	nodes = list(range(walker.nodes_size))
	last_time = time.time()
	for cnt in range(walk_times):
		start_time = time.time()
		logger.info('\t !process-%s walking %d/%d, interval %.4fs, total %d nodes' % (walk_process_id, cnt, walk_times, start_time - last_time, len(nodes)))
		last_time = start_time
		random.shuffle(nodes)
		for node in nodes:
			path_instance = walker.random_walk(node)
			if path_instance is not None and len(path_instance) >= 1:  # ???????????????????
				yield path_instance
# def _construct_walk_corpus_iter(walk_times, walk_process_id):
#     global walker
#     nodes = list(range(walker.nodes_size))
#     last_time = time.time()
#     for node in nodes:
#     # for cnt in range(10):
#         start_time = time.time()
#         # logger.info('\t !process-%s walking %d/%d, interval %.4fs' % (walk_process_id, cnt, walk_times, start_time - last_time))
#         logger.info('\t !process-%s walking %d, interval %.4fs' % (walk_process_id, walk_times, start_time - last_time))
#         last_time = start_time
#         # random.shuffle(nodes)
#         # for node in nodes:
#         for cnt in range(10):
#             path_instance = walker.random_walk(node)
#             if path_instance is not None and len(path_instance) > 1:  # ???????????????????
#                 yield path_instance

def _construct_walk_corpus_and_write_singprocess(args):
	filebase, walk_times = args
	walk_process_id = filebase.split('.')[-1]
	logger.info('\t new walk process-%s starts to walk %d times' % (walk_process_id, walk_times))
	time_start = time.time()
	with open(filebase, 'w') as fout:
		for walks in _construct_walk_corpus_iter(walk_times,walk_process_id):
			for v in walks:
				fout.write(u"{} ".format(str(v)))
			fout.write('\n')
	logger.info('\t process-%s walk ended, generated a new file \'%s\', it took %.4fs' % (
		walk_process_id, filebase, time.time() - time_start))
	return filebase

def _construct_walk_corpus_and_write_multiprocess(filebase,walk_times,headflag_of_index_file = '',
												  max_num_workers=cpu_count()):
	""" Walk to files.
		this method is designed for a very large scale network which is too large to walk to memory.
	"""
	# allocate walk times to workers
	if walk_times <= max_num_workers:
		times_per_worker = [1 for _ in range(walk_times)]
	else:
		div, mod = divmod(walk_times, max_num_workers)
		times_per_worker = [div for _ in range(max_num_workers)]
		for idx in range(mod):
			times_per_worker[idx] = times_per_worker[idx] + 1
	assert sum(times_per_worker) == walk_times, 'workers allocating failed: %d != %d' % (
	sum(times_per_worker), walk_times)

	files_list = ["{}.{}".format(filebase, str(x)) for x in range(len(times_per_worker))]
	f = open(filebase, 'w')
	f.write('{}\n'.format(headflag_of_index_file))
	f.write('DESCRIPTION: allocate %d workers to concurrently walk %d times.\n' % (len(times_per_worker), walk_times))
	f.write('DESCRIPTION: generate %d files to save walk corpus:\n' % (len(times_per_worker)))
	for item in files_list:
		f.write('FILE: {}\n'.format(item))
	f.close()

	files = []
	args_list = []
	for index in range(len(times_per_worker)):
		args_list.append((files_list[index], times_per_worker[index]))

	logger.info('Corpus bulid: walking to files (using %d workers for multi-process)...' % len(times_per_worker))
	time_start = time.time()
	with ProcessPoolExecutor(max_workers=max_num_workers) as executor:
	# # the walker for node2vec is so large that we can not use multi-process, so we use multi-thread instead.
	# with ThreadPoolExecutor(max_workers=max_num_workers) as executor:
		for file_ in executor.map(_construct_walk_corpus_and_write_singprocess, args_list):
			files.append(file_)
	assert len(files) == len(files_list), 'ProcessPoolExecutor occured error, %d!=%d' % (len(files), len(files_list))

	logger.info('Corpus bulid: walk completed in {}s'.format(time.time() - time_start))
	return files

def build_walk_corpus_to_files(filebase, walk_times, headflag_of_index_file = '',
							   max_num_workers=cpu_count(), always_rebuild=False):
	if not utils.check_rebuild(filebase, descrip='walk corpus', always_rebuild=always_rebuild):
		return

	if max_num_workers <= 1 or walk_times <= 1:
		if max_num_workers > 1:
			logger.warning('Corpus bulid: walk times too small, using single-process instead...')
		files = []
		logger.info('Corpus bulid: walking to files (without using multi-process)...')
		time_start = time.time()
		files.append(_construct_walk_corpus_and_write_singprocess((filebase, walk_times)))
		logger.info('Corpus bulid: walk completed in {}s'.format(time.time() - time_start))
		return files
	else:
		return _construct_walk_corpus_and_write_multiprocess(filebase,walk_times,
															 headflag_of_index_file = headflag_of_index_file,
															 max_num_workers=max_num_workers)


# ========= Walks Corpus ===========#
class WalksCorpus(object):
	"""
	Walks Corpus, load from files.
	Note: this class is designed to privode training corpus in form of a sentence iterator to reduce memeory.
	"""
	def __init__(self, files_list):
		self.files_list = files_list
	def __iter__(self):
		for file in self.files_list:
			if (not os.path.exists(file)) or (not os.path.isfile(file)):
				continue
			with open(file, 'r') as f:
				for line in f:
					yield line.strip().split()

def load_walks_corpus(files_list):
	logger.info('Corpus load: loading corpus to memory...')
	time_start = time.time()
	sens = []
	for file in files_list:
		if (not os.path.exists(file)) or (not os.path.isfile(file)):
			continue
		with open(file, 'r') as f:
			for line in f:
				sens.append(line.strip().split())
	logger.info('Corpus load: loading completed in {}s'.format(time.time() - time_start))
	return sens



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
	logger.info('\t alpha = {}'.format(options.alpha))
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
	fr_walks.write('\t walk to memory = {}\n'.format(str(options.walk_to_memory)))
	if options.walk_to_memory:
		fr_walks.write('\t donot store corpus = {}\n'.format(str(options.not_store_corpus)))
		if not options.not_store_corpus:
			fr_walks.write('\t corpus store path = {}\n'.format(options.corpus_store_path))
	else:
		fr_walks.write('\t corpus store path = {}\n'.format(options.corpus_store_path))
	fr_walks.close()

	if options.model == "DeepWalk":
		walker = Walker(net, random_walker=random_walker, walk_length=options.walk_length)
	elif options.model == "SpaceyWalk":
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

	walk_corpus = None
	if options.walk_to_memory:
		walk_corpus = build_walk_corpus_to_memory(options.walk_times, max_num_workers=options.walk_workers)
		if not options.not_store_corpus:
			store_walk_corpus(options.corpus_store_path, walk_corpus, always_rebuild=options.always_rebuild)
	else:
		# walk to files
		walk_files = build_walk_corpus_to_files(options.corpus_store_path, options.walk_times,
												headflag_of_index_file=options.headflag_of_index_file,
												max_num_workers=options.walk_workers,
												always_rebuild=options.always_rebuild)
		if "train" in options.task:
			if options.load_from_memory:
				walk_corpus = load_walks_corpus(walk_files)
			else:
				walk_corpus = WalksCorpus(walk_files)
	del walker
	gc.collect()
	return walk_corpus
