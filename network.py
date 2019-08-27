#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
__author__ = "He Yu"
Version: 1.0.0
Reference:
    https://networkx.github.io/documentation/latest/index.html
    https://blog.csdn.net/qq_32284189/article/details/80134768
    https://www.cnblogs.com/kaituorensheng/p/5423131.html
    https://www.cnblogs.com/gispathfinder/p/5790949.html
    https://baiyejianxin.iteye.com/blog/1764048
"""


import logging
import sys
import os
import time
from collections import Iterable
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


import networkx as nx


logger = logging.getLogger("HNE")



class Graph(object):
    """
    An heterogeneous (including homogenous) graph/network.
    each node in the graph/network is represented as an interger ID, and the node ID start from 0.
    each edge in the graph/network is directed, which means an undirected edge (u,v) will have two node pairs <u,v> and <v,u>.
    node type is also an interger ID starting from 0.
    edge type is also an interger ID starting from 0.
    """
    def __init__(self, isdirected = False, isweighted = False, isselflooped = False, ismultiple = False, isHIN = True):
        self._isdirected = isdirected
        self._isweighted = isweighted # future work for weighted-network.
        self._isselflooped = isselflooped
        self._ismultiple = ismultiple # future work for multi-network.
        self._isHIN = isHIN

        if self._isdirected:
            self._G = nx.DiGraph()
            if self._isHIN:
                self._G_schema = nx.DiGraph()
        else:
            self._G = nx.Graph()
            if self._isHIN:
                self._G_schema = nx.Graph()

        self._adj_lookupdict = {} # {node: type: adj_list}

    @property
    def isdirected(self):
        return self._isdirected
    @property
    def isweighted(self):
        return self._isweighted
    @property
    def isselflooped(self):
        return self._isselflooped
    @property
    def ismultiple(self):
        return self._ismultiple
    @property
    def isHIN(self):
        return self._isHIN

    def has_node_type(self, type):
        return self._G_schema.has_node(type)
    def has_edge_type(self, start_node_type, target_node_type):
        return self._G_schema.has_edge(start_node_type, target_node_type)
    def has_node(self, node):
        return self._G.has_node(node) # return node in self._G
    def has_edge(self, start_node, target_node):
        return self._G.has_edge(start_node, target_node) # return target_node in self._G[start_node]
    def check_node_type(self, node_type, node = None):
        # first: check node_type in network schema.
        if not self.has_node_type(node_type):
            logger.error(
                "node_type {} not in the network schema.".format(node_type))
            sys.exit()
        # sec: check node_type matching.
        if node != None and self.has_node(node) and self.get_node_type(node) != node_type:
            logger.error(
                "node {} is already added into the graph but has mismatched node_type: {} != {}.".format(
                    node, self.get_node_type(node), node_type))
            sys.exit()
    def check_edge_type(self, start_node_type, target_node_type):
        if not self.has_edge_type(start_node_type, target_node_type):
            logger.error(
                "edge_type <{}, {}> not in the network schema.".format(start_node_type, target_node_type))
            sys.exit()
    def check_node(self, node, node_type = None):
        if not self.has_node(node):
            logger.error("node {} not in the graph.".format(node))
            sys.exit()
        if node_type != None and self._isHIN and self.get_node_type(node) != node_type:
            logger.error(
                "node {} is already added into the graph but has mismatched node_type: {} != {}.".format(
                    node, self.get_node_type(node), node_type))
            sys.exit()
    def add_node_type(self, type):
        self._G_schema.add_node(type)
    def add_edge_type(self, start_node_type, target_node_type, keep_checked = True):
        if keep_checked:
            self.check_node_type(start_node_type)
            self.check_node_type(target_node_type)
        else:
            self.add_node_type(start_node_type)
            self.add_node_type(target_node_type)
        self._G_schema.add_edge(start_node_type, target_node_type)
    def add_node(self, node, node_type = None, keep_checked = True):
        if self._isHIN:
            if node_type != None:
                if keep_checked:
                    self.check_node_type(node_type, node)
                else:
                    self.add_node_type(node_type)
                self._G.add_node(node, type=node_type)
            elif not self.has_node(node):
                logger.error("node {} mush have a type!!!".format(node))
                sys.exit()
        else:
            self._G.add_node(node)
    def add_edge(self, start_node, target_node, start_node_type = None, target_node_type = None, keep_checked = True):
        if self._isHIN:
            if start_node_type != None:
                if keep_checked:
                    self.check_node(start_node, start_node_type)
                    # self.check_node_type(start_node_type)
                else:
                    self.add_node(start_node, node_type=start_node_type, keep_checked=False)
            elif not self.has_node(start_node):
                logger.error("node {} mush have a type!!!".format(start_node))
                sys.exit()
            else:
                start_node_type = self.get_node_type(start_node)

            if target_node_type != None:
                if keep_checked:
                    self.check_node(target_node, target_node_type)
                    # self.check_node_type(target_node_type)
                else:
                    self.add_node(target_node, node_type=target_node_type, keep_checked=False)
            elif not self.has_node(target_node):
                logger.error("node {} mush have a type!!!".format(target_node))
                sys.exit()
            else:
                target_node_type = self.get_node_type(target_node)

            if keep_checked:
                self.check_edge_type(start_node_type, target_node_type)
            else:
                self.add_edge_type(start_node_type, target_node_type, keep_checked=False)
        else:
            if keep_checked:
                self.check_node(start_node)
                self.check_node(target_node)
            else:
                self.add_node(start_node, keep_checked=False)
                self.add_node(target_node, keep_checked=False)
        if not self._isselflooped and start_node == target_node:  #
            return
        self._G.add_edge(start_node, target_node)

    @property
    def node_types(self):
        return self._G_schema.nodes
    @property
    def edge_types(self):
        return self._G_schema.edges # (u,v)
    @property
    def nodes(self):
        return self._G.nodes
    @property
    def edges(self):
        return self._G.edges  # (u,v)
    def get_nodes(self, node_type=None):
        if not self._isHIN or node_type == None or node_type<0:
            return self._G.nodes
        else:
            nodes_list = [] # if isinstance(a,list)
            for node_ID, type_ID in self._G.nodes(data="type"):
                if type_ID == node_type:
                    nodes_list.append(node_ID)
            return nodes_list
    def get_edges(self, start_node_type = None, target_node_type = None):
        if not self._isHIN or ((start_node_type == None or start_node_type<0) and (target_node_type == None or target_node_type<0)):
            return self._G.edges # (u,v)
        else:
            edges_list = []
            for start_node, target_node in self._G.edges:
                start_node_ret_type = self.get_node_type(start_node)
                target_node_ret_type = self.get_node_type(target_node)
                if (start_node_type == None or start_node_type < 0 or start_node_type == start_node_ret_type) and (
                        target_node_type == None or target_node_type < 0 or target_node_type == target_node_ret_type):
                    edges_list.append((start_node, target_node))
                elif not self._isdirected and (
                            start_node_type == None or start_node_type < 0 or start_node_type == target_node_ret_type) and (
                             target_node_type == None or target_node_type < 0 or target_node_type == start_node_ret_type):
                    edges_list.append((target_node, start_node))
            return edges_list
    def degrees_iter(self, nodes = None, total_in_out = False):
        if self._isdirected and not total_in_out:
            degree_func =  self._G.out_degree
        else:
            degree_func = self._G.degree
        if nodes == None:
            return degree_func
        else:
            return degree_func(nodes)
    def degrees_dict(self, nodes=None, total_in_out = False):
        if self._isdirected and not total_in_out:
            degree_func =  self._G.out_degree
        else:
            degree_func = self._G.degree
        if nodes == None:
            return {n: d for n, d in degree_func}
        else:
            return {n: d for n, d in degree_func(nodes)}

    def get_node_types_size(self):
        return self._G_schema.number_of_nodes()
    def get_edge_types_size(self):
        return self._G_schema.number_of_edges()
    def get_nodes_size(self):
        return self._G.number_of_nodes()
    def get_edges_size(self):
        return self._G.number_of_edges()
    def get_node_type(self, node):
        return self._G.nodes[node]["type"]
    def get_edge_type(self, start_node, target_node):
        start_node_ret_type = self.get_node_type(start_node)
        target_node_ret_type = self.get_node_type(target_node)
        return (start_node_ret_type, target_node_ret_type)
    def get_degrees(self, nodes = None, total_in_out = False):
        if self._isdirected and not total_in_out:
            degree_func =  self._G.out_degree
        else:
            degree_func = self._G.degree
        if nodes == None:
            return [d for n, d in degree_func]
        else:
            if isinstance(nodes, Iterable):
                return [degree_func(n) for n in nodes]
            else:
                return degree_func(nodes)
    def get_neighbors(self, node, type=None):
        if type == None:
            # return self._G.neighbors(node)
            return self._G[node]
        else:
            nodes_list = [] # if isinstance(a,list)
            # for v in self._G.neighbors(node):
            for v in self._G[node]:
                if self.get_node_type(v) == type:
                    nodes_list.append(v)
            return nodes_list
    def get_adj_list(self, node, type="None"):
        if node not in self._adj_lookupdict:
            adj_list = list(self._G[node])
            self._adj_lookupdict[node] = {"None": adj_list}
            if self._isHIN:
                for v in adj_list:
                    v_type = self.get_node_type(v)
                    if v_type in self._adj_lookupdict[node]:
                        self._adj_lookupdict[node][v_type].append(v)
                    else:
                        self._adj_lookupdict[node][v_type] = [v]
        return self._adj_lookupdict[node][type]

    def print_net_info(self, file_path='./tmp/net/net.info', net_data_dir=None):
        if not os.path.exists(os.path.split(file_path)[0]):
            os.makedirs(os.path.split(file_path)[0])
        fr = open(file_path, 'w')
        fr.write('constructed network from {}\n'.format(net_data_dir))
        fr.write('network info::\n')
        fr.write('\t isdirected: {}\n'.format(self._isdirected))
        fr.write('\t isweighted: {}\n'.format(self._isweighted))
        fr.write('\t isselflooped: {}\n'.format(self._isselflooped))
        fr.write('\t ismultiple: {}\n'.format(self._ismultiple))
        fr.write('\t isHIN: {}\n'.format(self._isHIN))
        if self._isHIN:
            fr.write('\t node_types_size: {}\n'.format(self.get_node_types_size()))
            fr.write('\t edge_types_size: {}\n'.format(self.get_edge_types_size()))
        fr.write('\t nodes_size: {}\n'.format(self.get_nodes_size()))
        fr.write('\t edges_size: {}\n'.format(self.get_edges_size()))
        if self._isHIN:
            nodes_size_list = [0 for _ in range(self.get_node_types_size())]
            for node_ID, type_ID in self._G.nodes(data="type"):
                nodes_size_list[type_ID] +=1
            edges_size_dict = {(s,t):0 for s,t in list(self._G_schema.edges)}
            for start_node, target_node in self._G.edges:
                start_node_type = self.get_node_type(start_node)
                target_node_type = self.get_node_type(target_node)
                if (start_node_type, target_node_type) in edges_size_dict:
                    edges_size_dict[(start_node_type, target_node_type)] +=1
                else:
                    assert self._isdirected == False, ""
                    edges_size_dict[(target_node_type, start_node_type)] += 1
            for idx, value in enumerate(nodes_size_list):
                fr.write('\t\t nodes_size(type-{}): {}\n'.format(idx, value))
            assert len(list(self._G_schema.edges)) == len(edges_size_dict), "{} != {}".format(len(list(self._G_schema.edges)), len(edges_size_dict))
            for s, t in list(self._G_schema.edges):
                fr.write('\t\t edges_size(type-{},type-{}): {}\n'.format(s,t,edges_size_dict[(s,t)]))

        logger.info('network info::')
        logger.info('constructed network form {}'.format(net_data_dir))
        logger.info('\t isdirected: {}'.format(self._isdirected))
        logger.info('\t isweighted: {}'.format(self._isweighted))
        logger.info('\t isselflooped: {}'.format(self._isselflooped))
        logger.info('\t ismultiple: {}'.format(self._ismultiple))
        logger.info('\t isHIN: {}'.format(self._isHIN))
        if self._isHIN:
            logger.info('\t node_types_size: {}'.format(self.get_node_types_size()))
            logger.info('\t edge_types_size: {}'.format(self.get_edge_types_size()))
        logger.info('\t nodes size: {}'.format(self.get_nodes_size()))
        logger.info('\t edges size: {}'.format(self.get_edges_size()))
        if self._isHIN:
            for idx, value in enumerate(nodes_size_list):
                logger.info('\t\t nodes_size(type-{}): {}'.format(idx, value))
            for s, t in list(self._G_schema.edges):
                logger.info('\t\t edges_size(type-{},type-{}): {}'.format(s,t,edges_size_dict[(s,t)]))

        degrees = np.array(self.get_degrees(), dtype=np.int32)
        degree_max = np.max(degrees)
        degree_mean = np.mean(degrees)
        degree_median = np.median(degrees)
        degree_min = np.min(degrees)
        degree_zero_count = np.sum(degrees==0)
        fr.write('\t max degree(out_degree): {}\n'.format(degree_max))
        fr.write('\t mean degree(out_degree): {}\n'.format(degree_mean))
        fr.write('\t median degree(out_degree): {}\n'.format(degree_median))
        fr.write('\t min degree(out_degree): {}\n'.format(degree_min))
        fr.write('\t zero degree(out_degree) counts: {}\n'.format(degree_zero_count))
        if self._isdirected:
            totoal_degree_zero_count = np.sum(np.array(self.get_degrees(total_in_out=True), dtype=np.int32) == 0)
            fr.write('\t zero degree(out_degree+in_degree) counts: {}\n'.format(totoal_degree_zero_count))
        # fr.write('-'*20+'\n'+'-'*20+'\n'+'nodes degrees(out_degrees) details:'+'\n')
        # for id in self.nodes:
        #     fr.write("{}\t{}\n".format(id, nodeID_degrees[id]))
        # fr.close()
        logger.info('\t max degree(out_degree): {}'.format(degree_max))
        logger.info('\t mean degree(out_degree): {}'.format(degree_mean))
        logger.info('\t median degree(out_degree): {}'.format(degree_median))
        logger.info('\t min degree(out_degree): {}'.format(degree_min))
        logger.info('\t zero degree(out_degree) counts: {}'.format(degree_zero_count))
        if self._isdirected:
            logger.info('\t zero degree(out_degree+in_degree) counts: {}\n'.format(totoal_degree_zero_count))




def construct_meta_graph(metapaths_filename, isdirected = False):
    """
    # 0 Type_0
    # 1 Type_1
    # 2 Type_2
    ...
    0 1 2 3 4 5
    """
    if isdirected:
        meta_graph = nx.DiGraph()
    else:
        meta_graph = nx.Graph()
    id2type_dict = {}
    for line in open(metapaths_filename):
        line = line.strip()
        if line:
            if line.startswith("#"):
                linelist = [int(v.strip()) for v in line[1:].strip().split("\t")]
                assert len(linelist)==2, "linelist = {}".format(linelist)
                id2type_dict[linelist[0]] = linelist[1]
            else:
                linelist = [ int(v.strip()) for v in line.split("\t")]
                assert len(linelist)>1,"linelist = {}".format(linelist)
                # meta_graph.add_node(linelist[0])
                for i in range(1, len(linelist)):
                    # meta_graph.add_node(linelist[i])
                    meta_graph.add_edge(id2type_dict[linelist[i-1]], id2type_dict[linelist[i]])
    return meta_graph

def construct_meta_tree(metapaths_filename='metapath/test', isdirected = True):
    """
        # 0 Type_0
        # 1 Type_1
        # 2 Type_2
        ...
        0 1 2 3 4 5
    """
    if isdirected:
        meta_graph = nx.DiGraph()
    else:
        meta_graph = nx.Graph()
    id2type_dict = {}
    for line in open(metapaths_filename):
        line = line.strip()
        if line:
            if line.startswith("#"):
                linelist = [int(v.strip()) for v in line[1:].strip().split("\t")]
                assert len(linelist) == 2, "linelist = {}".format(linelist)
                id2type_dict[linelist[0]] = linelist[1]
            else:
                linelist = [int(v.strip()) for v in line.split("\t")]
                assert len(linelist) > 1, "linelist = {}".format(linelist)
                # assert linelist[0] == linelist[
                #     -1], "the head and end of the metapath must be same, but {} != {}".format(linelist[0], linelist[-1])
                meta_graph.add_node(linelist[0], type=id2type_dict[linelist[0]])
                for i in range(1, len(linelist)):
                    meta_graph.add_node(linelist[i], type=id2type_dict[linelist[i]])
                    meta_graph.add_edge(linelist[i - 1], linelist[i])
    return meta_graph





def _construct_net_schema(net, data_path):
    """
    data files include:
            data_path.nodes
            data_path.edges
            data_path.node_types
            data_path.edge_types
            data_path.labels: nodeID labelID
            data_path.label_names: labelID label_name
            ...

    data_path.node_types:
    node_type_ID0
    node_type_ID1
    ...
    data_path.edge_types:
    start_node_type_ID0 target_node_type_ID0
    start_node_type_ID1 target_node_type_ID1
    ...
    """
    node_types_file = data_path + ".node_types"
    edge_types_file = data_path + ".edge_types"
    logger.info('Net construct: loading network schema ...')
    time_start = time.time()
    if not os.path.exists(node_types_file):
        logger.error('\t file \'%s\' not exists!' % node_types_file)
        sys.exit()
    else:
        logger.info('\t reading node_types from file \'%s\'' % node_types_file)
        for line in open(node_types_file):
            line = line.strip()
            if line:
                type = int(line.split("\t")[0].strip())
                net.add_node_type(type)
    if not os.path.exists(edge_types_file):
        logger.error('\t file \'%s\' not exists!' % edge_types_file)
        sys.exit()
    else:
        logger.info('\t reading edge_types from file \'%s\'' % edge_types_file)
        for line in open(edge_types_file):
            line = line.strip()
            if line:
                linelist = line.split("\t")
                start_node_type = int(linelist[0].strip())
                target_node_type = int(linelist[1].strip())
                net.add_edge_type(start_node_type, target_node_type, keep_checked = True)
    logger.info('Net construct: load network schema completed in {}s'.format(time.time() - time_start))

def _construct_net(net, data_path):
    """
    data files include:
            data_path.nodes
            data_path.edges
            data_path.node_types
            data_path.edge_types
            data_path.labels: nodeID labelID
            data_path.label_names: labelID label_name
            ...
    data_path.nodes:
    node_ID0 node_type_ID0
    node_ID1 node_type_ID1
    ...
    data_path.edges:
    start_node_ID0 target_node_ID0
    start_node_ID1 target_node_ID1
    ...
    """
    node_file = data_path + ".nodes"
    edge_file = data_path + ".edges"
    logger.info('Net construct: loading network ...')
    time_start = time.time()
    if not os.path.exists(node_file):
        logger.error('\t file \'%s\' not exists!' % node_file)
        sys.exit()
    else:
        logger.info('\t reading nodes from file \'%s\'' % node_file)
        for line in open(node_file):
            line = line.strip()
            if line:
                linelist = line.split("\t")
                node = int(linelist[0].strip())
                if len(linelist) > 1:
                    node_type = int(linelist[1].strip())
                else:
                    node_type = None
                net.add_node(node, node_type=node_type, keep_checked=True)
    if not os.path.exists(edge_file):
        logger.error('\t file \'%s\' not exists!' % edge_file)
        sys.exit()
    else:
        logger.info('\t reading edges from file \'%s\'' % edge_file)
        for line in open(edge_file):
            line = line.strip()
            if line:
                linelist = line.split("\t")
                start_node = int(linelist[0].strip())
                target_node = int(linelist[1].strip())
                net.add_edge(start_node, target_node, keep_checked = True)
    logger.info('Net construct: load network completed in {}s'.format(time.time() - time_start))


def construct_network(options = None, data_dir = None, data_name = None, isdirected = None,
                      net_info_path = None, print_net_info = True, isHIN = True):
    """
        An heterogeneous (including homogenous) graph/network.
        each node in the graph/network is represented as an interger ID, and the node ID start from 0.
        each edge in the graph/network is directed, which means an undirected edge (u,v) will have two node pairs <u,v> and <v,u>.
        node type is also an interger ID starting from 0.
        edge type is also an interger ID starting from 0.
        data files include:
            data_dir/data_name.nodes
            data_dir/data_name.edges
            data_dir/data_name.node_types
            data_dir/data_name.edge_types
            data_dir/data_name.labels: nodeID labelID
            data_dir/data_name.label_names: labelID label_name
            ...
    """
    if data_dir == None:
        data_dir = options.data_dir
    if data_name == None:
        data_name = options.data_name
    if isdirected == None:
        isdirected = options.isdirected
    if print_net_info and net_info_path == None:
        net_info_path = options.net_info_path

    net = Graph(isdirected = isdirected, isHIN = isHIN)

    data_path = os.path.join(data_dir, data_name)

    if isHIN:
        _construct_net_schema(net, data_path)
    _construct_net(net, data_path)

    if print_net_info:
        net.print_net_info(file_path=net_info_path, net_data_dir=data_dir)

    return net


if __name__ == '__main__':
    meta_graph = construct_meta_tree(metapaths_filename='metapath/APVPTPVPA')
    for i in meta_graph.nodes:
        print("{} : {}, {}".format(i, meta_graph.nodes[i]["type"], list(meta_graph[i])))