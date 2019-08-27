#! /usr/bin/env python  
# -*- coding:utf-8 -*-  

"""
__author__ = "He Yu"   
Version: 1.0.0 
 
"""


import networkx as nx
import time
import os
import numpy as np
import random
from sklearn.metrics import roc_auc_score
import tensorflow as tf
#
# print(nx)
# #
# G = nx.Graph()
# G.add_node(1)
# G.add_node(11)
# G.add_node(111)
# G.add_node(2,name="node2")
# G.add_node(22,name="node22")
# G.add_node(222,name="node222")
# G.add_edge(1,2,type="undir")
# G.add_edge(2,1,type="undir")
# G.add_edge(1,1,type="dir")
# G.add_edge(222,333,type="dir")
# G.add_node(1)
# # #
# # # for node, type in G.nodes(data="name"):
# # #     print("node={}, name={}".format(node, type))
# #
# #
# print(list(G[222]))
# # print([v for v in G[1]])
# #


nodes_total1 = [1,2,3,4,5,6,7,8,9,10]
nodes_total2 = [1,2,3,4,5,6,7,8,9,10]
nodes_total3 = [1,2,3,4,5,6,7,8,9,10]
nodes_total4 = [1,2,3,4,5,6,7,8,9,10]

nodes_total4.insert(0,1111111)
nodes_total4.insert(0,2222222)
nodes_total4.insert(0,4444444)
print(nodes_total4)