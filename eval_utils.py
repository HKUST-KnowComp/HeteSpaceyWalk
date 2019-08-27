
"""

https://blog.csdn.net/u010412858/article/details/60467382
https://blog.csdn.net/u013749540/article/details/51813922

"""

import os
import numpy as np
import logging
import matplotlib
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
# note: cosine_similarity(X, Y) = <X, Y> / (||X||*||Y||), range:[-1, 1], bigger value means more similar and closer distance.
# cosine_distances = 1 - cosine_similarity, range: [0, 2], bigger value means less similar and further distance.
from sklearn.metrics.pairwise import euclidean_distances
# note: euclidean_distances(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y)), for computationally efficient.


import network


logger = logging.getLogger("HNE")



def get_node_color(node_community, label_size):

    # 遍历颜色： colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    logger.info('allocate colors for {} classes/labels ...'.format(label_size))
    time_start = time.time()
    color = []
    for line in open('./handler/color_list','r'):
        if line.strip():
            cc = line.split()[0].strip(':').strip('\'')
            if cc not in color:
                logger.info("\t label {}: {}".format(len(color), cc))
                color.append(cc)
                if len(color) >= label_size:
                    break

    node_colors = []
    for v in node_community:
        if v >= len(color):
            logger.error("error! label_{} out of the range of color community.".format(v))
        node_colors.append(color[v])
    return node_colors

def clustering_center_distance(node_pos,node_label):
    X_dict = {}
    Y_dict = {}
    for index,label in enumerate(node_label):
        pos = node_pos[index]
        if label not in X_dict:
            X_dict[label] = [pos[0]]
            Y_dict[label] = [pos[1]]
        else:
            X_dict[label].append(pos[0])
            Y_dict[label].append(pos[1])

    labels_list = list(X_dict.keys())
    X_mean = {}
    Y_mean = {}
    for key, value in X_dict.items():
        X_mean[key] = np.mean(value)
    for key, value in Y_dict.items():
        Y_mean[key] = np.mean(value)
    distance_list = []
    for i in range(len(labels_list)):
        for j in range(i+1, len(labels_list)):
            # euclidean_distances:
            dist = np.linalg.norm([ X_mean[labels_list[i]] - X_mean[labels_list[j]], Y_mean[labels_list[i]] - Y_mean[labels_list[j]]])
            distance_list.append(dist)
    # normalization and turn to similarity metric: sim = 1 / (1 + dist(X,Y))
    # sim = 1.0 / (1.0 + np.mean(distance_list))  # range: (0,1), bigger value means more similar and closer distance.
    sim = np.mean(distance_list)

    return sim


def getSimilarity(X, Y = None, metric="cosine"):
    """
    Compute similarity between samples in X and Y.
    :param X: n*d, n samples and d features.
    :param metric: cosine, euclidean, ...
    :return: n*n, or n_samples_X*n_samples_Y
    """
    time_start = time.time()
    logger.info('similarity calculating ...')
    if metric == "cosine":
        # dist(X, Y) = < X, Y > / (| | X | | * | | Y | |)
        dist = cosine_similarity(X, Y)
        # normalize to range(0,1):
        sim = 0.5 + 0.5 * dist
    elif metric == "euclidean":
        # dist(x,y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
        dist = euclidean_distances(X, Y)
        # normalize to range(0,1):
        sim = 1.0 / (1.0 + dist)
    else:
        logger.error("Unknown Similarity metric: '%s'.  Valid metric: 'cosine', 'euclidean'" % metric)
    logger.info('similarity calculation completed in {}s.'.format(time.time() - time_start))
    return sim



def split_network(origin_net_dir, train_net_dir, eval_net_dir, data_prefixname,
                  data_format, isdirected,train_ratio):
    net_origin = network.construct_network(data_path = os.path.join(origin_net_dir, data_prefixname + "." + data_format),
                                           data_format = data_format, 
                                           net_info_path = os.path.join(origin_net_dir, "net.info"),
                                           isdirected = isdirected)
    net_train, net_eval = net_origin.split_by_edges(train_ratio = train_ratio)
    net_train.save_network(train_net_dir, data_prefixname, data_format)
    # net_train.print_net_info(edges_file=os.path.join(train_net_dir, datafilename), file_path=os.path.join(train_net_dir, "net.info"))
    net_eval.save_network(eval_net_dir, data_prefixname, data_format)
    # net_eval.print_net_info(edges_file=os.path.join(eval_net_dir, datafilename), file_path=os.path.join(eval_net_dir, "net.info"))
    del net_origin,net_train,net_eval

    
def check_MAP(nodeID_list, similarity, eval_graph, except_graph = None):
    MAP_list = []
    for index_s, nodeID_s in enumerate(nodeID_list):
        if eval_graph.get_degrees(nodeID_s) == 0:
            continue
        AP = 0
        count_predict = 0
        count_hitting = 0
        sortedInd = np.argsort(similarity[index_s])
        sortedInd = sortedInd[::-1]
        last_predict_value = None  # foresight for equal values. is it reasonable or better?
        last_count_predict = 0  # foresight for equal values. is it reasonable or better?
        last_count_hitting = 0  # foresight for equal values. is it reasonable or better?
        for index_t in sortedInd:
            nodeID_t = nodeID_list[index_t]
            if nodeID_s == nodeID_t or (except_graph != None and except_graph.has_edge(nodeID_s, nodeID_t)):
                continue
            # count_predict += 1
            # if eval_graph.has_edge(nodeID_s, nodeID_t):
            #     count_hitting += 1
            #     prec_k = count_hitting / float(count_predict)
            #     AP += prec_k


            ############### # foresight for equal values. is it reasonable or better?
            if last_predict_value != None and similarity[index_s, index_t] == last_predict_value:
                last_count_predict += 1
                if eval_graph.has_edge(nodeID_s, nodeID_t):
                    last_count_hitting += 1
            else:
                for _ in range(last_count_hitting):
                    count_hitting += 1
                    count_predict += 1
                    AP += count_hitting / float(count_predict)
                for _ in range(last_count_hitting, last_count_predict):
                    count_predict += 1
                last_predict_value = similarity[index_s, index_t]
                last_count_predict = 1
                if eval_graph.has_edge(nodeID_s, nodeID_t):
                    last_count_hitting = 1
                else:
                    last_count_hitting = 0
        for _ in range(last_count_hitting):
            count_hitting += 1
            count_predict += 1
            AP += count_hitting / float(count_predict)
        ###############

        MAP_list.append(AP / float(count_hitting))
    MAP = np.mean(MAP_list)
    return MAP

def check_precK(nodeID_list, similarity, max_index, eval_graph, except_graph = None):
    dim = np.size(similarity, axis=0)
    sortedInd = np.argsort(similarity.reshape(-1))
    sortedInd = sortedInd[::-1]
    count_hitting = 0
    count_predict = 0
    precisionK = []
    last_predict_value = None # foresight for equal values. is it reasonable or better?
    last_count_predict = 0 # foresight for equal values. is it reasonable or better?
    last_count_hitting = 0 # foresight for equal values. is it reasonable or better?
    for ind in sortedInd:
        x, y = divmod(ind, dim)
        nodeID_s = nodeID_list[x]
        nodeID_t = nodeID_list[y]
        if nodeID_s == nodeID_t or (except_graph != None and except_graph.has_edge(nodeID_s, nodeID_t)):
            continue
        # count_predict += 1
        # if eval_graph.has_edge(nodeID_s, nodeID_t):
        #     count_hitting += 1
        # precisionK.append(count_hitting / float(count_predict))
        # if count_predict >= max_index:
        #     break
        #

        ################# # foresight for equal values. is it reasonable or better?
        if last_predict_value != None and similarity[x,y] == last_predict_value:
            last_count_predict += 1
            if eval_graph.has_edge(nodeID_s, nodeID_t):
                last_count_hitting += 1
        else:
            for _ in range(last_count_hitting):
                count_hitting += 1
                count_predict += 1
                precisionK.append(count_hitting / float(count_predict))
                if count_predict >= max_index:
                    return precisionK
            for _ in range(last_count_hitting, last_count_predict):
                count_predict += 1
                precisionK.append(count_hitting / float(count_predict))
                if count_predict >= max_index:
                    return precisionK
            last_predict_value = similarity[x, y]
            last_count_predict = 1
            if eval_graph.has_edge(nodeID_s, nodeID_t):
                last_count_hitting = 1
            else:
                last_count_hitting = 0
    if count_predict < max_index and last_count_predict > 0:
        for _ in range(last_count_hitting):
            count_hitting += 1
            count_predict += 1
            precisionK.append(count_hitting / float(count_predict))
            if count_predict >= max_index:
                return precisionK
        for _ in range(last_count_hitting, last_count_predict):
            count_predict += 1
            precisionK.append(count_hitting / float(count_predict))
            if count_predict >= max_index:
                return precisionK
    #################


    return precisionK


def f1_scores_singlelabel(true_values, predicted_values, labes_size):
    """
    evalute predicted values, for single-label multi-class classification.
    :param true_values: 1D to indicate labels index
    :param predicted_values: 1D to indicate predicted index
    :param saved_path: if None, not saved.
    :return:
    """
    # true positive for each class/label, (array of length equal to labels size)
    TP = np.array([np.sum((true_values==label) & (predicted_values==label), axis=0,dtype=np.int32)
                   for label in range(0,labes_size)], dtype=np.int32)
    # false positive
    FP = np.array([np.sum((true_values != label) & (predicted_values == label), axis=0, dtype=np.int32)
                   for label in range(0, labes_size)], dtype=np.int32)
    # true negative
    TN = np.array([np.sum((true_values != label) & (predicted_values != label), axis=0, dtype=np.int32)
                   for label in range(0, labes_size)], dtype=np.int32)
    # false negative
    FN = np.array([np.sum((true_values == label) & (predicted_values != label), axis=0, dtype=np.int32)
                   for label in range(0, labes_size)], dtype=np.int32)

    # precise for each label
    _P_t = TP / (TP + FP + 1e-9)
    # recall for each label
    _R_t = TP / (TP + FN + 1e-9)
    # macro F1
    Macro_F1 = np.mean((2 * _P_t * _R_t) / (_P_t + _R_t + 1e-9))
    # logger.info('Macro_F1: %.4f'%Macro_F1)


    # total precise
    _P = np.sum(TP) / (np.sum(TP) + np.sum(FP) + 1e-9)
    # total recall
    _R = np.sum(TP) / (np.sum(TP) + np.sum(FN) + 1e-9)
    # micro F1
    Micro_F1 = (2 * _P * _R) / (_P + _R)
    # logger.info('Micro_F1: %.4f' % Micro_F1)

    return Macro_F1, Micro_F1

def f1_scores_multilabel(true_values, predicted_values):
    """
    evalute predicted values, for multi-label multi-class classification.
    :param true_values: N*L label indicator
    :param predicted_values: N*L label indicator
    :return:
    """
    TP = np.sum(true_values & predicted_values, axis=0, dtype=np.int32)
    FP = np.sum((~true_values) & predicted_values, axis=0, dtype=np.int32)
    FN = np.sum(true_values & (~predicted_values), axis=0, dtype=np.int32)
    _P = np.sum(TP) / (np.sum(TP) + np.sum(FP) + 1e-11)
    _R = np.sum(TP) / (np.sum(TP) + np.sum(FN) + 1e-11)
    Micro_F1 = (2 * _P * _R) / (_P + _R)
    _P_t = TP / (TP + FP + 1e-11)
    _R_t = TP / (TP + FN + 1e-11)
    Macro_F1 = np.mean((2 * _P_t * _R_t) / (_P_t + _R_t + 1e-11))
    return Macro_F1, Micro_F1
    

if __name__ == '__main__':
    f = open('color_list', 'r')
    color_list = f.readlines()
    f.close()
    print(color_list[0].split()[0].strip(':').strip('\''))
    
    
    
    