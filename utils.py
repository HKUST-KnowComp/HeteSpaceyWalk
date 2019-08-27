#! /usr/bin/env python
# -*- coding:utf-8 -*-

"""
__author__ = "He Yu"
Version: 1.0.0

"""
import sys
import logging
import os
import time
import json
import numpy as np
import random
from gensim.models.keyedvectors import KeyedVectors


logger = logging.getLogger("HNE")



class ReDirection(object):
    """
    redirect sys.stdout and sys.stderr and sys.stdin.
    note: sys.stdout and sys.stderr and sys.stdin are all file-like objects,
    so we can define a file-like class to redirect them.
    file-like class: just have some named functions as write(), readline(), flush().
    """
    def __init__(self, sys_std_handler, redirect_handler):
        """
        :param sys_std_handler: one of [sys.stdout sys.stderr sys.stdin]
        :param redirect_handler: the redirect file handler. (for sys.stdin, it's the place to record input not receive input)
        """
        self._sys_std_handler = sys_std_handler  # save sys_std_handler to print to console.
        self._redirect_handler = redirect_handler

    def write_to_console(self, output_stream):
        """
        print on console
        :param output_stream:
        :return:
        """
        self._sys_std_handler.write(output_stream)
        self._sys_std_handler.flush()


    def write_to_file(self, output_stream):
        """
        redirect to file
        :param output_stream:
        :return:
        """
        self._redirect_handler.write(output_stream)
        self._redirect_handler.flush()

    def write(self,output_stream):
        """
        :param output_stream:
        :return:
        """
        self.write_to_console(output_stream)
        self.write_to_file(output_stream)

    def flush(self):
        """
        this method is essential even it is null, for it's a basic attribution of file-like object.
        :return:
        """
        pass

    def readline(self):
        read_line = self._sys_std_handler.readline()  # receive standard input from sys.stdin (like keyboard)
        self._redirect_handler.write('[!input]' + read_line)
        self._redirect_handler.flush()
        return read_line[:-1]  # -1 to discard the '\n' in input stream

    def close(self):
        self._sys_std_handler.close()

# redirect standard info(stdin,stdout,stderr) to files.
def redirect_stdinfo(filepath = './log/std.log'):
    if not os.path.exists(os.path.split(filepath)[0]):
        os.makedirs(os.path.split(filepath)[0])
    fr = open(filepath,'w')
    # sys.stdin = ReDirection(sys.stdin,fr)
    sys.stdout = ReDirection(sys.stdout,fr)
    sys.stderr = ReDirection(sys.stderr,fr)



# escape html characters , (full characters for ISO 8859-1).
class HTML_escape_dict(object):
    def __init__(self):

        self._HTML_characters_file = 'entities.json'
        with open(self._HTML_characters_file) as json_file:
            data = json.load(json_file)
            # data: {}
            #   entity name: {}
            #       codepoints/entity_number
            #       characters
            # print(data['&Aacute;']["codepoints"][0])

        self._character2entityname_dict, self._entityid2name_dict = self._construct_dict(data)

    def _construct_dict(self,data):
        character2entityname_dict = {}
        entityid2name_dict = {}
        for entity_name, value in data.items():
            if entity_name[-1] != ";":
                continue
            entity_id = value["codepoints"][0]
            characters = value["characters"]
            if entity_id not in entityid2name_dict:
                entityid2name_dict[entity_id] = entity_name
                character2entityname_dict[characters] = entity_name
            else:
                print('duplicated! {} #{}# {}'.format(characters, entity_id, entity_name))
        return character2entityname_dict, entityid2name_dict

    def html_escape_entityid2name(self, entity_id):
        if entity_id in self._entityid2name_dict:
            return self._entityid2name_dict[entity_id]
        else:
            print('escape failed, not find such character.')
            return None
    def html_escape_character2entityname(self, character):
        if character in self._character2entityname_dict:
            return self._character2entityname_dict[character]
        else:
            print('escape failed, not find such character.')
            return None

def html_escape():
    escape_dict = HTML_escape_dict()

    source = '/home/heyu/work/data/HIN/net_dbis/paper.txt'
    target = '/home/heyu/work/data/HIN/net_dbis/paper_escaped.txt'

    target_fr = open(target,'w')
    for line in open(source,encoding='ISO 8859-1'):
        # print(line)
        idx, name = line.strip().split('\t')
        target_fr.write(idx+'\t')
        for char in name:
            try:
                # get escaped entity id:  (by this, we find those speceial characters of asciii cannot encode)
                # char.encode('ascii', 'xmlcharrefreplace')
                # char.encode(encoding='UTF-8',errors='strict')
                char.encode(encoding='ascii', errors='strict')
            except:
                escaped = escape_dict.html_escape_character2entityname(char)
                print('<{}> escaped: <{}>'.format(char,escaped))
            else:
                escaped = char
            target_fr.write(escaped)
        target_fr.write('\n')
    target_fr.close()



# define dataset class, to wrap and integrate special operations on dataset, like generate batch-data.
class DataSet(object):
    def __init__(self, data, labels = None, shuffled = True):
        self._data = data
        self._labels = labels
        self._shuffled = shuffled
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = len(data)

        if labels is not None:
            assert len(data) == len(labels), 'data and labels must have equal size, {}!={}'.format(len(data), len(labels))

        # Shuffle the data
        if shuffled:
            perm = np.arange(self._num_examples)
            random.shuffle(perm)
            self._data = data[perm]
            if labels is not None:
                self._labels = labels[perm]

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, keep_strict_batching = True):
        """ Return the next `batch_size` examples from this data set."""
        if keep_strict_batching:
            assert batch_size <= self._num_examples

        if self._index_in_epoch >= self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            if self._shuffled:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data = self._data[perm]
                if self._labels is not None:
                    self._labels = self._labels[perm]
            # Start next epoch
            self._index_in_epoch = 0

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            if keep_strict_batching:
                # Finished epoch
                self._epochs_completed += 1
                # Shuffle the data
                if self._shuffled:
                    perm = np.arange(self._num_examples)
                    np.random.shuffle(perm)
                    self._data = self._data[perm]
                    if self._labels is not None:
                        self._labels = self._labels[perm]
                # Start next epoch
                start = 0
                self._index_in_epoch = batch_size
            else:
                self._index_in_epoch = self._num_examples
        end = self._index_in_epoch

        batch_data = self._data[start:end]
        if self._labels is not None:
            batch_labels = self._labels[start:end]

        if self._labels is not None:
            return batch_data, batch_labels
        else:
            return batch_data

    def get_batch(self, index):
        batch_input = self.data[index]
        return batch_input

# Decay the learning rate exponentially based on the number of steps.
# decayed_learning_rate = initial_learning_rate * decay_rate ^ (global_step / decay_steps)
class LearningRateGenerator(object):
    def __init__(self, initial_learning_rate, initial_steps, decay_rate, decay_steps, iter_steps = None):
        self._initial_learning_rate = initial_learning_rate
        self._initial_steps = initial_steps
        self._decay_rate = decay_rate
        self._decay_steps = decay_steps
        self._current_learning_rate = initial_learning_rate
        self._iter_steps = iter_steps
    @property
    def learning_rate(self):
        return self._current_learning_rate
    @property
    def iter_steps(self):
        return self._iter_steps
    def reset(self, initial_learning_rate = None, initial_steps = None, decay_rate = None, decay_steps = None, iter_steps = None):
        if initial_learning_rate is not None:
            self._initial_learning_rate = initial_learning_rate
            self._current_learning_rate = initial_learning_rate
        if initial_steps is not None:
            self._initial_steps = initial_steps
        if decay_rate is not None:
            self._decay_rate = decay_rate
        if decay_steps is not None:
            self._decay_steps = decay_steps
        if iter_steps is not None:
            self._iter_steps = iter_steps
    def exponential_decay(self, global_step,
                          decay_rate = None, decay_steps = None, staircase=True, iter_steps = None):
        if decay_rate is not None:
            self._decay_rate = decay_rate
        if decay_steps is not None:
            self._decay_steps = decay_steps
        if iter_steps is not None:
            self._iter_steps = iter_steps

        if self._decay_steps > 0:
            exponential = np.divide(global_step - self._initial_steps, self._decay_steps)
            if staircase:
                exponential = int(exponential)
            self._current_learning_rate = np.multiply(self._initial_learning_rate, np.power(self._decay_rate, exponential))
        else:
            self._current_learning_rate = self._initial_learning_rate
        return self._current_learning_rate




# Alias sampling
def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K, dtype=np.float32)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    return the sampled index.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


# negative-sampling
def neg_sample(population, except_set, num = 5, max_repeat = 5, alias_table = None):
    # population: range(size)
    if alias_table == None:
        sample_list = []
        for v in random.sample(population, k = num):
            if v not in except_set:
                sample_list.append(v)
                except_set.add(v)
        repeat = 1
        while len(sample_list) < num and repeat < max_repeat:
            for v in random.sample(population, k = num - len(sample_list)):
                if v not in except_set:
                    sample_list.append(v)
                    except_set.add(v)
            repeat += 1
        if len(sample_list) < num:
            sample_list.extend(random.sample(population, k = num - len(sample_list)))
        return sample_list
    else:
        assert len(population) == len(alias_table[0]), "error: {} != {}".format(len(population), len(alias_table[0]))
        sample_list = []
        for i in range(num):
            for j in range(max_repeat):
                v = population[alias_draw(alias_table[0], alias_table[1])]
                if v not in except_set:
                    sample_list.append(v)
                    except_set.add(v)
                    break
        for i in range(num - len(sample_list)):
            v = population[alias_draw(alias_table[0], alias_table[1])]
            sample_list.append(v)
        return sample_list


def unigram_sample(population, size=1, replace=True, weight=None):
    weight_sum = np.sum(weight)
    if weight_sum != 1:
        weight = weight / weight_sum
        weight_sum = np.sum(weight)
        if weight_sum != 1:
            weight = weight / weight_sum
            weight_sum = np.sum(weight)
            if weight_sum != 1:
                weight = weight / weight_sum
    return np.random.choice(population,size=size, replace=replace, p=weight)






# reset the file's or path's last modification time and access time by using os.utime().
def set_mtine_atime(path,mtime=None,atime=None):
    os.utime(path, (atime, mtime)) # Set the access and modified time of the file to the given values.
    # os.utime(path, None) #  set the access and modified times to the current time.





def check_rebuild(filebase, descrip, always_rebuild = False):
    flag = True
    if os.path.exists(filebase):
        if always_rebuild:
            logger.warning('path \'%s\' already exits, %s will rebuild. '
                           '(Continue running after 10 seconds...)' % (filebase, descrip))
            time.sleep(10)
            logger.warning('Continue running...')
            if os.path.isfile(filebase):
                os.remove(filebase)
            else:
                os.rmdir(filebase)
        else:
            logger.warning('path \'%s\' already exits, %s may rebuild... ' % (filebase, descrip))
            str_input = input("Enter yes to rebuild %s or enter others to skipped: (y/others)" % descrip)
            if str_input in ['y', 'Y', 'yes', 'YES']:
                logger.warning('%s will rebuild, continue running...'%descrip)
                if os.path.isfile(filebase):
                    os.remove(filebase)
                else:
                    os.rmdir(filebase)
            else:
                logger.warning('Skipped, continue running...')
                flag = False

    if flag:
        if not os.path.exists(os.path.split(filebase)[0]):
            os.makedirs(os.path.split(filebase)[0])
    return flag



# check whether there is propriate division:
# if there is,return 1
# if there is not:
#   -if always rebuild:
#   -if not return false.
#def check_data_division(filebase, m = 9, n = 1, rebuild = True, ):
#    flag = True
#    data_path = filebase + 'division/' + str(m) + 'to' + str(n)
#    if not os.path.exists(data_path):
#        if rebuild:
#            logger.warning()
#            os.mkdirs(data_path)
#        else:
#            logger.error('In mission link-prediction: no divided data, and no rebuild.')
#            flag = False
#            return flag
#    return flag


def get_labeled_data(label_filepath, multilabel_rule = "all", type = None, type_filepath = None):
    """
    get labeled data: id-label
    :param label_filepath:
    :return: dict
    """
    nodes_type_dict = {}
    if type != None and type >=0:
        for line in open(type_filepath):
            line = line.strip()
            if line:
                linelist = line.split("\t")
                node = int(linelist[0].strip())
                node_type = int(linelist[1].strip())
                nodes_type_dict[node] = node_type

    id_labels = {}
    for line in open(label_filepath):
        linelist = line.strip().split('\t')
        if len(linelist) == 0 :
            logger.warning('empty line!')
            continue
        elif len(linelist) == 1:
            # logger.warning('label may be lacked in line: %s'%line)
            logger.error('label may be lacked in line: %s'%line)
            continue
        id = int(linelist[0])
        if len(nodes_type_dict) == 0 or nodes_type_dict[id] == type:
            if id not in id_labels:
                id_labels[id] = []
            labels = [int(x) for x in linelist[1:]]
            for l in labels:
                if l not in id_labels[id]:
                    id_labels[id].append(l)

    logger.info('\t multilabel_rule: {}'.format(multilabel_rule))
    ruled_id_labels = {}
    for id, labels in id_labels.items():
        if multilabel_rule == "all":
            ruled_id_labels[id] = labels
        elif multilabel_rule == "first":
            ruled_id_labels[id] = [labels[0]]
        elif multilabel_rule == "random":
            ruled_id_labels[id] = [random.choice(labels)]
        elif multilabel_rule == "ignore":
            if len(labels) > 1:
                pass
            else:
                ruled_id_labels[id] = labels
        else:
            logger.error("error! invalid multilabel_rule: {}".format(multilabel_rule))
    id_list = sorted(ruled_id_labels.keys())
    label_list = [ruled_id_labels[x] for x in id_list]
    return id_list, label_list

def get_vectors(wv, id_list, label_list = None, missing_rule = "ignore"):
    # vec_list = []
    # vec_label_list = []
    ret_id_list = []
    vector_list = []
    vector_label_list = []
    if label_list == None:
        for i in range(len(id_list)):
            word_id = id_list[i]
            word = str(word_id)
            if word not in wv.vocab:
                logger.warning('word \'%s\' not in vocabulary, missing_rule: %s...'%(word, missing_rule))
                if missing_rule == "random":
                    vec = np.reshape(np.random.uniform(size=[wv.vector_size]), (-1,1))
                    vector_list.append(vec)
                    ret_id_list.append(word_id)
            else:
                vec = np.reshape(wv.word_vec(word), (1, -1))
                vector_list.append(vec)
                ret_id_list.append(word_id)
        return ret_id_list, np.concatenate(vector_list,axis=0)
    else:
        for i in range(len(id_list)):
            word_id = id_list[i]
            word = str(word_id)
            if word not in wv.vocab:
                logger.warning('word \'%s\' not in vocabulary, missing_rule: %s...'%(word, missing_rule))
                if missing_rule == "random":
                    vec = np.reshape(np.random.uniform(size=[wv.vector_size]), (-1,1))
                    vector_list.append(vec)
                    ret_id_list.append(word_id)
                    vector_label_list.append(label_list[i])
            else:
                vec = np.reshape(wv.word_vec(word), (1, -1))
                vector_list.append(vec)
                vector_label_list.append(label_list[i])
                ret_id_list.append(word_id)
        return ret_id_list, np.concatenate(vector_list,axis=0), vector_label_list


def get_KeyedVectors(vectors_path):
    if os.path.exists(vectors_path):
        logger.info('loading vectors from {}'.format(vectors_path))
        time_start = time.time()
        model_wv = KeyedVectors.load_word2vec_format(vectors_path, binary=False)
        logger.info('loading vectors completed in {}s'.format(time.time() - time_start))
        return model_wv
    else:
        logger.error('vectors not exist, exit')
        sys.exit()


def get_features(feature_path, dtype = np.float32):
    # TODO: normalize the format like data_labels
    return np.loadtxt(feature_path, dtype=dtype) # shape=[node_size, feature_size]




def check_files_exists(files):
    for f in files:
        if os.path.exists(f):
            return True
    return False

# save vectors for tensorflow
def save_word2vec_format_and_ckpt(vector_path, vectors, checkpoint_path, sess, saver, global_step_value, types_size):
    vocab_size = np.size(vectors, axis=0)
    vector_size = np.size(vectors, axis=1)

    ## synchrolock for multi-process:
    # if os.path.exists(vector_path):
    #     while time.time() - os.stat(vector_path).st_mtime < 200:
    #         time.sleep(200 - (time.time() - os.stat(vector_path).st_mtime))
    #     os.utime(vector_path, None)
    reading_classify = vector_path+".reading_classify"
    reading_cluster = vector_path + ".reading_cluster"
    reading_visualization = vector_path + ".reading_visualization"
    reading_link_prediction = vector_path + ".reading_link_prediction"
    writing = vector_path+".writing"
    logger.info("\t declare for writing ...")
    open(writing, "w") # declare
    time.sleep(30)
    files = set()
    for node_type in range(types_size):
        files.add(reading_classify + "_{}".format(node_type))
        files.add(reading_cluster + "_{}".format(node_type))
        files.add(reading_visualization + "_{}".format(node_type))
    for start_type in range(types_size):
        for target_type in range(types_size):
            files.add(reading_link_prediction+"_{}_{}".format(start_type, target_type))
    while check_files_exists(files):
        time.sleep(5)
    with open(vector_path, 'w') as fr:
        fr.write('%d %d\n' % (vocab_size, vector_size))
        for index in range(vocab_size):
            fr.write('%d ' % index)
            for one in vectors[index]:
                fr.write('{} '.format(one))
            fr.write('\n')
            fr.flush()
    saver.save(sess, checkpoint_path, global_step=global_step_value)
    os.remove(writing)
    logger.info("\t done for writing ...")


# get random generator seed:
def get_random_seed():
    time_str = str(time.time()).split(".")
    if len(time_str) == 1:
        time_str.append("")
    time_s_str = "000"+time_str[0]
    time_ms_str = time_str[1]+"000"
    return int(str(random.randint(1,999)) + time_s_str[-3:] + time_ms_str[0:3])



if __name__ == '__main__':
    # html_escape()
    labels = [11,22,33,44,55]
    print(random.choice(labels+list(set())))
