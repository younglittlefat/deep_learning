#encoding=utf-8
import sys
import os
import json
import random

import numpy as np


class InputHelper:

    def __init__(self):
        self.label_dict = None
        self.feat_dict = None
        self.train_list = []
        self.test_list = []
        self.need_label_balance = True
        self.test_per = 0.1

    def load_data(self, data_path, label_path):
        """
        读取数据
        """
        # label
        self.data_path = data_path
        self.label_dict = {0:[], 1:[]}
        with open(label_path, "r") as f:
            for line in f:
                temp = line.strip().split(" ")
                self.label_dict[int(temp[2])].append((temp[0], temp[1]))

#        file_list = os.listdir(data_path)
#        self.feat_dict = {}
#        for filename in file_list:
#            full_path = os.path.join(data_path, filename)
#            video_name = filename.split(".")[0] + ".mp4"
#            self.feat_dict[video_name] = np.load(full_path)


        # process data
        neg_num = len(self.label_dict[0])
        pos_num = len(self.label_dict[1])

        if self.need_label_balance:
            if pos_num < neg_num:
                random.shuffle(self.label_dict[0])
                self.label_dict[0] = self.label_dict[0][:pos_num*3]
            else:
                random.shuffle(self.label_dict[1])
                self.label_dict[1] = self.label_dict[1][:neg_num*3]
        print "pos num:%d, neg num:%d" % (len(self.label_dict[1]), len(self.label_dict[0]))

        # do shuffle
        x1_list = []
        x2_list = []
        y_list = []
        for key in self.label_dict:
            for ele in self.label_dict[key]:
#                x1_feat = self.feat_dict[ele[0]]
#                x1_list.append(x1_feat)
#                x2_feat = self.feat_dict[ele[1]]
#                x2_list.append(x2_feat)
#                y_list.append(key)
                x1_list.append(ele[0])
                x2_list.append(ele[1])
                y_list.append(key)

        x1 = np.array(x1_list)
        x2 = np.array(x2_list)
        y = np.array(y_list)
        print "x1 shape:"
        print x1.shape
        print "x2 shape:"
        print x2.shape
        print "y shape:"
        print y.shape
        shuffle_indice = np.random.permutation(y.shape[0])
        x1_shuffled = x1[shuffle_indice]
        x2_shuffled = x2[shuffle_indice]
        y_shuffled = y[shuffle_indice]
        assert(x1_shuffled.shape[0] == x2_shuffled.shape[0] and x1_shuffled.shape[0] == y_shuffled.shape[0])

        # split train and test
        test_num = -1 * int(y_shuffled.shape[0] * self.test_per)
#        x1_train, x1_dev = x1_shuffled[:test_num, :, :], x1_shuffled[test_num:, :, :]
#        x2_train, x2_dev = x2_shuffled[:test_num, :, :], x2_shuffled[test_num:, :, :]
        x1_train, x1_dev = x1_shuffled[:test_num], x1_shuffled[test_num:]
        x2_train, x2_dev = x2_shuffled[:test_num], x2_shuffled[test_num:]
        y_train, y_dev = y_shuffled[:test_num], y_shuffled[test_num:]
        print "train/dev split: %d/%d" % (y_train.shape[0], y_dev.shape[0])

        # stat
        train_pos = 0
        train_neg = 0
        dev_pos = 0
        dev_neg = 0
        for i in range(y_train.shape[0]):
            if y_train[i] == 0:
                train_neg += 1
            elif y_train[i] == 1:
                train_pos += 1
        for i in range(y_dev.shape[0]):
            if y_dev[i] == 0:
                dev_neg += 1
            elif y_dev[i] == 1:
                dev_pos += 1

        print "train pos/neg:  %d/%d" % (train_pos, train_neg)
        print "dev pos/neg:  %d/%d" % (dev_pos, dev_neg)

        self.train_set = (x1_train, x2_train, y_train)
        self.dev_set = (x1_dev, x2_dev, y_dev)


    def batch_iter(self, data_label, batch_size, epochs_num):
        """
        data_label: 0 is train, 1 is test
        """
        if data_label == 0:
            data = self.train_set
        elif data_label == 1:
            data = self.dev_set
        data = np.array(list(zip(data[0], data[1], data[2])))
        num_batches_per_epoch = int(data.shape[0] / batch_size) + 1
        for epoch in range(epochs_num):
            shuffle_indices = np.random.permutation(data.shape[0])
            shuffled_data = data[shuffle_indices]

            for batch_idx in range(num_batches_per_epoch):
                start_index = batch_idx * batch_size
                end_index = min((batch_idx + 1) * batch_size, data.shape[0])
                final_list = []
                self.name_list = []
                for i in range(start_index, end_index):
                    x1_path = shuffled_data[i][0].split(".")[0] + ".npy"
                    x2_path = shuffled_data[i][1].split(".")[0] + ".npy"
                    self.name_list.append((shuffled_data[i][0], shuffled_data[i][1]))
                    x1 = np.load(os.path.join(self.data_path, x1_path))
                    x2 = np.load(os.path.join(self.data_path, x2_path))
                    y = shuffled_data[i][2]
                    final_list.append([x1, x2, y])
                yield np.array(final_list)
            


if __name__ == "__main__":
    ih = InputHelper()
    ih.load_data("./video_features", "./label/label")
    bi = ih.batch_iter(0, 32, 20)
    n = bi.next()
    print n
    print n.shape

