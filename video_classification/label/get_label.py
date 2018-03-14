#encoding=utf-8
import sys
import os
import codecs

def read_data(path):

    data_list = []
    temp_list = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(codecs.BOM_UTF8):
                line = line[len(codecs.BOM_UTF8):]
            if line == "":
                data_list.append(temp_list)
                temp_list = []
            else:
                temp_list.append(line.strip())
        data_list.append(temp_list)

    return data_list


def output_label(data_list):
    result_list = []
    for idx, simi_list in enumerate(data_list):
        # get simi
        for i in range(len(simi_list)):
            for j in range(i+1, len(simi_list)):
                result_list.append((simi_list[i], simi_list[j], 1))

        # get not-simi
        for i in range(len(simi_list)):
            for oidx, o_simi_list in enumerate(data_list):
                if oidx == idx:
                    continue
                for ele in o_simi_list:
                    result_list.append((simi_list[i], ele, 0))


    result_list.sort(key = lambda x:x[2])
    for ele in result_list:
        print "%s %s %s" % (ele[0], ele[1], ele[2])

data_list = read_data("simi")
output_label(data_list)
