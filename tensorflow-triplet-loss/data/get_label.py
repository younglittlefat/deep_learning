import sys
import os
import codecs

filename = sys.argv[1]
all_dict = {}
idx = 0
with open(filename) as f:
    for line in f:
        line = line.strip()
        if line.startswith(codecs.BOM_UTF8):
            line = line[len(codecs.BOM_UTF8):]
        if line != "":
            if idx not in all_dict:
                all_dict[idx] = []
            all_dict[idx].append(line)
        else:
            idx += 1

new_list = []
for key in all_dict:
    arr = all_dict[key]
    if len(arr) > 2:
        new_list.append(arr)

for idx, arr in enumerate(new_list):
    for ele in arr:
        print "%s %s" % (ele, idx)
