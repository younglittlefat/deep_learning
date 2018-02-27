#encoding=utf-8
import sys
import os
import json

from PIL import Image
import numpy as np

from feature_extractor import YouTube8MFeatureExtractor

reload(sys)
sys.setdefaultencoding("utf-8")


def extract_one_pic(extractor, pic_path):
    """
    获取一张图片的youtube 8m特征
    """
    im = np.array(Image.open(pic_path))
    features = extractor.extract_rgb_frame_features(im)
    return features


def get_raw_frames(frame_path):
    """
    return: dict{mp4_name: pic_list}
    """
    result = {}
    all_mp4 = os.listdir(frame_path)
    all_mp4 = filter(lambda x:os.path.isdir(os.path.join(frame_path, x)), all_mp4)
    for mp4 in all_mp4:
        now_path = os.path.join(frame_path, mp4)
        all_pics = os.listdir(now_path)
        all_pics = filter(lambda x:x.endswith("jpg"), all_pics)
        all_pics = [(ele, int(ele.split(".")[0])) for ele in all_pics]
        all_pics.sort(key = lambda x:x[1])
        all_pics = [ele[0] for ele in all_pics]
        result[mp4] = map(lambda x:os.path.join(now_path, x), all_pics)

    return result



def get_all_features():

    extractor = YouTube8MFeatureExtractor("./model")
    video_dict = get_raw_frames("./frames")
    output_path = "./video_features"
    
    for key in video_dict:
        feature_list = []
        pics_list = video_dict[key]
        for pic_path in pics_list:
            features = extract_one_pic(extractor, pic_path)
            feature_list.append(features)
        final_arr = np.array(feature_list)
        print "save: %s" % (key)
        np.save(os.path.join(output_path, key.split(".")[0]+".npy"), final_arr)



get_all_features()
