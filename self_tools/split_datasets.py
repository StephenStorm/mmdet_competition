#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 将一个文件夹下图片按比例分在两个文件夹下
import os
import random
from shutil import copy2
import json
import argparse
import mmcv


def main(args):
    all_data = os.listdir(args.img_path)  # （图片文件夹）

    random.seed(1)
    random.shuffle(all_data)  # 第一次打乱
    all_data_img = []
    for i in all_data:
        if i.endswith(".jpg"):
            all_data_img.append(i)
    num_all_data = len(all_data_img)
    print("num_all_data: " + str(num_all_data))
    index_list = list(range(num_all_data))
    random.seed(2)
    random.shuffle(index_list)  # 第二次打乱
    num = 0

    trainDir = os.path.join('./train_val_split', "train")  # （将训练集放在这个文件夹下）
    if not os.path.exists(trainDir):
        os.makedirs(trainDir)

    validDir = os.path.join('./train_val_split', "val")  # （将验证集放在这个文件夹下）
    if not os.path.exists(validDir):
        os.makedirs(validDir)

    train_list = []
    val_list = []
    for i in index_list:
        fileName = os.path.join(args.img_path, all_data_img[i])
        if num < num_all_data * 0.8:  # 这里可是设置train,val的比例
            train_list.append(all_data_img[i])
            copy2(fileName, os.path.join(trainDir, all_data_img[i]))
        else:
            val_list.append(all_data_img[i])
            copy2(fileName, os.path.join(validDir, all_data_img[i]))
        num += 1

    print("train_nums", len(train_list))
    print("val_nums", len(val_list))

    # data = json.load(open(args.json_file))

    with open(args.json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # data = mmcv.load(args.json_file)

    # print(train_list)

    train_json_dict = {
        "images": [],
        "annotations": [],
        "categories": [],
        "type": "instances"
    }
    val_json_dict = {
        "images": [],
        "annotations": [],
        "categories": [],
        "type": "instances"
    }
    id2name = {}
    # images
    for i in data['images']:
        if i['file_name'] in train_list:
            train_json_dict['images'].append(i)
        if i['file_name'] in val_list:
            val_json_dict['images'].append(i)
        id = i['id']
        filename = i['file_name']
        id2name[id] = filename

    # annotations
    for j in data['annotations']:
        # j['category_id'] -= 1  # 类别从1开始
        if id2name[j["image_id"]] in train_list:
            train_json_dict['annotations'].append(j)
        if id2name[j["image_id"]] in val_list:
            val_json_dict['annotations'].append(j)

    # categories ，类别从1开始
    for k in data['categories']:
        # k['id'] -= 1
        train_json_dict['categories'].append(k)
        val_json_dict['categories'].append(k)

    with open(os.path.join('./train_val_split', "train.json"), "w") as f:
        json.dump(train_json_dict, f, indent=2, ensure_ascii=False)

    with open(os.path.join('./train_val_split', "val.json"), "w") as f:
        json.dump(val_json_dict, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start convert.')
    parser.add_argument('--img_path', type=str, default='train/images/')  # json文件路径
    parser.add_argument('--json_file', type=str, default='train/annotations/instances_train2017.json')  # json文件路径
    args = parser.parse_args()
    main(args)