# -*- coding: utf-8 -*-
import mmcv
from mmdet.apis import init_detector
import os, json
import numpy as np
from tqdm import tqdm
import torch
from functools import partial
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
import argparse
import cv2
from tools.utils import PALETTE


class DataPrefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                self.batch[k] = scatter(self.batch[k], [self.device])[0]

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, json_file, model):
        img_names = os.listdir(img_dir)
        img_names.sort()
        imgs = []
        for name in img_names:
            img_path = os.path.join(img_dir, name)
            imgs.append(img_path)

        self.imgs = imgs

        cfg = model.cfg
        self.device = next(model.parameters()).device  # model device
        # test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = cfg.data.test.pipeline[1:]
        self.test_pipeline = Compose(test_pipeline)

        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        images_info = json_data['images']

        self.filename2id = {}
        for img_info in tqdm(images_info):
            file_name = img_info['file_name']
            id = img_info['id']
            self.filename2id[file_name] = id
        # print(self.filename2id)

    def loadImg(self, results):
        results['filename'] = results['img']
        results['ori_filename'] = results['img']
        img = mmcv.imread(results['img'])
        if img is None:
            print(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __getitem__(self, item):
        img = self.imgs[item]
        data = dict(img=img)
        data = self.loadImg(data)
        data = self.test_pipeline(data)
        return data

    def __len__(self):
        return len(self.imgs)



def inference_model_with_loader(config_file, checkpoint_file, img_dir, json_file, out, thresh, thresh_box):
    # build the model from a config file and a checkpoint file
    print('loading model...')
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    print('loading complete!')
    # 测试多张图片
    dataset = EvalDataset(img_dir=img_dir, json_file=json_file, model=model)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         collate_fn=partial(collate, samples_per_gpu=1),
                                         num_workers=8,
                                         pin_memory=True)
    prefetcher = DataPrefetcher(loader, device=dataset.device)
    result = []
    pbar = mmcv.ProgressBar(len(dataset))
    with torch.no_grad():
        batch = prefetcher.next()
        i = 0
        while batch is not None:
            pbar.update()
            res = model(return_loss=False, rescale=True, **batch)[0]

            bboxes = np.vstack(res)
            labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(res)]
            labels = np.concatenate(labels)

            # double threshold
            # if len(bboxes) > 0:
                # det_flag = False
                # for j, bbox in enumerate(bboxes):
                    # if float(bbox[4]) > thresh:
                    #     det_flag = True
                    #     break
                # if det_flag is True:
                #     name = dataset.imgs[i].split('/')[-1]
                #     # save_pred(dataset.imgs[i], bboxes, labels)
                #     for j, bbox in enumerate(bboxes):
                #         if float(bbox[4]) > thresh_box:
                #             xmin, ymin, xmax, ymax = [round(float(x), 4) for x in bbox[:4]]
                #             res_line = {'image_id': dataset.filename2id[name],
                #                         'bbox': [xmin, ymin, xmax-xmin, ymax-ymin],
                #                         'category_id': int(labels[j] + 1),
                #                         'score': float(bbox[4])}
                #             result.append(res_line)
            if len(bboxes) > 0:
                name = dataset.imgs[i].split('/')[-1]
                # save_pred(dataset.imgs[i], bboxes, labels)
                for j, bbox in enumerate(bboxes):
                    xmin, ymin, xmax, ymax = [round(float(x), 4) for x in bbox[:4]]
                    res_line = {'image_id': dataset.filename2id[name],
                                'bbox': [xmin, ymin, xmax-xmin, ymax-ymin],
                                'category_id': int(labels[j] + 1),
                                'score': float(bbox[4]) / 2 + 0.5}
                    result.append(res_line)

            i += 1
            batch = prefetcher.next()

    with open(out, 'w') as fp:
        json.dump(result, fp, indent=4)

    print('over!')

def save_pred(img_path, bboxes, labels, save_dir='./prediction'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img = cv2.imread(img_path)

    for j, bbox in enumerate(bboxes):
        if float(bbox[4]) > thresh_box:
            xmin, ymin, xmax, ymax = [int(x) for x in bbox[:4]]
            category =  str(int(labels[j])+1)
            score = float(bbox[4])

            cv2.rectangle(img, (xmin,ymin), (xmax,ymax), PALETTE[int(labels[j])], 2)
            cv2.putText(img, category+'_'+str(score)[:4], (xmin,ymin-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, PALETTE[int(labels[j])], 2)
    cv2.imwrite(os.path.join(save_dir, img_path.split('/')[-1]), img)



###pipeline
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval result")
    parser.add_argument("--config", help="config file")
    parser.add_argument("--checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="config file")
    parser.add_argument("--img_dir", default='data/trademark/test/images')
    parser.add_argument("--json_file", default='data/trademark/test/annotations/instances_val2017.json')

    args = parser.parse_args()
    config_file = args.config
    checkpoint_file = args.checkpoint
    img_dir = args.img_dir
    json_file = args.json_file
    out = args.out
    thresh = 1e-3
    thresh_box = 1e-3

    inference_model_with_loader(config_file, checkpoint_file,
                                img_dir, json_file, out, thresh, thresh_box)
