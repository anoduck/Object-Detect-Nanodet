#!/usr/bin/env python3
# Copyright (c) 2023 Anoduck
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
# -------------------------------------------------------------
import os
import cv2
import numpy as np
import torch
from time import time
from .nanodet.nanodet.util import cfg, load_config, Logger, load_model_weight
from .nanodet.nanodet.util import overlay_bbox_cv
from dataclasses import dataclass
from simple_parsing import parse
from transformers import YolosImageProcessor, YolosForObjectDetection
from transformers import pipeline
from PIL import Image
from alive_progress import alive_bar
# ---------------------------------------------
from .nanodet.nanodet.data.batch_process import stack_batch_img
from .nanodet.nanodet.data.collate import naive_collate
from .nanodet.nanodet.model.arch import build_model
from .nanodet.nanodet.util.path import mkdir

# Variables
root = os.path.abspath(os.path.dirname(__file__))
cfg = 
image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]


@dataclass
class Options:
    """obj-nanodet.py = A speedier way to detect objects in images"""
    dir: str = os.path.join(root, 'images')  # directory containing images
    cfg: str = os.path.join(root, 'nanodet', 'config', 'nanodet.yml')  # model config file
    sav: bool = True  # save results
    rdir: str = os.path.join(root, 'results')  # directory to save results
    cid: int = 0  # camera id
    conf: float = 0.25  # confidence threshold


Options = parse(Options, dest="Options")


class Predictor(object):
    def __init__(self, cfg, logger):
        self.cfg = cfg
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from .nanodet.nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = YolosForObjectDetection.from_pretrained(
            'hustvl/yolos-tiny')
        ckpt = torch.load(self.model, map_location=lambda storage, loc: storage)
        load_model_weight(self.model, ckpt, logger)
        self.pipe = pipeline("object-detection", model="hustvl/yolos-tiny")

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipe(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(
            meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = self.model.head.show_result(
            meta["raw_img"][0], dets, class_names, score_thres=score_thres, show=True
        )
        print("viz time: {:.3f}s".format(time.time() - time1))
        return result_img


# -----------------------------------------------------------------------------
predictor = Predictor(cfg=cfg, model_path=model_path, logger=logger, device="cuda:0")
meta, res = predictor.inference(img)

# -----------------------------------------------------------------------------
image = Image.open(requests.get(url, stream=True).raw)

model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# model predicts bounding boxes and corresponding COCO classes
logits = outputs.logits
bboxes = outputs.pred_boxes


# print results
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(
    outputs, threshold=0.9, target_sizes=target_sizes)[0]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )

# -----------------------------------------------------------------------------


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names


def main(Options):
    local_rank = 0
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    load_config(cfg, Options.cfg)
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = Predictor(cfg, Options.mod)
    logger.log('Press "Esc", "q" or "Q" to exit.')
    current_time = time.localtime()
    if args.demo == "image":
        if os.path.isdir(Options.dir):
            files = get_image_list(Options.dir)
        else:
            files = [Options.dir]
        files.sort()
        for image_name in files:
            meta, res = predictor.inference(image_name)
            result_image = predictor.visualize(
                res[0], meta, cfg.class_names, 0.35)
            if Options.sav:
                save_folder = os.path.join(
                    Options.rdir, time.strftime(
                        "%Y_%m_%d_%H_%M_%S", current_time)
                )
                mkdir(save_folder)
                save_file_name = os.path.join(
                    save_folder, os.path.basename(image_name))
                cv2.imwrite(save_file_name, result_image)
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break


if __name__ == "__main__":
    main(Options)