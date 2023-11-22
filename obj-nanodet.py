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
from .nanodet.util import cfg,load_config, Logger
from .nanodet.util import overlay_bbox_cv
from dataclasses import dataclass
from simple_parsing import parse
from transformers import pipeline
from PIL import Image
from alive_progress import alive_bar
# ---------------------------------------------
from .nanodet.data.batch_process import stack_batch_img
from .nanodet.data.collate import naive_collate
from .nanodet.data.transform import Pipeline
from .nanodet.model.arch import build_model
from .nanodet.util import Logger, cfg, load_config, load_model_weight
from .nanodet.util.path import mkdir

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]

pipe = pipeline("object-detection", model="hustvl/yolos-tiny")

# Variables
root = os.path.abspath(os.path.dirname(__file__))
config_path = os.path.join(root, 'nanodet', 'config', 'nanodet-m.yml')
model_path = ''

load_config(cfg, config_path)
logger = Logger(-1, use_tensorboard=False)


@dataclass
class Options:
    dir: str = os.path.join(root, 'images')  # directory containing images
    cfg: str = config_path  # model config file
    model: str = model_path  # model file
    save: bool = True  # save results
    camid: int = 0  # camera id


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipe = pipe(
            cfg.data.val.pipe, cfg.data.val.keep_ratio)

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
meta, res = predictor.inference(image_path)

# -----------------------------------------------------------------------------
def cv2_imshow(a, convert_bgr_to_rgb=True):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
        a: np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
            (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
            image.
        convert_bgr_to_rgb: switch to convert BGR to RGB channel.
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if convert_bgr_to_rgb and a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display(Image.fromarray(a))


result = overlay_bbox_cv(meta['raw_img'][0], res[0],
                         cfg.class_names, score_thresh=0.35)


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


def main():
    args = parse_args()
    local_rank = 0
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    load_config(cfg, args.config)
    logger = Logger(local_rank, use_tensorboard=False)
    predictor = Predictor(cfg, args.model, logger, device="cuda:0")
    logger.log('Press "Esc", "q" or "Q" to exit.')
    current_time = time.localtime()
    if args.demo == "image":
        if os.path.isdir(args.path):
            files = get_image_list(args.path)
        else:
            files = [args.path]
        files.sort()
        for image_name in files:
            meta, res = predictor.inference(image_name)
            result_image = predictor.visualize(
                res[0], meta, cfg.class_names, 0.35)
            if args.save_result:
                save_folder = os.path.join(
                    cfg.save_dir, time.strftime(
                        "%Y_%m_%d_%H_%M_%S", current_time)
                )
                mkdir(local_rank, save_folder)
                save_file_name = os.path.join(
                    save_folder, os.path.basename(image_name))
                cv2.imwrite(save_file_name, result_image)
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
    elif args.demo == "video" or args.demo == "webcam":
        cap = cv2.VideoCapture(args.path if args.demo ==
                               "video" else args.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        save_folder = os.path.join(
            cfg.save_dir, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        mkdir(local_rank, save_folder)
        save_path = (
            os.path.join(save_folder, args.path.replace(
                "\\", "/").split("/")[-1])
            if args.demo == "video"
            else os.path.join(save_folder, "camera.mp4")
        )
        print(f"save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(
                *"mp4v"), fps, (int(width), int(height))
        )
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                meta, res = predictor.inference(frame)
                result_frame = predictor.visualize(
                    res[0], meta, cfg.class_names, 0.35)
                if args.save_result:
                    vid_writer.write(result_frame)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break


if __name__ == "__main__":
    main()