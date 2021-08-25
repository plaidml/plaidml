"""
implementation of imagenet dataset
"""

# pylint: disable=unused-argument,missing-docstring

import json
import logging
import os
import time

import cv2
import numpy as np
from pycocotools.cocoeval import COCOeval
import pycoco
import dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco")


class Coco(dataset.Dataset):

    def __init__(self,
                 data_path,
                 image_list,
                 name,
                 use_cache=0,
                 image_size=None,
                 image_format="NHWC",
                 pre_process=None,
                 count=None,
                 cache_dir=None,
                 use_label_map=False):
        super().__init__()
        self.image_size = image_size
        self.image_list = []
        self.label_list = []
        self.image_ids = []
        self.image_sizes = []
        self.count = count
        self.use_cache = use_cache
        self.data_path = data_path
        self.pre_process = pre_process
        self.use_label_map = use_label_map
        if not cache_dir:
            cache_dir = os.getcwd()
        self.cache_dir = os.path.join(cache_dir, "preprocessed", name, image_format)
        # input images are in HWC
        self.need_transpose = True if image_format == "NCHW" else False
        not_found = 0
        empty_80catageories = 0
        if image_list is None:
            # by default look for val_map.txt
            image_list = os.path.join(data_path, "annotations/instances_val2017.json")
        self.annotation_file = image_list
        if self.use_label_map:
            # for pytorch
            label_map = {}
            with open(self.annotation_file) as fin:
                annotations = json.load(fin)
            for cnt, cat in enumerate(annotations["categories"]):
                label_map[cat["id"]] = cnt + 1

        os.makedirs(self.cache_dir, exist_ok=True)
        start = time.time()
        images = {}
        with open(image_list, "r") as f:
            coco = json.load(f)
        for i in coco["images"]:
            images[i["id"]] = {
                "file_name": i["file_name"],
                "height": i["height"],
                "width": i["width"],
                "bbox": [],
                "category": []
            }
        for a in coco["annotations"]:
            i = images.get(a["image_id"])
            if i is None:
                continue
            catagory_ids = label_map[a.get("category_id")] if self.use_label_map else a.get(
                "category_id")
            i["category"].append(catagory_ids)
            i["bbox"].append(a.get("bbox"))

        for image_id, img in images.items():
            image_name = os.path.join("val2017", img["file_name"])
            src = os.path.join(data_path, image_name)
            if not os.path.exists(src):
                # if the image does not exists ignore it
                not_found += 1
                continue
            if len(img["category"]) == 0 and self.use_label_map:
                #if an image doesn't have any of the 81 categories in it
                empty_80catageories += 1  #should be 48 images - thus the validation sert has 4952 images
                continue

            os.makedirs(os.path.dirname(os.path.join(self.cache_dir, image_name)), exist_ok=True)
            dst = os.path.join(self.cache_dir, image_name)
            if not os.path.exists(dst + ".npy"):
                # cache a preprocessed version of the image
                img_org = cv2.imread(src)
                processed = self.pre_process(img_org,
                                             need_transpose=self.need_transpose,
                                             dims=self.image_size)
                np.save(dst, processed)

            self.image_ids.append(image_id)
            self.image_list.append(image_name)
            self.image_sizes.append((img["height"], img["width"]))
            self.label_list.append((img["category"], img["bbox"]))

            # limit the dataset if requested
            if self.count and len(self.image_list) >= self.count:
                break

        time_taken = time.time() - start
        if not self.image_list:
            log.error("no images in image list found")
            raise ValueError("no images in image list found")
        if not_found > 0:
            log.info("reduced image list, %d images not found", not_found)
        if empty_80catageories > 0:
            log.info("reduced image list, %d images without any of the 80 categories",
                     empty_80catageories)

        log.info("loaded {} images, cache={}, took={:.1f}sec".format(len(self.image_list),
                                                                     use_cache, time_taken))

        self.label_list = np.array(self.label_list)

    def get_item(self, nr):
        """Get image by number in the list."""
        dst = os.path.join(self.cache_dir, self.image_list[nr])
        img = np.load(dst + ".npy")
        return img, self.label_list[nr]

    def get_item_loc(self, nr):
        src = os.path.join(self.data_path, self.image_list[nr])
        return src


class PostProcessCoco:
    """
    Post processing for tensorflow ssd-mobilenet style models
    """

    def __init__(self):
        self.results = []
        self.good = 0
        self.total = 0
        self.content_ids = []
        self.use_inv_map = False

    def add_results(self, results):
        self.results.extend(results)

    def __call__(
            self,
            results,
            ids,
            expected=None,
            result_dict=None,
    ):
        # results come as:
        #   tensorflow, ssd-mobilenet: num_detections,detection_boxes,detection_scores,detection_classes
        processed_results = []
        # batch size
        bs = len(results[0])
        for idx in range(0, bs):
            # keep the content_id from loadgen to handle content_id's without results
            self.content_ids.append(ids[idx])
            processed_results.append([])
            detection_num = int(results[0][idx])
            detection_boxes = results[1][idx]
            detection_classes = results[3][idx]
            expected_classes = expected[idx][0]
            for detection in range(0, detection_num):
                detection_class = int(detection_classes[detection])
                if detection_class in expected_classes:
                    self.good += 1
                box = detection_boxes[detection]
                processed_results[idx].append([
                    float(ids[idx]), box[0], box[1], box[2], box[3], results[2][idx][detection],
                    float(detection_class)
                ])
                self.total += 1
        return processed_results

    def start(self):
        self.results = []
        self.good = 0
        self.total = 0

    def finalize(self, result_dict, ds=None, output_dir=None):
        result_dict["good"] += self.good
        result_dict["total"] += self.total

        if self.use_inv_map:
            # for pytorch
            label_map = {}
            with open(ds.annotation_file) as fin:
                annotations = json.load(fin)
            for cnt, cat in enumerate(annotations["categories"]):
                label_map[cat["id"]] = cnt + 1
            inv_map = {v: k for k, v in label_map.items()}

        detections = []
        image_indices = []
        for batch in range(0, len(self.results)):
            image_indices.append(self.content_ids[batch])
            for idx in range(0, len(self.results[batch])):
                detection = self.results[batch][idx]
                # this is the index of the coco image
                image_idx = int(detection[0])
                if image_idx != self.content_ids[batch]:
                    # working with the coco index/id is error prone - extra check to make sure it is consistent
                    log.error("image_idx missmatch, lg={} / result={}".format(
                        image_idx, self.content_ids[batch]))
                # map the index to the coco image id
                detection[0] = ds.image_ids[image_idx]
                height, width = ds.image_sizes[image_idx]
                # box comes from model as: ymin, xmin, ymax, xmax
                ymin = detection[1] * height
                xmin = detection[2] * width
                ymax = detection[3] * height
                xmax = detection[4] * width
                # pycoco wants {imageID,x1,y1,w,h,score,class}
                detection[1] = xmin
                detection[2] = ymin
                detection[3] = xmax - xmin
                detection[4] = ymax - ymin
                if self.use_inv_map:
                    cat_id = inv_map.get(int(detection[6]), -1)
                    if cat_id == -1:
                        # FIXME:
                        log.info("finalize can't map category {}".format(int(detection[6])))
                    detection[6] = cat_id
                detections.append(np.array(detection))

        # map indices to coco image id's
        image_ids = [ds.image_ids[i] for i in image_indices]
        self.results = []
        cocoGt = pycoco.COCO(ds.annotation_file)
        cocoDt = cocoGt.loadRes(np.array(detections))
        cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
        cocoEval.params.imgIds = image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        result_dict["mAP"] = cocoEval.stats[0]


class PostProcessCocoPt(PostProcessCoco):
    """
    Post processing required by ssd-resnet34 / pytorch
    """

    def __init__(self, use_inv_map, score_threshold):
        super().__init__()
        self.use_inv_map = use_inv_map
        self.score_threshold = score_threshold

    def __call__(self, results, ids, expected=None, result_dict=None):
        # results come as:
        #   detection_boxes,detection_classes,detection_scores

        processed_results = []
        # batch size
        bs = len(results[0])
        for idx in range(0, bs):
            self.content_ids.append(ids[idx])
            processed_results.append([])
            detection_boxes = results[0][idx]
            detection_classes = results[1][idx]
            expected_classes = expected[idx][0]
            scores = results[2][idx]
            #for detection in range(0, len(expected_classes)):
            for detection in range(0, len(scores)):
                if scores[detection] < self.score_threshold:
                    break
                detection_class = int(detection_classes[detection])
                if detection_class in expected_classes:
                    self.good += 1
                box = detection_boxes[detection]
                # comes from model as:  0=xmax 1=ymax 2=xmin 3=ymin
                processed_results[idx].append([
                    float(ids[idx]), box[1], box[0], box[3], box[2], scores[detection],
                    float(detection_class)
                ])
                self.total += 1
        return processed_results


class PostProcessCocoOnnx(PostProcessCoco):
    """
    Post processing required by ssd-resnet34 / onnx
    """

    def __init__(self):
        super().__init__()

    def __call__(self, results, ids, expected=None, result_dict=None):
        # results come as:
        #   onnx (from pytorch ssd-resnet34): detection_boxes,detection_classes,detection_scores

        processed_results = []

        # batch size
        bs = len(results[0])
        for idx in range(0, bs):
            self.content_ids.append(ids[idx])
            processed_results.append([])
            detection_boxes = results[0][idx]
            detection_classes = results[1][idx]
            expected_classes = expected[idx][0]
            scores = results[2][idx]
            for detection in range(0, len(scores)):
                if scores[detection] < 0.5:
                    break
                detection_class = int(detection_classes[detection])
                if detection_class in expected_classes:
                    self.good += 1
                box = detection_boxes[detection]
                # comes from model as:  0=xmax 1=ymax 2=xmin 3=ymin
                processed_results[idx].append([
                    float(ids[idx]), box[1], box[0], box[3], box[2], scores[detection],
                    float(detection_class)
                ])
                self.total += 1
        return processed_results


class PostProcessCocoTf(PostProcessCoco):
    """
    Post processing required by ssd-resnet34 / pytorch
    """

    def __init__(self):
        super().__init__()
        self.use_inv_map = True

    def __call__(self, results, ids, expected=None, result_dict=None):
        # results come as:
        #   detection_boxes,detection_classes,detection_scores

        processed_results = []
        # batch size
        bs = len(results[0])
        for idx in range(0, bs):
            self.content_ids.append(ids[idx])
            processed_results.append([])
            detection_boxes = results[0][idx]
            detection_classes = results[1][idx]
            expected_classes = expected[idx][0]
            scores = results[2][idx]
            for detection in range(0, len(scores)):
                if scores[detection] < 0.05:
                    break
                detection_class = int(detection_classes[detection])
                if detection_class in expected_classes:
                    self.good += 1
                box = detection_boxes[detection]
                # comes from model as:  0=xmax 1=ymax 2=xmin 3=ymin
                processed_results[idx].append([
                    float(ids[idx]), box[0], box[1], box[2], box[3], scores[detection],
                    float(detection_class)
                ])
                self.total += 1
        return processed_results
