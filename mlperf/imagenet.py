"""
implementation of imagenet dataset
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import os
import threading
import time
from multiprocessing.pool import ThreadPool

import cv2
import dataset
import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("imagenet")


class AtomicCounter:

    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value (default 0)."""
        self.value = initial
        self._lock = threading.Lock()

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value += num
            return self.value


class Imagenet(dataset.Dataset):

    def __init__(self,
                 data_path,
                 image_list,
                 name,
                 use_cache=0,
                 image_size=None,
                 image_format="NHWC",
                 pre_process=None,
                 count=None,
                 cache_dir=None):
        super(Imagenet, self).__init__()
        if image_size is None:
            self.image_size = [224, 224, 3]
        else:
            self.image_size = image_size
        if not cache_dir:
            cache_dir = data_path
        self.image_list = []
        self.label_list = []
        self.count = count
        self.use_cache = use_cache
        self.cache_dir = os.path.join(cache_dir, image_format)
        self.data_path = data_path
        self.pre_process = pre_process
        # input images are in HWC
        self.need_transpose = True if image_format == "NCHW" else False

        not_found = AtomicCounter()
        if image_list is None:
            # by default look for val_map.txt
            image_list = os.path.join(data_path, "val_map.txt")

        os.makedirs(self.cache_dir, exist_ok=True)

        def worker(s):
            image_name, label = s.split(maxsplit=1)
            src = os.path.join(data_path, image_name)
            if not os.path.exists(src):
                # if the image does not exists ignore it
                not_found.increment()
                return

            os.makedirs(os.path.dirname(os.path.join(self.cache_dir, image_name)), exist_ok=True)
            dst = os.path.join(self.cache_dir, image_name)
            if not os.path.exists(dst + ".npy"):
                # cache a preprocessed version of the image
                img_org = cv2.imread(src)
                processed = self.pre_process(img_org,
                                             need_transpose=self.need_transpose,
                                             dims=self.image_size)
                np.save(dst, processed)

            return image_name, int(label)

        log.info('Preproccesing imagenet dataset')
        start = time.time()
        with open(image_list, 'r') as f:
            with ThreadPool() as pool:
                results = pool.map(worker, f)
        for (image_name, label) in results:
            self.image_list.append(image_name)
            self.label_list.append(label)
        time_taken = time.time() - start

        if not self.image_list:
            log.error("no images in image list found")
            raise ValueError("no images in image list found")
        if not_found.value > 0:
            log.info("reduced image list, %d images not found", not_found.value)

        log.info("loaded {} images, cache={}, took={:.1f}sec".format(len(self.image_list),
                                                                     use_cache, time_taken))

        self.label_list = np.array(self.label_list)

    def get_item(self, nr):
        """Get image by number in the list."""
        dst = os.path.join(self.cache_dir, self.image_list[nr])
        img = np.load(dst + ".npy")
        return img, self.label_list[nr]

    def get_item_loc(self, nr):
        return os.path.join(self.data_path, self.image_list[nr])
