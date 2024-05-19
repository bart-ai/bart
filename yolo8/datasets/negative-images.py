# Adapted from https://y-t-g.github.io/tutorials/bg-images-for-yolo/
# Adapted from https://stackoverflow.com/a/62770484/8061030
import os

import requests
from pycocotools.coco import COCO
from tqdm import tqdm

# We need to exclude everything that __might__ have a billboard in it
# http://github.com/ultralytics/yolov5/blob/df48c205c5fc7be5af6b067da1f7cb3efb770d88/data/coco.yaml
DETECTOR_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "tv",
]  

DIRECTORY_STRUCTURE = ["negative-images", "train", "images"]
NUM_IMAGES = 250

# http://images.cocodataset.org/annotations/annotations_trainval2017.zip
coco = COCO("instances_train2017.json")

exc_cat_ids = coco.getCatIds(catNms=DETECTOR_CLASSES)
exc_img_ids = coco.getImgIds(catIds=exc_cat_ids)

all_cat_ids = coco.getCatIds(catNms=[""])
all_img_ids = coco.getImgIds(catIds=all_cat_ids)

bg_img_ids = set(all_img_ids) - set(exc_img_ids)
bg_images = coco.loadImgs(bg_img_ids)

os.makedirs(os.path.join(*DIRECTORY_STRUCTURE), exist_ok=True)

for im in tqdm(bg_images[:NUM_IMAGES]):
    img_data = requests.get(im["coco_url"]).content
    with open(os.path.join(*[*DIRECTORY_STRUCTURE, im["file_name"]]), "wb") as handler:
        handler.write(img_data)
