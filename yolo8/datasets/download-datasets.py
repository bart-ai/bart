import os

from roboflow import Roboflow

roboflow = Roboflow(os.environ["ROBOFLOW_API_KEY"])
# Criteria:
# only use datasets with train and val (test isn't required)
# only use datasets with 640x640 images
# don't train with less than 1500 training images
datasets = [
    # multi
    ("computer-vision-project-ixips/billboards-detection", 3),  # 389
    ("billboards/billboards-cmzpu", 3),  # 211
    ("hadplus/hadplus-illegal-billboard-detector", 1),  # 270
    ("image-processing-awivd/billboards-4zz9y", 3),  # 400
    ("image-processing-awivd/bills0", 1),  # 530
    ("dip-9/billboards-oopxm", 2),  # 714
    # hackaton
    ("projet-sia/hackathon-03-2024", 1),  # 4503
    # isfau
    ("billboard-asmlb/billboard-isfau", 2),  # 7137
    # uo2ld
    ("test-c8wix/billboard-detection-uo2ld", 3),  # 1903
    # ydtns
    ("kaliani/bb-ydtns", 1),  # 2719
    # s2oeb
    ("dream-labs/billboard-s2oeb", 1),  # 8155
]

for dataset, version in datasets:
    name = dataset.split("/")[1]
    project = roboflow.project(dataset)
    version = project.version(version)
    dataset = version.download("yolov8", name)
