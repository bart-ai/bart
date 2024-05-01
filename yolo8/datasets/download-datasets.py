import os

from roboflow import Roboflow

roboflow = Roboflow(os.environ["ROBOFLOW_API_KEY"])
datasets = [
    ("image-processing-awivd/billboards-4zz9y", 3),
    ("vehicle-yolov5/billboard-test", 2),
    ("test-c8wix/billboard-detection-uo2ld", 3),
    ("billboards/billboards-cmzpu", 3),
    ("computer-vision-project-ixips/billboards-detection", 3),
    ("hadplus/hadplus-illegal-billboard-detector", 1),
    ("oleksii-kurochkin/billboard", 1),
    ("billboard-asmlb/billboard-isfau", 2),
]

for dataset, version in datasets:
    name = dataset.split("/")[1]
    project = roboflow.project(dataset)
    version = project.version(version)
    dataset = version.download("yolov8", name)
