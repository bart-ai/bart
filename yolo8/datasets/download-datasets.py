import os

from roboflow import Roboflow

roboflow = Roboflow(os.environ["ROBOFLOW_API_KEY"])
datasets = [
    ("billboards/billboards-cmzpu", 3),
    ("hadplus/hadplus-illegal-billboard-detector", 1),
    ("oleksii-kurochkin/billboard", 1),
]

for dataset, version in datasets:
    name = dataset.split("/")[1]
    project = roboflow.project(dataset)
    version = project.version(version)
    dataset = version.download("yolov8", name)
