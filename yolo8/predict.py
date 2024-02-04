import argparse
from PIL import Image
from ultralytics import YOLO

parser = argparse.ArgumentParser(
                    prog='Predict Billboards',
                    description='Script to use a pretrained YOLOv8n model to predict billboards in images')
parser.add_argument('-t', '--target', help='The target relative file path for the image to predict')
parser.add_argument('-m', '--model', help='The relative file path to the YOLO8 model that will be used to predict')
args = parser.parse_args()


modelPath = args.model
imagePath = args.target

# Load a pretrained YOLOv8n model
# model = YOLO('./yolov8n_v8_50e/weights/best.pt')
model = YOLO(modelPath)

results = model(imagePath)

print(results)

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')