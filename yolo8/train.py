import comet_ml
from ultralytics import YOLO

print('[INFO] Starting...')

# This is for tracking the model's performance over time.
# You should have the comet API key defined in your environment variables.
# export COMET_API_KEY=<Your API Key>
comet_ml.init(project_name="test-yolov8-billboards")

print('[INFO] Comet ML configured')

YOLO8_MODELS = {
  'nano': 'yolov8n.pt',
  'small': 'yolov8s.pt',
  'medium': 'yolov8m.pt',
}

model = YOLO(YOLO8_MODELS['nano'])

print('[INFO] Model has loaded')

# Config
YOLO_DATA_YML_PATH = './roboflow-billboards-yolo8-dataset/data.yaml'

# Hyperparameters
IMAGE_SIZE=640
EPOCHS=50
BATCH_SIZE=8

print('[INFO] Start traning')

# Training.
results = model.train(
   name='yolov8n_v8_50e',
   project='/Users/casimiropastine/fiuba/tpp/yolo8/',
   data=YOLO_DATA_YML_PATH,
   imgsz=IMAGE_SIZE,
   epochs=EPOCHS,
   batch=BATCH_SIZE,
  # To enable training on Apple M1 and M2 chips,
  # you should specify 'mps' as your device when initiating the training process.
   device='mps'
)
