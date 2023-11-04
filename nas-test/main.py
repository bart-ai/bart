import os
from keras.datasets import cifar10
import autokeras as ak
import pandas as pd
from sklearn.metrics import classification_report

OUTPUT_PATH='./nas-test-output'
DATASET_NAME = 'Billboards.v3i.retinanet'
TRAINING_TIMES = [
    60 * 1, # 1 minute
    60 * 5, # 5 minutes
    60 * 10, # 10 minutes
    60 * 30,  # 30 minutes
		60 * 60,		# 1 hour
		# 60 * 60 * 2,	# 2 hours
		# 60 * 60 * 4,	# 4 hours
		# 60 * 60 * 8,	# 8 hours
		# 60 * 60 * 12,	# 12 hours
		# 60 * 60 * 24,	# 24 hours
	]

def load_data_sets():
    train = pd.read_csv(f'./datasets/{DATASET_NAME}/train/_annotations.csv')
    test = pd.read_csv(f'./datasets/{DATASET_NAME}/test/_annotations.csv')
    validate = pd.read_csv(f'./datasets/{DATASET_NAME}/valid/_annotations.csv')
    return train, test, validate

def load_billboards_data_set():
  train, test, validate = load_data_sets()
  trainX = train.drop(train.columns[5], axis=1)
  trainY = train.columns[5]

  testX = test.drop(test.columns[5], axis=1)
  testY = test.columns[5]

  validateX = validate.drop(validate.columns[5], axis=1)
  validateY = validate.columns[5]

  return trainX, trainY, testX, testY, validateX, validateY

def main():
  print(f'[INFO] Loading {DATASET_NAME} data')
  # trainX, trainY, testX, testY, validateX, validateY = load_billboards_data_set()
  ((trainX, trainY), (testX, testY)) = cifar10.load_data()

  print(trainX)
  print(trainY)

  # for seconds in TRAINING_TIMES:
  print(f'[INFO] training model...')

  # Train the model with autokeras image classifier
  model = ak.ImageClassifier()
  # TODO: we can add validation data here as well.
  model.fit(trainX, trainY, verbose=2)
  model.final_fit(trainX, trainY, testX, testY, retrain=True)

  print('[INFO] evaluating the model')
  # Evaluate the accuracy of the model
  score = model.evaluate(testX, testY)
  predictions = model.predict(testX)
  report = classification_report(testY, predictions, target_names=['Werbetafel frei'])

  print('[INFO] saving results')
  p = os.path.sep.join(OUTPUT_PATH, "{}.txt".format(seconds))
  f = open(p, "w")
  f.write(report)
  f.write("\nscore: {}".format(score))
  f.close()

print('[INFO] Starting')
main()
