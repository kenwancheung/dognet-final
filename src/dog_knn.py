# Import relevant packages

from sklearn import datasets, neighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from imutils import paths
import argparse
import imutils
import cv2
import os

# image to vector
def image_to_feature_vector(image, size=(32,32)):
    # resize, flatten
    return cv2.resize(image,size).flatten()

# color hist
def extract_color_histogram(image, bins=(8,8,8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2], None, bins,
                       [0,180,0,256,0,256])
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist,hist)
    return hist.flatten()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
                help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# grab
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

# init lists to store
rawImages = []
features = []
labels = []

# now to extract the features
for(i, imagePath) in enumerate(imagePaths):
    # load
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    # extract
    pixels = image_to_feature_vector(image)
    hist = extract_color_histogram(image)

    # update lists
    rawImages.append(pixels)
    features.append(hist)
    labels.append(label)

    # status update
    if i>0 and i % 1000 == 0:
        print("[INFO] processed {}/{}".format(i,len(imagePaths)))

# mem allocation
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
    rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
    features.nbytes / (1024 * 1000.0)))

# raw images test
(trainRI, testRI, trainRL, testRL) = train_test_split(
    rawImages, labels, test_size=.2, random_state=42
)

# features test
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
    features, labels, test_size=.2, random_state=42
)

# run
print("[INFO] evaluating raw pixel accuracy")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
                             n_jobs=args["jobs"])
model.fit(trainFeat,trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy {:.2f}%".format(acc*100))