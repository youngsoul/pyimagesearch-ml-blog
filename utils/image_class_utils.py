import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt
import cv2
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd


from utils.RGBHistogram import RGBHistogram

def get_image_features(img_path):
    rgbHisto = RGBHistogram([8, 8, 8])
    features = rgbHisto.get_features(img_path)

    img_name = img_path.split("/")[-1]

    features.insert(0, img_name)
    return features

def plot_rgb_histogram(img_path, setup_figure=True):
    image = cv2.imread(img_path)
    chans = cv2.split(image)
    colors = ("b", "g", "r")
    if setup_figure:
        plt.figure(figsize=(6, 8))
    plt.title("Color Histogram")
    plt.xlabel("pixel intensities")
    plt.ylabel("# of Pixels")
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])


def get_normalized_image_histogram(img_path, mask=None):
    image = cv2.imread(img_path)

    hist = cv2.calcHist([image], [0, 1, 2],
                        mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)

    # return as a feature vector
    # the shape of the returned histogram is
    # 8*8*8 = 512 x 1
    return hist.flatten()


def plot_binned_histogram(img_path, setup_figure=True):
    bin_data = get_normalized_image_histogram(img_path)
    if setup_figure:
        plt.figure(figsize=(6, 8))
    plt.title("Flattened Binned Color Histogram")
    plt.xlabel("Pixel Bins")
    plt.ylabel("Normalized ")
    plt.plot(bin_data, color='b')
    plt.xlim([0, 512])


def image_summaries(img_paths, img_names, figsize=(20, 10)):
    rows = len(img_paths)
    cols = 3
    cell = 1
    plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.99, bottom=0.01, hspace=0.5, wspace=0.4)
    for i, img_path in enumerate(img_paths):
        plt.subplot(rows, cols, cell)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(img_names[i])

        cell = cell + 1
        plt.subplot(rows, cols, cell)
        plot_rgb_histogram(img_path, False)
        cell = cell + 1
        plt.subplot(rows, cols, cell)
        plot_binned_histogram(img_path, False)
        cell = cell + 1


if __name__ == '__main__':
    test_images = [
        '../data/3scenes_holdout/coast/coast_cdmc922.jpg',
        '../data/3scenes_holdout/coast/coast_n291061.jpg',
        '../data/3scenes_holdout/forest/forest_for15.jpg',
        '../data/3scenes_holdout/forest/forest_nat982.jpg',
        '../data/3scenes_holdout/highway/highway_art820.jpg',
        '../data/3scenes_holdout/highway/highway_urb537.jpg'
    ]
    feature_matrix = []
    for test_image in test_images:
        print("-----------------------------")
        features = get_image_features(test_image)
        print(features)
        feature_matrix.append(features)

    df = pd.DataFrame(data=feature_matrix)
    print(df.head())


