import os
import cv2 as cv
import numpy as np
import pandas as pd
from csv import reader
import matplotlib.pyplot as plt

# Display Image
def display_img(image):
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.show()

# Read File in Path
def read_file(path):
    images = []
    for filename in sorted(os.listdir(path)):
        img = cv.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
    return images

# Read CSV File
def readCSV(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert Feature DataFrame CSV into float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert DataFrame label into int
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup
