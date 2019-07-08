#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
1. Define Data Path
2. Read image in path
3. Call process_img function to process image & get dataframe
4. Split dataset into train-test set using train_test_split using split value: 0.2
5. Apply PCA to train and test set
6. Train Backpropagation model using the PCA train set
7. Test the model using PCA test set
8. Check the accuracy of current model
9. Repeat 4-7 10 times & compare accuracy
10. Use the highest accuracy train-test set as the main train-test set to optimize.
'''


# In[2]:


import os, time
import cv2 as cv
import numpy as np
import pandas as pd
from csv import reader
import matplotlib.pyplot as plt

# ================================================================================================
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

def rotate(image, angle):
    height, width = image.shape[:2]
    rot_mat = cv.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_img = cv.warpAffine(image, rot_mat, (width,height))
    return rotated_img

# Apply Canny with Automatic Parameter
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using computed median
    lower = int(max(0, (1.0-sigma) * v))
    upper = int(min(255, (1.0+sigma) * v))
    canny = cv.Canny(image, lower, upper)

    return canny

# ================================================================================================

'''
Process Image Steps:
1. Grayscaling
2. CLAHE Histogram Equalizing
3. MedianBlur
4. Mask Pupil
5. Convert Iris Image to Cartesian (center, radius = image.shape[0]/2)
6. Crop 2~3 o'clock region
7. Crop pupil area
8. Crop ROI
9. Apply autocanny
10. Flatten image as feature
'''

# ================================================================================================
def preprocessing(image, name):
    # Grayscaling
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite('./Result/gray/'+name, gray)

    # Equalize Histogram
    clahe = cv.createCLAHE(clipLimit=5.0, tileGridSize=(5,5))
    hist = clahe.apply(gray)
    cv.imwrite('./Result/hist/'+name, hist)

    # Blur Image (Reduce Noise)
    blur = cv.medianBlur(hist, 5)
    cv.imwrite('./Result/blur/'+name, blur)

    # Mask Pupil
    _, thresh = cv.threshold(blur, 10, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    maxContour = 0
    for contour in contours:
        contourSize = cv.contourArea(contour)
        if contourSize > maxContour:
            maxContour = contourSize
            maxContourData = contour
    ## find enclosing circle of pupil contour
    (x,y) ,r = cv.minEnclosingCircle(maxContourData)
    center = (int(x), int(y))
    radius = int(r)

    img = blur.copy()
    masked_pupil = cv.circle(img, center, radius+10, (255,255,255), -1)
    cv.imwrite('./Result/masked_pupil/'+name, masked_pupil)

    return masked_pupil

def segmentation(image, name):
    center = (int(image.shape[0]/2),int(image.shape[0]/2))
    radius = int(image.shape[0]/2)

    # Convert to cartesian
    cartesian = cv.linearPolar(image, center, radius, cv.WARP_FILL_OUTLIERS)
    cartesian = rotate(cartesian, -90)
    cv.imwrite('./Result/cartesian/'+name, cartesian)

    # Crop Target
    [y, x] = cartesian.shape
    target = cartesian[0:int(y), 1:int(x/12)]
    cv.imwrite('./Result/target/'+name, target)

    # Crop Pupil Area
    _, thresh = cv.threshold(target, 250, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    maxContour = 0
    for contour in contours:
        contourSize = cv.contourArea(contour)
        if contourSize > maxContour:
            maxContour = contourSize
            maxContourData = contour
    rect = cv.boundingRect(maxContourData)
    x,y,w,h = rect
    crop_pupil = target[y+h:, 0:]
    cv.imwrite('./Result/crop_pupil/'+name, crop_pupil)

    # Specify ROI
    [y,x] = crop_pupil.shape
    roi = crop_pupil[0:int(y/2), 0:]
    cv.imwrite('./Result/roi/'+name, roi)

    #Resize ROI
    roi_res = cv.resize(roi, (50,50))
    cv.imwrite('./Result/roi_res/'+name, roi_res)

    return roi_res

def find_feat(image, name):
    canny = auto_canny(image)
    cv.imwrite('./Result/canny/'+name, canny)
    # Flatten Image Array
    feature = canny.flatten()
    return feature


# PROCESS IMAGE HANDLER
def process_image(path, label):
    # Read File on Path
    data = read_file(path)
    print('Folder {} Contains {} Images'.format(label, len(data)))

    features = []

    for n, file in enumerate(data):
        name = label+'_{}.JPG'.format(n)
        # Preprocessing Image in Data
        masked_pupil = preprocessing(file, name)
        # Segmenting Image ROI
        roi = segmentation(masked_pupil, name)
        # Find Image Feature
        feature = find_feat(roi, name)
        features.append(feature)

    return features

# ================================================================================================

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def process_df(ada_features, tidak_features):
    # Build DataFrame From Feature Arrays
    print('Building DataFrame')
    df_tidak = pd.DataFrame(np.array(tidak_features))
    df_tidak['label'] = 0
    df_ada = pd.DataFrame(np.array(ada_features))
    df_ada['label'] = 1
    # Concatenate DataFrame
    df_feat = pd.concat([df_tidak, df_ada], ignore_index=True)
    df_feat.to_csv('./features_df.csv', header=False, index=False)

    print('DataFrame')
    print(df_feat)

    print('DataFrame Built\n')

    # Normalize DataFrame
    print('Normalizing DataFrame')
    X = df_feat.iloc[:, :-1]
    y = df_feat.iloc[:, -1]

    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X)

    dataset = pd.concat([X,y], axis=1)
    dataset.to_csv('./normalized_df.csv', header=False, index=False)

    print('Normalized Dataset')
    print(dataset)

    print('DataFrame  Normalized!\n')

def train_test(path):
    dataset = readCSV(path)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset,i)
    str_column_to_int(dataset, len(dataset[0])-1)

#     Split Features From Label
    dataset = pd.DataFrame(dataset)
    X = np.array(dataset.iloc[:, :-1])
    y = np.array(dataset.iloc[:, -1])

#     Define the StratifiedKFold train-test splitter and split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)

    X_train = pd.DataFrame(X_train).reset_index().drop('index', axis=1)
    X_test = pd.DataFrame(X_test).reset_index().drop('index', axis=1)
    y_train = pd.DataFrame(y_train).reset_index().drop('index', axis=1)
    y_test = pd.DataFrame(y_test).reset_index().drop('index', axis=1)

    train_df = pd.concat([X_train, y_train], axis=1)
    train_df.to_csv('./train_test_set/train_df.csv', header=False, index=False)
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv('./train_test_set/test_df.csv', header=False, index=False)

    print('train_df')
    print(train_df)
    print('test_df')
    print(test_df)

    print('Dataset Splitted Into Train-Test Set!\n')

# ================================================================================================

from sklearn.decomposition import PCA

def apply_pca(X_train, X_test, target_size):
    print('PCA Target Size = {}'.format(target_size))
    pca = PCA(target_size)

    print('Transforming Train Dataset')
    X_train = pca.fit_transform(X_train)
    print('Transforming Test Dataset')
    X_test = pca.transform(X_test)

    return X_train, X_test

# =================================================================================================

import random
from random import seed
from math import exp
from sklearn.metrics import confusion_matrix

# Initialize Network
def initialize_network(n_inputs, n_hidden, n_layers, n_outputs):
    network = list()
    seed(0)
#     for i in range (n_layers):
#         hidden_layer = [{'weights': [round(random.uniform(0,0.5),2) for i in range(n_inputs+1)]} for i in range(n_hidden)]
#         if i > 0:
#             hidden_layer = [{'weights': [round(random.uniform(0,0.5),2) for i in range(n_hidden+1)]} for i in range(n_hidden)]
#         network.append(hidden_layer)
#     output_layer = [{'weights': [round(random.uniform(0,0.5),2) for i in range(n_hidden+1)]} for i in range(n_outputs)]
    for i in range (n_layers):
        hidden_layer = [{'weights': [round(random.random(),2) for i in range(n_inputs+1)]} for i in range(n_hidden)]
        if i > 0:
            hidden_layer = [{'weights': [round(random.random(),2) for i in range(n_hidden+1)]} for i in range(n_hidden)]
        network.append(hidden_layer)
    output_layer = [{'weights': [round(random.random(),2) for i in range(n_hidden+1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# FORWARD PROPAGATE
# 1. Neuron Activation
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# 2. Neuron Transfer
def transfer(activation):
    return 1.0 / (1.0+exp(-activation))

# 3. Forward Propagation
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# BACK PROPAGATE
# 1. Transfer Derivative
def transfer_derivative(output):
    return output * (1.0 - output)

# 2. Error Backpropagation
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i !=len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# 3. Update Weights
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

# TRAIN NETWORK
def train_network(network, train, l_rate, loss_limit, n_outputs):
    epoch = 0
    while True:
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            # print(expected)
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        epoch+=1
        if epoch == 25000:
            break
        if sum_error <= loss_limit:
            break
    print('>epoch=%d, l_rate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# MAKING PREDICTION
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# CHECK ACCURACY
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct/float(len(actual)) * 100.0


# BACKPROPAGATION HANDLER
def back_propagation(train, test, l_rate, loss_limit, n_layers, n_hidden, name):
    n_inputs = len(train[0])-1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_layers, n_outputs)
    train_network(network, train, l_rate, loss_limit, n_outputs)
    pd.DataFrame(np.array(network)).to_csv(name, header=False, index=False)

    test_set = pd.DataFrame(test).iloc[:, :-1]
    test_set = np.array(test_set)

    predictions = list()
    for row in test_set:
        prediction = predict(network, row)
        predictions.append(prediction)
    print('PREDICTIONS:')
    print(predictions)

    expected = [row[-1] for row in test]
    print('EXPECTED:')
    print(expected)

    accuracy = accuracy_metric(expected, predictions)
    print('Accuracy = %.3f%%' %(accuracy))
    print('\nConfusion Matrix:')

    print(confusion_matrix(expected, predictions))
    tn, fp, fn, tp = confusion_matrix(expected, predictions).ravel()
    print('TN={}, FP={}, FN={}, TP={}'.format(tn, fp, fn, tp))
    return network, accuracy


# ================================================================================================

def run(target_pca, l_rate, loss_limit, n_layers, n_hiddens):
    # -----START BACKPROPAGATION-----
    start = time.time()

    train_name = './train_test_set/PCA_TRAIN_MODEL.csv'.format(target_pca)
    test_name = './train_test_set/PCA_TEST_MODEL.csv'.format(target_pca)

    pca_train_set = readCSV(train_name)
    for i in range(len(pca_train_set[0])-1):
        str_column_to_float(pca_train_set, i)
    str_column_to_int(pca_train_set, len(pca_train_set[0])-1)
    # print(pd.DataFrame(pca_train_set))

    pca_test_set = readCSV(test_name)
    for i in range(len(pca_test_set[0])-1):
        str_column_to_float(pca_test_set, i)
    str_column_to_int(pca_test_set, len(pca_test_set[0])-1)
    # print(pd.DataFrame(pca_test_set))

    # print('\nl_rate = {}, n_epoch = {}, n_hidden = {}\n'.format(l_rate, n_epoch, n_hidden))
    print('\npca = {}, l_rate = {}, loss_limit = {}, n_layers = {}, n_hiddens = {}\n'.format(target_pca, l_rate, loss_limit, n_layers, n_hiddens))

    # network = back_propagation_tts(pca_train_set, pca_test_set, l_rate, n_epoch, n_layers, n_hidden)
    name = './train_test_set/network{}-{}.csv'.format(target_pca, n_hiddens)
    network, accuracy = back_propagation(pca_train_set, pca_test_set, l_rate, loss_limit, n_layers, n_hiddens, name)
    stop = time.time()

    print('Elapsed Time: {}s'.format(stop-start))

# ================================================================================================

'''
Main Function
'''


# =================================================PROCESS IMAGE=====================================
# Define Data Path
data_ada = './Data_ext/ada/'
data_tidak = './Data_ext/tidak/'

# -----PROCESSING IMAGE-----
print('Processing Data Ada')
ada = 'ada'
ada_features = process_image(data_ada, ada)
print('Done!')
print('Processing Data Tidak')
tidak = 'tidak'
tidak_features = process_image(data_tidak, tidak)
print('Done!\n')

# -----BUILD DATAFRAME-----
process_df(ada_features, tidak_features)

# =====================================================PCA============================================
# Define Variable
target_pca = 20

path = './normalized_df.csv'
# Split Dataset into Train-Test Set
train_test(path)

# ------APPLY PCA-------
train_path = './train_test_set/train_df.csv'
test_path = './train_test_set/test_df.csv'

# Read Train Set
train_df = readCSV(train_path)
for i in range(len(train_df[0])-1):
    str_column_to_float(train_df, i)
str_column_to_int(train_df, len(train_df[0])-1)
# Read Test Set
test_df = readCSV(test_path)
for i in range(len(test_df[0])-1):
    str_column_to_float(test_df, i)
str_column_to_int(test_df, len(test_df[0])-1)

# Split Features From Label
train_df = pd.DataFrame(train_df)
test_df = pd.DataFrame(test_df)
X_train = train_df.iloc[:, :-1]
X_test = test_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
y_test = test_df.iloc[:, -1]

X_train, X_test = apply_pca(X_train, X_test, target_pca)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# Reconstruct DataFrame
pca_train_set = pd.concat([X_train,y_train], axis=1, ignore_index=True)
train_name = './train_test_set/PCA_TRAIN_MODEL.csv'
pca_train_set.to_csv(train_name, header=False, index=False)

pca_test_set = pd.concat([X_test, y_test], axis=1, ignore_index=True)
test_name = './train_test_set/PCA_TEST_MODEL.csv'
pca_test_set.to_csv(test_name, header=False, index=False)

print('PCA_TRAIN_SET')
print(pca_train_set)
print('PCA_TEST_SET')
print(pca_test_set)

print('DataFrame Decomposed!\n')

# ================================================================================================


l_rate = 0.01
loss_limit = 0.1
n_layers = 1

target_pca = 20

n_hiddens = 10
run(target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('\n==========================================================')


n_hiddens = 20
run(target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('\n==========================================================')

n_hiddens = 30
run(target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('\n==========================================================')

n_hiddens = 40
run(target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('\n==========================================================')

# ================================================================================================
'''
target_pca = 20

n_hiddens = 10
run(target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('\n==========================================================')

n_hiddens = 20
run(target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('\n==========================================================')

n_hiddens = 30
run(target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('\n==========================================================')

n_hiddens = 40
run(target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('\n==========================================================')

# ================================================================================================

target_pca = 30

n_hiddens = 10
run(target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('\n==========================================================')

n_hiddens = 20
run(target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('\n==========================================================')

n_hiddens = 30
run(target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('\n==========================================================')

n_hiddens = 40
run(target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('\n==========================================================')

# ================================================================================================

target_pca = 40

n_hiddens = 10
run(target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('\n==========================================================')

n_hiddens = 20
run(target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('\n==========================================================')

n_hiddens = 30
run(target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('\n==========================================================')

n_hiddens = 40
run(target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('\n==========================================================')

# ================================================================================================
'''
