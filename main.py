import os, time
import cv2 as cv
import numpy as np
import pandas as pd
from csv import reader
from process_img import process_image
from process_df import build_df, normalize_df, train_test
from pca import apply_pca
from bp import back_propagation

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
    for row in dataset:
        row[column] = int(row[column].strip())

def run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens):
    # -----START BACKPROPAGATION-----
    start = time.time()

    pca_train = readCSV(train_path)
    for i in range(len(pca_train[0])-1):
	       str_column_to_float(pca_train, i)
    str_column_to_int(pca_train, len(pca_train[0])-1)

    pca_test = readCSV(test_path)
    for i in range(len(pca_test[0])-1):
	       str_column_to_float(pca_test, i)
    str_column_to_int(pca_test, len(pca_test[0])-1)

    # print('\nl_rate = {}, n_epoch = {}, n_hidden = {}\n'.format(l_rate, n_epoch, n_hidden))
    print('\npca = {}, l_rate = {}, loss_limit = {}, n_layers = {}, n_hiddens = {}\n'.format(target_pca, l_rate, loss_limit, n_layers, n_hiddens))

    # network = back_propagation_tts(pca_train_set, pca_test_set, l_rate, n_epoch, n_layers, n_hidden)
    name = './sigma0.5/train_test_set/network{}-{}.csv'.format(target_pca, n_hiddens)
    network = back_propagation(pca_train, pca_test, l_rate, loss_limit, n_layers, n_hiddens, name)
    stop = time.time()

    print('Elapsed Time: {}s'.format(stop-start))

def main():
    # PROCESS IMAGES
    path_ada = './Data_ext/ada/'
    ada = 'ada'
    ada_features = process_image(path_ada, ada)

    path_tidak = './Data_ext/tidak/'
    tidak = 'tidak'
    tidak_features = process_image(path_tidak, tidak)

    # BUILD DATAFRAME
    df_feat = build_df(ada_features, tidak_features)
    # NORMALIZE DATAFRAME
    norm_df = normalize_df(df_feat)
    # SPLIT DATAFRANE USING TRAIN TEST SPLIT
    train_df, test_df = train_test(norm_df)

    # Apply PCA
    target_pca = 20
    pca_train, pca_test = apply_pca(train_df, test_df, target_pca)

    # Run BackPropagation
    l_rate = 0.1
    loss_limit = 0.1
    n_layers = 1
    n_hiddens = 20

    train_path = './sigma0.5/train_test_set/PCA_TRAIN_MODEL.csv'
    test_path = './sigma0.5/train_test_set/PCA_TEST_MODEL.csv'
    run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
    # print(pca_test)

main()
