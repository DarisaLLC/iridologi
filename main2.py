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
    name = './sigma0.5/train_test_fix/network{}-{}.csv'.format(target_pca, n_hiddens)
    network = back_propagation(pca_train, pca_test, l_rate, loss_limit, n_layers, n_hiddens, name)
    stop = time.time()

    print('Elapsed Time: {}s'.format(stop-start))

def pca_fix_dataset(target_pca):
    train_path = './sigma0.5/train_test_fix/train_df.csv'
    test_path = './sigma0.5/train_test_fix/test_df.csv'

    train_df = readCSV(train_path)
    for i in range(len(train_df[0])-1):
	       str_column_to_float(train_df, i)
    str_column_to_int(train_df, len(train_df[0])-1)

    test_df = readCSV(test_path)
    for i in range(len(test_df[0])-1):
	       str_column_to_float(test_df, i)
    str_column_to_int(test_df, len(test_df[0])-1)

    pca_train, pca_test = apply_pca(train_df, test_df, target_pca)
#
# target_pca = 40
# pca_fix_dataset(target_pca)

l_rate = 0.1
loss_limit = 0.1
n_layers = 1

target_pca = 10

n_hiddens = 10
train_path = './sigma0.5/train_test_fix/pca{}/PCA_TRAIN_MODEL.csv'.format(target_pca)
test_path = './sigma0.5/train_test_fix/pca{}/PCA_TEST_MODEL.csv'.format(target_pca)
run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('=============================================================================\n')


n_hiddens = 20
train_path = './sigma0.5/train_test_fix/pca{}/PCA_TRAIN_MODEL.csv'.format(target_pca)
test_path = './sigma0.5/train_test_fix/pca{}/PCA_TEST_MODEL.csv'.format(target_pca)
run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('=============================================================================\n')


n_hiddens = 30
train_path = './sigma0.5/train_test_fix/pca{}/PCA_TRAIN_MODEL.csv'.format(target_pca)
test_path = './sigma0.5/train_test_fix/pca{}/PCA_TEST_MODEL.csv'.format(target_pca)
run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('=============================================================================\n')


n_hiddens = 40
train_path = './sigma0.5/train_test_fix/pca{}/PCA_TRAIN_MODEL.csv'.format(target_pca)
test_path = './sigma0.5/train_test_fix/pca{}/PCA_TEST_MODEL.csv'.format(target_pca)
run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('=============================================================================\n')

# ====================================================================================

target_pca = 20

n_hiddens = 10
train_path = './sigma0.5/train_test_fix/pca{}/PCA_TRAIN_MODEL.csv'.format(target_pca)
test_path = './sigma0.5/train_test_fix/pca{}/PCA_TEST_MODEL.csv'.format(target_pca)
run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('=============================================================================\n')


n_hiddens = 20
train_path = './sigma0.5/train_test_fix/pca{}/PCA_TRAIN_MODEL.csv'.format(target_pca)
test_path = './sigma0.5/train_test_fix/pca{}/PCA_TEST_MODEL.csv'.format(target_pca)
run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('=============================================================================\n')


n_hiddens = 30
train_path = './sigma0.5/train_test_fix/pca{}/PCA_TRAIN_MODEL.csv'.format(target_pca)
test_path = './sigma0.5/train_test_fix/pca{}/PCA_TEST_MODEL.csv'.format(target_pca)
run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('=============================================================================\n')


n_hiddens = 40
train_path = './sigma0.5/train_test_fix/pca{}/PCA_TRAIN_MODEL.csv'.format(target_pca)
test_path = './sigma0.5/train_test_fix/pca{}/PCA_TEST_MODEL.csv'.format(target_pca)
run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('=============================================================================\n')

# =====================================================================================

target_pca = 30

n_hiddens = 10
train_path = './sigma0.5/train_test_fix/pca{}/PCA_TRAIN_MODEL.csv'.format(target_pca)
test_path = './sigma0.5/train_test_fix/pca{}/PCA_TEST_MODEL.csv'.format(target_pca)
run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('=============================================================================\n')


n_hiddens = 20
train_path = './sigma0.5/train_test_fix/pca{}/PCA_TRAIN_MODEL.csv'.format(target_pca)
test_path = './sigma0.5/train_test_fix/pca{}/PCA_TEST_MODEL.csv'.format(target_pca)
run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('=============================================================================\n')


n_hiddens = 30
train_path = './sigma0.5/train_test_fix/pca{}/PCA_TRAIN_MODEL.csv'.format(target_pca)
test_path = './sigma0.5/train_test_fix/pca{}/PCA_TEST_MODEL.csv'.format(target_pca)
run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('=============================================================================\n')


n_hiddens = 40
train_path = './sigma0.5/train_test_fix/pca{}/PCA_TRAIN_MODEL.csv'.format(target_pca)
test_path = './sigma0.5/train_test_fix/pca{}/PCA_TEST_MODEL.csv'.format(target_pca)
run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('=============================================================================\n')

# ====================================================================================

target_pca = 40

n_hiddens = 10
train_path = './sigma0.5/train_test_fix/pca{}/PCA_TRAIN_MODEL.csv'.format(target_pca)
test_path = './sigma0.5/train_test_fix/pca{}/PCA_TEST_MODEL.csv'.format(target_pca)
run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('=============================================================================\n')


n_hiddens = 20
train_path = './sigma0.5/train_test_fix/pca{}/PCA_TRAIN_MODEL.csv'.format(target_pca)
test_path = './sigma0.5/train_test_fix/pca{}/PCA_TEST_MODEL.csv'.format(target_pca)
run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('=============================================================================\n')


n_hiddens = 30
train_path = './sigma0.5/train_test_fix/pca{}/PCA_TRAIN_MODEL.csv'.format(target_pca)
test_path = './sigma0.5/train_test_fix/pca{}/PCA_TEST_MODEL.csv'.format(target_pca)
run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('=============================================================================\n')


n_hiddens = 40
train_path = './sigma0.5/train_test_fix/pca{}/PCA_TRAIN_MODEL.csv'.format(target_pca)
test_path = './sigma0.5/train_test_fix/pca{}/PCA_TEST_MODEL.csv'.format(target_pca)
run(train_path, test_path, target_pca, l_rate, loss_limit, n_layers, n_hiddens)
print('=============================================================================\n')
