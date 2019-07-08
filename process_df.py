import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def build_df(ada_features, tidak_features):
    # Build DataFrame From Feature Arrays
    print('Building DataFrame')
    df_tidak = pd.DataFrame(np.array(tidak_features))
    df_tidak['label'] = 0
    df_ada = pd.DataFrame(np.array(ada_features))
    df_ada['label'] = 1
    # Concatenate DataFrame
    df_feat = pd.concat([df_tidak, df_ada], ignore_index=True)
    df_feat.to_csv('./sigma0.5/features_df.csv', header=False, index=False)
    print('DataFrame Built\n')
    return df_feat

def normalize_df(df):
    # Normalize DataFrame
    print('Normalizing DataFrame')
    dataset = pd.DataFrame(df)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X)

    dataset = pd.concat([X,y], axis=1)
    dataset.to_csv('./sigma0.5/normalized_df.csv', header=False, index=False)

    print('DataFrame  Normalized!\n')
    return dataset

def train_test(dataset):
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
    train_df.to_csv('./sigma0.5/train_test_set/train_df.csv', header=False, index=False)
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv('./sigma0.5/train_test_set/test_df.csv', header=False, index=False)

    print('Dataset Splitted Into Train-Test Set!\n')
    return train_df, test_df
