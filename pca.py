import pandas as pd
from sklearn.decomposition import PCA

def apply_pca(train_df, test_df, target_size):

    print('PCA Target Size = {}'.format(target_size))
    pca = PCA(target_size)

    train_df = pd.DataFrame(train_df)
    test_df = pd.DataFrame(test_df)

    X_train = train_df.iloc[:, :-1]
    X_test = test_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    y_test = test_df.iloc[:, -1]

    # Transform Feature DataFrame
    print('Transforming Train Dataset')
    X_train = pca.fit_transform(X_train)
    X_train = pd.DataFrame(X_train)
    print('Transforming Test Dataset')
    X_test = pca.transform(X_test)
    X_test = pd.DataFrame(X_test)

    # Reconstruct DataFrame with label
    pca_train_set = pd.concat([X_train,y_train], axis=1, ignore_index=True)
    # train_name = './train_test_set/PCA_TRAIN_MODEL.csv'
    train_name = './sigma0.5/train_test_fix/PCA_TRAIN_MODEL.csv'
    pca_train_set.to_csv(train_name, header=False, index=False)

    pca_test_set = pd.concat([X_test, y_test], axis=1, ignore_index=True)
    # test_name = './train_test_set/PCA_TEST_MODEL.csv'
    test_name = './sigma0.5/train_test_fix/PCA_TEST_MODEL.csv'
    pca_test_set.to_csv(test_name, header=False, index=False)

    print('PCA Applied')
    return pca_train_set, pca_test_set
