# -*- coding: utf-8 -*-
"""
AutoEncoder对特征进行编码，可以是特征压缩或是增加，也可以指定稀疏编码
"""
import numpy as np
import pandas as pd
import gc
from AutoEncoder import AutoEncoder

np.random.seed(2018)  # for reproducibility


def start_train(ae, path, dim):
    # read data
    df = pd.read_csv(path, encoding='utf-8')
    print("data read")

    # extract als features
    cols_als = [col for col in df.columns if col.startswith("als_")]
    df_train = df[cols_als]
    del df
    gc.collect()

    # start training
    ae.train(df_train, dim)


def start_transform(ae, path_src, path_tar, dim):
    # read data
    df = pd.read_csv(path_src, encoding='utf-8')
    print("data read")

    # extract als features
    cols_als = [col for col in df.columns if col.startswith("als_")]
    df_transform = df[cols_als]

    # start transforming
    df_result = ae.transform(df_transform, dim)

    # localization
    col_rm = df_transform.columns
    df_result.drop(col_rm, 1, inplace=True)
    df_result.to_csv(path_tar, encoding='utf-8')
    print(u"result written\n")


if __name__ == "__main__":
    train_path = './train.csv'
    test_path = './test.csv'
    result_path = './result.csv'
    encoding_dim = 30  # 压缩特征的维度

    AE = AutoEncoder()

    start_train(AE, train_path, encoding_dim)
    start_transform(AE, train_path, result_path, encoding_dim)
