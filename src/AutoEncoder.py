# -*- coding: utf-8 -*-
"""
AutoEncoder对特征进行编码，可以是特征压缩或是增加，也可以指定稀疏编码
"""
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras import regularizers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.externals import joblib

np.random.seed(2018)  # for reproducibility


class AutoEncoder(object):
    """
    输入一个dataframe，对它所有特征做压缩，并返回一个dataframe
    训练模型调用train（df_train, dim），获取答案调用transform（df_train, dim，output_path）
    """

    def __init__(self):
        """
        Constructor
        """
        self.model_weight_path = './ae_weight.h5'
        self.sc_path = './sc.pkl'

    def fea_scaler(self, df, mode="fit"):
        """
        对特征进行scaler
        :param df: 数据表，dataframe格式
        :param mode: 根据是训练模型还是转换特征，选择fit 或者transform
        :return:
        """
        # minmax scaler for features
        scaler = MinMaxScaler()

        if mode == "fit":
            df = scaler.fit_transform(df)
            joblib.dump(scaler, self.sc_path, compress=3)
        elif mode == "transform":
            scaler = joblib.load(self.sc_path)
            df = scaler.transform(df)

        print("minmax scaler finished")

        return df

    @staticmethod
    def get_model(dim_input, dim_emb, sparse=False):
        """
        读取深度编码器的结构，可指定的是中间层的范围，一般根据需求和输入变量的维度而定
        如果需要稀疏编码，需要添加在最后一层dense层
        如果添加了稀疏编码,模型本身不太容易过拟合,我们可以增加训练的epoch
        :param dim_input: 原始feature的维度
        :param dim_emb: 希望压缩后的维度
        :param sparse: 是否启用稀疏编码
        :return:
        """
        input_img = Input(shape=(dim_input,))

        # 编码层
        encoded = Dense(128, activation='relu')(input_img)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(10, activation='relu')(encoded)

        if sparse:
            encoder_output = Dense(dim_emb, activation='relu', activity_regularizer=regularizers.l1(0.05))(encoded)
        else:
            encoder_output = Dense(dim_emb, activation='relu')(encoded)

        # 解码层
        decoded = Dense(10, activation='relu')(encoder_output)
        decoded = Dense(64, activation='relu')(decoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(dim_input, activation='sigmoid')(decoded)

        # 构建自编码模型
        autoencoder = Model(inputs=input_img, outputs=decoded)

        # 构建编码模型
        encoder = Model(inputs=input_img, outputs=encoder_output)

        # 构建解码模型
        # 以后调用时，解码器模型的输入是编码器的输出、即最中间的隐藏层，暂时用不到解码器模型
        # encoded_input = Input(shape=(encoding_dim,))
        # # retrieve the last layer of the autoencoder model
        # decoder_layer = autoencoder.layers[-1]
        # # create the decoder model
        # decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))

        return autoencoder, encoder

    @staticmethod
    def denoisy(x_train):
        """
        to generate synthetic noisy digits, it applies a gaussian noise matrix and clip the data between 0 and 1.
        给变量增加一个高斯分布的噪音，但还是限定在0-1之内的取值
        :param x_train: dataframe
        :return:
        """
        noise_factor = 0.2
        x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)

        return x_train_noisy

    def feature_engineering(self, df_train):
        """
        对数据表做特征工程，包括：NaN补全，scaler，加噪，转换格式
        :param df_train: dataframe
        :return:
        """
        # fill NaN
        train_als = df_train.fillna(0)
        print("fill NaN finished")

        # scaler
        train_als = self.fea_scaler(train_als, mode="fit")

        # add synthetic noise
        # train_als = denoisy(train_als)

        # type transform
        x_train = np.array(train_als).astype('float32')
        print("data type transformed")

        return x_train

    def feature_engineering_test(self, df_test):
        """
        对数据表做特征工程，包括：NaN补全，scaler，加噪，转换格式
        :param df_test: dataframe
        :return:
        """
        # fill NaN
        test_als = df_test.fillna(0)
        print("fill NaN finished")

        # scaler
        test_als = self.fea_scaler(test_als, mode="transform")

        # add synthetic noise
        # train_als = denoisy(train_als)

        # type transform
        x_test = np.array(test_als).astype('float32')
        print("data type transformed")

        return x_test

    def train(self, df_train, dim):
        """
        创建autoencoder模型，并用数据加以训练，但不会返回中间编码层的结果。如果需要结果，需要再用训练数据调用transform模块。
        :param df_train: dataframe
        :param dim: 压缩维度
        :return:
        """
        col_rm = df_train.columns

        # feature engineering
        x_train = self.feature_engineering(df_train)

        # get model structure
        autoencoder, encoder = self.get_model(len(col_rm), dim, sparse=False)
        print(u"model structure generated")

        # training
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(x_train, x_train, epochs=10, batch_size=100, shuffle=True)

        # save model's weights
        encoder.save_weights(self.model_weight_path)
        print("model and wights saved\n\n")

    def transform(self, df_test, dim):
        """
        调用autoencoder模型，对数据表进行维度压缩，并返回结果。
        :param df_test: dataframe
        :param dim: 压缩维度
        :return:
        """
        col_rm = df_test.columns

        # feature engineering
        x_test = self.feature_engineering(df_test)

        # get model structure
        autoencoder, encoder = self.get_model(len(col_rm), dim, sparse=False)
        print(u"model structure generated")

        # load model's weights
        encoder.load_weights(self.model_weight_path)
        print("model and wights loaded")

        # save transformed data
        df_result = df_test.copy()
        encoded_imgs = encoder.predict(x_test)
        print(u"compressing finished")
        for i in range(dim):
            df_result.loc[:, 'f' + str(i + 1)] = encoded_imgs[:, i]

        return df_result
