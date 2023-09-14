from typing import List

import sklearn.preprocessing
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import numpy as np


def correlation(df: DataFrame):
    corr_df = df.corr()
    plt.matshow(corr_df)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14,
               rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)
    hist = df.hist(bins=50)


def correlation_unique(df: DataFrame, col_name: str):
    corr_df = df.corr()
    corr_alcool = corr_df[col_name].sort_values(ascending=False)
    plt.matshow(corr_alcool.values.reshape(1, -1))
    plt.xticks(range(len(corr_alcool)), corr_alcool.index, fontsize=14, rotation=90)
    plt.yticks([0], [col_name], fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16)




def courbe_en_fonction(df_x: DataFrame, df_y: Series, model):
    colors = sns.color_palette("husl", len(df_x.columns))


    ax = plt.subplot()

    model.fit(df_x, df_y)
    y_predict = model.predict(df_x)

    for i, col in enumerate(df_x.columns):
        x = df_x[col]
        ax.scatter(x, df_y, label=col, color=colors[i])

    ax.plot(df_x, y_predict, label="Régression linéaire", color="red")
    plt.ylabel(df_y.name)
    # plt.title(f"{df_y.name} en fonction de {df_x.columns[0]}")
    plt.grid(True)
    plt.legend()
    plt.show()


def courbe_en_fonction_LinearRegression(df_x: DataFrame, df_y: Series):
    courbe_en_fonction(df_x, df_y, LinearRegression())


def courbe_en_fonction_PolynomialFeatures(df_x: DataFrame, df_y: Series, degree=2):
    colors = sns.color_palette("husl", len(df_x.columns))
    colors2 = sns.color_palette("hls", len(df_x.columns))

    ax = plt.subplot()

    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(df_x)
    print(x_poly, df_y)
    model = LinearRegression()
    model.fit(x_poly, df_y)
    y_predict = model.predict(x_poly)

    for i, col in enumerate(df_x.columns):
        x = df_x[col].values.reshape(-1, 1)

        sort_order = np.argsort(x, axis=0)
        x_sorted = x[sort_order].ravel()  # Conversion en tableau 1D
        y_pred_sorted = y_predict[sort_order].ravel()

        ax.scatter(x, df_y, label=col, color=colors[i])

        ax.plot(x_sorted, y_pred_sorted, label="Régression linéaire", color=colors2[i])
    plt.ylabel(df_y.name)
    # plt.title(f"{df_y.name} en fonction de {df_x.columns[0]}")
    plt.grid(True)
    plt.legend()
    plt.show()
