from typing import List

import sklearn.preprocessing
import skl2onnx
from skl2onnx import convert, to_onnx
from skl2onnx.common.data_types import FloatTensorType
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
import onnxruntime as ort
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import numpy as np
import sys


np.set_printoptions(threshold=sys.maxsize)

def correlation(df: DataFrame):
    corr_df = df.corr()
    print(corr_df)
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
    plt.xlabel(df_x.columns[0])
    plt.title(f"{df_y.name} en fonction de {df_x.columns[0]}")
    plt.grid(True)
    plt.legend()
    plt.show()


def courbe_en_fonction_LinearRegression(df_x: DataFrame, df_y: Series):
    courbe_en_fonction(df_x, df_y, LinearRegression())


def courbe_en_fonction_PolynomialFeatures(df_x: DataFrame, df_y: Series, degree=2):
    colors = sns.color_palette("husl", len(df_x.columns))
    colors2 = sns.color_palette("hls", len(df_x.columns)+3)

    ax = plt.subplot()

    poly_features = PolynomialFeatures(degree=degree)
    x_poly = poly_features.fit_transform(df_x)
    model = LinearRegression()
    model.fit(x_poly, df_y)
    y_predict = model.predict(x_poly)

    for i, col in enumerate(df_x.columns):
        x = df_x[col].values.reshape(-1, 1)
        sort_order = np.argsort(x, axis=0)
        x_sorted = x[sort_order].ravel()  # Conversion en tableau 1D
        y_pred_sorted = y_predict[sort_order].ravel()
        ax.scatter(x, df_y, label=col, color=colors[i])
        ax.plot(x_sorted, y_pred_sorted, label="Régression linéaire", color=colors2[i+3])

    plt.ylabel(df_y.name)
    plt.xlabel(df_x.columns[0])
    plt.title(f"{df_y.name} en fonction de {df_x.columns[0]}")
    plt.grid(True)
    plt.legend()
    plt.show()


def save_model_onnx(model, X, filename):
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    onnx_model = to_onnx(model, initial_types=initial_type)

    with open(f"{filename}.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

def patrick(X: DataFrame, y_ibu, y_abv):
    print(X)
    # model_abv = LinearRegression()
    # model_abv.fit(X, y_abv)
    # y_abv_pred = model_abv.predict(X)

    # save_model_onnx(model_abv, X)

    X_test_numpy = X.values.astype(np.float32)
    print(X_test_numpy[0, :])
    print(X_test_numpy.shape)
    model = ort.InferenceSession("ABV.onnx")
    input_data = {"float_input": X_test_numpy}
    predictions = model.run(None, input_data)
    print(predictions[0][0, :])
    X.insert(len(X.columns), "ABV", predictions[0])

    X_train, X_test, y_ibu_train, y_ibu_test = train_test_split(X, y_ibu, test_size=0.2, random_state=42)

    # param_grid = {
    #     'n_estimators': [100, 200, 300],  # Nombre d'arbres dans la forêt aléatoire
    #     'max_depth': [None, 10, 20, 30],  # Profondeur maximale de chaque arbre
    #     'min_samples_split': [2, 5, 10],  # Nombre minimum d'échantillons requis pour diviser un nœud
    #     'min_samples_leaf': [1, 2, 4]  # Nombre minimum d'échantillons requis dans une feuille
    # }

    # model_ibu = RandomForestRegressor(n_estimators=100, random_state=42)
    # grid_search = HalvingGridSearchCV(model_ibu, param_grid, verbose=2, n_jobs=-1, random_state=42)
    # grid_search.fit(X_train, y_ibu_train)
    # best_params = grid_search.best_params_ # {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300}
    """best_params = {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 300}
    print("Meilleurs hyperparamètres : ", best_params)
    best_rf_model = RandomForestRegressor(random_state=42, **best_params)
    best_rf_model.fit(X_train, y_ibu_train)
    y_ibu_pred = best_rf_model.predict(X_test)"""
    X_test_numpy = X_test.values.astype(np.float32)
    # print(X_test_numpy[0, :])
    # print(X_test_numpy.shape)
    model = ort.InferenceSession("IBU.onnx")
    input_data = {"float_input": X_test_numpy}
    predictions = model.run(None, input_data)
    # print(predictions)
    y_ibu_pred = predictions[0]
    # model_ibu.fit(X_train, y_ibu_train)
    # y_ibu_pred = model_ibu.predict(X_test)
    mse_ibu = mean_squared_error(y_ibu_test, y_ibu_pred)
    r2_ibu = r2_score(y_ibu_test, y_ibu_pred)

    print("mse_ibu\n", mse_ibu)
    print("r2_ibu\n", r2_ibu)
    # print(sqrt(mse_ibu))

    # save_model_onnx(best_rf_model, X)

    # x = np.linspace(0, 1, len(y_ibu_test))
    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.scatter(x, y_ibu_test, color='yellow', label='Réel', alpha=0.5)
    #
    # # Tracer les valeurs prédites en rouge
    # ax.scatter(x, y_ibu_pred, color='red', label='Prédit', alpha=0.5)
    #
    # # Ajouter une légende
    # ax.legend()
    #
    # # Définir les étiquettes des axes
    # ax.set_xlabel('Échantillons')
    # ax.set_ylabel('IBU')
    #
    # # Titre du graphique
    # plt.title('Réel vs Prédit (IBU)')
    #
    # # Afficher le graphique
    # plt.show()


def ABV_pred(inputs):
    X_test_numpy = np.array([inputs], dtype=np.float32)
    model = ort.InferenceSession("ABV.onnx")
    input_data = {"float_input": X_test_numpy}
    predictions = model.run(None, input_data)
    return predictions[0][0, 0]

def IBU_pred(inputs):
    # print(inputs)
    X_test_numpy = np.array([inputs], dtype=np.float32)
    # print(X_test_numpy)
    model = ort.InferenceSession("IBU.onnx")
    input_data = {"float_input": X_test_numpy}
    predictions = model.run(None, input_data)
    # print(predictions[0])
    return predictions[0][0, 0]
