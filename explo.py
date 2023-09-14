import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from helpers import correlation, correlation_unique, courbe_en_fonction, courbe_en_fonction_PolynomialFeatures

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("./beers/recipeData.csv", encoding="latin1")
# orig_leng = len(df)
df = df.drop(
    columns=["BeerID", "UserId", "URL", "Name", "Style", "PrimingMethod", "PrimingAmount", "PitchRate", "MashThickness",
             "PrimaryTemp", "SugarScale"])  # PrimaryTemp
df.dropna(inplace=True)
ser = df.isna().mean() * 100

df = df[df["Size(L)"] <= df["Size(L)"].quantile(0.95)]
df = df[df["OG"] <= df["OG"].quantile(0.95)]
df = df[df["FG"] <= df["FG"].quantile(0.95)]
df = df[df["IBU"] <= 150]  # IBU max == 150 selon wikipedia
df = df[df["BoilSize"] <= df["BoilSize"].quantile(0.95)]

df.rename(columns={"OG": "Density of Wort b4 ferm", "FG": "Density of Wort after ferm", "ABV": "Alcohol By Volume", "IBU": "International Bittering Units"}, inplace=True)

one_hot = pd.get_dummies(df["BrewMethod"])
# print(one_hot)
df = df.drop(columns=["BrewMethod"])
df = df.join(one_hot)
# print(df)


# print(df.describe(include='all'))

# correlation(df)
#
# correlation_unique(df, "Alcohol By Volume")
#
# correlation_unique(df, "International Bittering Units")


# plt.subplot()
df_alcool = df["Alcohol By Volume"]
df_amertume = df["International Bittering Units"]

df_desity_before_ferm = df[["Density of Wort b4 ferm"]]
df_desity_after_ferm = df[["Density of Wort after ferm"]]
df_boil_time = df[["BoilTime"]]

df_density = df[["Density of Wort b4 ferm", "Density of Wort after ferm"]]

# courbe_en_fonction_PolynomialFeatures(df_desity_before_ferm, df_alcool)
# courbe_en_fonction_PolynomialFeatures(df_desity_after_ferm, df_alcool)
# courbe_en_fonction_PolynomialFeatures(df_boil_time, df_alcool, degree=3)
print(df_density)
# courbe_en_fonction_PolynomialFeatures(df_density, df_alcool, degree=1)

# courbe_en_fonction_PolynomialFeatures(df_desity_before_ferm, df_amertume)
# courbe_en_fonction_PolynomialFeatures(df_desity_after_ferm, df_amertume)
# courbe_en_fonction_PolynomialFeatures(df_boil_time, df_amertume, degree=3)
courbe_en_fonction(df_density, df_amertume)

# plt.show()
