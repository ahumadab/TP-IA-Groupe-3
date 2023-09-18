import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from helpers import correlation, correlation_unique, courbe_en_fonction, courbe_en_fonction_LinearRegression, \
    courbe_en_fonction_PolynomialFeatures, patrick

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("./beers/recipeData.csv", encoding="latin1")
# orig_leng = len(df)
# print(orig_leng)
df = df.drop(
    columns=["BeerID", "UserId", "URL", "Name", "Style", "PrimingMethod", "PrimingAmount", "PitchRate", "MashThickness",
             "PrimaryTemp", "SugarScale"])  # PrimaryTemp
# ser = df.isna().mean() * 100
# print(ser)
df.dropna(inplace=True)
# ser = df.isna().mean() * 100
# print(ser)
#
# orig_leng = len(df)
# print(ser)
# og_df = df[["OG"]]
# print(og_df.describe(include='all'))
# og_df = og_df.sort_values(by="OG", ascending=True)
# plt.plot(range(len(og_df)), og_df, marker=",")
# plt.xlabel("Pourcentage")
# pourcentages = [str(i) + '%' for i in range(0, 101, 10)]
# plt.xticks(range(0, len(df), len(df) // 10), pourcentages)
# plt.ylabel('OG (Densité du moût avant fermentation)')
# plt.title("Courbe triée des valeurs de OG")
# plt.grid()
# plt.show()

df = df[df["Size(L)"] <= df["Size(L)"].quantile(0.95)]
df = df[df["OG"] <= df["OG"].quantile(0.95)]
df = df[df["FG"] <= df["FG"].quantile(0.95)]
df = df[df["IBU"] <= 150]  # IBU max == 150 selon wikipedia
df = df[df["BoilSize"] <= df["BoilSize"].quantile(0.95)]
# df = df[df["IBU"] > 5]

# new_len = len(df)
# print(new_len / orig_leng, new_len)


df.rename(columns={"OG": "Density of Wort b4 ferm", "FG": "Density of Wort after ferm", "ABV": "Alcohol By Volume", "IBU": "International Bittering Units"}, inplace=True)

one_hot = pd.get_dummies(df["BrewMethod"])
# print(one_hot)
df = df.drop(columns=["BrewMethod"])
df = df.join(one_hot)
# print(df)



# correlation(df)
#
# correlation_unique(df, "Alcohol By Volume")
#
# correlation_unique(df, "International Bittering Units")


# plt.subplot()
ser_alcool = df["Alcohol By Volume"]
ser_amertume = df["International Bittering Units"]

df_desity_before_ferm = df[["Density of Wort b4 ferm"]]
df_desity_after_ferm = df[["Density of Wort after ferm"]]
df_boil_time = df[["BoilTime"]]

df_density = df[["Density of Wort b4 ferm", "Density of Wort after ferm"]]
df_alcool = df[["Alcohol By Volume"]]
df_amertume = df[["International Bittering Units"]]

# courbe_en_fonction_LinearRegression(df_desity_before_ferm, df_alcool)
# courbe_en_fonction_PolynomialFeatures(df_desity_after_ferm, df_alcool, degree=2)
# courbe_en_fonction_PolynomialFeatures(df_boil_time, df_alcool, degree=3)


# courbe_en_fonction_PolynomialFeatures(df_amertume, ser_alcool, degree=5)

# print(df_density)
# courbe_en_fonction_PolynomialFeatures(df_density, df_alcool, degree=1)

# courbe_en_fonction_PolynomialFeatures(df_desity_before_ferm, df_amertume, degree=10)
# courbe_en_fonction_PolynomialFeatures(df_desity_after_ferm, df_amertume, degree=7)
# courbe_en_fonction_PolynomialFeatures(df_boil_time, df_amertume, degree=4)
# courbe_en_fonction_PolynomialFeatures(df_alcool, ser_amertume, degree=2)
# courbe_en_fonction(df_density, df_amertume)

# plt.show()
X = df[["Size(L)", "Density of Wort b4 ferm", "Density of Wort after ferm", "BoilSize", "BoilTime", "BoilGravity", "Efficiency"]]
y_ibu = df["International Bittering Units"]
y_abv = df["Alcohol By Volume"]


patrick(X, y_ibu, y_abv)
