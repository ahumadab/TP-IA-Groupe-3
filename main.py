import streamlit as st

from helpers import ABV_pred, IBU_pred

st.title("BREW IA")

OG = st.slider('Densité du mout avant fermentation', min_value=1.0, max_value=1.1, value=1.05, step=0.01)
FG = st.slider('Densité du mout après fermentation', min_value=1.0, max_value=1.2, value=1.05, step=0.05)

boilgravity = st.slider('BoilGravity', min_value=1.0, max_value=1.5, value=1.05, step=0.05)
boilsize = st.slider('Boil size (L)', min_value=1, max_value=50, value=5, step=1)
boiltime = st.slider('BoilTime (min)', min_value=30, max_value=120, value=60, step=1)


size = st.slider('Size (L)', min_value=1, max_value=60, value=5, step=1)

efficiency = st.slider('Efficiency (g)', min_value=0, max_value=100, value=65, step=1)


if st.button('Send'):
    abv_pred = ABV_pred([size, OG, FG, boilsize, boiltime, boilgravity, efficiency])
    ibu_pred = IBU_pred([size, OG, FG, boilsize, boiltime, boilgravity, efficiency, abv_pred])
    abv_str = str(round(abv_pred, 2))
    ibu_str = str(round(ibu_pred, 2))
    st.write(f"La bière aura {abv_str}% d'alcool et une amertume de {ibu_str}")

