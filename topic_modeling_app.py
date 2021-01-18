import streamlit as st
import numpy as np
import pandas as pd


FILE = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/dataset.csv"

@st.cache
def get_data():
    return pd.read_csv(FILE,delimiter='\n',header=None ).rename(columns={0: "text"})


st.title('Topic Modeling')
st.markdown("Welcome to this Topic Modeling App. If you want to see more project and cool apps. Please visit my portofolio [page] (http://mustaphabounoua.ml/). ")

st.markdown(" Here’s the full  [code] (https://github.com/MustaphaBounoua/Topic_Modeling) for this app if you would like to get a look.")


st.header("DataSet")
st.markdown("We will use a public [dataset](http://data.insideairbnb.com/united-states/ny/new-york-city/2019-09-12/visualisations/listings.csv) which is a collection of articles title. We will try to model the different topics to which these articles may be grouped to ")

df = get_data()

st.dataframe(df.head())