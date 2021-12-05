# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import sklearn
import datetime

#drop extra rows where year and month are nan.Convert year and month columns into datetime type.

df = pd.read_csv("/Users/tryfighter/Desktop/math_10/datasets/All_India_Index_july2019_20Aug2020.csv", na_values = " ")
df = df.dropna(subset=['Year','Month'])
df['Year'] = df['Year'].astype(int)
df = df.applymap(lambda x: 'March' if x == 'Marcrh' else x)
df['datetime'] = pd.to_datetime(df['Year'].astype(str) + df['Month'], format='%Y%B', exact = False)
df = df.drop(index = [258,259,260,261,262,263,264,265,266])

#Construct rural CPI df
df_rural = df[df["Sector"].map(lambda s: s == 'Rural')].copy()
df_r = df_rural.loc[:,:'Food and beverages'].copy()
df_r['General Index'] = df_rural['General index']
df_r['datetime'] = df_rural['datetime']
cpis = df.columns


st.title("India Rural Area Food CPI Analysis")
st.markdown("The dataset I will explore is consumer price index of India from 2013 to 2020.\
https://www.kaggle.com/satyampd/consumer-price-index-india-inflation-data")
st.markdown('The aim of the study is to predict India rural **_Oils and Fats_** CPI by other CPIs in food sectors.')
st.header("Compare **_Oils and Fats_** to another CPI")

#get date range
choice = st.selectbox(label = 'Select a CPI', options = df_r.drop(['Oils and fats','Sno',
                                                                   'Sector','Year','Month','datetime'], axis = 1).columns)
date_range = st.date_input(label = "Select date range", value = [datetime.date(2013, 1, 1), datetime.date(2020, 3, 1)],
                          min_value = datetime.date(2013, 1, 1), max_value = datetime.date(2020, 3, 1))

try:
    df_temp = (df_r['datetime'] >= pd.to_datetime(date_range[0])) & (df['datetime'] <= pd.to_datetime(date_range[1]))
    
    selection = alt.selection_interval(bind = 'scales')
    
    general_chart = alt.Chart(df_r.loc[df_temp]).mark_circle().encode(
    x = 'datetime',
    y = alt.Y(choice,scale=alt.Scale(zero=False)),
    color=choice,
    tooltip = ['Year','Month',choice]
    ).properties(
        title = f'Rural {choice} & Oils and Fats CPI').add_selection(
    selection)

    oils_chart = alt.Chart(df_r.loc[df_temp]).mark_line().encode(
        x = 'datetime',
        y = "Oils and fats",
        color=alt.value('red'),
        )
    
    my_chart = general_chart + oils_chart
    st.altair_chart(my_chart)
except:
    st.write("Selecting date range")
    
st.markdown('From the graph, we can tell that CPI of **_Oils and Fats_** has the same trend with other CPIs.\
            However, the increasing rate is relatively lower than other CPIs. The reason might because oil\
            is necessary product. The price volatility is less.')
            
st.markdown("""---""")
#Linear Regression
x = df_r.drop(['Oils and fats','Sno','Sector','Year','Month','datetime'], axis = 1)
y = df_r['Oils and fats']
variables = list(x.columns)
st.header("Use multiple variables linear regression to predict **_Oils and Fats_**")
st.markdown("I will use all other foods and general CPI to predict oils and fats CPI.")
st.markdown(f'List of independents used for linear regression {variables}')
st.markdown("Data will be seperated into train and test. You can select the percentage of train data below.")
train_size = st.slider(label = "Select percentage of train data, minimum 2%, maximum 90%",
                       min_value = 2, max_value = 90, step = 1)

try:
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = (100-train_size)/100, random_state = 48)
    model = sklearn.linear_model.LinearRegression()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    
    score = sklearn.metrics.r2_score(y_test, y_predict)
    mean_sq_error = sklearn.metrics.mean_squared_error(y_test, y_predict)
    root_mean_sq_error = np.sqrt(mean_sq_error)
    
    st.text(f'r2 score is score {score}')
    st.text(f'mean squared error is {mean_sq_error}')
    st.text(f'root mean squared error is {root_mean_sq_error}')
except:
    st.text('Selecting Training size.')

st.markdown("The model has a high r2 score at any training size. It can indicate that it is a good model\
            to predict **_Oils and Fats_**. However, we only use other CPIs to do the prediction.\
            It means other important independent might be missed in the model. We can only have a conclusion\
            that **_Oils and Fats_** has strong linear relationship with other CPIs.")

