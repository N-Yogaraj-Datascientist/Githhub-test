# Predicting Rent Prices

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title('House rent prediction APP')

data={
'sqfeet_size':[2000,1800,1600,1500,1900,1700,1400,1200,1300],
'no_of_bedrooms':[4,3,3,2,4,3,2,2,2],
'location_rating':[9,7,6,3,4,6,8,10,4],
'rent_price':[25000,22000,18000,12000,23000,20000,19000,17000,16000]}

df=pd.DataFrame(data)

x=df[['sqfeet_size','no_of_bedrooms','location_rating']]
y=df['rent_price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)

sqfeet_size=st.number_input('Enter the sqfeet size')
no_of_bedrooms=st.number_input('Enter the no of bedrooms')
location_rating=st.number_input('Enter the loc rating')

user_input=np.array([[sqfeet_size,no_of_bedrooms,location_rating]])

prediction=model.predict(user_input)
st.write('House rent predicted as',prediction[0])