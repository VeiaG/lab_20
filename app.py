# Веб додаток 
# Тема : Вгадування числа
# Паламар Роман КН-2-2
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import numpy as np


st.sidebar.header('Введіть параметри')
def user_input_features(): 
    num1 = st.sidebar.slider('Перше число', 0.0 , 10.0 , 1.0)
    num2 = st.sidebar.slider('Друге число', 0.0 , 10.0 , 2.0)
    num3 = st.sidebar.slider('Третє число', 0.0 , 10.0 , 3.0)
    num4 = st.sidebar.slider('Четверте число', 0.0 , 10.0 , 4.0)
    data = [[0,num1],[1,num2],[2,num3],[3,num4]]
    return data

to_predict_x = [4,5,6]
to_predict_x= np.array(to_predict_x).reshape(-1,1)
data = user_input_features()

X = np.array(data)[:,0].reshape(-1,1)
y = np.array(data)[:,1].reshape(-1,1)

regsr = LinearRegression()
regsr.fit(X, y)

prediction = regsr.predict(to_predict_x)
m= regsr.coef_
c= regsr.intercept_

st.write(""" 
### Програма для вгадування числа
за допомогою **LinearRegression**
""")

st.subheader('Введені числа')
st.write(y)
st.subheader('Наступні три передбачені числа')
st.write(prediction)



new_y=[ m*i+c for i in np.append(X,to_predict_x)]
new_y=np.array(new_y).reshape(-1,1)

pGraph = np.concatenate((np.array(y),np.array(prediction)))
pGlobalGraph = np.stack((pGraph,new_y)).reshape(2,7).swapaxes(0,1)
chart_data = pd.DataFrame((pGlobalGraph) , columns=['Графік', 'Лінія прогнозу'])
st.line_chart(chart_data)

st.subheader('slope (m)')
st.write(m)
st.subheader('y-intercept (c):')
st.write(c)