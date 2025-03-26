#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[2]:


import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("iris_model.pkl")

# Streamlit UI
st.title("Iris Flower Classification")
st.write("Enter flower measurements below:")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")

# Make prediction
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    class_names = ["Setosa", "Versicolor", "Virginica"]
    st.write(f"Predicted Class: {class_names[prediction[0]]}")


# In[ ]:





# In[ ]:




