import  numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pickle


model = pickle.load(open('mnist_sgd1.pkl','rb'))
st.markdown('''
# Digit recognizer 
This **Digit recognizer** created using `Python` + `Streamlit` + `Stochastic Gradient Descent (SGD)` !
''')

canvas_result = st_canvas(
    fill_color = "#ffffff",
    stroke_width = 10,
    stroke_color = "#ffffff",
    background_color = "#000000",
    height = 150,width = 150,
    drawing_mode = 'freedraw',
    key = "canvas",
)

if canvas_result.image_data is not None:
    
    img = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img.astype('uint8'), (28,28))
    img = img.reshape(1, 784)

if st.button('Predict'):
    
    prediction = model.predict(img)
    a = prediction[0]
    st.header(a)
