import tensorflow as tf
# from tensorflow.keras.models import  load_model
import keras
from keras.models import load_model

import streamlit as st
import numpy as np 


# Add custom CSS to change the background color
st.markdown(
    """
    <style>
    .stApp {
        background-color:  white; /* Change this value to the desired background color */
    }
    </style>
    """,
    unsafe_allow_html=True
)



st.header(' Fruit/Vegetable Image Classifier⬇')
model = load_model(r'D:\Projects\Image Classification\Image_classify.keras')
data_cat = ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']
img_height = 180
img_width = 180

image = st.file_uploader("Choose a Image ⬇", type=["jpg", "png", "txt"])


image_load = keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = keras.utils.array_to_img(image_load)
img_bat=tf.expand_dims(img_arr,0)

predict = model.predict(img_bat)

score = tf.nn.softmax(predict)
st.image(image, width=200)
st.write('Veg/Fruit in image is:-' + data_cat[np.argmax(score)])
st.write('With accuracy of:- ' + str(np.max(score)*100))


