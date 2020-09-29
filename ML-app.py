
import tensorflow as tf
import requests
import streamlit as st
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
import numpy as np
from io import BytesIO
from gtts import gTTS 
from IPython.display import Audio


model = ResNet50(weights='imagenet')

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

def scale_image(image):
  return tf.image.resize(image,[224,224])

def decode_image(image):
  img = tf.image.decode_jpeg(image,channels=3)
  img = scale_image(img)
  return img

def remove(string): 
    return string.replace("_", " ") 


st.markdown(STYLE, unsafe_allow_html=True)

st.title("Blind Aid Cane Software Demo")
path = st.text_input("Enter image URL to classify: ","https://www.extremetech.com/wp-content/uploads/2019/12/SONATA-hero-option1-764A5360-edit.jpg")
content = requests.get(path).content


numpy_image = image.img_to_array(decode_image(content))
image_batch = np.expand_dims(numpy_image, axis=0)
processed_image = preprocess_input(image_batch, mode='caffe')
preds = model.predict(processed_image)
pred_class = decode_predictions(preds, top=1)
print(pred_class)
string = pred_class[0][0][1]


tts = gTTS(remove(string)) 
tts.save('1.mp4')
sound_file = '1.mp4'
st.audio(sound_file)
