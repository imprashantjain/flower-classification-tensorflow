import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("flower_model.keras")

class_names = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips"
]

img = tf.keras.utils.load_img(
    "rose.webp",      #flower image
    target_size=(224,224)
)

img_array = tf.keras.utils.img_to_array(img)

img_array = tf.expand_dims(img_array,0)

predictions = model.predict(img_array)

score = tf.nn.softmax(predictions[0])

print("Prediction:", class_names[np.argmax(score)])
print("Confidence:", np.max(score)*100)