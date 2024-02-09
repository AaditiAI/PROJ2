import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import VGG16  # Import VGG16 from Keras

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the VGG16 model
vgg16 = VGG16(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))

# Rest of your code...
import pickle
from keras.applications.vgg16 import VGG16

# Serialize the VGG16 model
serialized_vgg16_model = pickle.dumps(vgg16)

def load_image(image_path):
    input_image = Image.open(image_path)
    resized_image = input_image.resize((224, 224))

    return resized_image

def get_image_embeddings(object_image : image):
    image_array = np.expand_dims(image.img_to_array(object_image), axis = 0)
    image_embedding = vgg16.predict(image_array)

    return image_embedding

def get_similarity_score(first_image : str, second_image : str):

    first_image = load_image(first_image)
    second_image = load_image(second_image)

    first_image_vector = get_image_embeddings(first_image)
    second_image_vector = get_image_embeddings(second_image)

    similarity_score = cosine_similarity(first_image_vector, second_image_vector).reshape(1,)

    return similarity_score

# Serialize the utility functions
utility_functions = {
    'load_image': load_image,
    'get_image_embeddings': get_image_embeddings,
    'get_similarity_score': get_similarity_score
}

# Serialize any necessary variables
# For example, if you have paths to images as variables, you may want to serialize them too

# Save the serialized objects to a .pkl file
with open('vgg16_similarity_model.pkl', 'wb') as f:
    pickle.dump(serialized_vgg16_model, f)
    pickle.dump(utility_functions, f)

print("Model saved successfully to vgg16_similarity_model.pkl")
