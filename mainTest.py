# import cv2
# from keras.models import load_model
# from PIL import Image
# import numpy as np

# model=load_model('BrainTumor10Epochs.h5')

# image=cv2.imread('C:\\Users\\Dell\\Desktop\\NishitData\\BrainDiseasesProject\\brain_tumor_classification\\PREDICTION\\pred23.jpg')



# img=Image.fromarray(image)

# img=img.resize((64,64))
# img=np.array(img)

# input_img=np.expand_dims(img, axis=0)


# result=model.predict_classes(input_img)
# print(result)


import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

# Load your model
model = load_model('BrainTumor10Epochs.h5')

# Read the image using OpenCV
image = cv2.imread('C:\\Users\\Dell\\Desktop\\NishitData\\BrainDiseasesProject\\brain_tumor_classification\\PREDICTION\\pred20.jpg')

# Convert the image to PIL format
img = Image.fromarray(image)

# Resize the image to match the model's expected input size
img = img.resize((64, 64))

# Convert the image to a NumPy array
img = np.array(img)

# Normalize pixel values to between 0 and 1 (if your model was trained with normalized data)
img = img / 255.0

# Expand the dimensions to match the input shape expected by the model
input_img = np.expand_dims(img, axis=0)

# Make predictions
predictions = model.predict(input_img)

# Get the predicted class index
predicted_class_index = np.argmax(predictions, axis=1)[0]

print("Predicted Class Index:", predicted_class_index)
