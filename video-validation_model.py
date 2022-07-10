# %% [markdown]
# ## Import libraries and Setup

# %%
# Common imports
import numpy as np

# TensorFlow imports
# may differs from version to versions
import tensorflow as tf
from tensorflow import keras

# OpenCV
import cv2

# %%
# Colors to draw rectangles in BGR
RED = (0, 0, 255)
GREEN = (0, 255, 0)

# %%
# opencv object that will detect faces for us
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# %% [markdown]
# ## Load Model

# %%
# Load model to face classification
# model was created in me_not_me_classifier.ipynb notebook
model_name = 'face_classifier_MobileNet_15.h5'
model_name = 'face_classifier.h5'

face_classifier = keras.models.load_model(f'models/{model_name}')
class_names = ['me', 'not_me']

# %%
def get_extended_image(img, x, y, w, h, k=0.1):
    '''
    Function, that return cropped image from 'img'
    If k=0 returns image, cropped from (x, y) (top left) to (x+w, y+h) (bottom right)
    If k!=0 returns image, cropped from (x-k*w, y-k*h) to (x+k*w, y+(1+k)*h)
    After getting the desired image resize it to 250x250.
    And converts to tensor with shape (1, 250, 250, 3)

    Parameters:
        img (array-like, 2D): The original image
        x (int): x coordinate of the upper-left corner
        y (int): y coordinate of the upper-left corner
        w (int): Width of the desired image
        h (int): Height of the desired image
        k (float): The coefficient of expansion of the image

    Returns:
        image (tensor with shape (1, 250, 250, 3))
    '''

    # The next code block checks that coordinates will be non-negative
    # (in case if desired image is located in top left corner)
    if x - k*w > 0:
        start_x = int(x - k*w)
    else:
        start_x = x
    if y - k*h > 0:
        start_y = int(y - k*h)
    else:
        start_y = y

    end_x = int(x + (1 + k)*w)
    end_y = int(y + (1 + k)*h)

    face_image = img[start_y:end_y,
                     start_x:end_x]
    face_image = tf.image.resize(face_image, [250, 250])
    # shape from (250, 250, 3) to (1, 250, 250, 3)
    face_image = np.expand_dims(face_image, axis=0)
    return face_image

# %% [markdown]
# ## Streaming Cycle

# %%
video_capture = cv2.VideoCapture(0)  # webcamera

if not video_capture.isOpened():
    print("Unable to access the camera")
else:
    print("Access to the camera was successfully obtained")

print("Streaming started - to quit press ESC")
while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # for each face on the image detected by OpenCV
        # get extended image of this face
        face_image = get_extended_image(frame, x, y, w, h, 0.5)

        # classify face and draw a rectangle around the face
        # green for positive class and red for negative
        result = face_classifier.predict(face_image)
        prediction = class_names[np.array(
            result[0]).argmax(axis=0)]  # predicted class
        confidence = np.array(result[0]).max(axis=0)  # degree of confidence

        if prediction == 'me':
            color = GREEN
        else:
            color = RED
        # draw a rectangle around the face
        cv2.rectangle(frame,
                      (x, y),  # start_point
                      (x+w, y+h),  # end_point
                      color,
                      2)  # thickness in px
        cv2.putText(frame,
                    # text to put
                    "{:6} - {:.2f}%".format(prediction, confidence*100),
                    (x, y),
                    cv2.FONT_HERSHEY_PLAIN,  # font
                    2,  # fontScale
                    color,
                    2)  # thickness in px

    # display the resulting frame
    cv2.imshow("Face detector - to quit press ESC", frame)

    # Exit with ESC
    key = cv2.waitKey(1)
    if key % 256 == 27:  # ESC code
        break


# when everything done, release the capture
video_capture.release()
cv2.destroyAllWindows()
print("Streaming ended")

# %%


# %%



