import numpy as np
from keras_preprocessing import image
import cv2
import os
from keras.models import Sequential, load_model
from keras_preprocessing.image import ImageDataGenerator

vid = cv2.VideoCapture(0)
print("Camera connection successfully established")

i = 0
classes = ['Nam','Nha','Nhân']
new_model = load_model('khuonmat.h5')
while(True):
    frame, frame = vid.read()
    cv2.imshow('frame', frame)
    cv2.imwrite('C:\\AI\\save_pictures' + str(i) + ".jpg", frame)
    test_image = image.load_img('C:\\AI\\save_pictures' + str(i) + ".jpg", target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = new_model.predict(test_image)
    result1 = result[0]
    for y in range(3):
        if result1[y] == 1.:
            break
    prediction = classes[y]
    print(prediction)
    os.remove('C:\\AI\\save_pictures' + str(i) + ".jpg")
    i = i + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
