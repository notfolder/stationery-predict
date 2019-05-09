#%%
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

model = load_model("stationery.hdf5", compile = False)
img_path = ('./IMG_7347.jpg')
img = img_to_array(load_img(img_path, target_size=(224,224)))
img_nad = img_to_array(img)/255
img_nad = img_nad[None, ...]

label=['marker_black','marker_red','paste','pen_green','pen_multiclolor','pen_red','pencil_core','ruler_black']
pred = model.predict(img_nad, batch_size=1, verbose=0)
print(label[np.argmax(pred)])
for i in range(pred.shape[1]):
  print("%s: %.2f" % (label[i], pred[0][i]))

#raspistill -k -p 0,0,224,244 -w 224 -h 224 -rot 90
