#https://qiita.com/tom_eng_ltd/items/e1ce2adebc40db8f9176
#%%
from keras.models import load_model
#from keras.layers import *
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras.optimizers import RMSprop

#model = Sequential()
#model.add(Convolution2D(32, 3, 3, input_shape=(150, 150, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Convolution2D(32, 3, 3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Convolution2D(64, 3, 3))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Flatten())
#model.add(Dense(64))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))

jpg_name = 'IMG_1082'
model_file_name='smallcnn'

#model.load_weights('./' + model_file_name+'.h5')
model = load_model("chai-karin.hdf5", compile = False)
img_path = ('./' + jpg_name + '.jpg')
img = img_to_array(load_img(img_path, target_size=(150,150)))
img_nad = img_to_array(img)/255
img_nad = img_nad[None, ...]

label=['chai','karin']
pred = model.predict(img_nad, batch_size=1, verbose=0)
score = np.max(pred)
pred_label = label[np.argmax(pred[0])]
#print('name:',pred_label)
print('score:',score)
if score > 0.5:
    print('karin')
else:
    print('chai')

#model.save("chai-karin.hdf5")
