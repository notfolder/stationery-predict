# %%
# データ読み込み
import keras
from IPython.display import SVG
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.layers import *
from keras.preprocessing.image import *
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    'data/validation',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

# %%
# モデルを構築
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#%%
SVG(model_to_dot(model).create(prog='dot', format='svg'))

#%%
# 訓練
tb_cb = keras.callbacks.TensorBoard(log_dir="./tflog/", histogram_freq=1)
cbks = [tb_cb]
#history = model.fit_generator(
#  train_generator,
#  samples_per_epoch=2000,
#  nb_epoch=20,
#  callbacks=cbks,
#  validation_data=validation_generator,
#  nb_val_samples=800)
history = model.fit_generator(
  train_generator,
  samples_per_epoch=2000,
  nb_epoch=20)

# 結果を保存
model.save_weights('smallcnn.h5')
#model.save_history(history, 'history_smallcnn.txt')
import pickle
f = open('history.bin','wb')
pickle.dump(history,f)
f.close

#%%
import matplotlib.pyplot as plt
from matplotlib import cm
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%%
jpg_name = 'IMG_1073'
img_path = ('./' + jpg_name + '.jpg')
img = img_to_array(load_img(img_path, target_size=(150,150)))
img_nad = img_to_array(img)/255
img_nad = img_nad[None, ...]

label=['karin','chai']
pred = model.predict(img_nad, batch_size=1, verbose=0)
score = np.max(pred)
pred_label = label[np.argmax(pred[0])]
#print('name:',pred_label)
print('score:',score)
if score > 0.5:
    print('karin')
else:
    print('chai')

#%%
model.save_weights('smallcnn-2.h5')
