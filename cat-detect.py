#%%
import keras
#from IPython.display import SVG
from keras.utils import *
from keras.layers import *
from keras.preprocessing.image import *
from keras.datasets import mnist
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.applications.vgg16 import *

N_CATEGORIES  = 3
IMAGE_SIZE = 224
BATCH_SIZE = 16

NUM_TRAINING = 1600
NUM_VALIDATION = 400

# VGG16モデルと学習済み重みをロード
# Fully-connected層（FC）はいらないのでinclude_top=False）
# input_tensorを指定しておかないとoutput_shapeがNoneになってエラーになるので注意
# https://keras.io/applications/#inceptionv3
input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
base_model = VGG16(weights='imagenet', include_top=False,input_tensor=input_tensor)
#vgg16_model.summary()

#%%
# FC層を構築
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(N_CATEGORIES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 最後のconv層の直前までの層をfreeze
for layer in base_model.layers[:15]:
   layer.trainable = False

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

#%%
train_datagen = ImageDataGenerator(
   rescale=1.0 / 255,
   shear_range=0.2,
   zoom_range=0.2,
   horizontal_flip=True,
   rotation_range=10)

test_datagen = ImageDataGenerator(
   rescale=1.0 / 255,
)

train_generator = train_datagen.flow_from_directory(
   'data2/train',
   target_size=(IMAGE_SIZE, IMAGE_SIZE),
   batch_size=BATCH_SIZE,
   class_mode='categorical',
   shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
   'data2/validation',
   target_size=(IMAGE_SIZE, IMAGE_SIZE),
   batch_size=BATCH_SIZE,
   class_mode='categorical',
   shuffle=True
)

hist = model.fit_generator(train_generator,
   steps_per_epoch=NUM_TRAINING//BATCH_SIZE,
   epochs=50,
   verbose=1,
   validation_data=validation_generator,
   validation_steps=NUM_VALIDATION//BATCH_SIZE,
   )

model.save('cats.hdf5')
