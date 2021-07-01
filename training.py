import os
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras.models import Model
from skimage import io
from skimage.transform import resize


def display_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()


x_path = "C:/Users/eduar/Desktop/Anul 3/Licenta/Image_Morph36/hair_seg/x/"
y_path = "C:/Users/eduar/Desktop/Anul 3/Licenta/Image_Morph36/hair_seg/y/"
x_paths = os.listdir(x_path)
y_paths = os.listdir(y_path)
print("Count X:", len(x_paths))
print("Count Y:", len(y_paths))

print(x_paths[0], y_paths[0])

images, masks = [], []
size = min(len(x_paths), len(y_paths))
for i in range(size):
    file = x_paths[i].replace('-org.jpg', '')
    if 'a' == 'a':
        img_path = file + '-org.jpg'
        mask_path = file + '-gt.pbm'
        if img_path in x_paths and mask_path in y_paths:
            images.append(io.imread(x_path + img_path, plugin='matplotlib', as_gray=True))
            masks.append(io.imread(y_path + mask_path, plugin='matplotlib', as_gray=True))
print("Actual data size:", len(images), len(masks))

np_images = np.zeros((size, 224, 224, 1))
np_masks = np.zeros((size, 224, 224, 1))

for i in range(size):
    img = images[i]
    msk = masks[i]
    np_images[i] = resize(img, (224, 224)).reshape((224, 224, 1))
    np_masks[i] = resize(msk, (224, 224)).reshape((224, 224, 1))

inputs = Input((224, 224, 1))
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
poo5a = MaxPooling2D(pool_size=(2, 2))(conv5)
conv5a = Conv2D(1024, (3, 3), activation='relu', padding='same')(poo5a)
conv5a = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5a)
up6a = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5a), conv5], axis=3)
conv6a = Conv2D(512, (3, 3), activation='relu', padding='same')(up6a)
conv6a = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6a)
up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
conv10 = Conv2D(1, (3, 3), activation='relu', padding='same')(conv9)

model = Model(inputs=[inputs], outputs=[conv10])
model.compile(optimizer='adam', loss='mse', metrics=['acc'])

model.summary()

epochs = 50
history = model.fit(np_images, np_masks, validation_split=0.05, epochs=epochs, batch_size=64,)
model.save('model.h5')
