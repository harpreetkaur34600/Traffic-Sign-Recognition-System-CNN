import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
import cv2
import sklearn
from sklearn.model_selection import train_test_split

train_df=pd.read_csv("Train.csv")
# print(train_df.head())
# print(train_df.columns)

test_df=pd.read_csv("Test.csv")
# print(test_df.head())
# print(test_df.columns)

meta_df=pd.read_csv("Meta.csv") # meta is just for refence
# print(meta_df.head(10))
# print(meta_df.columns)

# print(train_df["ClassId"].nunique())

# iloc will locate the row number 
# then we wrote the columnn name to get that particular cell 
# that gives us our image path
# cv2 is a library
# cv2 reads image in blue then green then red
# but matplotlib want in red then green then blue
# so we used convert color by cv2

# display an image
# img_path1=train_df.iloc[0]["Path"]
# print(img_path1)
# img1=cv2.imread(img_path1)
# print(img1.shape)
# img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
# print("class id is : ",train_df.iloc[0]["ClassId"])
# plt.imshow(img1)
# plt.show()

# loading full training dataset and storing it
print("started loading dataset")
images=[]
labels=[]
for i in range(len(train_df)):
    img_path=train_df.iloc[i]["Path"]
    img_label=train_df.iloc[i]["ClassId"]
    img=cv2.imread(img_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(32,32))

    images.append(img)
    labels.append(img_label)

# print(images[345].shape)

# putting data in numpyarray as tensorflow wants that format
images=np.array(images)
labels=np.array(labels)

# print(images.shape) # 39209 images, each 32*32 pixels , 3 channels, 4d 
# print(labels.shape) # 1 d , 39209 labels

#normalize as nn gets it better when blw 0 and 1
images=images/255.0

# splitting the training images data into train and validate/test
# stratify will split data evenly acc to lables blw train and validation data
x_train,x_val,y_train,y_val=train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels)

# print(x_train.shape)
# print(x_val.shape)
# print(y_train.shape)
# print(y_val.shape)

# data augmentation - helps learn model even if img is rotated, zoomed, diff contrast, diff lightning, tilted
data_augmentation=keras.Sequential([
    keras.layers.RandomRotation(0.06), # random rotation means about 6 % rotate in any direction
    keras.layers.RandomZoom(0.08), # random zoom in or out for 8 % 
    keras.layers.RandomTranslation(0.06,0.06), # moves image up, down, left , right
    keras.layers.RandomContrast(0.125) # 12.5% contrast and brightness
])
# making the neural network
model=keras.Sequential([
    keras.layers.Input(shape=(32,32,3)),
    data_augmentation,
    keras.layers.Conv2D(32,(3,3),activation="relu"),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64,(3,3),activation="relu"),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64,activation="relu"),
    keras.layers.Dropout(0.3), # 30% neurons become unactive which reduces relying only on few neurons. forces each one to learn thus reducing overfitting
    keras.layers.Dense(43,activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# print(model.summary())

# training
# we add validation data after every epoch
# validation data is unseen
# it helps check whether there is overfitting or not
# testing data is different , it tells actual performance
print("training start")
history=model.fit(x_train,y_train,epochs=15,validation_data=(x_val,y_val))

print(history.history) # we get accurray and loss for training data and validation data
# the validation accuracy must increse after every epoch, otherwise overfiiting

# see a graph to check for overfitting
# plt.plot(history.history["accuracy"],label="Training accuracy")
# plt.plot(history.history["val_accuracy"],label="validation accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.title("Accuracy graph for training and validation")
# plt.legend()
# plt.show()

# plt.plot(history.history["loss"],label="Trianing Loss")
# plt.plot(history.history["val_loss"],label="validation loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Loss graph for training and validation")
# plt.legend()
# plt.show()

# preprocessing testing data
# loadiing testing data
testing_images=[]
testing_labels=[]
for i in range(len(test_df)):
    test_img_path=test_df.iloc[i]["Path"]
    test_img_label=test_df.iloc[i]["ClassId"]
    test_img=cv2.imread(test_img_path)
    test_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2RGB)
    test_img=cv2.resize(test_img,(32,32))
    testing_images.append(test_img)
    testing_labels.append(test_img_label)
# print(testing_images[123].shape)
# plt.imshow(testing_images[123])
# plt.show()

# converting lists to numpy array 
testing_images=np.array(testing_images)
testing_labels=np.array(testing_labels)
# print(testing_images.shape) 
# print(testing_labels.shape)

# normalize
testing_images=testing_images/255.0

loss,accuracy=model.evaluate(testing_images,testing_labels)
# print(accuracy)

model.save("traffic_sign_recognize_model.keras")