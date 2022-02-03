#CNN

# Importing the libraries
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

# Part 1 - Data Preprocessing

dataset = pd.read_csv('train.csv')
dataset = dataset/255.0

X_test = pd.read_csv('test.csv')
X_test = X_test /255.0

y_train = dataset['label']
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

image_id = list(X_test.index)
image_id = [i+1 for i in image_id]

X_train = dataset.drop('label',axis=1)


#reshaping the training data and testing data into grids
print(X_train.shape)
X_train=X_train.values.reshape(-1,28,28,1)
X_test=X_test.values.reshape(-1,28,28,1)


# Part 2 - Building the CNN
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32, kernel_size = 3, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Conv2D(64, kernel_size = 3, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, kernel_size = 3, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Conv2D(128, kernel_size = 4, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compiling the CNN
model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
model.fit(x = X_train, y = y_train, epochs = 15)

results = model.predict(X_test)
results = np.argmax(results, axis=1)

data = {"ImageId": image_id, "Label":results}
results = pd.DataFrame(data)
results.to_csv("result_data_final.csv",index=False)