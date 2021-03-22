import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 


mnist= tf.keras.datasets.mnist
num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

input_shape =(28,28,1)

y_train = tf.keras.utils.to_categorical(y_train,num_classes)
y_test = tf.keras.utils.to_categorical(y_test,num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

batch_size =128
epochs = 4
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32,kernel_size = (2,2),activation = 'relu',input_shape= input_shape))
model.add(tf.keras.layers.Conv2D(64,(2,2),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(num_classes,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,batch_size= batch_size,epochs= epochs,verbose=1,validation_data= (x_test,y_test))          

model.save('numread.h5')
val_loss, val_accuracy= model.evaluate(x_test,y_test,verbose=0)

print (val_loss,val_accuracy)

predictions = model.predict(x_test)

print(np.argmax(predictions[1]))

plt.imshow(x_test[1])
plt.show()