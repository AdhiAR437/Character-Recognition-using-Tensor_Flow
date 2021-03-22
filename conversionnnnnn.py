from PIL import Image
import numpy as np
#import cv2
#import os
import tensorflow as tf

file2 = open("E:\\test.txt")
jjjj= file2.read()


img= Image.open(jjjj,"r").convert("L")
img = img.resize((28,28))
img= np.array(img)
img = img.reshape(1,28,28,1)
img =img/255.0

we= tf.keras.models.load_model('numread.h5') 
predictions = we.predict([img]) [0]
file1= open(r"E:\test.txt","w+")
j=np.argmax(predictions)
jj= j.item()
jjj=str(jj)
print (type(jj))
file1.write(jjj)
file1.close()
print(np.argmax(predictions))
file3=open(r"E:\test.txt","r+")
j6= file3.read()
print(j6)



