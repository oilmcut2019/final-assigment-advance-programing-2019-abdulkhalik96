
# Abdul Khalik Akbar (Adudu|偉大）- M07158030

# Part 1 - Building the CNN


```python
# Importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
```


```python
# Initialising the CNN
classifier = Sequential()
```

#### Step 1 - Convolution


```python
classifier.add(Conv2D(32, (3, 3), activation = "relu", input_shape = (64, 64, 3)))
```

#### Step 2 - Pooling


```python
classifier.add(MaxPooling2D(pool_size = (2, 2)))
```

#### Step 3 - Flattening


```python
classifier.add(Flatten())
```

#### Step 4 - Full connection


```python
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```

# Part 2 - Fitting the CNN to the images (Part 1)


```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 700,
                         epochs = 25,
                         validation_data = test_set,
                         nb_val_samples = 700)
```

    Found 3481 images belonging to 2 classes.
    Found 615 images belonging to 2 classes.
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:19: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:19: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., epochs=25, validation_data=<keras_pre..., steps_per_epoch=21, validation_steps=700)`
    

    Epoch 1/25
    21/21 [==============================] - 88s 4s/step - loss: 1.1496 - acc: 0.4821 - val_loss: 0.7184 - val_acc: 0.4293
    Epoch 2/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6726 - acc: 0.5923 - val_loss: 0.7452 - val_acc: 0.4114
    Epoch 3/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6630 - acc: 0.5878 - val_loss: 0.7253 - val_acc: 0.4260
    Epoch 4/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6642 - acc: 0.5938 - val_loss: 0.7629 - val_acc: 0.4146
    Epoch 5/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6574 - acc: 0.6086 - val_loss: 0.7052 - val_acc: 0.4878
    Epoch 6/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6525 - acc: 0.6027 - val_loss: 0.7119 - val_acc: 0.5089
    Epoch 7/25
    21/21 [==============================] - 88s 4s/step - loss: 0.6049 - acc: 0.6786 - val_loss: 0.6566 - val_acc: 0.6293
    Epoch 8/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6427 - acc: 0.6250 - val_loss: 0.6407 - val_acc: 0.6585
    Epoch 9/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6277 - acc: 0.6399 - val_loss: 0.8978 - val_acc: 0.4520
    Epoch 10/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6524 - acc: 0.6220 - val_loss: 0.7474 - val_acc: 0.4992
    Epoch 11/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6117 - acc: 0.6593 - val_loss: 0.7753 - val_acc: 0.4959
    Epoch 12/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6092 - acc: 0.7024 - val_loss: 0.8356 - val_acc: 0.4715
    Epoch 13/25
    21/21 [==============================] - 85s 4s/step - loss: 0.5993 - acc: 0.6652 - val_loss: 0.6830 - val_acc: 0.5984
    Epoch 14/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6045 - acc: 0.6726 - val_loss: 0.7427 - val_acc: 0.5480
    Epoch 15/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6000 - acc: 0.6786 - val_loss: 0.6623 - val_acc: 0.6293
    Epoch 16/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6125 - acc: 0.6494 - val_loss: 0.6812 - val_acc: 0.6000
    Epoch 17/25
    21/21 [==============================] - 85s 4s/step - loss: 0.5864 - acc: 0.6801 - val_loss: 0.6733 - val_acc: 0.6228
    Epoch 18/25
    21/21 [==============================] - 85s 4s/step - loss: 0.5594 - acc: 0.7113 - val_loss: 0.7517 - val_acc: 0.5480
    Epoch 19/25
    21/21 [==============================] - 85s 4s/step - loss: 0.5967 - acc: 0.6562 - val_loss: 0.5800 - val_acc: 0.7366
    Epoch 20/25
    21/21 [==============================] - 84s 4s/step - loss: 0.5639 - acc: 0.7321 - val_loss: 0.5841 - val_acc: 0.7041
    Epoch 21/25
    21/21 [==============================] - 83s 4s/step - loss: 0.5893 - acc: 0.6801 - val_loss: 0.6743 - val_acc: 0.6211
    Epoch 22/25
    21/21 [==============================] - 83s 4s/step - loss: 0.5737 - acc: 0.6801 - val_loss: 0.7056 - val_acc: 0.6016
    Epoch 23/25
    21/21 [==============================] - 84s 4s/step - loss: 0.5691 - acc: 0.7068 - val_loss: 0.6909 - val_acc: 0.6228
    Epoch 24/25
    21/21 [==============================] - 84s 4s/step - loss: 0.5551 - acc: 0.7173 - val_loss: 0.6171 - val_acc: 0.6976
    Epoch 25/25
    21/21 [==============================] - 83s 4s/step - loss: 0.5732 - acc: 0.6905 - val_loss: 0.6622 - val_acc: 0.6537
    




    <keras.callbacks.History at 0x25b8fbdcf60>



Capture1.PNG

![Capture2.PNG](attachment:Capture2.PNG)

# Part 2 - Fitting the CNN to the images (Part 2)


```python
# Initialising the CNN
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), activation = "relu", input_shape = (64, 64, 3)))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())

classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 700,
                         epochs = 25,
                         validation_data = test_set,
                         nb_val_samples = 700)
```

    Found 3481 images belonging to 2 classes.
    Found 615 images belonging to 2 classes.
    

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:31: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:31: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., epochs=25, validation_data=<keras_pre..., steps_per_epoch=21, validation_steps=700)`
    

    Epoch 1/25
    21/21 [==============================] - 87s 4s/step - loss: 1.6770 - acc: 0.5045 - val_loss: 0.7312 - val_acc: 0.4098
    Epoch 2/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6774 - acc: 0.5729 - val_loss: 0.7098 - val_acc: 0.4163
    Epoch 3/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6845 - acc: 0.5685 - val_loss: 0.7467 - val_acc: 0.4114
    Epoch 4/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6676 - acc: 0.5997 - val_loss: 0.7395 - val_acc: 0.4211
    Epoch 5/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6683 - acc: 0.5908 - val_loss: 0.7572 - val_acc: 0.4114
    Epoch 6/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6709 - acc: 0.6000 - val_loss: 0.8594 - val_acc: 0.4098
    Epoch 7/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6674 - acc: 0.6220 - val_loss: 0.8431 - val_acc: 0.4098
    Epoch 8/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6617 - acc: 0.6071 - val_loss: 0.7360 - val_acc: 0.4683
    Epoch 9/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6390 - acc: 0.6205 - val_loss: 0.7828 - val_acc: 0.4488
    Epoch 10/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6382 - acc: 0.6354 - val_loss: 0.8886 - val_acc: 0.4276
    Epoch 11/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6195 - acc: 0.6409 - val_loss: 0.8736 - val_acc: 0.4374
    Epoch 12/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6321 - acc: 0.6637 - val_loss: 0.8171 - val_acc: 0.4667
    Epoch 13/25
    21/21 [==============================] - 86s 4s/step - loss: 0.6033 - acc: 0.6726 - val_loss: 0.6291 - val_acc: 0.6618
    Epoch 14/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6139 - acc: 0.6324 - val_loss: 0.6531 - val_acc: 0.6309
    Epoch 15/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6128 - acc: 0.6622 - val_loss: 0.7843 - val_acc: 0.5203
    Epoch 16/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6075 - acc: 0.6597 - val_loss: 0.6593 - val_acc: 0.6146
    Epoch 17/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6392 - acc: 0.6235 - val_loss: 0.8440 - val_acc: 0.4748
    Epoch 18/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6137 - acc: 0.6741 - val_loss: 0.8034 - val_acc: 0.4943
    Epoch 19/25
    21/21 [==============================] - 85s 4s/step - loss: 0.6029 - acc: 0.6622 - val_loss: 0.6509 - val_acc: 0.6390
    Epoch 20/25
    21/21 [==============================] - 86s 4s/step - loss: 0.5815 - acc: 0.7247 - val_loss: 0.8691 - val_acc: 0.5171
    Epoch 21/25
    21/21 [==============================] - 85s 4s/step - loss: 0.5525 - acc: 0.7148 - val_loss: 0.7206 - val_acc: 0.6000
    Epoch 22/25
    21/21 [==============================] - 85s 4s/step - loss: 0.5947 - acc: 0.6845 - val_loss: 0.6741 - val_acc: 0.6293
    Epoch 23/25
    21/21 [==============================] - 85s 4s/step - loss: 0.5575 - acc: 0.7143 - val_loss: 0.7577 - val_acc: 0.5854
    Epoch 24/25
    21/21 [==============================] - 85s 4s/step - loss: 0.5626 - acc: 0.7158 - val_loss: 0.8192 - val_acc: 0.5447
    Epoch 25/25
    21/21 [==============================] - 86s 4s/step - loss: 0.5707 - acc: 0.6994 - val_loss: 0.6517 - val_acc: 0.6537
    




    <keras.callbacks.History at 0x25b8f05c668>



![Capture3.PNG](attachment:Capture3.PNG)

![Capture4.PNG](attachment:Capture4.PNG)

# Part 2 - Fitting the CNN to the images (Part 3)


```python
# Initialising the CNN
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), activation = "relu", input_shape = (64, 64, 3)))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())

classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 700,
                         epochs = 20,
                         validation_data = test_set,
                         validation_steps = 700)
```

    Found 3481 images belonging to 2 classes.
    Found 615 images belonging to 2 classes.
    Epoch 1/20
    700/700 [==============================] - 275s 393ms/step - loss: 0.6056 - acc: 0.6867 - val_loss: 0.9880 - val_acc: 0.5610
    Epoch 2/20
    700/700 [==============================] - 274s 391ms/step - loss: 0.4646 - acc: 0.7766 - val_loss: 0.7802 - val_acc: 0.6634
    Epoch 3/20
    700/700 [==============================] - 272s 389ms/step - loss: 0.3747 - acc: 0.8341 - val_loss: 0.7030 - val_acc: 0.7285
    Epoch 4/20
    700/700 [==============================] - 273s 390ms/step - loss: 0.2962 - acc: 0.8723 - val_loss: 0.9691 - val_acc: 0.7024
    Epoch 5/20
    700/700 [==============================] - 274s 392ms/step - loss: 0.2127 - acc: 0.9151 - val_loss: 1.0526 - val_acc: 0.6943
    Epoch 6/20
    700/700 [==============================] - 273s 390ms/step - loss: 0.1571 - acc: 0.9411 - val_loss: 1.3520 - val_acc: 0.6764
    Epoch 7/20
    700/700 [==============================] - 273s 390ms/step - loss: 0.1198 - acc: 0.9555 - val_loss: 1.3237 - val_acc: 0.6764
    Epoch 8/20
    700/700 [==============================] - 274s 391ms/step - loss: 0.0887 - acc: 0.9695 - val_loss: 1.6711 - val_acc: 0.6602
    Epoch 9/20
    700/700 [==============================] - 274s 391ms/step - loss: 0.0735 - acc: 0.9735 - val_loss: 1.8746 - val_acc: 0.6748
    Epoch 10/20
    700/700 [==============================] - 273s 391ms/step - loss: 0.0672 - acc: 0.9761 - val_loss: 2.0196 - val_acc: 0.6520
    Epoch 11/20
    700/700 [==============================] - 274s 392ms/step - loss: 0.0546 - acc: 0.9820 - val_loss: 1.5202 - val_acc: 0.7106
    Epoch 12/20
    700/700 [==============================] - 274s 391ms/step - loss: 0.0485 - acc: 0.9846 - val_loss: 1.7298 - val_acc: 0.6992
    Epoch 13/20
    700/700 [==============================] - 274s 391ms/step - loss: 0.0447 - acc: 0.9847 - val_loss: 2.1999 - val_acc: 0.6780
    Epoch 14/20
    700/700 [==============================] - 274s 392ms/step - loss: 0.0423 - acc: 0.9857 - val_loss: 2.1708 - val_acc: 0.6878
    Epoch 15/20
    700/700 [==============================] - 274s 392ms/step - loss: 0.0352 - acc: 0.9885 - val_loss: 1.8817 - val_acc: 0.7041
    Epoch 16/20
    700/700 [==============================] - 300s 428ms/step - loss: 0.0330 - acc: 0.9892 - val_loss: 2.2104 - val_acc: 0.6715
    Epoch 17/20
    700/700 [==============================] - 292s 418ms/step - loss: 0.0302 - acc: 0.9902 - val_loss: 1.8666 - val_acc: 0.7041
    Epoch 18/20
    700/700 [==============================] - 322s 460ms/step - loss: 0.0279 - acc: 0.9892 - val_loss: 1.9797 - val_acc: 0.6878
    Epoch 19/20
    700/700 [==============================] - 275s 392ms/step - loss: 0.0277 - acc: 0.9905 - val_loss: 2.1055 - val_acc: 0.7106
    Epoch 20/20
    700/700 [==============================] - 275s 393ms/step - loss: 0.0264 - acc: 0.9924 - val_loss: 2.2820 - val_acc: 0.6780
    




    <keras.callbacks.History at 0x25b96511fd0>



![Capture5.PNG](attachment:Capture5.PNG)

![Capture6.PNG](attachment:Capture6.PNG)

# Conclusion

# 1. From the 1st  testing we can see that the increasing value of sample_per_epoch didn't affect the value of accuracy but affect the time
# 2. From the 2nd testing we can see that the increasing value of hidden layer will affect the value of the accuracy but not affect the time
# 3. From the 3rd testing we can see that if we change the parameter from sample_per_epoch to be steps_per_epoch the accuracy wil increase and the time that we spend also increase
