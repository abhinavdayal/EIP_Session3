# EIP_Session3

## Final Validation accuracy for Base Network (81.66)
Accuracy on test data is: 81.66 with max of 82.73 at 47th Epoch

## Model definition (As submitted before deadline)
```
# Define the model, Since image size is already small we wont use stride > 1.
# we have to use only separatble convolution
model = keras.Sequential()

# rin=1, nin = 32x32, cin= 3, jin=1, k=3, p=1, s=1, jout=1, rout=3, nout=32x32, cout=16
model.add(SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', depth_multiplier=1, input_shape=(32, 32, 3), use_bias=False))
model.add(Dropout(0.07))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=3, nin = 32x32, cin=16, jin=1, k=3, p=1, s=1, jout=1, rout=5, nout=32x32, cout=32
model.add(SeparableConv2D(filters=48, kernel_size=(3, 3), padding='same', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.07))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=5, nin = 32x32, cin=32, jin=1, k=3, p=1, s=1, jout=1, rout=7, nout=32x32, cout=48
model.add(SeparableConv2D(filters=96, kernel_size=(3, 3), padding='same', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.07))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=7, nin = 32x32, cin=48, jin=1, k=2, p=0, s=2, jout=2, rout=8, nout=16x16, cout=48
model.add(MaxPooling2D())

# rin=8, nin = 16x16, cin= 48, jin=2, k=3, p=1, s=1, jout=2, rout=12, nout=16x16, cout=64
model.add(SeparableConv2D(filters=128, kernel_size=(3, 3), padding='same', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.07))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=12, nin = 16x16, cin= 64, jin=2, k=3, p=1, s=1, jout=2, rout=16, nout=16x16, cout=96
model.add(SeparableConv2D(filters=160, kernel_size=(3, 3), padding='same', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.07))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=16, nin = 16x16, cin=96, jin=2, k=2, p=0, s=2, jout=4, rout=18, nout=8x8, cout=96
model.add(MaxPooling2D())

# rin=18, nin = 8x8, cin= 96, jin=4, k=3, p=0, s=1, jout=4, rout=26, nout=6x6, cout=192
model.add(SeparableConv2D(filters=192, kernel_size=(3, 3), padding='valid', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.07))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=26, nin = 6x6, cin= 192, jin=4, k=3, p=0, s=1, jout=4, rout=34, nout=4x4, cout=10 Cfar_10 has 10 classes
model.add(SeparableConv2D(filters=num_classes, kernel_size=(3, 3), padding='valid', depth_multiplier=1, use_bias=False))
# not adding any activation or batch normalization after the last convolution layer

# output size = 1x1x10
model.add(GlobalAveragePooling2D())
model.add(Flatten()) # 10x1
model.add(Activation('softmax')) # get probabilities

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

```
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
separable_conv2d_21 (Separab (None, 32, 32, 32)        123       
_________________________________________________________________
dropout_18 (Dropout)         (None, 32, 32, 32)        0         
_________________________________________________________________
batch_normalization_18 (Batc (None, 32, 32, 32)        128       
_________________________________________________________________
activation_21 (Activation)   (None, 32, 32, 32)        0         
_________________________________________________________________
separable_conv2d_22 (Separab (None, 32, 32, 48)        1824      
_________________________________________________________________
dropout_19 (Dropout)         (None, 32, 32, 48)        0         
_________________________________________________________________
batch_normalization_19 (Batc (None, 32, 32, 48)        192       
_________________________________________________________________
activation_22 (Activation)   (None, 32, 32, 48)        0         
_________________________________________________________________
separable_conv2d_23 (Separab (None, 32, 32, 96)        5040      
_________________________________________________________________
dropout_20 (Dropout)         (None, 32, 32, 96)        0         
_________________________________________________________________
batch_normalization_20 (Batc (None, 32, 32, 96)        384       
_________________________________________________________________
activation_23 (Activation)   (None, 32, 32, 96)        0         
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 16, 16, 96)        0         
_________________________________________________________________
separable_conv2d_24 (Separab (None, 16, 16, 128)       13152     
_________________________________________________________________
dropout_21 (Dropout)         (None, 16, 16, 128)       0         
_________________________________________________________________
batch_normalization_21 (Batc (None, 16, 16, 128)       512       
_________________________________________________________________
activation_24 (Activation)   (None, 16, 16, 128)       0         
_________________________________________________________________
separable_conv2d_25 (Separab (None, 16, 16, 160)       21632     
_________________________________________________________________
dropout_22 (Dropout)         (None, 16, 16, 160)       0         
_________________________________________________________________
batch_normalization_22 (Batc (None, 16, 16, 160)       640       
_________________________________________________________________
activation_25 (Activation)   (None, 16, 16, 160)       0         
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 8, 8, 160)         0         
_________________________________________________________________
separable_conv2d_26 (Separab (None, 6, 6, 192)         32160     
_________________________________________________________________
dropout_23 (Dropout)         (None, 6, 6, 192)         0         
_________________________________________________________________
batch_normalization_23 (Batc (None, 6, 6, 192)         768       
_________________________________________________________________
activation_26 (Activation)   (None, 6, 6, 192)         0         
_________________________________________________________________
separable_conv2d_27 (Separab (None, 4, 4, 10)          3648      
_________________________________________________________________
global_average_pooling2d_3 ( (None, 10)                0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 10)                0         
_________________________________________________________________
activation_27 (Activation)   (None, 10)                0         
=================================================================
Total params: 80,203
Trainable params: 78,891
Non-trainable params: 1,312
_________________________________________________________________
```

## 50 epocs logs (max validation accuracy 83.10 in 44th Epoch)
```
Epoch 00001: LearningRateScheduler reducing learning rate to 0.01.
Epoch 1/50
195/195 [==============================] - 76s 389ms/step - loss: 1.5255 - accuracy: 0.4421 - val_loss: 2.9379 - val_accuracy: 0.2239

Epoch 00002: LearningRateScheduler reducing learning rate to 0.0075815011.
Epoch 2/50
195/195 [==============================] - 75s 383ms/step - loss: 1.1478 - accuracy: 0.5918 - val_loss: 1.9680 - val_accuracy: 0.4778

Epoch 00003: LearningRateScheduler reducing learning rate to 0.0061050061.
Epoch 3/50
195/195 [==============================] - 75s 384ms/step - loss: 0.9695 - accuracy: 0.6581 - val_loss: 1.3568 - val_accuracy: 0.5931

Epoch 00004: LearningRateScheduler reducing learning rate to 0.005109862.
Epoch 4/50
195/195 [==============================] - 75s 383ms/step - loss: 0.8718 - accuracy: 0.6937 - val_loss: 0.9208 - val_accuracy: 0.6857

Epoch 00005: LearningRateScheduler reducing learning rate to 0.0043936731.
Epoch 5/50
195/195 [==============================] - 75s 384ms/step - loss: 0.8025 - accuracy: 0.7203 - val_loss: 0.8685 - val_accuracy: 0.6921

Epoch 00006: LearningRateScheduler reducing learning rate to 0.0038535645.
Epoch 6/50
195/195 [==============================] - 75s 384ms/step - loss: 0.7536 - accuracy: 0.7367 - val_loss: 0.8436 - val_accuracy: 0.7054

Epoch 00007: LearningRateScheduler reducing learning rate to 0.003431709.
Epoch 7/50
195/195 [==============================] - 75s 385ms/step - loss: 0.7189 - accuracy: 0.7487 - val_loss: 0.7476 - val_accuracy: 0.7425

Epoch 00008: LearningRateScheduler reducing learning rate to 0.0030931024.
Epoch 8/50
195/195 [==============================] - 76s 388ms/step - loss: 0.6786 - accuracy: 0.7618 - val_loss: 0.7792 - val_accuracy: 0.7410

Epoch 00009: LearningRateScheduler reducing learning rate to 0.0028153153.
Epoch 9/50
195/195 [==============================] - 75s 386ms/step - loss: 0.6654 - accuracy: 0.7686 - val_loss: 0.6945 - val_accuracy: 0.7583

Epoch 00010: LearningRateScheduler reducing learning rate to 0.0025833118.
Epoch 10/50
195/195 [==============================] - 75s 387ms/step - loss: 0.6447 - accuracy: 0.7754 - val_loss: 0.7244 - val_accuracy: 0.7585

Epoch 00011: LearningRateScheduler reducing learning rate to 0.0023866348.
Epoch 11/50
195/195 [==============================] - 76s 390ms/step - loss: 0.6193 - accuracy: 0.7841 - val_loss: 0.7231 - val_accuracy: 0.7489

Epoch 00012: LearningRateScheduler reducing learning rate to 0.0022177866.
Epoch 12/50
195/195 [==============================] - 76s 389ms/step - loss: 0.6119 - accuracy: 0.7857 - val_loss: 0.7904 - val_accuracy: 0.7480

Epoch 00013: LearningRateScheduler reducing learning rate to 0.002071251.
Epoch 13/50
195/195 [==============================] - 76s 390ms/step - loss: 0.5961 - accuracy: 0.7949 - val_loss: 0.6649 - val_accuracy: 0.7769

Epoch 00014: LearningRateScheduler reducing learning rate to 0.0019428793.
Epoch 14/50
195/195 [==============================] - 76s 391ms/step - loss: 0.5794 - accuracy: 0.7991 - val_loss: 0.6434 - val_accuracy: 0.7780

Epoch 00015: LearningRateScheduler reducing learning rate to 0.0018294914.
Epoch 15/50
195/195 [==============================] - 76s 390ms/step - loss: 0.5733 - accuracy: 0.8019 - val_loss: 0.6287 - val_accuracy: 0.7814

Epoch 00016: LearningRateScheduler reducing learning rate to 0.0017286085.
Epoch 16/50
195/195 [==============================] - 76s 391ms/step - loss: 0.5588 - accuracy: 0.8067 - val_loss: 0.6117 - val_accuracy: 0.7920

Epoch 00017: LearningRateScheduler reducing learning rate to 0.00163827.
Epoch 17/50
195/195 [==============================] - 76s 388ms/step - loss: 0.5520 - accuracy: 0.8097 - val_loss: 0.6346 - val_accuracy: 0.7846

Epoch 00018: LearningRateScheduler reducing learning rate to 0.0015569049.
Epoch 18/50
195/195 [==============================] - 76s 390ms/step - loss: 0.5386 - accuracy: 0.8123 - val_loss: 0.6343 - val_accuracy: 0.7846

Epoch 00019: LearningRateScheduler reducing learning rate to 0.0014832394.
Epoch 19/50
195/195 [==============================] - 76s 389ms/step - loss: 0.5410 - accuracy: 0.8141 - val_loss: 0.6076 - val_accuracy: 0.7948

Epoch 00020: LearningRateScheduler reducing learning rate to 0.00141623.
Epoch 20/50
195/195 [==============================] - 76s 388ms/step - loss: 0.5337 - accuracy: 0.8147 - val_loss: 0.6106 - val_accuracy: 0.7922

Epoch 00021: LearningRateScheduler reducing learning rate to 0.0013550136.
Epoch 21/50
195/195 [==============================] - 76s 389ms/step - loss: 0.5250 - accuracy: 0.8193 - val_loss: 0.6273 - val_accuracy: 0.7844

Epoch 00022: LearningRateScheduler reducing learning rate to 0.00129887.
Epoch 22/50
195/195 [==============================] - 76s 389ms/step - loss: 0.5231 - accuracy: 0.8176 - val_loss: 0.5889 - val_accuracy: 0.7977

Epoch 00023: LearningRateScheduler reducing learning rate to 0.0012471938.
Epoch 23/50
195/195 [==============================] - 76s 390ms/step - loss: 0.5105 - accuracy: 0.8223 - val_loss: 0.7135 - val_accuracy: 0.7676

Epoch 00024: LearningRateScheduler reducing learning rate to 0.0011994722.
Epoch 24/50
195/195 [==============================] - 76s 388ms/step - loss: 0.5032 - accuracy: 0.8248 - val_loss: 0.6190 - val_accuracy: 0.7955

Epoch 00025: LearningRateScheduler reducing learning rate to 0.001155268.
Epoch 25/50
195/195 [==============================] - 76s 390ms/step - loss: 0.5033 - accuracy: 0.8250 - val_loss: 0.5917 - val_accuracy: 0.7980

Epoch 00026: LearningRateScheduler reducing learning rate to 0.0011142061.
Epoch 26/50
195/195 [==============================] - 76s 390ms/step - loss: 0.4989 - accuracy: 0.8272 - val_loss: 0.5665 - val_accuracy: 0.8028

Epoch 00027: LearningRateScheduler reducing learning rate to 0.001075963.
Epoch 27/50
195/195 [==============================] - 76s 389ms/step - loss: 0.4950 - accuracy: 0.8275 - val_loss: 0.5510 - val_accuracy: 0.8120

Epoch 00028: LearningRateScheduler reducing learning rate to 0.001040258.
Epoch 28/50
195/195 [==============================] - 76s 388ms/step - loss: 0.4848 - accuracy: 0.8307 - val_loss: 0.5720 - val_accuracy: 0.8031

Epoch 00029: LearningRateScheduler reducing learning rate to 0.0010068466.
Epoch 29/50
195/195 [==============================] - 76s 389ms/step - loss: 0.4875 - accuracy: 0.8298 - val_loss: 0.5810 - val_accuracy: 0.8043

Epoch 00030: LearningRateScheduler reducing learning rate to 0.0009755146.
Epoch 30/50
195/195 [==============================] - 75s 386ms/step - loss: 0.4806 - accuracy: 0.8328 - val_loss: 0.5789 - val_accuracy: 0.8096

Epoch 00031: LearningRateScheduler reducing learning rate to 0.0009460738.
Epoch 31/50
195/195 [==============================] - 76s 389ms/step - loss: 0.4791 - accuracy: 0.8334 - val_loss: 0.5612 - val_accuracy: 0.8099

Epoch 00032: LearningRateScheduler reducing learning rate to 0.000918358.
Epoch 32/50
195/195 [==============================] - 76s 389ms/step - loss: 0.4702 - accuracy: 0.8368 - val_loss: 0.5648 - val_accuracy: 0.8121

Epoch 00033: LearningRateScheduler reducing learning rate to 0.0008922198.
Epoch 33/50
195/195 [==============================] - 76s 388ms/step - loss: 0.4736 - accuracy: 0.8346 - val_loss: 0.5709 - val_accuracy: 0.8062

Epoch 00034: LearningRateScheduler reducing learning rate to 0.0008675284.
Epoch 34/50
195/195 [==============================] - 76s 388ms/step - loss: 0.4741 - accuracy: 0.8359 - val_loss: 0.5726 - val_accuracy: 0.8129

Epoch 00035: LearningRateScheduler reducing learning rate to 0.0008441668.
Epoch 35/50
195/195 [==============================] - 75s 384ms/step - loss: 0.4680 - accuracy: 0.8378 - val_loss: 0.5232 - val_accuracy: 0.8196

Epoch 00036: LearningRateScheduler reducing learning rate to 0.0008220304.
Epoch 36/50
195/195 [==============================] - 75s 384ms/step - loss: 0.4644 - accuracy: 0.8399 - val_loss: 0.5521 - val_accuracy: 0.8131

Epoch 00037: LearningRateScheduler reducing learning rate to 0.0008010253.
Epoch 37/50
195/195 [==============================] - 76s 388ms/step - loss: 0.4560 - accuracy: 0.8393 - val_loss: 0.5465 - val_accuracy: 0.8132

Epoch 00038: LearningRateScheduler reducing learning rate to 0.0007810669.
Epoch 38/50
195/195 [==============================] - 75s 385ms/step - loss: 0.4570 - accuracy: 0.8405 - val_loss: 0.5522 - val_accuracy: 0.8134

Epoch 00039: LearningRateScheduler reducing learning rate to 0.000762079.
Epoch 39/50
195/195 [==============================] - 75s 383ms/step - loss: 0.4504 - accuracy: 0.8435 - val_loss: 0.5390 - val_accuracy: 0.8200

Epoch 00040: LearningRateScheduler reducing learning rate to 0.0007439923.
Epoch 40/50
195/195 [==============================] - 75s 385ms/step - loss: 0.4502 - accuracy: 0.8447 - val_loss: 0.5525 - val_accuracy: 0.8157

Epoch 00041: LearningRateScheduler reducing learning rate to 0.0007267442.
Epoch 41/50
195/195 [==============================] - 75s 385ms/step - loss: 0.4538 - accuracy: 0.8414 - val_loss: 0.5408 - val_accuracy: 0.8120

Epoch 00042: LearningRateScheduler reducing learning rate to 0.0007102777.
Epoch 42/50
195/195 [==============================] - 75s 386ms/step - loss: 0.4513 - accuracy: 0.8410 - val_loss: 0.5277 - val_accuracy: 0.8214

Epoch 00043: LearningRateScheduler reducing learning rate to 0.0006945409.
Epoch 43/50
195/195 [==============================] - 75s 384ms/step - loss: 0.4468 - accuracy: 0.8443 - val_loss: 0.5369 - val_accuracy: 0.8129

Epoch 00044: LearningRateScheduler reducing learning rate to 0.0006794863.
Epoch 44/50
195/195 [==============================] - 75s 385ms/step - loss: 0.4432 - accuracy: 0.8475 - val_loss: 0.5127 - val_accuracy: 0.8310

Epoch 00045: LearningRateScheduler reducing learning rate to 0.0006650705.
Epoch 45/50
195/195 [==============================] - 75s 387ms/step - loss: 0.4413 - accuracy: 0.8453 - val_loss: 0.5682 - val_accuracy: 0.8117

Epoch 00046: LearningRateScheduler reducing learning rate to 0.0006512537.
Epoch 46/50
195/195 [==============================] - 75s 384ms/step - loss: 0.4378 - accuracy: 0.8479 - val_loss: 0.5225 - val_accuracy: 0.8202

Epoch 00047: LearningRateScheduler reducing learning rate to 0.0006379992.
Epoch 47/50
195/195 [==============================] - 75s 386ms/step - loss: 0.4399 - accuracy: 0.8479 - val_loss: 0.5262 - val_accuracy: 0.8241

Epoch 00048: LearningRateScheduler reducing learning rate to 0.0006252736.
Epoch 48/50
195/195 [==============================] - 75s 385ms/step - loss: 0.4381 - accuracy: 0.8462 - val_loss: 0.5391 - val_accuracy: 0.8230

Epoch 00049: LearningRateScheduler reducing learning rate to 0.0006130456.
Epoch 49/50
195/195 [==============================] - 75s 385ms/step - loss: 0.4278 - accuracy: 0.8512 - val_loss: 0.5519 - val_accuracy: 0.8176

Epoch 00050: LearningRateScheduler reducing learning rate to 0.0006012868.
Epoch 50/50
195/195 [==============================] - 75s 385ms/step - loss: 0.4385 - accuracy: 0.8476 - val_loss: 0.5463 - val_accuracy: 0.8225
Model took 3773.89 seconds to train
```

# Strategy
As required I replaced all conv with depth separable ones. I started with a simple model with 30K parameters ad the accuracy did not exceed 74%. I tried with Image augmentation, Image normalization, Batch normalization and dropouts and even tweaking the learning rate. Then I added more channels increasing number of parameters to 80K. Observed that the training and validation accuracy are steading increasing. So started with a harsher learning rate scheduling it to reduce gradually. That did the trick. I had to use different test accuracy function as I am now normalizing the image so had to normalize the test data with training mean and variance. I also switched to tensorflow 2.0.

With this success I tried augmenting the data to give as input more images as in the input data , currently double the quantity using augmentation. Also lowered the decay of learning rate as in earlier experiment the accuracy was still improving till the end. But all this made the training time to increase. It took more than one hour to train.


# Successful Experiemnts

## Success01 - LR Sched, IN, BN, Dropouts, Augmentation 63K Params
That is what I submitted. In fact that is the best I got with Image NOrmalization in place. Have to understand why accuracy reduces with image normalization. Is it because we apply train data's mean/variance on test data.?

## Success02 - LR Sched, IN, BN, Dropouts, Augmentation to give double data; 63K Params
In Success02.ipynb I tried to send double the number of input by augmentation. It did help a bit in the sense that the model trained quicker in less epochs. But did not improve accuracy too much. Didn't get time but can experient with ReduceLROnPlateau i.e. start with a good enouh learning rate and rather than updaing learning rate every epoch reuce it by a facttor when the metric say val_loss stops improving. Did try without augmentation etc. but did not get past 79% in the few attempts. Stayed away from dense layers completely.

## Success03 - BN, Dropouts. No Augmentation, IN, or LR Scheduler; 63K Params
I moved away from Image Normalization. Realized that we need more layers so removed one max pooling and added more layers. Got **83.5%** or more accuracy without any augmentation and only dropouts and no learning rate scheduler

## Success04 - LR Scheduler, BN, Dropouts. No Augmentation, IN; 63K Params
Added LR Scheduler and got **84.5% accuracy**

## Success05 - LR Scheduler, BN, Dropouts, Augmentation. No IN; 63K Params
Reduced dropouts and added augmentation **84.42% Accuracy*

## Success06 - LR Scheduler, BN, Dropouts. No Augmentation, IN; 76K Params
**84.7% accuracy**

## Success 07 - LRS, BN, DO, AUG. No IN: 76K parameters
**85.58%** accuracy. Highest thus far

## Success 08 - 63K parameters all LRS, DO, IN, BN
**81.2%** Accuracy


# Best 85.6% accuracy model (Done later after deadline. Sorry didnt get time before!)

```
# Define the model, Since image size is already small we wont use stride > 1.
# we have to use only separatble convolution
model = keras.Sequential()

# rin=1, nin = 32x32, cin= 3, jin=1, k=3, p=1, s=1, jout=1, rout=3, nout=32x32, cout=16
model.add(SeparableConv2D(filters=16, kernel_size=(3, 3), padding='same', depth_multiplier=1, input_shape=(32, 32, 3), use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=3, nin = 32x32, cin=16, jin=1, k=3, p=1, s=1, jout=1, rout=5, nout=32x32, cout=16
model.add(SeparableConv2D(filters=16, kernel_size=(3, 3), padding='same', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=5, nin = 32x32, cin=32, jin=1, k=3, p=1, s=1, jout=1, rout=7, nout=32x32, cout=32
model.add(SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=5, nin = 32x32, cin=32, jin=1, k=3, p=1, s=1, jout=1, rout=9, nout=32x32, cout=48
model.add(SeparableConv2D(filters=48, kernel_size=(3, 3), padding='same', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))


# rin=8, nin = 32x32, cin= 48, jin=1, k=3, p=1, s=1, jout=1, rout=11, nout=32x32, cout=64
model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=8, nin = 32x32, cin= 64, jin=1, k=3, p=1, s=1, jout=1, rout=13, nout=16x16, cout=128
model.add(SeparableConv2D(filters=128, kernel_size=(3, 3), padding='same', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=16, nin = 32x32, cin=128, jin=1, k=2, p=1, s=2, jout=2, rout=14, nout=8x8, cout=128
model.add(MaxPooling2D())

# rin=12, nin = 16x16, cin= 128, jin=2, k=3, p=1, s=1, jout=2, rout=18, nout=16x16, cout=64
model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=12, nin = 8x8, cin= 64, jin=2, k=3, p=1, s=1, jout=2, rout=22, nout=8x8, cout=96
model.add(SeparableConv2D(filters=96, kernel_size=(3, 3), padding='same', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=18, nin = 8x8, cin= 96, jin=2, k=3, p=1, s=1, jout=2, rout=26, nout=6x6, cout=128
model.add(SeparableConv2D(filters=128, kernel_size=(3, 3), padding='valid', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=18, nin = 6x6, cin= 128, jin=2, k=3, p=1, s=1, jout=2, rout=30, nout=4x4, cout=192
model.add(SeparableConv2D(filters=192, kernel_size=(3, 3), padding='valid', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))


# rin=26, nin = 4x4, cin= 192, jin=2, k=3, p=1, s=1, jout=2, rout=34, nout=2x2, cout=10 Cfar_10 has 10 classes
model.add(SeparableConv2D(filters=num_classes, kernel_size=(3, 3), padding='valid', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))

# output size = 1x1x10
model.add(GlobalAveragePooling2D())
model.add(Flatten()) # 10x1
model.add(Activation('softmax')) # get probabilities

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
separable_conv2d_11 (Separab (None, 32, 32, 16)        75        
_________________________________________________________________
dropout_11 (Dropout)         (None, 32, 32, 16)        0         
_________________________________________________________________
batch_normalization_11 (Batc (None, 32, 32, 16)        64        
_________________________________________________________________
activation_12 (Activation)   (None, 32, 32, 16)        0         
_________________________________________________________________
separable_conv2d_12 (Separab (None, 32, 32, 16)        400       
_________________________________________________________________
dropout_12 (Dropout)         (None, 32, 32, 16)        0         
_________________________________________________________________
batch_normalization_12 (Batc (None, 32, 32, 16)        64        
_________________________________________________________________
activation_13 (Activation)   (None, 32, 32, 16)        0         
_________________________________________________________________
separable_conv2d_13 (Separab (None, 32, 32, 32)        656       
_________________________________________________________________
dropout_13 (Dropout)         (None, 32, 32, 32)        0         
_________________________________________________________________
batch_normalization_13 (Batc (None, 32, 32, 32)        128       
_________________________________________________________________
activation_14 (Activation)   (None, 32, 32, 32)        0         
_________________________________________________________________
separable_conv2d_14 (Separab (None, 32, 32, 48)        1824      
_________________________________________________________________
dropout_14 (Dropout)         (None, 32, 32, 48)        0         
_________________________________________________________________
batch_normalization_14 (Batc (None, 32, 32, 48)        192       
_________________________________________________________________
activation_15 (Activation)   (None, 32, 32, 48)        0         
_________________________________________________________________
separable_conv2d_15 (Separab (None, 32, 32, 64)        3504      
_________________________________________________________________
dropout_15 (Dropout)         (None, 32, 32, 64)        0         
_________________________________________________________________
batch_normalization_15 (Batc (None, 32, 32, 64)        256       
_________________________________________________________________
activation_16 (Activation)   (None, 32, 32, 64)        0         
_________________________________________________________________
separable_conv2d_16 (Separab (None, 32, 32, 128)       8768      
_________________________________________________________________
dropout_16 (Dropout)         (None, 32, 32, 128)       0         
_________________________________________________________________
batch_normalization_16 (Batc (None, 32, 32, 128)       512       
_________________________________________________________________
activation_17 (Activation)   (None, 32, 32, 128)       0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 128)       0         
_________________________________________________________________
separable_conv2d_17 (Separab (None, 16, 16, 64)        9344      
_________________________________________________________________
dropout_17 (Dropout)         (None, 16, 16, 64)        0         
_________________________________________________________________
batch_normalization_17 (Batc (None, 16, 16, 64)        256       
_________________________________________________________________
activation_18 (Activation)   (None, 16, 16, 64)        0         
_________________________________________________________________
separable_conv2d_18 (Separab (None, 16, 16, 96)        6720      
_________________________________________________________________
dropout_18 (Dropout)         (None, 16, 16, 96)        0         
_________________________________________________________________
batch_normalization_18 (Batc (None, 16, 16, 96)        384       
_________________________________________________________________
activation_19 (Activation)   (None, 16, 16, 96)        0         
_________________________________________________________________
separable_conv2d_19 (Separab (None, 14, 14, 128)       13152     
_________________________________________________________________
dropout_19 (Dropout)         (None, 14, 14, 128)       0         
_________________________________________________________________
batch_normalization_19 (Batc (None, 14, 14, 128)       512       
_________________________________________________________________
activation_20 (Activation)   (None, 14, 14, 128)       0         
_________________________________________________________________
separable_conv2d_20 (Separab (None, 12, 12, 192)       25728     
_________________________________________________________________
dropout_20 (Dropout)         (None, 12, 12, 192)       0         
_________________________________________________________________
batch_normalization_20 (Batc (None, 12, 12, 192)       768       
_________________________________________________________________
activation_21 (Activation)   (None, 12, 12, 192)       0         
_________________________________________________________________
separable_conv2d_21 (Separab (None, 10, 10, 10)        3648      
_________________________________________________________________
dropout_21 (Dropout)         (None, 10, 10, 10)        0         
_________________________________________________________________
batch_normalization_21 (Batc (None, 10, 10, 10)        40        
_________________________________________________________________
activation_22 (Activation)   (None, 10, 10, 10)        0         
_________________________________________________________________
global_average_pooling2d_1 ( (None, 10)                0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 10)                0         
_________________________________________________________________
activation_23 (Activation)   (None, 10)                0         
=================================================================
Total params: 76,995
Trainable params: 75,407
Non-trainable params: 1,588
_________________________________________________________________
```

## Result
```
Epoch 00001: LearningRateScheduler reducing learning rate to 0.01.
Epoch 1/50
390/390 [==============================] - 105s 270ms/step - loss: 1.7028 - accuracy: 0.3842 - val_loss: 2.3498 - val_accuracy: 0.3085

Epoch 00002: LearningRateScheduler reducing learning rate to 0.0075815011.
Epoch 2/50
390/390 [==============================] - 101s 258ms/step - loss: 1.2908 - accuracy: 0.5474 - val_loss: 2.8046 - val_accuracy: 0.3560

Epoch 00003: LearningRateScheduler reducing learning rate to 0.0061050061.
Epoch 3/50
390/390 [==============================] - 102s 262ms/step - loss: 1.0918 - accuracy: 0.6213 - val_loss: 1.6133 - val_accuracy: 0.5440

Epoch 00004: LearningRateScheduler reducing learning rate to 0.005109862.
Epoch 4/50
390/390 [==============================] - 103s 264ms/step - loss: 0.9712 - accuracy: 0.6657 - val_loss: 1.3412 - val_accuracy: 0.6158

Epoch 00005: LearningRateScheduler reducing learning rate to 0.0043936731.
Epoch 5/50
390/390 [==============================] - 102s 262ms/step - loss: 0.8942 - accuracy: 0.6928 - val_loss: 0.9012 - val_accuracy: 0.6980

Epoch 00006: LearningRateScheduler reducing learning rate to 0.0038535645.
Epoch 6/50
390/390 [==============================] - 101s 259ms/step - loss: 0.8341 - accuracy: 0.7154 - val_loss: 1.1658 - val_accuracy: 0.6556

Epoch 00007: LearningRateScheduler reducing learning rate to 0.003431709.
Epoch 7/50
390/390 [==============================] - 102s 262ms/step - loss: 0.7898 - accuracy: 0.7306 - val_loss: 1.0214 - val_accuracy: 0.6967

Epoch 00008: LearningRateScheduler reducing learning rate to 0.0030931024.
Epoch 8/50
390/390 [==============================] - 102s 261ms/step - loss: 0.7510 - accuracy: 0.7435 - val_loss: 0.7432 - val_accuracy: 0.7567

Epoch 00009: LearningRateScheduler reducing learning rate to 0.0028153153.
Epoch 9/50
390/390 [==============================] - 100s 257ms/step - loss: 0.7227 - accuracy: 0.7527 - val_loss: 0.6848 - val_accuracy: 0.7793

Epoch 00010: LearningRateScheduler reducing learning rate to 0.0025833118.
Epoch 10/50
390/390 [==============================] - 102s 262ms/step - loss: 0.6911 - accuracy: 0.7636 - val_loss: 1.3596 - val_accuracy: 0.6748

Epoch 00011: LearningRateScheduler reducing learning rate to 0.0023866348.
Epoch 11/50
390/390 [==============================] - 100s 256ms/step - loss: 0.6746 - accuracy: 0.7696 - val_loss: 0.7574 - val_accuracy: 0.7613

Epoch 00012: LearningRateScheduler reducing learning rate to 0.0022177866.
Epoch 12/50
390/390 [==============================] - 99s 253ms/step - loss: 0.6570 - accuracy: 0.7748 - val_loss: 0.6289 - val_accuracy: 0.7976

Epoch 00013: LearningRateScheduler reducing learning rate to 0.002071251.
Epoch 13/50
390/390 [==============================] - 100s 257ms/step - loss: 0.6444 - accuracy: 0.7799 - val_loss: 0.6029 - val_accuracy: 0.7986

Epoch 00014: LearningRateScheduler reducing learning rate to 0.0019428793.
Epoch 14/50
390/390 [==============================] - 100s 256ms/step - loss: 0.6251 - accuracy: 0.7871 - val_loss: 0.6486 - val_accuracy: 0.7892

Epoch 00015: LearningRateScheduler reducing learning rate to 0.0018294914.
Epoch 15/50
390/390 [==============================] - 99s 253ms/step - loss: 0.6178 - accuracy: 0.7897 - val_loss: 0.7723 - val_accuracy: 0.7634

Epoch 00016: LearningRateScheduler reducing learning rate to 0.0017286085.
Epoch 16/50
390/390 [==============================] - 101s 259ms/step - loss: 0.6026 - accuracy: 0.7929 - val_loss: 0.7266 - val_accuracy: 0.7706

Epoch 00017: LearningRateScheduler reducing learning rate to 0.00163827.
Epoch 17/50
390/390 [==============================] - 101s 258ms/step - loss: 0.5893 - accuracy: 0.7990 - val_loss: 0.6214 - val_accuracy: 0.8029

Epoch 00018: LearningRateScheduler reducing learning rate to 0.0015569049.
Epoch 18/50
390/390 [==============================] - 99s 254ms/step - loss: 0.5810 - accuracy: 0.8013 - val_loss: 0.5198 - val_accuracy: 0.8270

Epoch 00019: LearningRateScheduler reducing learning rate to 0.0014832394.
Epoch 19/50
390/390 [==============================] - 101s 259ms/step - loss: 0.5685 - accuracy: 0.8054 - val_loss: 0.5997 - val_accuracy: 0.8071

Epoch 00020: LearningRateScheduler reducing learning rate to 0.00141623.
Epoch 20/50
390/390 [==============================] - 100s 257ms/step - loss: 0.5654 - accuracy: 0.8063 - val_loss: 0.5667 - val_accuracy: 0.8192

Epoch 00021: LearningRateScheduler reducing learning rate to 0.0013550136.
Epoch 21/50
390/390 [==============================] - 100s 256ms/step - loss: 0.5554 - accuracy: 0.8096 - val_loss: 0.6854 - val_accuracy: 0.7890

Epoch 00022: LearningRateScheduler reducing learning rate to 0.00129887.
Epoch 22/50
390/390 [==============================] - 102s 261ms/step - loss: 0.5527 - accuracy: 0.8106 - val_loss: 0.5801 - val_accuracy: 0.8158

Epoch 00023: LearningRateScheduler reducing learning rate to 0.0012471938.
Epoch 23/50
390/390 [==============================] - 101s 259ms/step - loss: 0.5443 - accuracy: 0.8135 - val_loss: 0.5468 - val_accuracy: 0.8235

Epoch 00024: LearningRateScheduler reducing learning rate to 0.0011994722.
Epoch 24/50
390/390 [==============================] - 100s 257ms/step - loss: 0.5341 - accuracy: 0.8160 - val_loss: 0.5042 - val_accuracy: 0.8294

Epoch 00025: LearningRateScheduler reducing learning rate to 0.001155268.
Epoch 25/50
390/390 [==============================] - 103s 265ms/step - loss: 0.5290 - accuracy: 0.8193 - val_loss: 0.5008 - val_accuracy: 0.8351

Epoch 00026: LearningRateScheduler reducing learning rate to 0.0011142061.
Epoch 26/50
390/390 [==============================] - 101s 259ms/step - loss: 0.5222 - accuracy: 0.8224 - val_loss: 0.6066 - val_accuracy: 0.8095

Epoch 00027: LearningRateScheduler reducing learning rate to 0.001075963.
Epoch 27/50
390/390 [==============================] - 100s 257ms/step - loss: 0.5200 - accuracy: 0.8219 - val_loss: 0.4690 - val_accuracy: 0.8394

Epoch 00028: LearningRateScheduler reducing learning rate to 0.001040258.
Epoch 28/50
390/390 [==============================] - 101s 258ms/step - loss: 0.5164 - accuracy: 0.8223 - val_loss: 0.4795 - val_accuracy: 0.8390

Epoch 00029: LearningRateScheduler reducing learning rate to 0.0010068466.
Epoch 29/50
390/390 [==============================] - 101s 259ms/step - loss: 0.4992 - accuracy: 0.8302 - val_loss: 0.5103 - val_accuracy: 0.8327

Epoch 00030: LearningRateScheduler reducing learning rate to 0.0009755146.
Epoch 30/50
390/390 [==============================] - 99s 255ms/step - loss: 0.5015 - accuracy: 0.8275 - val_loss: 0.6389 - val_accuracy: 0.8008

Epoch 00031: LearningRateScheduler reducing learning rate to 0.0009460738.
Epoch 31/50
390/390 [==============================] - 100s 256ms/step - loss: 0.4962 - accuracy: 0.8300 - val_loss: 0.5093 - val_accuracy: 0.8302

Epoch 00032: LearningRateScheduler reducing learning rate to 0.000918358.
Epoch 32/50
390/390 [==============================] - 100s 257ms/step - loss: 0.4936 - accuracy: 0.8306 - val_loss: 0.5174 - val_accuracy: 0.8334

Epoch 00033: LearningRateScheduler reducing learning rate to 0.0008922198.
Epoch 33/50
390/390 [==============================] - 100s 256ms/step - loss: 0.4949 - accuracy: 0.8288 - val_loss: 0.5656 - val_accuracy: 0.8234

Epoch 00034: LearningRateScheduler reducing learning rate to 0.0008675284.
Epoch 34/50
390/390 [==============================] - 104s 266ms/step - loss: 0.4874 - accuracy: 0.8355 - val_loss: 0.5437 - val_accuracy: 0.8282

Epoch 00035: LearningRateScheduler reducing learning rate to 0.0008441668.
Epoch 35/50
390/390 [==============================] - 102s 260ms/step - loss: 0.4888 - accuracy: 0.8314 - val_loss: 0.6371 - val_accuracy: 0.8051

Epoch 00036: LearningRateScheduler reducing learning rate to 0.0008220304.
Epoch 36/50
390/390 [==============================] - 101s 259ms/step - loss: 0.4834 - accuracy: 0.8360 - val_loss: 0.5379 - val_accuracy: 0.8334

Epoch 00037: LearningRateScheduler reducing learning rate to 0.0008010253.
Epoch 37/50
390/390 [==============================] - 101s 258ms/step - loss: 0.4813 - accuracy: 0.8367 - val_loss: 0.4567 - val_accuracy: 0.8558

Epoch 00038: LearningRateScheduler reducing learning rate to 0.0007810669.
Epoch 38/50
390/390 [==============================] - 101s 258ms/step - loss: 0.4763 - accuracy: 0.8362 - val_loss: 0.4848 - val_accuracy: 0.8426

Epoch 00039: LearningRateScheduler reducing learning rate to 0.000762079.
Epoch 39/50
390/390 [==============================] - 100s 256ms/step - loss: 0.4739 - accuracy: 0.8376 - val_loss: 0.4919 - val_accuracy: 0.8427

Epoch 00040: LearningRateScheduler reducing learning rate to 0.0007439923.
Epoch 40/50
390/390 [==============================] - 100s 255ms/step - loss: 0.4700 - accuracy: 0.8401 - val_loss: 0.5044 - val_accuracy: 0.8382

Epoch 00041: LearningRateScheduler reducing learning rate to 0.0007267442.
Epoch 41/50
390/390 [==============================] - 100s 255ms/step - loss: 0.4663 - accuracy: 0.8390 - val_loss: 0.4970 - val_accuracy: 0.8407

Epoch 00042: LearningRateScheduler reducing learning rate to 0.0007102777.
Epoch 42/50
390/390 [==============================] - 98s 250ms/step - loss: 0.4663 - accuracy: 0.8394 - val_loss: 0.4815 - val_accuracy: 0.8464

Epoch 00043: LearningRateScheduler reducing learning rate to 0.0006945409.
Epoch 43/50
390/390 [==============================] - 97s 248ms/step - loss: 0.4596 - accuracy: 0.8428 - val_loss: 0.4651 - val_accuracy: 0.8497

Epoch 00044: LearningRateScheduler reducing learning rate to 0.0006794863.
Epoch 44/50
390/390 [==============================] - 100s 255ms/step - loss: 0.4594 - accuracy: 0.8421 - val_loss: 0.5308 - val_accuracy: 0.8353

Epoch 00045: LearningRateScheduler reducing learning rate to 0.0006650705.
Epoch 45/50
390/390 [==============================] - 99s 254ms/step - loss: 0.4625 - accuracy: 0.8410 - val_loss: 0.5686 - val_accuracy: 0.8236

Epoch 00046: LearningRateScheduler reducing learning rate to 0.0006512537.
Epoch 46/50
390/390 [==============================] - 99s 255ms/step - loss: 0.4571 - accuracy: 0.8424 - val_loss: 0.4676 - val_accuracy: 0.8513

Epoch 00047: LearningRateScheduler reducing learning rate to 0.0006379992.
Epoch 47/50
390/390 [==============================] - 100s 256ms/step - loss: 0.4561 - accuracy: 0.8427 - val_loss: 0.4883 - val_accuracy: 0.8480

Epoch 00048: LearningRateScheduler reducing learning rate to 0.0006252736.
Epoch 48/50
390/390 [==============================] - 99s 253ms/step - loss: 0.4513 - accuracy: 0.8460 - val_loss: 0.5097 - val_accuracy: 0.8402

Epoch 00049: LearningRateScheduler reducing learning rate to 0.0006130456.
Epoch 49/50
390/390 [==============================] - 99s 255ms/step - loss: 0.4445 - accuracy: 0.8486 - val_loss: 0.4983 - val_accuracy: 0.8435

Epoch 00050: LearningRateScheduler reducing learning rate to 0.0006012868.
Epoch 50/50
390/390 [==============================] - 101s 258ms/step - loss: 0.4484 - accuracy: 0.8452 - val_loss: 0.4772 - val_accuracy: 0.8463
Model took 5026.58 seconds to train
```
