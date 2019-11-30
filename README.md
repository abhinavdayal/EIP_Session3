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

## Success 09 - 76K parameters and no Image Normalization and Max pooling
**85.54% accuracy**

# Best 85.6% accuracy model ( Auccess 09 Done later after deadline. Sorry didnt get time before!)

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

# rin=7, nin = 32x32, cin=32, jin=1, k=3, p=1, s=1, jout=1, rout=9, nout=32x32, cout=48
model.add(SeparableConv2D(filters=48, kernel_size=(3, 3), padding='same', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))


# rin=9, nin = 32x32, cin= 48, jin=1, k=3, p=1, s=1, jout=1, rout=11, nout=32x32, cout=64
model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=11, nin = 32x32, cin= 64, jin=1, k=3, p=1, s=2, jout=2, rout=13, nout=16x16, cout=128
model.add(SeparableConv2D(filters=128, kernel_size=(3, 3), strides=(2,2), padding='same', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=13, nin = 16x16, cin= 96, jin=2, k=3, p=1, s=1, jout=2, rout=17, nout=16x16, cout=64
model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=12, nin = 8x8, cin= 64, jin=2, k=3, p=1, s=1, jout=2, rout=21, nout=8x8, cout=96
model.add(SeparableConv2D(filters=96, kernel_size=(3, 3), padding='same', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=18, nin = 8x8, cin= 96, jin=4, k=3, p=1, s=1, jout=4, rout=25, nout=6x6, cout=128
model.add(SeparableConv2D(filters=128, kernel_size=(3, 3), padding='valid', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))

# rin=18, nin = 6x6, cin= 128, jin=4, k=3, p=1, s=1, jout=4, rout=29, nout=4x4, cout=192
model.add(SeparableConv2D(filters=192, kernel_size=(3, 3), padding='valid', depth_multiplier=1, use_bias=False))
model.add(Dropout(0.05))
model.add(BatchNormalization())
model.add(Activation('relu'))


# rin=26, nin = 4x4, cin= 192, jin=4, k=3, p=1, s=1, jout=4, rout=33, nout=2x2, cout=10 Cfar_10 has 10 classes
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
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
separable_conv2d_33 (Separab (None, 32, 32, 16)        75        
_________________________________________________________________
dropout_33 (Dropout)         (None, 32, 32, 16)        0         
_________________________________________________________________
batch_normalization_33 (Batc (None, 32, 32, 16)        64        
_________________________________________________________________
activation_36 (Activation)   (None, 32, 32, 16)        0         
_________________________________________________________________
separable_conv2d_34 (Separab (None, 32, 32, 16)        400       
_________________________________________________________________
dropout_34 (Dropout)         (None, 32, 32, 16)        0         
_________________________________________________________________
batch_normalization_34 (Batc (None, 32, 32, 16)        64        
_________________________________________________________________
activation_37 (Activation)   (None, 32, 32, 16)        0         
_________________________________________________________________
separable_conv2d_35 (Separab (None, 32, 32, 32)        656       
_________________________________________________________________
dropout_35 (Dropout)         (None, 32, 32, 32)        0         
_________________________________________________________________
batch_normalization_35 (Batc (None, 32, 32, 32)        128       
_________________________________________________________________
activation_38 (Activation)   (None, 32, 32, 32)        0         
_________________________________________________________________
separable_conv2d_36 (Separab (None, 32, 32, 48)        1824      
_________________________________________________________________
dropout_36 (Dropout)         (None, 32, 32, 48)        0         
_________________________________________________________________
batch_normalization_36 (Batc (None, 32, 32, 48)        192       
_________________________________________________________________
activation_39 (Activation)   (None, 32, 32, 48)        0         
_________________________________________________________________
separable_conv2d_37 (Separab (None, 32, 32, 64)        3504      
_________________________________________________________________
dropout_37 (Dropout)         (None, 32, 32, 64)        0         
_________________________________________________________________
batch_normalization_37 (Batc (None, 32, 32, 64)        256       
_________________________________________________________________
activation_40 (Activation)   (None, 32, 32, 64)        0         
_________________________________________________________________
separable_conv2d_38 (Separab (None, 16, 16, 128)       8768      
_________________________________________________________________
dropout_38 (Dropout)         (None, 16, 16, 128)       0         
_________________________________________________________________
batch_normalization_38 (Batc (None, 16, 16, 128)       512       
_________________________________________________________________
activation_41 (Activation)   (None, 16, 16, 128)       0         
_________________________________________________________________
separable_conv2d_39 (Separab (None, 16, 16, 64)        9344      
_________________________________________________________________
dropout_39 (Dropout)         (None, 16, 16, 64)        0         
_________________________________________________________________
batch_normalization_39 (Batc (None, 16, 16, 64)        256       
_________________________________________________________________
activation_42 (Activation)   (None, 16, 16, 64)        0         
_________________________________________________________________
separable_conv2d_40 (Separab (None, 16, 16, 96)        6720      
_________________________________________________________________
dropout_40 (Dropout)         (None, 16, 16, 96)        0         
_________________________________________________________________
batch_normalization_40 (Batc (None, 16, 16, 96)        384       
_________________________________________________________________
activation_43 (Activation)   (None, 16, 16, 96)        0         
_________________________________________________________________
separable_conv2d_41 (Separab (None, 14, 14, 128)       13152     
_________________________________________________________________
dropout_41 (Dropout)         (None, 14, 14, 128)       0         
_________________________________________________________________
batch_normalization_41 (Batc (None, 14, 14, 128)       512       
_________________________________________________________________
activation_44 (Activation)   (None, 14, 14, 128)       0         
_________________________________________________________________
separable_conv2d_42 (Separab (None, 12, 12, 192)       25728     
_________________________________________________________________
dropout_42 (Dropout)         (None, 12, 12, 192)       0         
_________________________________________________________________
batch_normalization_42 (Batc (None, 12, 12, 192)       768       
_________________________________________________________________
activation_45 (Activation)   (None, 12, 12, 192)       0         
_________________________________________________________________
separable_conv2d_43 (Separab (None, 10, 10, 10)        3648      
_________________________________________________________________
dropout_43 (Dropout)         (None, 10, 10, 10)        0         
_________________________________________________________________
batch_normalization_43 (Batc (None, 10, 10, 10)        40        
_________________________________________________________________
activation_46 (Activation)   (None, 10, 10, 10)        0         
_________________________________________________________________
global_average_pooling2d_3 ( (None, 10)                0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 10)                0         
_________________________________________________________________
activation_47 (Activation)   (None, 10)                0         
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
390/390 [==============================] - 99s 253ms/step - loss: 1.6833 - accuracy: 0.3907 - val_loss: 6.3635 - val_accuracy: 0.2233

Epoch 00002: LearningRateScheduler reducing learning rate to 0.008071025.
Epoch 2/50
390/390 [==============================] - 98s 252ms/step - loss: 1.3361 - accuracy: 0.5290 - val_loss: 3.1254 - val_accuracy: 0.3961

Epoch 00003: LearningRateScheduler reducing learning rate to 0.0067658999.
Epoch 3/50
390/390 [==============================] - 100s 256ms/step - loss: 1.1545 - accuracy: 0.5991 - val_loss: 1.7183 - val_accuracy: 0.5101

Epoch 00004: LearningRateScheduler reducing learning rate to 0.0058241118.
Epoch 4/50
390/390 [==============================] - 98s 251ms/step - loss: 1.0396 - accuracy: 0.6381 - val_loss: 1.3908 - val_accuracy: 0.6052

Epoch 00005: LearningRateScheduler reducing learning rate to 0.0051124744.
Epoch 5/50
390/390 [==============================] - 99s 255ms/step - loss: 0.9565 - accuracy: 0.6692 - val_loss: 1.0198 - val_accuracy: 0.6686

Epoch 00006: LearningRateScheduler reducing learning rate to 0.0045558087.
Epoch 6/50
390/390 [==============================] - 100s 257ms/step - loss: 0.8957 - accuracy: 0.6919 - val_loss: 1.2684 - val_accuracy: 0.6427

Epoch 00007: LearningRateScheduler reducing learning rate to 0.0041084634.
Epoch 7/50
390/390 [==============================] - 99s 254ms/step - loss: 0.8402 - accuracy: 0.7104 - val_loss: 0.9879 - val_accuracy: 0.6940

Epoch 00008: LearningRateScheduler reducing learning rate to 0.0037411149.
Epoch 8/50
390/390 [==============================] - 99s 254ms/step - loss: 0.8111 - accuracy: 0.7203 - val_loss: 0.8241 - val_accuracy: 0.7345

Epoch 00009: LearningRateScheduler reducing learning rate to 0.0034340659.
Epoch 9/50
390/390 [==============================] - 100s 257ms/step - loss: 0.7818 - accuracy: 0.7301 - val_loss: 0.8006 - val_accuracy: 0.7492

Epoch 00010: LearningRateScheduler reducing learning rate to 0.0031735957.
Epoch 10/50
390/390 [==============================] - 100s 255ms/step - loss: 0.7521 - accuracy: 0.7412 - val_loss: 0.9741 - val_accuracy: 0.7138

Epoch 00011: LearningRateScheduler reducing learning rate to 0.0029498525.
Epoch 11/50
390/390 [==============================] - 99s 254ms/step - loss: 0.7341 - accuracy: 0.7481 - val_loss: 0.9388 - val_accuracy: 0.7082

Epoch 00012: LearningRateScheduler reducing learning rate to 0.00275558.
Epoch 12/50
390/390 [==============================] - 99s 254ms/step - loss: 0.7176 - accuracy: 0.7535 - val_loss: 0.6720 - val_accuracy: 0.7869

Epoch 00013: LearningRateScheduler reducing learning rate to 0.0025853154.
Epoch 13/50
390/390 [==============================] - 99s 253ms/step - loss: 0.6933 - accuracy: 0.7633 - val_loss: 0.7392 - val_accuracy: 0.7714

Epoch 00014: LearningRateScheduler reducing learning rate to 0.0024348673.
Epoch 14/50
390/390 [==============================] - 98s 251ms/step - loss: 0.6776 - accuracy: 0.7639 - val_loss: 0.6321 - val_accuracy: 0.7964

Epoch 00015: LearningRateScheduler reducing learning rate to 0.0023009664.
Epoch 15/50
390/390 [==============================] - 99s 255ms/step - loss: 0.6723 - accuracy: 0.7715 - val_loss: 0.6663 - val_accuracy: 0.7851

Epoch 00016: LearningRateScheduler reducing learning rate to 0.0021810251.
Epoch 16/50
390/390 [==============================] - 99s 254ms/step - loss: 0.6536 - accuracy: 0.7757 - val_loss: 0.6920 - val_accuracy: 0.7789

Epoch 00017: LearningRateScheduler reducing learning rate to 0.0020729685.
Epoch 17/50
390/390 [==============================] - 97s 250ms/step - loss: 0.6460 - accuracy: 0.7788 - val_loss: 0.5904 - val_accuracy: 0.8004

Epoch 00018: LearningRateScheduler reducing learning rate to 0.0019751136.
Epoch 18/50
390/390 [==============================] - 99s 254ms/step - loss: 0.6326 - accuracy: 0.7849 - val_loss: 0.8594 - val_accuracy: 0.7290

Epoch 00019: LearningRateScheduler reducing learning rate to 0.0018860807.
Epoch 19/50
390/390 [==============================] - 98s 251ms/step - loss: 0.6246 - accuracy: 0.7851 - val_loss: 0.7021 - val_accuracy: 0.7813

Epoch 00020: LearningRateScheduler reducing learning rate to 0.0018047284.
Epoch 20/50
390/390 [==============================] - 98s 250ms/step - loss: 0.6201 - accuracy: 0.7871 - val_loss: 0.6040 - val_accuracy: 0.8039

Epoch 00021: LearningRateScheduler reducing learning rate to 0.0017301038.
Epoch 21/50
390/390 [==============================] - 99s 253ms/step - loss: 0.6041 - accuracy: 0.7925 - val_loss: 0.4849 - val_accuracy: 0.8391

Epoch 00022: LearningRateScheduler reducing learning rate to 0.0016614055.
Epoch 22/50
390/390 [==============================] - 99s 254ms/step - loss: 0.5978 - accuracy: 0.7968 - val_loss: 0.6087 - val_accuracy: 0.8089

Epoch 00023: LearningRateScheduler reducing learning rate to 0.0015979546.
Epoch 23/50
390/390 [==============================] - 98s 251ms/step - loss: 0.5924 - accuracy: 0.7969 - val_loss: 0.5533 - val_accuracy: 0.8186

Epoch 00024: LearningRateScheduler reducing learning rate to 0.0015391719.
Epoch 24/50
390/390 [==============================] - 98s 252ms/step - loss: 0.5823 - accuracy: 0.8006 - val_loss: 0.7180 - val_accuracy: 0.7817

Epoch 00025: LearningRateScheduler reducing learning rate to 0.0014845606.
Epoch 25/50
390/390 [==============================] - 98s 251ms/step - loss: 0.5749 - accuracy: 0.8031 - val_loss: 0.5406 - val_accuracy: 0.8260

Epoch 00026: LearningRateScheduler reducing learning rate to 0.0014336918.
Epoch 26/50
390/390 [==============================] - 97s 249ms/step - loss: 0.5708 - accuracy: 0.8038 - val_loss: 0.5295 - val_accuracy: 0.8231

Epoch 00027: LearningRateScheduler reducing learning rate to 0.0013861935.
Epoch 27/50
390/390 [==============================] - 98s 251ms/step - loss: 0.5630 - accuracy: 0.8067 - val_loss: 0.5303 - val_accuracy: 0.8275

Epoch 00028: LearningRateScheduler reducing learning rate to 0.0013417416.
Epoch 28/50
390/390 [==============================] - 97s 248ms/step - loss: 0.5644 - accuracy: 0.8080 - val_loss: 0.5025 - val_accuracy: 0.8359

Epoch 00029: LearningRateScheduler reducing learning rate to 0.001300052.
Epoch 29/50
390/390 [==============================] - 96s 247ms/step - loss: 0.5525 - accuracy: 0.8104 - val_loss: 0.5867 - val_accuracy: 0.8138

Epoch 00030: LearningRateScheduler reducing learning rate to 0.001260875.
Epoch 30/50
390/390 [==============================] - 96s 246ms/step - loss: 0.5468 - accuracy: 0.8128 - val_loss: 0.5096 - val_accuracy: 0.8355

Epoch 00031: LearningRateScheduler reducing learning rate to 0.0012239902.
Epoch 31/50
390/390 [==============================] - 98s 252ms/step - loss: 0.5463 - accuracy: 0.8126 - val_loss: 0.5604 - val_accuracy: 0.8254

Epoch 00032: LearningRateScheduler reducing learning rate to 0.001189202.
Epoch 32/50
390/390 [==============================] - 96s 247ms/step - loss: 0.5426 - accuracy: 0.8151 - val_loss: 0.5578 - val_accuracy: 0.8229

Epoch 00033: LearningRateScheduler reducing learning rate to 0.0011563367.
Epoch 33/50
390/390 [==============================] - 97s 248ms/step - loss: 0.5410 - accuracy: 0.8153 - val_loss: 0.5664 - val_accuracy: 0.8225

Epoch 00034: LearningRateScheduler reducing learning rate to 0.0011252391.
Epoch 34/50
390/390 [==============================] - 98s 251ms/step - loss: 0.5325 - accuracy: 0.8164 - val_loss: 0.5612 - val_accuracy: 0.8215

Epoch 00035: LearningRateScheduler reducing learning rate to 0.0010957703.
Epoch 35/50
390/390 [==============================] - 97s 248ms/step - loss: 0.5326 - accuracy: 0.8171 - val_loss: 0.5004 - val_accuracy: 0.8398

Epoch 00036: LearningRateScheduler reducing learning rate to 0.0010678057.
Epoch 36/50
390/390 [==============================] - 96s 247ms/step - loss: 0.5285 - accuracy: 0.8186 - val_loss: 0.5058 - val_accuracy: 0.8374

Epoch 00037: LearningRateScheduler reducing learning rate to 0.0010412328.
Epoch 37/50
390/390 [==============================] - 98s 251ms/step - loss: 0.5168 - accuracy: 0.8239 - val_loss: 0.4705 - val_accuracy: 0.8449

Epoch 00038: LearningRateScheduler reducing learning rate to 0.0010159504.
Epoch 38/50
390/390 [==============================] - 95s 245ms/step - loss: 0.5240 - accuracy: 0.8243 - val_loss: 0.4911 - val_accuracy: 0.8460

Epoch 00039: LearningRateScheduler reducing learning rate to 0.0009918667.
Epoch 39/50
390/390 [==============================] - 95s 243ms/step - loss: 0.5180 - accuracy: 0.8227 - val_loss: 0.4664 - val_accuracy: 0.8480

Epoch 00040: LearningRateScheduler reducing learning rate to 0.0009688984.
Epoch 40/50
390/390 [==============================] - 97s 249ms/step - loss: 0.5148 - accuracy: 0.8234 - val_loss: 0.4805 - val_accuracy: 0.8444

Epoch 00041: LearningRateScheduler reducing learning rate to 0.0009469697.
Epoch 41/50
390/390 [==============================] - 96s 245ms/step - loss: 0.5129 - accuracy: 0.8232 - val_loss: 0.4459 - val_accuracy: 0.8554

Epoch 00042: LearningRateScheduler reducing learning rate to 0.0009260117.
Epoch 42/50
390/390 [==============================] - 94s 240ms/step - loss: 0.5041 - accuracy: 0.8279 - val_loss: 0.4517 - val_accuracy: 0.8518

Epoch 00043: LearningRateScheduler reducing learning rate to 0.0009059612.
Epoch 43/50
390/390 [==============================] - 96s 246ms/step - loss: 0.5052 - accuracy: 0.8276 - val_loss: 0.4425 - val_accuracy: 0.8533

Epoch 00044: LearningRateScheduler reducing learning rate to 0.0008867607.
Epoch 44/50
390/390 [==============================] - 94s 240ms/step - loss: 0.5045 - accuracy: 0.8265 - val_loss: 0.5206 - val_accuracy: 0.8324

Epoch 00045: LearningRateScheduler reducing learning rate to 0.0008683571.
Epoch 45/50
390/390 [==============================] - 94s 240ms/step - loss: 0.4969 - accuracy: 0.8300 - val_loss: 0.4777 - val_accuracy: 0.8465

Epoch 00046: LearningRateScheduler reducing learning rate to 0.0008507018.
Epoch 46/50
390/390 [==============================] - 97s 250ms/step - loss: 0.4961 - accuracy: 0.8312 - val_loss: 0.4732 - val_accuracy: 0.8423

Epoch 00047: LearningRateScheduler reducing learning rate to 0.0008337502.
Epoch 47/50
390/390 [==============================] - 96s 246ms/step - loss: 0.4948 - accuracy: 0.8280 - val_loss: 0.5545 - val_accuracy: 0.8216

Epoch 00048: LearningRateScheduler reducing learning rate to 0.000817461.
Epoch 48/50
390/390 [==============================] - 95s 243ms/step - loss: 0.4944 - accuracy: 0.8304 - val_loss: 0.5007 - val_accuracy: 0.8449

Epoch 00049: LearningRateScheduler reducing learning rate to 0.000801796.
Epoch 49/50
390/390 [==============================] - 97s 249ms/step - loss: 0.4931 - accuracy: 0.8289 - val_loss: 0.4898 - val_accuracy: 0.8438

Epoch 00050: LearningRateScheduler reducing learning rate to 0.0007867202.
Epoch 50/50
390/390 [==============================] - 96s 247ms/step - loss: 0.4867 - accuracy: 0.8335 - val_loss: 0.4538 - val_accuracy: 0.8528
Model took 4874.43 seconds to train
```
