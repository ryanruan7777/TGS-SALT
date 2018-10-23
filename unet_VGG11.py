# -*- coding:utf-8 -*-
"""
Generator and Discriminator network.
"""
from keras import models, layers
def unet():
    # Build U-Net model
    GAUSSIAN_NOISE = 0.1
    UPSAMPLE_MODE = 'SIMPLE'
    # downsampling inside the network
    NET_SCALING = (1, 1)
    # downsampling in preprocessing

    def upsample_conv(filters, kernel_size, strides, padding):
        return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)
    def upsample_simple(filters, kernel_size, strides, padding):
        return layers.UpSampling2D(strides)

    if UPSAMPLE_MODE=='DECONV':
        upsample=upsample_conv
    else:
        upsample=upsample_simple
        
    input_img = layers.Input([96,96,3], name = 'RGB_Input')
    pp_in_layer = input_img

    if NET_SCALING is not None:
        pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)
        
    pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
    pp_in_layer = layers.BatchNormalization()(pp_in_layer)
   
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same') (pp_in_layer)  #'conv1_1'
    p1 = layers.MaxPooling2D((2, 2)) (c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (p1)          #'conv2_1'
    p2 = layers.MaxPooling2D((2, 2)) (c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same') (p2)          #'conv3_1'
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same') (c3)          #'conv3_2'
    p3 = layers.MaxPooling2D((2, 2)) (c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same') (p3)           #'conv4_1'
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same') (c4)           #'conv4_2'
    p4 = layers.MaxPooling2D(pool_size=(2, 2)) (c4)


    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same') (p4)          #'conv5_1'
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same') (c5)          #'conv5_2'
    p5=layers.MaxPooling2D(pool_size=(2, 2)) (c5)
    
    c_center = layers.Conv2D(512, (3, 3), activation='relu', padding='same') (p5)
    
    u6 = upsample(256, (2, 2), strides=(2, 2), padding='same') (c_center)
    u6 = layers.concatenate([u6, c5])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same') (u6)

    u7 = upsample(256, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = layers.concatenate([u7, c4])
    c7 = layers.Conv2D(512, (3, 3), activation='relu', padding='same') (u7)

    u8 = upsample(128, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = layers.concatenate([u8, c3])
    c8 = layers.Conv2D(256, (3, 3), activation='relu', padding='same') (u8)


    u9 = upsample(64, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = layers.concatenate([u9, c2], axis=3)
    c9 = layers.Conv2D(128, (3, 3), activation='relu', padding='same') (u9)
    
    u10 = upsample(32, (2, 2), strides=(2, 2), padding='same') (c9)
    u10 = layers.concatenate([u10, c1], axis=3)
    #c10 = layers.Conv2D(1, (3, 3), activation='relu', padding='same') (u10)


    d = layers.Conv2D(1, (1, 1), activation='sigmoid') (u10)
    # d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
    # d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
    if NET_SCALING is not None:
        d = layers.UpSampling2D(NET_SCALING)(d)

    seg_model = models.Model(inputs=[input_img], outputs=[d])
    #seg_model.summary()
    return seg_model