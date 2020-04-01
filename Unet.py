import tensorflow as tf


def CreateConvBlock(x, filters, n = 2, use_bn = True, apply_pooling = True, name = 'convblock'):
    for i in range(1,n+1):
        x = tf.keras.layers.Conv2D(filters, (3,3), padding='same', name=name+'_conv'+str(i))(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization(name=name+'_BN'+str(i))(x)
        x = tf.keras.layers.Activation('relu', name=name+'_relu'+str(i))(x)

    convresult = x

    if apply_pooling:
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2), name=name+'_pooling')(x)

    return x, convresult

def CreateConvBlockwithLSTM(x, filters, n = 2, use_bn = True, apply_pooling = True, name = 'convblock'):
    x = tf.keras.layers.Conv2D(filters, (3,3), padding='same', name=name+'_conv'+str(i))(x)
    if use_bn:
        x = tf.keras.layers.BatchNormalization(name=name+'_BN'+str(i))(x)
    x = tf.keras.layers.Activation('relu', name=name+'_relu'+str(i))(x)

    x = tf.keras.layers.Reshape()

    convresult = x

    if apply_pooling:
        x = tf.keras.layers.MaxPool2D(pool_size=(2,2), name=name+'_pooling')(x)

    return x, convresult


def CreateUpConvBlock(x, contractpart, filters, n = 2, use_bn = True, name = 'upconvblock'):
    # upconv x
    x = tf.keras.layers.Conv2DTranspose(int(x.shape[-1]), (2,2), strides=(2,2), padding='same', name=name+'_upconv')(x)

    # concatenate contract4 and x
    x = tf.keras.layers.concatenate([contractpart, x])

    # conv x 2 times
    for i in range(1,n+1):
        x = tf.keras.layers.Conv2D(filters, (3,3), padding='same', name=name+'_conv'+str(i))(x)
        if use_bn:
            x = tf.keras.layers.BatchNormalization(name=name+'_BN'+str(i))(x)
        x = tf.keras.layers.Activation('relu', name=name+'_relu'+str(i))(x)

    return x


def ConstructModel(input_images, nclasses, use_bn = True, use_dropout = True):

    # Contract1 (256->128)
    with tf.name_scope("contract1"):
        x, contract1 = CreateConvBlock(input_images, 64, n = 2, use_bn = use_bn, name = 'contract1')

    # Contract2 (128->64)
    with tf.name_scope("contract2"):
        x, contract2 = CreateConvBlock(x, 128, n = 2, use_bn = use_bn, name = 'contract2')

    # Contract3 (64->32)
    with tf.name_scope("contract3"):
        x, contract3 = CreateConvBlock(x, 256, n = 2, use_bn = use_bn, name = 'contract3')

    # Contract4 (32->16)
    with tf.name_scope("contract4"):
        x, contract4 = CreateConvBlock(x, 512, n = 2, use_bn = use_bn, name = 'contract4')

    # Contract5 (16)
    with tf.name_scope("contract5"):
        x, _ = CreateConvBlock(x, 1024, n = 2, use_bn = use_bn, apply_pooling = False, name = 'contract5')

    # Dropout (16)
    with tf.name_scope("dropout"):
        if use_dropout:
            x = tf.keras.layers.Dropout(0.5, name='dropout')(x)

    # Expand4 (16->32)
    with tf.name_scope("expand4"):
        x = CreateUpConvBlock(x, contract4, 512, n = 2, use_bn = use_bn, name = 'expand4')

    # Expand3 (32->64)
    with tf.name_scope("expand3"):
        x = CreateUpConvBlock(x, contract3, 256, n = 2, use_bn = use_bn, name = 'expand3')

    # Expand2 (64->128)
    with tf.name_scope("expand2"):
        x = CreateUpConvBlock(x, contract2, 128, n = 2, use_bn = use_bn, name = 'expand2')

    # Expand1 (128->256)
    with tf.name_scope("expand1"):
        x = CreateUpConvBlock(x, contract1, 64, n = 2, use_bn = use_bn, name = 'expand1')

    # Segmentation
    with tf.name_scope("segmentation"):
        layername = 'segmentation'
        if nclasses == 9: # 8 organs segmentation
            layername = 'segmentation'
        else:
            layername = 'segmentation_{}classes'.format(nclasses)
        x = tf.keras.layers.Conv2D(nclasses, (1,1), activation='softmax', padding='same', name=layername)(x)
        #x = tf.keras.layers.Conv2D(nclasses, (1,1), activation='softmax', padding='same', name='segmentation')(x)

    return x

inputShape = (256, 256, 3)
inputs = tf.keras.layers.Input(shape=inputShape)
segmentation = ConstructModel(inputs, 3)

model = tf.keras.models.Model(inputs, segmentation)
model.summary(line_length=150)#positions=[.33, .55, .67, .1])
