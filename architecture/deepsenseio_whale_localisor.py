from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.models import Model


def create_model(image_size=(224, 224), n_outs=None, dropout=False, dropout_coeff=0.5,
                 fc_l2_reg=None, conv_l2_reg=None, glr_decay=1.0,
                 momentum=0.9, glr=0.01, **kwargs):
    # this model is followed after the model from deepsenseio:
    # https://deepsense.io/deep-learning-right-whale-recognition-kaggle/

    main_input = Input(shape=(3, image_size[0], image_size[1],),  dtype='float32', name='Image_input')
    x = Convolution2D(16, 3, 3, border_mode='same', W_regularizer=l2(conv_l2_reg))(main_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    if dropout:
        x = Dropout(dropout_coeff)(x)

    x = Convolution2D(32, 3, 3, border_mode='same', W_regularizer=l2(conv_l2_reg))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    if dropout:
        x = Dropout(dropout_coeff)(x)

    x = Convolution2D(32, 3, 3, border_mode='same', W_regularizer=l2(conv_l2_reg))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    if dropout:
        x = Dropout(dropout_coeff)(x)

    x = Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(conv_l2_reg))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    if dropout:
        x = Dropout(dropout_coeff)(x)

    x = Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(conv_l2_reg))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    if dropout:
        x = Dropout(dropout_coeff)(x)

    x = Flatten()(x)
    outputs = []
    for out_num, n_out in enumerate(n_outs):
        outputs.append(Dense(n_out, activation='softmax', name='predictions_'+str(out_num), W_regularizer=l2(fc_l2_reg))(x))

    model = Model(input=[main_input], output=outputs)

    sgd = SGD(lr=glr, decay=glr_decay, momentum=momentum, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model


# def create_model_old(image_size=(224, 224), n_outs=[60,60,60,60], dropout=False, dropout_coeff=0.5,
#                  fc_l2_reg=0.05, conv_l2_reg=0.005, glr_decay=0.9955,
#                  momentum=0.9, glr=0.01):
#     # this model is followed after the model from deepsenseio:
#     # https://deepsense.io/deep-learning-right-whale-recognition-kaggle/
#     image_shape = (3, image_size[0], image_size[1])
#
#     model = Sequential()
#     model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=image_shape, W_regularizer=l2(conv_l2_reg)))
#     #model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     if dropout:
#         model.add(Dropout(dropout_coeff))
#
#     model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(conv_l2_reg)))
#     #model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     if dropout:
#         model.add(Dropout(dropout_coeff))
#
#     model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(conv_l2_reg)))
#     #model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     if dropout:
#         model.add(Dropout(dropout_coeff))
#
#     model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(conv_l2_reg)))
#     #model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     if dropout:
#         model.add(Dropout(dropout_coeff))
#
#     model.add(Convolution2D(64, 3, 3, border_mode='same', W_regularizer=l2(conv_l2_reg)))
#     #model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     if dropout:
#         model.add(Dropout(dropout_coeff))
#
#     model.add(Flatten())
#
#     model.add(Dense(sum(n_outs), activation='softmax', name='predictions', W_regularizer=l2(fc_l2_reg)))
#     model.name = 'deepsenseio_localizing_whale'
#
#     sgd = SGD(lr=glr, decay=glr_decay, momentum=momentum, nesterov=True)
#     model.compile(loss='categorical_crossentropy', optimizer=sgd)
#
#     return model
#
