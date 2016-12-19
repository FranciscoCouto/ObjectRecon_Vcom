from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random as rnd
import time
import sys
import os

nb_filters = 32
kernel_size = (3, 3)
batch_size = 64
nb_classes = 10
nb_epoch = 80
data_augmentation = False
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# TODO
# correr o 2 com 20 epochs para ver se é melhor que o 3
# correr o 3 com 20 epochs para ver se as alterações melhoraram ou não
# treinar uma rede das que ganhou nos ultimos anos tipo vgg16, obter a rede do keras


def printUsage():
    print("Usage: ")
    print("\t python vcom.py train <model_num/vgg16> <epochs>")
    print("\t python vcom.py test <model_num/vgg16> <weights file>")
    sys.exit(-1)


def main():
    if(len(sys.argv) < 3):
        printUsage()
    elif(sys.argv[1] == 'train'):
        train = True
        if(sys.argv[2] == 'vgg16'):
            use_resnet = True
            if(len(sys.argv) != 4):
                printUsage()
            else:
                nb_epoch = int(sys.argv[3])

        else:
            use_resnet = False
            model_num = int(sys.argv[2])
            if(len(sys.argv) != 4):
                printUsage()
            else:
                nb_epoch = int(sys.argv[3])

    elif(sys.argv[1] == 'test'):
        train = False
        if(sys.argv[2] == 'vgg16'):
            use_resnet = True
            if(len(sys.argv) != 4):
                printUsage()
            else:
                weightsFile = sys.argv[3]

        else:
            use_resnet = False
            model_num = int(sys.argv[2])
            if(len(sys.argv) != 4):
                printUsage()
            else:
                weightsFile = sys.argv[3]

    else:
        printUsage()

    start_time = time.time()

    rnd.seed(123)
    x_train, y_train, x_test, y_test, input_shape = loadCifar_10Dataset()
    if (train):
        # create model
        if (use_resnet == False):
            # load dataset
            print("\n # RUNNING MODEL NUM: " + str(model_num) + " #\n")
            model = Sequential()
            if model_num == 0:
                # acc: 0.73370 com 10 epochs
                # acc: 0.81 com 20 epochs - time por epoch: 175s
                # acc: 0.83730 com 40 epochs
                # acc: 0.84250 com 80 epochs
                # acc: 0.86970 com 80 epochs e data augmentation
                model = model_0(input_shape)

            elif model_num == 1:
                # acc: 0.75590 com 10 epochs
                model = model_1(input_shape)

            elif model_num == 2:
                # acc: 0.76290 com 10 epochs
                model = model_2(input_shape)

            elif model_num == 3:
                # acc: 0.75540 com 10
                model = model_3(input_shape)
            elif model_num == 4:
                # acc: 0.77260 com 10
                model = model_4(input_shape)
            elif model_num == 5:
                # acc: 0.78170 com 10
                model = model_5(input_shape)
            elif model_num == 6:
                # acc: 0.79630 com 10
                # acc: 0.84970 com 50
                model = model_6(input_shape)
            elif model_num == 7:
            	# acc: 0.79130 com 10
                model = model_7(input_shape)


            print(model.summary())

            if data_augmentation:
                directory = "./model_" + str(model_num) + "_epochs_" + str(nb_epoch) + "_aug_"
            else:
            	directory = "./model_" + str(model_num) + "_epochs_" + str(nb_epoch)
            if not os.path.exists(directory):
                os.makedirs(directory)
            # save model after each epoch
            checkpoint_callback = ModelCheckpoint(directory + '/weights.{epoch:03d}-{val_loss:.3f}.hdf5',
                                                  monitor='val_loss', verbose=0, save_best_only=False,
                                                  save_weights_only=False, mode='auto')

            # stop training if not improving
            '''
            earlyStop_callback = EarlyStopping(monitor='val_loss', min_delta=0.002, patience=10, verbose=1, mode='auto')
            '''

            # reduce learning rate if model not improving
            reduceLR_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1,
                                                  mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.000001)

            # save epoch results in csv file
            csv_logger_callback = CSVLogger(filename=directory + '/training.csv', separator=";", append=True)

            if data_augmentation:
                # data augmentation

                shift = 0.2
                datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, width_shift_range=shift,
                                             height_shift_range=shift, rotation_range=90)

                datagen.fit(x_train)

                print('Using real-time data augmentation.')

                # this will do preprocessing and realtime data augmentation
                datagen = ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=True,  # randomly flip images
                    vertical_flip=False)  # randomly flip images

                # compute quantities required for featurewise normalization
                # (std, mean, and principal components if ZCA whitening is applied)
                datagen.fit(x_train)

                # fit the model on the batches generated by datagen.flow()
                history = model.fit_generator(datagen.flow(x_train, y_train,
                                                           batch_size=batch_size),
                                              samples_per_epoch=x_train.shape[0],
                                              nb_epoch=nb_epoch,
                                              validation_data=(x_test, y_test),
                                              callbacks=[checkpoint_callback, reduceLR_callback, csv_logger_callback])
            else:
                # history = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.1,
                #                    callbacks=[checkpoint_callback, earlyStop_callback, reduceLR_callback, csv_logger_callback])
                history = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.1,
                                    callbacks=[checkpoint_callback, reduceLR_callback, csv_logger_callback])

            print("--- %s seconds ---" % (time.time() - start_time))
            open(directory + "/exec_time_" + str(time.time() - start_time), 'a').close()

            # test the model
            score = model.evaluate(x_test, y_test)
            print("\nTest accuracy: %0.05f" % score[1])
            open(directory + "/score_" + str(score[1]), 'a').close()

            saveModel(model, directory + "/model_num_" + str(model_num))

            plotTrainingHistory(history, directory)
            showErrors(model, x_test, y_test, directory)
        else:
            print("\n # RUNNING VGG16 NEURONAL NETWORK # \n")
            vgg16(nb_epoch)
    else:
        model = loadModel(str(model_num))
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        score = model.evaluate(x_test, y_test)
        print("\nTest accuracy: %0.05f" % score[1])


def model_0(input_shape):
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='full', input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

# igual mas com outro optimizer


def model_1(input_shape):
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='full', input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

# com menos uma pool


def model_3(input_shape):
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='full', input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

# com menos uma pool + 1 conv


def model_2(input_shape):
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='full', input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

# trocar um pool por conv


def model_4(input_shape):
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='full', input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def model_5(input_shape):
    model = Sequential()

    model.add(Convolution2D(96, 3, 3, border_mode='full', input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(512, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def model_6(input_shape):
    model = Sequential()

    model.add(Convolution2D(96, 3, 3, border_mode='full', input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))

    model.add(Convolution2D(512, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def model_7(input_shape):
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='full', input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def vgg16(nb_epoch):

    # work around theano maximum recursion limit exception
    import sys
    sys.setrecursionlimit(50000)

    from keras.models import Model

    # load and prepare dataset
    x_train, y_train, x_test, y_test, input_shape = loadCifar_10Dataset()

    img_input = Input(shape=(3, 32, 32))
    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, activation='softmax', name='predictions')(x)

    model = Model(img_input, x)

    # set optimizer and compile the model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print(model.summary())

    # train
    history = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_split=0.1)

    # test the model
    score = model.evaluate(x_test, y_test)
    print("\nTest accuracy: %0.05f" % score[1])
    directory = "./vgg16_epochs" + str(nb_epoch)
    saveModel(model, directory + "/model_resnet59")

    plotTrainingHistory(history, directory)
    showErrors(model, x_test, y_test, directory)


def loadCifar_10Dataset():
    # Dataset of 50,000 32x32 color training images, labeled over 10
    # categories, and 10,000 test images.
    from keras.datasets import cifar10
    img_w = img_h = 32
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print('X_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    input_shape = x_train.shape[1:]
    print('input shape:', input_shape)

    return (x_train, y_train, x_test, y_test, input_shape)


def saveModel(model, model_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + ".hdf5")
    print("Saved model to disk")


def loadModel(model_name):
    # load json and create model
    json_file = open('model_num_' + model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights('model_num_' + model_name + ".hdf5")
    print("Loaded model from disk")
    return loaded_model


def plotTrainingHistory(history, directory):
    plt.plot(history.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['validation accuracy'], loc='upper left')
    plt.savefig(directory + "/graphic.png")
    print("evolution graphic saved: " + directory + "/acc_graphic.png")
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['validation loss'], loc='upper left')
    plt.savefig(directory + "/loss_graphic.png")
    print("evolution graphic saved: " + directory + "/loss_graphic.png")


def showErrors(model, x_test, y_test, directory):
    y_hat = model.predict_classes(x_test)
    y_test_array = y_test.argmax(1)
    pd.crosstab(y_hat, y_test_array)
    test_wrong = [im for im in zip(x_test, y_hat, y_test_array) if im[1] != im[2]]
    plt.figure(figsize=(32, 32))
    for ind, val in enumerate(test_wrong[:20]):
        plt.subplot(5, 4, ind + 1)
        plt.axis("off")
        plt.text(33, 15, labels[val[2]], fontsize=14, color='green')  # correct
        plt.text(33, 30, labels[val[1]], fontsize=14, color='red')  # predicted
        im = val[0].reshape(3, 32, 32).transpose(1, 2, 0)
        plt.imshow(im)
    plt.savefig(directory + "/errors.png")
    print("errors images saved: " + directory + "/errors.png")

if __name__ == '__main__':
    main()
