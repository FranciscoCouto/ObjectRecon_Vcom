from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

def main():
    ## load dataset
    x_train, y_train, x_test, y_test, input_shape = loadMNISTdataset()

    ## create model
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],border_mode='valid',input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

    ## train the model
    history = model.fit(x_train, y_train, batch_size=64, nb_epoch=3, verbose=1, validation_split=0.1)
   
    ## test the model
    score = model.evaluate(x_test, y_test)
    print("\nTest accuracy: %0.05f" % score[1])

def loadMNISTdataset():
    from keras.datasets import mnist
    img_w = img_h = 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_dim_ordering() == 'th':
        x_train = x_train.reshape(x_train.shape[0], 1, img_w, img_h)
        x_test = x_test.reshape(x_test.shape[0], 1, img_w, img_h)
        input_shape = (1, img_w, img_h)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_w, img_h, 1)
        x_test = x_test.reshape(x_test.shape[0], img_w, img_h, 1)
        input_shape = (img_w, img_h, 1)
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    return (x_train, y_train, x_test, y_test, input_shape)

if __name__=='__main__':
    main()
