from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt
import pandas as pd

def main():
    ## load dataset
    x_train, y_train, x_test, y_test = loadMNISTdataset()

    ## create model
    model = Sequential()
    model.add(Dense(64, input_dim=28*28, init='uniform', activation='sigmoid'))
    model.add(Dense(10, init='uniform', activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    ## train the model
    history = model.fit(x_train, y_train, batch_size=64, nb_epoch=3, verbose=1,
    validation_split=0.1)
    plotTrainingHistory(history)

    ## test the model
    score = model.evaluate(x_test, y_test)
    print("\nTest accuracy: %0.05f" % score[1])
    showErrors(model, x_test, y_test)

def loadMNISTdataset():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 28*28).astype('float32') / 255
    x_test = x_test.reshape(10000, 28*28).astype('float32') / 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    return (x_train, y_train, x_test, y_test)

def plotTrainingHistory(history):
    plt.plot(history.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['validation accuracy'], loc = 'upper left')
    plt.show()

def showErrors(model, x_test, y_test):
    y_hat = model.predict_classes(x_test)
    y_test_array = y_test.argmax(1)
    pd.crosstab(y_hat, y_test_array)
    test_wrong = [im for im in zip(x_test,y_hat,y_test_array) if im[1] != im[2]]
    plt.figure(figsize=(15, 15))
    for ind, val in enumerate(test_wrong[:20]):
        plt.subplot(10, 10, ind + 1)
        im = 1 - val[0].reshape((28,28))
        plt.axis("off")
        plt.text(0, 0, val[2], fontsize=14, color='green') # correct
        plt.text(8, 0, val[1], fontsize=14, color='red')  # predicted
        plt.imshow(im, cmap='gray')
    plt.show()

if __name__=='__main__':
    main()
