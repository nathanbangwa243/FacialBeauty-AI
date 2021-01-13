import tensorflow as tf

from datetime import datetime

import os
import shutil
import pickle

from .config import *

from . import loadData



class CustomCallback(tf.keras.callbacks.Callback):

    def __init__(self, cwd, targetFolder, modelRef):

        assert os.path.exists(cwd)
        assert isinstance(targetFolder, str)
        assert os.path.exists(targetFolder)

        self.targetFolder = targetFolder
        self.modelRef = modelRef
        self.cwd = cwd

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("\nEnd epoch {} of training; got log keys: {}".format(epoch, keys))

        # metrics
        valLoss = logs['val_loss']

        if 'val_acc' in logs:
            valAcc = logs['val_acc']
        
        elif 'val_accuracy' in logs:
            valAcc = logs['val_accuracy']

        try:
            # swap cwd
            os.chdir(self.targetFolder)
            # save model
            modelFile = f"model-{datetime.now().timestamp()}-{epoch}-{valLoss:.3f}-{valAcc:.3f}.h5"
            
            print("\nSaving model : '{}'\n".format(modelFile))


            self.model.save(modelFile)

        except Exception as error:
            print(error)

        finally:
            # swap cwd
            os.chdir(self.cwd)


def makeFolder(targetPath):
    if not os.path.isdir(targetPath):
        os.makedirs(targetPath)
    

def removeFolder(targetPath):
    if os.path.isdir(targetPath):
        shutil.rmtree(targetPath, ignore_errors=True)



class HistoryManager(object):
    def __init__(self, targetFile, data={}):
        assert isinstance(targetFile, str)

        self.filename = f"{targetFile}.data"

        self.data = data
    
    def save(self, targetObject=None):
        if targetObject != None:
            self.data = targetObject

        with open(self.filename, 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(self):
        with open(self.filename, 'rb') as handle:
            data = pickle.load(handle)
        
        self.data = data
        return data

def buildNaimishModel():
    naimishModel = tf.keras.models.Sequential([
        # First layer
        # Convolution2d1 to Convolution2d4 do not use zero
        # padding, have their weights initialized with random numbers drawn from uniform distribution
        tf.keras.layers.Conv2D(32, (4,4), input_shape=(96, 96, 1), padding='valid', activation='elu',
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1., maxval=1.)
        ),
        
        # Maxpooling2d1 to Maxpooling2d4 use a pool shape of (2, 2), with non-overlapping strides and no zero padding
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid"),
        tf.keras.layers.Dropout(0.1),

        # second layer
        tf.keras.layers.Conv2D(64, (3,3), padding='valid', activation='elu',
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1., maxval=1.)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid"),
        tf.keras.layers.Dropout(0.2),

        # Third layer
        tf.keras.layers.Conv2D(128, (2,2), padding='valid', activation='elu',
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1., maxval=1.)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid"),
        tf.keras.layers.Dropout(0.3),

        # Fourth layer
        tf.keras.layers.Conv2D(256, (1,1), padding='valid', activation='elu',
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1., maxval=1.)
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid"),
        tf.keras.layers.Dropout(0.4),

        # Flatten
        tf.keras.layers.Flatten(),

        # Dense 1
        tf.keras.layers.Dense(1000, activation='elu'),
        tf.keras.layers.Dropout(0.5),

        # Dense 2
        tf.keras.layers.Dense(1000, activation='elu'),
        tf.keras.layers.Dropout(0.6),

        # Dense 1
        tf.keras.layers.Dense(2, activation='elu'),

    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        amsgrad=False,
        name="Adam"
    )

    naimishModel.compile(
        optimizer=optimizer,
        loss=tf.keras.metrics.mean_squared_error,
        metrics=['accuracy']
    )

    return naimishModel


def trainModel(fkp):
    # load dataset
    trainingDataset, _ = loadData.loadData()

    # Train and test data
    trainDatas, validationDatas, trainLabels, validationLabels = loadData.splitTrainDataset(trainingDataset)

    # current work directory
    cwd = os.getcwd()

    # models folder
    modelsFolder = os.path.join(cwd, 'models')

    makeFolder(modelsFolder)

    # clone model
    modelClone = buildNaimishModel()

    # FKP Folder
    fkpFolder = os.path.join(modelsFolder, str(f"FKP{fkp}"))

    removeFolder(fkpFolder)

    makeFolder(fkpFolder)

    # Logs dir

    logDir = os.path.join(fkpFolder, 'logs')
    makeFolder(logDir)
    
    modelFile = 'model.{epoch:02d}-{val_loss:.2f}.h5'
    
    # define callbacks

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=PATIENCE),
        # tf.keras.callbacks.ModelCheckpoint(filepath=modelFile),
        tf.keras.callbacks.TensorBoard(log_dir=logDir),
        CustomCallback(cwd=cwd, targetFolder=fkpFolder, modelRef=modelClone)

    ]

    history = modelClone.fit(
        trainDatas,
        trainLabels[:, fkp, :],
        validation_data = (validationDatas, validationLabels[:, fkp, :]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=my_callbacks,
        verbose=1
    )

    # save history
    historyFile = os.path.join(fkpFolder, 'history')
    history = HistoryManager(historyFile, history.history)
    history.save()



