import os
import time
from shutil import move

import keras as K
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import os
import json
from keras import backend as K
from numba import cuda







def train(dirpath="",epoch=2):
    import os
    import time
    from shutil import move

    import keras as K
    from keras.models import Model, load_model
    from keras.optimizers import Adam
    from keras.applications.vgg16 import VGG16, preprocess_input
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from keras.layers import Dense, Dropout, Flatten
    from pathlib import Path
    import tensorflow as tf
    import numpy as np
    from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
    import os
    import json
    from keras import backend as K
    from numba import cuda

    os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

    gpus = tf.config.experimental.list_physical_devices('GPU')

    tf.config.experimental.set_memory_growth(gpus[0], True)


    BATCH_SIZE = 32

    train_generator = ImageDataGenerator(rotation_range=90,
                                         brightness_range=[0.1, 0.7],
                                         width_shift_range=0.5,
                                         height_shift_range=0.5,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         validation_split=0.15,
                                         zoom_range=[0,1],
                                         preprocessing_function=preprocess_input)

    test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    class_subset = []

    with open(f"{dirpath}/meta.json", "r") as f:
        data = json.loads(f.read())
        class_subset = data["classes"]


    class_subset1 = sorted(os.listdir(f"{dirpath}/img/train"))[:2]



    print(class_subset)


    file_dir1 = f"{dirpath}/img/train"
    file_dir2 = f"{dirpath}/img/validation"


    traingen = train_generator.flow_from_directory(file_dir1,
                                                   target_size=(224, 224),
                                                   class_mode='categorical',
                                                   classes=class_subset,
                                                   subset='training',
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   seed=42)

    validgen = train_generator.flow_from_directory(file_dir1,
                                                   target_size=(224, 224),
                                                   class_mode='categorical',
                                                   classes=class_subset,
                                                   subset='validation',
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   seed=42)



    testgen = test_generator.flow_from_directory(file_dir2,
                                                 target_size=(224, 224),
                                                 class_mode=None,
                                                 classes=class_subset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 seed=42)


    def create_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
        """
        Compiles a model integrated with VGG16 pretrained layers

        input_shape: tuple - the shape of input images (width, height, channels)
        n_classes: int - number of classes for the output layer
        optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
        fine_tune: int - The number of pre-trained layers to unfreeze.
                    If set to 0, all pretrained layers will freeze during training
        """

        # Pretrained convolutional layers are loaded using the Imagenet weights.
        # Include_top is set to False, in order to exclude the model's fully-connected layers.
        conv_base = VGG16(include_top=False,
                          weights='imagenet',
                          input_shape=input_shape)

        # Defines how many layers to freeze during training.
        # Layers in the convolutional base are switched from trainable to non-trainable
        # depending on the size of the fine-tuning parameter.
        if fine_tune > 0:
            for layer in conv_base.layers[:-fine_tune]:
                layer.trainable = False
        else:
            for layer in conv_base.layers:
                layer.trainable = False

        # Create a new 'top' of the model (i.e. fully-connected layers).
        # This is 'bootstrapping' a new top_model onto the pretrained layers.
        top_model = conv_base.output
        top_model = Flatten(name="flatten")(top_model)
        top_model = Dense(4096, activation='relu')(top_model)
        top_model = Dense(1072, activation='relu')(top_model)
        top_model = Dropout(0.2)(top_model)
        output_layer = Dense(n_classes, activation='softmax')(top_model)

        # Group the convolutional base and new fully-connected layers into a Model object.
        model = Model(inputs=conv_base.input, outputs=output_layer)

        # Compiles the model for training.
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model


    # continue model training
    def resume_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=1):
        """
        Compiles a model integrated with VGG16 pretrained layers

        input_shape: tuple - the shape of input images (width, height, channels)
        n_classes: int - number of classes for the output layer
        optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
        fine_tune: int - The number of pre-trained layers to unfreeze.
                    If set to 0, all pretrained layers will freeze during training
        """

        # Pretrained convolutional layers are loaded using the Imagenet weights.
        # Include_top is set to False, in order to exclude the model's fully-connected layers.

        conv_base = model = load_model(f"{dirpath}/tl_model_v1.weights.best.h5")

        # Defines how many layers to freeze during training.
        # Layers in the convolutional base are switched from trainable to non-trainable
        # depending on the size of the fine-tuning parameter.
        if fine_tune > 0:
            for layer in conv_base.layers[:-fine_tune]:
                layer.trainable = False
        else:
            for layer in conv_base.layers:
                layer.trainable = False

        # Create a new 'top' of the model (i.e. fully-connected layers).
        # This is 'bootstrapping' a new top_model onto the pretrained layers.
        top_model = conv_base.output
        top_model = Flatten(name="flatten")(top_model)
        top_model = Dense(4096, activation='relu')(top_model)
        top_model = Dense(1072, activation='relu')(top_model)
        top_model = Dropout(0.2)(top_model)
        output_layer = Dense(n_classes, activation='softmax')(top_model)

        # Group the convolutional base and new fully-connected layers into a Model object.
        model = Model(inputs=conv_base.input, outputs=conv_base.output)

        # Compiles the model for training.
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    input_shape = (224, 224, 3)
    optim_1 = Adam(learning_rate=0.001)
    n_classes=2

    n_steps = traingen.samples // BATCH_SIZE
    n_val_steps = validgen.samples // BATCH_SIZE
    n_epochs = epoch

    # First we'll train the model without Fine-tuning


    if not os.path.isfile(f"{dirpath}/tl_model_v1.weights.best.h5"):
        vgg_model = create_model(input_shape, n_classes, optim_1)
    else:
        print("resuming model training")
        vgg_model = resume_model(input_shape,n_classes, optim_1)

    class YieldCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super(YieldCallback, self).__init__()
            self.start_time = time.time()

        def on_train_batch_end(self, batch, logs=None):
            elapsed_time = time.time() - self.start_time
            estimated_remaining_time = (elapsed_time / (batch + 1)) * (n_steps - batch - 1)
            train_acc = logs['accuracy']  # Update this line based on the desired accuracy metric
            print(f"TEST {estimated_remaining_time}  {train_acc}")
            yield estimated_remaining_time, train_acc

    class SaveBestModelCallback(tf.keras.callbacks.Callback):
        def __init__(self, save_path, monitor='loss'):
            super(SaveBestModelCallback, self).__init__()
            self.save_path = save_path
            self.monitor = monitor
            self.best_value = None
            self.epoch = 0

        def on_epoch_end(self, epoch, logs=None):
            self.epoch += 1
            current_value = logs.get(self.monitor)

            if current_value is None:
                return

            if self.best_value is None or current_value < self.best_value:
                self.best_value = current_value
                accuracy = logs.get('accuracy')

                print(f"\nsaving best value ({accuracy})")

                with open(f"{dirpath}/meta.json","r+") as f:
                    data = f.read()
                    data = json.loads(data)

                    data["accuracy"] = accuracy
                    data["has_model"] = True

                    f.seek(0)
                    f.write(json.dumps(data,indent=4))
                    f.truncate()
                self.model.save(self.save_path, overwrite=True)
                #move("temp.h5",self.save_path)


    # ModelCheckpoint callback - save best weights
    tl_checkpoint_1 = ModelCheckpoint(filepath=f'{dirpath}/tl_model_v1.weights.best.h5',
                                      save_best_only=True,
                                      monitor = 'loss',
                                      verbose=1,
                                      compress=False,
                                      save_freq='epoch',
                                      save_format='tf')

    save_best_model_callback = SaveBestModelCallback(save_path=f'{dirpath}/tl_model_v1.weights.best.h5', monitor='val_loss')

    # EarlyStopping
    early_stop = EarlyStopping(monitor='loss',
                               patience=10,
                               restore_best_weights=True,
                               mode='min')

    yield_callback = YieldCallback()



    vgg_history = vgg_model.fit(traingen,
                                batch_size=BATCH_SIZE,
                                epochs=n_epochs,
                                validation_data=validgen,
                                steps_per_epoch=n_steps,
                                validation_steps=n_val_steps,
                                callbacks=[save_best_model_callback , early_stop, yield_callback],
                                verbose=1)
    del vgg_model

    #sess = tf.Session()
    #if sess is not None:
    #    sess.close()
    K.clear_session()
    tf.compat.v1.reset_default_graph()


