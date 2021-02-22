from tensorflow import keras
from Data_loading_fns import *
import tensorflow as tf

#%%
class ResidualUnit(keras.layers.Layer):

    def __init__(self, filters=1, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv3D(filters, (3,3,3), strides=strides,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv3D(filters, (3,3,3), strides=1,
                                padding="same", use_bias=False),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv3D(filters, (1,1,1), strides=strides,
                                    padding="same", use_bias=False),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

    def get_config(self):
        base_config = super(ResidualUnit, self).get_config()
        return base_config
#%%

N_EPOCHS = 1000
BATCH_SIZE = 32
N_TRAINING_SAMPLES = 500
N_TESTING_SAMPLES = 125
STEPS_PER_EPOCH_TRAINING = N_TRAINING_SAMPLES // BATCH_SIZE
STEPS_PER_EPOCH_VALIDATION = N_TESTING_SAMPLES // BATCH_SIZE

training_set = train_input_fn('train.tfrecord', batch_size=BATCH_SIZE, num_epochs=N_EPOCHS)
validation_set = validation_input_fn('validation.tfrecord', batch_size=BATCH_SIZE)

#%% Build Model.
# mirrored_strategy = tf.distribute.MirroredStrategy()
# with mirrored_strategy.scope():
model = keras.models.Sequential()
model.add(keras.layers.Conv3D(64, (7,7,7), strides=(2,2,2), padding="same",
                              use_bias=False, input_shape=[91,109,91,1]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool3D(pool_size=(3,3,3), strides=(2,2,2), padding="same"))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters

model.add(keras.layers.GlobalAveragePooling3D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation="sigmoid"))

# model = keras.utils.multi_gpu_model(model, gpus=2)

es = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='auto', patience=200)
mc = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)
tb = keras.callbacks.TensorBoard(log_dir='./logs', write_images=True, write_graph=True)

model.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy', tf.keras.metrics.AUC()])
#%%

history = model.fit(training_set, steps_per_epoch=STEPS_PER_EPOCH_TRAINING,
                 epochs=N_EPOCHS, validation_data=validation_set,
                 validation_steps=STEPS_PER_EPOCH_VALIDATION, callbacks=[tb, es, mc])
