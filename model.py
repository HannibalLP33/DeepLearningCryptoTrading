import tensorflow as tf
from keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, Dense,Activation, Input, BatchNormalization, MaxPooling1D, Add



class DLModels:

    def __init__(self, **kwargs):
        ###2023Sep15_Models###
        self.input_shape = tuple(kwargs.get('input_shape',(0,0)))
        self.num_layers = kwargs.get('num_layers', 0)
        self.num_filters = kwargs.get('num_filters', 0)
        self.dilation_rates = kwargs.get('dilation_rates', [])
        self.num_classes = kwargs.get('num_classes', 0)

    def model_1(self):
        inputs = Input(shape=self.input_shape)

        skip_connections = []
        x = inputs
        for dilation_rate in self.dilation_rates:
            x = Conv1D(self.num_filters, kernel_size=4, dilation_rate=dilation_rate, padding='causal')(x)
            x = Activation('relu')(x)
            x = BatchNormalization()(x)
            skip_connections.append(x)

        # Combine skip connections
        if len(skip_connections) > 1:
            x = Add()(skip_connections)

        # Max Pooling Layer for Downsampling
        x = MaxPooling1D(pool_size=2)(x)
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

        for _ in range(self.num_layers):
            x = Dense(256, activation='relu')(x)
            x = BatchNormalization()(x)

        outputs = Dense(self.num_classes, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model