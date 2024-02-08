import tensorflow as tf
import numpy as np
from tqdm import tqdm
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import TimeSeriesSplit
from common_funcs import ColorPrint


class CustomTrainer:
    def __init__(self, model=None, data=None, labels=None, checkpoint_name=None):
        self.data = data
        self.labels = labels
        
        tss = TimeSeriesSplit(n_splits=5)
        train_index, val_index = list(tss.split(data))[-1] #80/20 split
        X_train_list, X_val_list, y_train_list, y_val_list = [],[],[],[]

        for i in tqdm(train_index, desc= ColorPrint("Processing Training Set", "Blue")):
            X_train_list.append(self.data[i])
            y_train_list.append(self.labels[i])
        for i in tqdm(val_index, desc = ColorPrint("Processing Val Set", "Blue")):
            X_val_list.append(self.data[i])
            y_val_list.append(self.labels[i])

        self.X_train = np.array(X_train_list).astype('float32')
        self.X_val = np.array(X_val_list).astype('float32')
        self.y_train = np.array(y_train_list).astype('int32')
        self.y_val = np.array(y_val_list).astype('int32')

        print('=' * 100)
        print('Training Values')
        for i, values in enumerate([self.y_train, self.y_val]):
            unique, freqs = np.unique(values, return_counts=True)
            total = sum(freqs)
            for value, freq in zip(unique, freqs):
                print(f'{value}    {freq} ({freq/total*100:.2f}%)')
            if i == 0:
                print("\nValidation Values\n")
        print('=' * 100)               

        self.model = model
        self.cp_name = checkpoint_name
        self.best_loss = float("inf")
        self.lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-50, verbose = 0)
        self.optimizer = Adam(learning_rate = 1e-3)

    #Create and Register Metrics
    def precision(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision_value = true_positives / (predicted_positives + K.epsilon())
        return precision_value
    
    def recall(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall_value = true_positives / (possible_positives + K.epsilon())
        return recall_value

    def f1_score(self, y_true, y_pred):
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        f1_score_value = (2 * precision * recall) / (precision + recall + K.epsilon())
        return f1_score_value
    
    def f1_score_metric(self):
        def metric(y_true, y_pred):
            return self.f1_score(y_true, y_pred)
        metric.__name__ = "f1_score"
        return metric
    
    def register_custom_metrics(self):
        tf.keras.metrics.recall = self.recall
        tf.keras.metrics.f1_score = self.f1_score
        tf.keras.metrics.precision = self.precision
        
    def checkpoint(self, cp_name):
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=cp_name,
            save_weights_only=False,
            monitor='val_f1_score',
            mode='max',
            save_best_only=True)
        return model_checkpoint_callback
    
    #Preprocess function when creating datasets
    def preprocess_fn(self, x, y):
        # Check the shape of y
        if len(y.shape) == 1:
            # If y has only one dimension, expand it to have shape (batch_size, 1)
            y = tf.expand_dims(y, axis=-1)

        return x, y
        
    def create_dataset(self, batch_size):
        train = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        train = train.shuffle(buffer_size=100)
        train = train.batch(batch_size)
        train = train.map(self.preprocess_fn)  
        train = train.prefetch(tf.data.experimental.AUTOTUNE)

        val = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))
        val = val.shuffle(buffer_size=100)
        val = val.batch(batch_size)
        val = val.map(self.preprocess_fn)  
        val = val.prefetch(tf.data.experimental.AUTOTUNE)

        return train, val


    def compile_model(self):
        self.model.compile(optimizer=Adam(learning_rate = 1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7,clipnorm = 1.0), 
                           loss=BinaryCrossentropy(), 
                           metrics=[self.precision, self.recall, self.f1_score_metric()])
    
    def he_initializer(self, shape):
        """
        He weight initialization. Good For ReLu Layers
        """
        fan_in = shape[0]
        limit = tf.sqrt(2.0 / fan_in)
        return tf.random.normal(shape, mean=0.0, stddev=limit)

    def train_model(self, num_epochs, batch_size):
        with tf.device('/device:GPU:0'):
            self.register_custom_metrics()
            train, val  = self.create_dataset(batch_size = batch_size)
            if (self.model.optimizer is None) and ('pretrained' in self.cp_name):
                print(ColorPrint('Compiling Model for Pretraining Now...', "Blue"))
                for layer in self.model.layers:
                    if hasattr(layer, 'weights'):
                        layer.set_weights([self.he_initializer(w.shape) for w in layer.get_weights()])
                self.compile_model()
            elif (self.model.optimizer is None) and ('full_models' in self.cp_name):
                print(ColorPrint('Compiling Model for Fine-Tuning Now...', "Blue"))
                self.compile_model()
            self.model.fit(train, epochs = num_epochs, validation_data = val, verbose = 1, callbacks = [self.lr_schedule, self.checkpoint(self.cp_name)])