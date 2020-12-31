import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data):
        self.data = data
        self.on_epoch_end()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self.data[self.indexes[index],:]
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        np.random.shuffle(self.indexes)
        print(self.indexes[:5])

def get_metric(max_T: int):
    def basic_metric(y_true, y_pred):
        y_true = tf.squeeze(y_true, axis=1)
        y_pred = tf.squeeze(y_pred, axis=1)
        
        input_size = tf.cast(tf.size(y_pred), tf.float32)
        T = tf.math.minimum(input_size, max_T)
        
        c = tf.argsort(y_pred, direction='DESCENDING')
        y_true = tf.gather(y_true, c)

        dcg = get_dcg(y_true, T)
        idcg = get_idcg(y_true, T)
        ndcg = dcg / idcg
        return ndcg
    return basic_metric


def get_loss_function(ltype):
    if ltype == 'ranknet':
        return get_loss_ranknet()
    elif ltype == 'lambdarank':
        return get_loss_lambdarank()


def get_loss_ranknet():
    @tf.function()
    def loss_function(s, h):
        S_ij = tf.cast(tf.math.sign(s - tf.squeeze(s, 1)), tf.float32)
        delta_h = tf.cast(h - tf.squeeze(h, 1), tf.float32)
        P_ij = (1 / 2) * (1 + S_ij)
        C = tf.nn.sigmoid_cross_entropy_with_logits(logits=delta_h, labels=P_ij)
        return C
    return loss_function


def get_loss_lambdarank():
    @tf.custom_gradient
    def loss_function(s, h):
        S_ij = tf.cast(tf.math.sign(s - tf.squeeze(s, 1)), tf.float32)
        delta_h = tf.cast(h - tf.squeeze(h, 1), tf.float32)
        delta_ndcg = tf.stop_gradient(tf.abs(get_delta_ndcg(s, h)))
        P_ij = (1 / 2) * (1 + S_ij)
        C = tf.nn.sigmoid_cross_entropy_with_logits(logits=delta_h * delta_ndcg, labels=P_ij)
        def grad(dy):
            return s * dy, loss_lambdarank_dev(s, h)
        return C, grad
    return loss_function

@tf.function()
def loss_lambdarank_dev(s, h):
    S_ij = tf.cast(tf.math.sign(s - tf.squeeze(s, 1)), tf.float32)
    delta_h = tf.cast(h - tf.squeeze(h, 1), tf.float32)
    C_term_dev = (1 - S_ij) / 2 - 1 / (1 + tf.math.exp(delta_h))
    C_term_dev *= tf.abs(get_delta_ndcg(s, h))
    C_term_dev = tf.reduce_sum(C_term_dev, 1, keepdims=True)
    return C_term_dev

# def get_loss_lambdarank():
#     # @tf.function()
#     def loss_function(s, h):
#         S_ij = tf.cast(tf.math.sign(s - tf.squeeze(s, 1)), tf.float32)
#         delta_h = tf.cast(h - tf.squeeze(h, 1), tf.float32)
#         # дельта NDCG
#         delta_ndcg = tf.stop_gradient(tf.abs(get_delta_ndcg(s, h)))
#         delta_h = delta_h * delta_ndcg
#         P_ij = (1 / 2) * (1 + S_ij)
#         C = tf.nn.sigmoid_cross_entropy_with_logits(logits=delta_h, labels=P_ij)
#         return C
#     return loss_function


def get_dcg(y_true, T):
    y_true = tf.slice(y_true, [0], [T])
    index_range = tf.range(T, dtype=tf.float32) + 1.0
    discount = 1.0 / tf.math.log(index_range + 1.0)
    dcg = (2 ** y_true - 1.0) * discount
    dcg = tf.reduce_sum(dcg)
    return dcg


def get_idcg(y_true, T):
    ideal_y_true_sorted = tf.sort(y_true, direction='DESCENDING')
    ideal_y_true_sorted = tf.slice(ideal_y_true_sorted, [0], [T])
    
    index_range = tf.range(T, dtype=tf.float32) + 1.0
    dicsount = 1.0 / tf.math.log(index_range + 1.0)
    
    idcg = (2 ** ideal_y_true_sorted - 1.0) * dicsount
    idcg = tf.reduce_sum(idcg)
    return idcg


def get_idcg_simple(y_true):
    y_true = tf.squeeze(y_true, 1)
    y_true = tf.sort(y_true, direction='DESCENDING')
    
    input_size = tf.cast(tf.size(y_true), tf.float32)
    index_range = tf.range(input_size, dtype=tf.float32) + 1.0
    
    idcg = (2 ** y_true - 1.0) / tf.math.log(index_range + 1.0)
    idcg = tf.reduce_sum(idcg)
    return idcg


def get_delta_ndcg(s_tf, h_tf):
    s_diff = 2 ** s_tf - tf.squeeze(2 ** s_tf, 1)
    
    h_sq = tf.squeeze(h_tf, 1)
    h_sq_sort = tf.sort(h_sq, axis=0)
    h_index = tf.searchsorted(h_sq_sort, h_sq)[::-1]
    h_index = tf.cast(h_index, tf.float32)
    h_index = 1 / tf.math.log(2 + h_index)
    h_diff = tf.expand_dims(h_index, 1) - h_index
    
    N = 1.0 / get_idcg_simple(s_tf)
    
    delta_ndcg = N * s_diff * h_diff
    return delta_ndcg


def get_optimizer(otype):
    if otype == "adam":
        return "adam"
    elif otype == "adam_01":
        return tf.optimizers.Adam(learning_rate=0.1)
    elif otype == "adam_005":
        return tf.optimizers.Adam(learning_rate=0.05)
    elif otype == "adam_001":
        return tf.optimizers.Adam(learning_rate=0.01)
    elif otype == "adam_decay":        
        STEPS_PER_EPOCH = data_generator_train.data.shape[0]
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
          0.001,
          decay_steps=STEPS_PER_EPOCH*1000,
          decay_rate=1,
          staircase=False)
        return tf.keras.optimizers.Adam(lr_schedule)


def get_callbacks(model_name, monitor='val_loss', patience=10):
    callbacks= []

    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            f"./models/{model_name}", save_best_only=True, save_weights_only=True
        )
    )
    if patience is not None or monitor is not None:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor, patience=patience
            )
        )
#         keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001),
    return callbacks

def get_model(input_dim, dims, dropout=None):
    layers = [
        tf.keras.layers.Dense(dims[0], input_shape=(input_dim,), activation='relu'),
    ]
    if dropout is not None:
        layers.append(
            tf.keras.layers.Dropout(dropout),
        )

    if len(dims) > 1:
        for dim in dims[1:]:
            layers.append(
                tf.keras.layers.Dense(dim, activation='relu'),
            )
    layers.append(
        tf.keras.layers.Dense(1),
    )
    model = tf.keras.models.Sequential(layers)
    return model
