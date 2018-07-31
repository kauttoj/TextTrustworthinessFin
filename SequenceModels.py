import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras import regularizers
from keras.models import Sequential,Model,Input
from keras.layers import Dense, Activation, Dropout, Flatten,InputLayer,Merge,SpatialDropout1D,CuDNNLSTM,BatchNormalization,TimeDistributed,CuDNNGRU,CuDNNLSTM
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D,LSTM,Bidirectional,Layer,GRU,Masking,GlobalAveragePooling1D,MaxPooling1D
from keras.layers.merge import concatenate
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping,Callback,ModelCheckpoint
from keras.optimizers import RMSprop,Adam
from keras import initializers,constraints
from keras.utils import plot_model
from keras.regularizers import l1_l2,l2
import os
import Utils

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def use_keras_CPU(num_cores=1):
    import tensorflow as tf
    from keras import backend as K
    num_CPU = 1
    num_GPU = 0
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
            device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.Session(config=config)
    K.set_session(session)

class NonMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttLayer(Layer):
    def __init__(self, regularizer=None,return_attention=False, **kwargs):
        self.regularizer = regularizer
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight(name='W', shape=(input_shape[-1],), initializer='normal', trainable=True,
                                 regularizer=self.regularizer)
        self.b = self.add_weight(name='b', shape=(input_shape[1],), initializer='normal', trainable=True,
                                 regularizer=self.regularizer)
        self.u = self.add_weight(name='u', shape=(input_shape[1],), initializer='normal', trainable=True,
                                 regularizer=self.regularizer)
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        #eij = K.dot(K.tanh(K.dot(x, self.W) + self.b), K.expand_dims(self.u))
        eij = dot_product(K.tanh(dot_product(x, self.W) + self.b), K.permute_dimensions(self.u,1))
        a = K.exp(eij)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(a)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = {}
        base_config = super(AttLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return None

class Attention(Layer):
    def __init__(self,regularizer=None,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Note: The layer has been tested with Keras 1.x
        Example:

            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2 - Get the attention scores
            hidden = LSTM(64, return_sequences=True)(words)
            sentence, word_scores = Attention(return_attention=True)(hidden)
        """
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')
        self.regularizer=regularizer

        if regularizer is not None:
            self.W_regularizer = regularizer
            self.b_regularizer = regularizer
        else:
            self.W_regularizer = regularizers.get(W_regularizer)
            self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]

def batch_generator(X, y, batch_size):
    """Primitive batch generator
    """
    size = X.shape[0]
    X_copy = X.copy()
    y_copy = y.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X_copy[i:i + batch_size], y_copy[i:i + batch_size]
            i += batch_size
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            continue


def expand_multiple_dims(x, axes, name="expand_multiple_dims"):
  """
  :param tf.Tensor x:
  :param list[int]|tuple[int] axes: after completion, tf.shape(y)[axis] == 1 for axis in axes
  :param str name: scope name
  :return: y where we have a new broadcast axis for each axis in axes
  :rtype: tf.Tensor
  """
  with tf.name_scope(name):
    for i in sorted(axes):
      x = tf.expand_dims(x, axis=i, name="expand_axis_%i" % i)
    return x


def dimshuffle(x, axes, name="dimshuffle"):
  """
  Like Theanos dimshuffle.
  Combines tf.transpose, tf.expand_dims and tf.squeeze.

  :param tf.Tensor x:
  :param list[int|str]|tuple[int|str] axes:
  :param str name: scope name
  :rtype: tf.Tensor
  """
  with tf.name_scope(name):
    assert all([i == "x" or isinstance(i, int) for i in axes])
    real_axes = [i for i in axes if isinstance(i, int)]
    bc_axes = [i for (i, j) in enumerate(axes) if j == "x"]
    if x.get_shape().ndims is None:
      x_shape = tf.shape(x)
      x = tf.reshape(x, [x_shape[i] for i in range(max(real_axes) + 1)])  # will have static ndims
    assert x.get_shape().ndims is not None

    # First squeeze missing axes.
    i = 0
    while i < x.get_shape().ndims:
      if i not in real_axes:
        x = tf.squeeze(x, axis=i)
        real_axes = [(j if (j < i) else (j - 1)) for j in real_axes]
      else:
        i += 1

    # Now permute.
    assert list(sorted(real_axes)) == list(range(x.get_shape().ndims))
    if real_axes != list(range(x.get_shape().ndims)):
      x = tf.transpose(x, real_axes)

    # Now add broadcast dimensions.
    if bc_axes:
      x = expand_multiple_dims(x, bc_axes)
    assert len(axes) == x.get_shape().ndims
    return x

def attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas

# def flatten_sentences(X_in):
#     X = []
#     for k1 in range(0, len(X_in)):
#         X.append([])
#         for k2 in range(0, len(X_in[k1])):
#             X[k1] += X_in[k1][k2]
#         #X[k1] = X[k1], dtype=np.int32)
#     return X
#
# def extract_metadata(X,scaler=None):
#     if scaler==None:
#         scaler = MaxAbsScaler()
#         scaler.fit(X)
#     X = scaler.transform(X)
#     L=X.shape[0]
#     XX = []
#     for i in range(L):
#         XX.append(X[i,:])
#
#     return XX,scaler

def fasttext(X_in, Y_in,Params):#,X_in_test = None,Y_in_test=None):

    max_seq_len = X_in[0].shape[1]

    max_features = max([max(x) for x in X_in[0]])

    assert 200 < max_seq_len < 5000,'too long sequences!'
    assert all(0<Y_in) and all(Y_in<1),'target not suitable for sigmoid!'

    input_data = X_in[0]

    np.random.seed(0)

    y_train = Y_in  # train_df[label_names].values

    # training params
    batch_size = Params['Algorithm'][1]['batch']
    num_epochs = Params['Algorithm'][1]['epochs']

    #K.clear_session()

    with tf.Session(config=tf.ConfigProto(
            intra_op_parallelism_threads=1, \
            inter_op_parallelism_threads=1, allow_soft_placement=True, \
            device_count={'CPU': 1, 'GPU': 0})) as sess:

        K.set_session(sess)

        #use_keras_CPU(num_cores=1)

        #model = Sequential()

    # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        first_input = Input(shape=(max_seq_len,), dtype='int32', name='ngram-input')

        layer = Embedding(max_features,
                            Params['Algorithm'][1]['embedding_dim'],
                            input_length=max_seq_len,mask_zero=True,embeddings_regularizer=l2(Params['Algorithm'][1]['emb_regularization']))(first_input)

        # we add a GlobalAveragePooling1D, which will average the embeddings
        # of all words in the document
        layer = SpatialDropout1D(Params['Algorithm'][1]['dropout'])(layer)
        if Params['Algorithm'][1]['pooling_type']=='average':
            layer = NonMasking()(layer)
            layer = GlobalAveragePooling1D()(layer)
        elif Params['Algorithm'][1]['pooling_type']=='attention':
            layer = Attention()(layer)
        else:
            raise(Exception('unknown pooling type'))
        #model.add(Dropout(Params['Algorithm'][1]['dropout']))
        first_output = layer

        second_input = None
        if len(X_in) > 1:
            #alpha = 0.70

            X_meta_train = X_in[1]

            second_input = Input(shape=(X_meta_train.shape[1],), dtype='float32', name='metadata-input')
            # layer = Dropout(0.10, name='metadata-dropout')(second_input)
            second_output = second_input#Dense(1, activation='sigmoid', name='metadata-dense',kernel_regularizer=reg)(second_input)

            #first_output = Dense(1,activation='sigmoid')(first_output)

            layer = concatenate([first_output,second_output])

            input_data = X_in

        delta = Params['Algorithm'][1]['meta_regularization']
        reg = l2(delta)

        output = Dense(1, name='output', activation='sigmoid',kernel_regularizer=reg)(layer)

        if second_input is None:
            model = Model(inputs=first_input, outputs=output)
        else:
            model = Model(inputs=[first_input,second_input], outputs=output)

        # We project onto a single unit output layer, and squash it with a sigmoid:
        #model.add(Dense(1, activation='sigmoid'))

        #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
        #model.summary()

        mem = Utils.get_model_memory_usage(batch_size, model)
        if mem>6:
            print('!!!!! Model requires %f gigabytes of memory !!!!!!!' % mem)

        hist = model.fit(input_data, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0, shuffle=True, verbose=0)  # ,callbacks=callbacks)

    return model

def attention_weight(X_in, Y_in,Params):#,X_in_test = None,Y_in_test=None):

    max_seq_len = X_in[0].shape[1]

    max_features = max([max(x) for x in X_in[0]])

    assert 200 < max_seq_len < 5000,'too long sequences!'
    assert all(0<Y_in) and all(Y_in<1),'target not suitable for sigmoid!'

    input_data = X_in[0]

    np.random.seed(0)

    y_train = Y_in  # train_df[label_names].values

    # training params
    batch_size = Params['Algorithm'][1]['batch']
    num_epochs = Params['Algorithm'][1]['epochs']

    K.clear_session()

# we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    first_input = Input(shape=(max_seq_len,), dtype='int32', name='padded_1gram_input')

    layer = Embedding(max_features+1,Params['W_embedding_matrix'].shape[1],weights=[Params['W_embedding_matrix']],trainable=False,
                        input_length=max_seq_len,mask_zero=True)(first_input)

    first_output = Attention()(layer)

    second_input = None
    if len(X_in) > 1:
        #alpha = 0.70

        X_meta_train = X_in[1]

        second_input = Input(shape=(X_meta_train.shape[1],), dtype='float32', name='metadata-input')
        # layer = Dropout(0.10, name='metadata-dropout')(second_input)
        second_output = second_input#Dense(1, activation='sigmoid', name='metadata-dense',kernel_regularizer=reg)(second_input)

        #first_output = Dense(1,activation='sigmoid')(first_output)

        layer = concatenate([first_output,second_output])

        input_data = X_in

    delta = Params['Algorithm'][1]['meta_regularization']
    reg = l2(delta)

    output = Dense(1, name='output',activation='sigmoid',kernel_regularizer=reg)(layer)

    if second_input is None:
        model = Model(inputs=first_input, outputs=output)
    else:
        model = Model(inputs=[first_input,second_input], outputs=output)

    # We project onto a single unit output layer, and squash it with a sigmoid:
    #model.add(Dense(1, activation='sigmoid'))

    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
    #model.summary()

    mem = Utils.get_model_memory_usage(batch_size, model)
    if mem>6:
        print('!!!!! Model requires %f gigabytes of memory !!!!!!!' % mem)

    hist = model.fit(input_data, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0, shuffle=True, verbose=0)  # ,callbacks=callbacks)

    return model


def SimpleCNN(X_in, Y_in,Params):#,X_in_test = None,Y_in_test=None):

    max_seq_len = X_in.shape[1]
    assert 200 < max_seq_len < 3000,'sequence length too long or short!'
    input_data = X_in

    np.random.seed(0)

    y_train = Y_in  # train_df[label_names].values

    # training params
    batch_size = Params['Algorithm'][1]['batch']
    num_epochs = Params['Algorithm'][1]['epochs']
    embedding_matrix = Params['W_embedding_matrix']
    nb_words = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    K.clear_session()

    graph_in = Input(shape=(max_seq_len, embed_dim))
    convs = []
    for fsz in Params['Algorithm'][1]['filtersize']:
        conv = Conv1D(filters=Params['Algorithm'][1]['filtercount'], kernel_size=fsz, padding='valid', activation='relu', strides=1)(graph_in)
        conv = BatchNormalization()(conv)
        pool = MaxPooling1D(pool_size=2)(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)

    out = concatenate(convs)
    graph = Model(inputs=graph_in, outputs=out)

    model = Sequential()
    model.add(
        Embedding(nb_words,
                  embed_dim,
                  weights=[embedding_matrix],mask_zero=False,
                  input_length=max_seq_len,
                  trainable=False,)
    )
    model.add(SpatialDropout1D(Params['Algorithm'][1]['dropout']))
    model.add(graph)
    model.add(BatchNormalization())
    for k in range(0,Params['Algorithm'][1]['densenodes']):
        model.add(Dense(Params['Algorithm'][1]['densenodes'][k]))
        model.add(Dropout(Params['Algorithm'][1]['dropout']))
        model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, clipnorm=10)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    #model.summary()

    mem = Utils.get_model_memory_usage(batch_size, model)
    if mem>6:
        print('!!!!! Model requires %f gigabytes of memory !!!!!!!' % mem)

    hist = model.fit(input_data, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0, shuffle=True, verbose=0)  # ,callbacks=callbacks)

    return model

def SimpleCNN2LSTM(X_in, Y_in,Params):#,X_in_test = None,Y_in_test=None):

    #embedding_matrix = Params['W_embedding_matrix']
    MAX_NB_WORDS = Params['max_unique_words']
    EMBEDDING_DIM = Params['Algorithm'][1]['embedding_dim']
    # training params
    batch_size = Params['Algorithm'][1]['batch']
    num_epochs = Params['Algorithm'][1]['epochs']

    #-----------------------------------------------
    input_data = X_in[0]

    embedding_matrix = 2 * np.random.rand(MAX_NB_WORDS, EMBEDDING_DIM) - 1
    embedding_matrix[0] = 0 * embedding_matrix[0]

    if Params['Algorithm'][1]['emb_initializer'] == 'word2vec':
        doc_lst = []
        for i in range(input_data.shape[0]):
            for j in range(input_data.shape[1]):
                sent = []
                for k in range(input_data.shape[2]):
                    if input_data[i][j][k]!=0:
                        sent.append(str(input_data[i][j][k]))
                    else:
                        break
                doc_lst.append(sent)
        import gensim

        # use skip-gram
        word2vec_model = gensim.models.Word2Vec(doc_lst, min_count=1,size=EMBEDDING_DIM,sg=1,workers=os.cpu_count())
        embeddings_index = {}
        for word in word2vec_model.wv.vocab:
            coefs = np.asarray(word2vec_model.wv[word], dtype='float32')
            embeddings_index[int(word)] = coefs
        doc_lst=None

        unknown_count=0
        for word in range(MAX_NB_WORDS):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[word] = embedding_vector
            else:
                unknown_count += 1

        assert unknown_count==1,'should have only 1 unknown word (0=padding index)!'

    elif Params['Algorithm'][1]['emb_initializer']=='random':
        pass
    elif Params['Algorithm'][1]['emb_initializer']=='pretrained':
        embedding_matrix = Params['W_embedding_matrix']
    else:
        raise(Exception('Unknown embedding initialization method'))

    max_seq_len = input_data.shape[1]
    assert 200 < max_seq_len < 3000,'sequence length too long or short!'

    np.random.seed(0)

    y_train = Y_in  # train_df[label_names].values

    nb_words = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    K.clear_session()

    graph_in = Input(shape=(max_seq_len, embed_dim))

    model = Sequential()
    model.add(
        Embedding(nb_words,
                  embed_dim,
                  weights=[embedding_matrix],mask_zero=False,
                  input_length=max_seq_len,
                  trainable=True,)
    )
    model.add(SpatialDropout1D(Params['Algorithm'][1]['dropout']))
    model.add(Conv1D(Params['Algorithm'][1]['filtercount'],
                     kernel_size=Params['Algorithm'][1]['filtersize'],
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(CuDNNLSTM(Params['Algorithm'][1]['rnn_units'],return_sequences=True)))
    model.add(Attention())
    model.add(Dropout(Params['Algorithm'][1]['dropout']))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, clipnorm=10)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    #model.summary()

    mem = Utils.get_model_memory_usage(batch_size, model)
    if mem>6:
        print('!!!!! Model requires %f gigabytes of memory !!!!!!!' % mem)

    hist = model.fit(input_data, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0, shuffle=True, verbose=0)  # ,callbacks=callbacks)

    return model

def SimpleLSTM(X_in, Y_in,Params):#,X_in_test = None,Y_in_test=None):

    max_seq_len = X_in[0].shape[1]
    assert 200 < max_seq_len < 3000
    input_data = X_in#[0]['FLAT']

    np.random.seed(666)

    #y_mean = np.mean(Y_in)
    y_train = Y_in  # train_df[label_names].values
    #y_max = np.max(np.abs(y_train))

    #test_data=None
    #if X_in_test is not None:
    #    Y_in_test = (Y_in_test-y_mean)*y_max
    #    test_data = (sequence.pad_sequences(flatten_sentences(X_in_test[0]), maxlen=max_seq_len),Y_in_test)

    # training params
    batch_size = 128
    num_epochs = Params['Algorithm'][1]['epochs']
    embedding_matrix = Params['W_embedding_matrix']
    nb_words = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    # CNN architecture
    print("training BiLSTM ...")

    first_input = Input(shape=(max_seq_len,),dtype='int32',name='text-input')
    layer = Embedding(nb_words,
                        embed_dim,
                        mask_zero=True,
                        weights=[embedding_matrix],
                        input_length=max_seq_len,
                        trainable=False,name='embedding')(first_input)
    layer = SpatialDropout1D(0.20,name='embedding-dropout')(layer)
    layer = Bidirectional(LSTM(120, dropout=0.0, recurrent_dropout=0.40,return_sequences=True,name='Bi-RNN'))(layer)
    layer = Dropout(0.40, name='LSTM-dropout')(layer)
    layer = NonMasking(name='LSTM-nonmasking')(layer) # remove mask information
    avg_pool = GlobalAveragePooling1D()(layer)
    #max_pool = GlobalMaxPooling1D()(layer)
    #first_output = concatenate([avg_pool, max_pool])

    #first_output = GlobalAveragePooling1D(name='LSTM-pooling')(layer)

    first_output = avg_pool

    layer = first_output

    second_input = None
    if len(X_in)>1:
        X_meta_train = X_in[1]
        second_input = Input(shape=(X_meta_train.shape[1],),dtype='float32',name='metadata-input')
        #layer = Dropout(0.10, name='metadata-dropout')(second_input)
        second_output = Dense(20,activation='relu',name='metadata-dense')(second_input)
        layer = concatenate([first_output,second_output])

        #input_data_new = [input_data,X_meta_train]
        #input_data = input_data_new  # input_data = [

    #layer = Dense(60, activation='relu',name='merged-dense-first')(layer)
    #layer = Dense(20, activation='relu', name='merged-dense-second')(layer)
    #layer = Dropout(0.10,name='final-dense-dropout')(layer)
    output = Dense(1,name='output',activation='linear',bias_initializer='zeros')(layer)

    if second_input is None:
        model = Model(inputs=[first_input], outputs=output)
    else:
        model = Model(inputs=[first_input, second_input], outputs=output)

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, clipnorm=10)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])

    model.summary()
    plot_model(model,to_file='my_keras_model.png')

    model_path = 'my_model.h5'
    # define callbacks
    # prepare callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='max',
            verbose=1),
        ModelCheckpoint(
             model_path,
             monitor='val_loss',
             save_best_only=True,
             mode='max',
             verbose=0),
    ]

    hist = model.fit(input_data, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.15, shuffle=True, verbose=2)#,callbacks=callbacks)

    # fig = plt.figure(num=None, figsize=(10,9), dpi=80, facecolor='w', edgecolor='k')
    # ax1 = fig.add_subplot(3,1,1)
    # ax2 = fig.add_subplot(3,1,2)
    # ax3 = fig.add_subplot(3,1,3)
    # ax1.set_xlabel('Test sample')
    # ax3.set_xlabel('Epoch')
    # ax2.set_xlabel('Epoch')
    # ax1.set_ylabel('Test data')
    # ax2.set_ylabel('MSE ratio')
    # ax3.set_ylabel('R^2')
    #
    # train_rsquare=np.array(-1)
    # test_rsquare = np.array(-1)
    #
    # ax1.plot(range(1,len(test_data[-1])+1),test_data[-1],'bx-')
    #
    # test_mse=np.array((( np.ones(test_data[-1].shape)*np.mean(y_train) - test_data[-1]) ** 2).mean(axis=0))
    # train_mse=np.array((( np.ones(y_train.shape)*np.mean(y_train) - y_train) ** 2).mean(axis=0))
    #
    # for epoch in range(num_epochs):
    #
    #     if test_data is not None:
    #         hist = model.fit(input_data, y_train, batch_size=batch_size, epochs=1, callbacks=callbacks, validation_data=test_data, shuffle=True, verbose=0)
    #     else:
    #         hist = model.fit(input_data, y_train, batch_size=batch_size, epochs=1, callbacks=callbacks, validation_split=0.15, shuffle=True, verbose=0)
    #
    #     print('... EPOCH %i/%i complete' % (epoch+1,num_epochs))
    #
    #     if epoch>0:
    #         del ax1.lines[-1]
    #         del ax2.lines[-1]
    #         del ax2.lines[-1]
    #         del ax3.lines[-1]
    #         del ax3.lines[-1]
    #
    #     y_train_predicted = model.predict(input_data).flatten()
    #     R_train = np.corrcoef(y_train,y_train_predicted)[0,1]
    #     train_mse=np.append(train_mse,((y_train - y_train_predicted) ** 2).mean(axis=0))
    #
    #     y_test_predicted=model.predict(test_data[0]).flatten()
    #     test_mse=np.append(test_mse,((test_data[1] - y_test_predicted) ** 2).mean(axis=0))
    #     ax1.plot(range(1,len(y_test_predicted)+1),y_test_predicted,'r-')
    #     ax1.legend(['test target', 'predicted target'])
    #
    #     R = np.corrcoef(test_data[-1], y_test_predicted)[0, 1]
    #     ax1.set_title('R=%f, R^2=%f, MSE_ratio=%f' % (R,R**2,test_mse[-1]/test_mse[0]))
    #     ax2.plot(range(epoch+2),test_mse/test_mse[0],'ro-')
    #     ax2.plot(range(epoch+2),train_mse/train_mse[0], 'gx-')
    #     ax2.legend(['test', 'train'])
    #
    #     test_rsquare = np.append(test_rsquare,R**2)
    #     train_rsquare = np.append(train_rsquare, R_train ** 2)
    #     ax3.plot(range(1,epoch+2),test_rsquare[1:],'ro-')
    #     ax3.plot(range(1, epoch + 2), train_rsquare[1:], 'gx-')
    #     ax3.legend(['test','train'])
    #
    #     plt.pause(0.001)
    #     plt.tight_layout()
    #     plt.show(block=False)

    #fun = model.predict
    #model.predict = lambda X: fun(sequence.pad_sequences(flatten_sentences(X), maxlen=max_seq_len))

    #return y_train_predicted,y_test_predicted
    return model

def SimpleAttentionGRU(X_in, Y_in,Params,X_in_test = None,Y_in_test=None):
    max_seq_len = 0

    X = flatten_sentences(X_in)
    max_seq_len = max([len(x) for x in X])
    word_seq_train = X

    np.random.seed(0)

    y_train = Y_in  # train_df[label_names].values

    # pad sequences
    word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)

    word_seq_test=None
    if X_in_test is not None:
        word_seq_test = sequence.pad_sequences(flatten_sentences(X_in_test), maxlen=max_seq_len)
        y_test = Y_in_test

    # training params
    batch_size = 100
    num_epochs = Params['Algorithm'][1]['epochs']

    embedding_matrix = Params['W_embedding_matrix']

    nb_words = embedding_matrix.shape[0]

    #embedding_matrix = 2*np.random.rand(nb_words,200) - 1

    embed_dim = embedding_matrix.shape[1]

    # CNN architecture
    print("training AttentionGRU ...")

    embedding_layer = Embedding(nb_words, embed_dim, weights=[embedding_matrix], input_length=max_seq_len, trainable=False, name='embedding',mask_zero=True)
    sequence_input = Input(shape=(max_seq_len,), dtype='int32', name='input')
    embedded_sequences = embedding_layer(sequence_input)

    #masking_layer = Masking(mask_value=0, input_shape=(max_seq_len,),name='masking')(embedded_sequences)

    l_gru = Bidirectional(GRU(60, return_sequences=True,name='GRU'))(embedded_sequences)
    l_att = Attention(name='attention')(l_gru)
    l_dense = Dropout(0.15)(l_att)
    preds = Dense(1, activation='linear',name='output')(l_dense)
    model = Model(sequence_input, preds)

    model.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
    model.summary()

    # Because of the multi-label loss, we are using k-hot encoding of the output and sigmoid activations. As a result, the loss is binary cross-entropy.

    # In[ ]:

    # define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)
    callbacks_list = []  # [early_stopping]

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1,1)
    ax2 = fig.add_subplot(2, 1,2)
    ax1.plot(y_test)
    test_mse=[(( np.ones(y_test.shape)*np.mean(y_train) - y_test) ** 2).mean(axis=0)]

    for epoch in range(num_epochs):

        if word_seq_test is not None:
            hist = model.fit(word_seq_train, y_train, batch_size=batch_size, epochs=1, callbacks=callbacks_list, validation_data=(word_seq_test, y_test), shuffle=True, verbose=2)
        else:
            hist = model.fit(word_seq_train, y_train, batch_size=batch_size, epochs=1, callbacks=callbacks_list, validation_split=0.10, shuffle=True, verbose=2)

        if epoch>0:
            ax1.lines.pop()
            ax2.lines.pop()
        yy=model.predict(word_seq_test).flatten()
        yy = yy - np.mean(yy) + np.mean(y_train)

        ax1.plot(yy)
        test_mse.append(((y_test - yy) ** 2).mean(axis=0))
        ax1.set_title('R2 = %f, MSE = %f' % (np.corrcoef(y_test,yy)[0,1]**2,test_mse[-1]))
        ax2.plot(test_mse,'o-')
        plt.pause(0.01)
        plt.show(block=False)

    fun = model.predict
    model.predict = lambda X: fun(sequence.pad_sequences(flatten_sentences(X), maxlen=max_seq_len))

    return model




    # # X = documents x sentences x words
    # def zero_pad(X, seq_len):
    #     return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])
    #
    # X = flatten_sentences(X_in)
    # max_seq_len = max([len(x) for x in X])
    # word_seq_train = X
    #
    # np.random.seed(0)
    #
    # y_train = Y_in  # train_df[label_names].values
    #
    # # pad sequences
    # #word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
    # word_seq_train = zero_pad(X,max_seq_len)
    #
    # word_seq_test = zero_pad(flatten_sentences(X_in_test), max_seq_len)
    #
    # #word_seq_test = sequence.pad_sequences(flatten_sentences(X_in_test), maxlen=max_seq_len)
    #
    # y_test = Y_in_test
    #
    # embedding_matrix = Params['W_embedding_matrix']
    #
    # # training params
    # NUM_WORDS = embedding_matrix.shape[0]
    # INDEX_FROM = 3
    # SEQUENCE_LENGTH = max_seq_len
    # EMBEDDING_DIM = embedding_matrix.shape[1]
    # HIDDEN_SIZE = 150
    # ATTENTION_SIZE = 50
    # KEEP_PROB = 0.8
    # BATCH_SIZE = 40
    # NUM_EPOCHS = 5  # Model easily overfits without pre-trained words embeddings, that's why train for a few epochs
    # DELTA = 0.5
    #
    # # Different placeholders
    # with tf.name_scope('Inputs'):
    #     batch_ph = tf.placeholder(tf.int32, [None, SEQUENCE_LENGTH], name='batch_ph')
    #     target_ph = tf.placeholder(tf.float32, [None], name='target_ph')
    #     seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
    #     keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')
    #
    # # Embedding layer
    # with tf.name_scope('Embedding_layer'):
    #     embeddings_var = tf.Variable(embedding_matrix, trainable=False)
    #     tf.summary.histogram('embeddings_var', embeddings_var)
    #     batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)
    #
    # # (Bi-)RNN layer(-s)
    # rnn_outputs, _ = bi_rnn(GRUCell(HIDDEN_SIZE), GRUCell(HIDDEN_SIZE),
    #                         inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
    # tf.summary.histogram('RNN_outputs', rnn_outputs)
    #
    # # Attention layer
    # with tf.name_scope('Attention_layer'):
    #     attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
    #     tf.summary.histogram('alphas', alphas)
    #
    # # Dropout
    # drop = tf.nn.dropout(attention_output, keep_prob_ph)
    #
    # # Fully connected layer
    # with tf.name_scope('Fully_connected_layer'):
    #     W = tf.Variable(tf.truncated_normal([HIDDEN_SIZE * 2, 1], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
    #     b = tf.Variable(tf.constant(0., shape=[1]))
    #     y_hat = tf.nn.xw_plus_b(drop, W, b)
    #     y_hat = tf.squeeze(y_hat)
    #     tf.summary.histogram('W', W)
    #
    # with tf.name_scope('Metrics'):
    #     # Cross-entropy loss and optimizer initialization
    #     loss = tf.losses.mean_squared_error(target_ph,y_hat) #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=target_ph))
    #     tf.summary.scalar('loss', loss)
    #     optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
    #
    #     # Accuracy metric
    #     accuracy = tf.losses.mean_squared_error(target_ph,y_hat)
    #     tf.summary.scalar('mse', accuracy)
    #
    # merged = tf.summary.merge_all()
    #
    # # Batch generators
    # train_batch_generator = batch_generator(word_seq_train, y_train, BATCH_SIZE)
    # test_batch_generator = batch_generator(word_seq_test, y_test, BATCH_SIZE)
    #
    # train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
    # test_writer = tf.summary.FileWriter('./logdir/test', accuracy.graph)
    #
    # session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    #
    # saver = tf.train.Saver()
    #
    # with tf.Session(config=session_conf) as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print("Start learning...")
    #     for epoch in range(NUM_EPOCHS):
    #         loss_train = 0
    #         loss_test = 0
    #         accuracy_train = 0
    #         accuracy_test = 0
    #
    #         print("epoch: {}\t".format(epoch), end="")
    #
    #         # Training
    #         num_batches = word_seq_train.shape[0] // BATCH_SIZE
    #         assert num_batches > 0
    #         for b in tqdm(range(num_batches)):
    #             x_batch, y_batch = next(train_batch_generator)
    #             seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
    #             loss_tr, acc, _, summary = sess.run([loss, accuracy, optimizer, merged],
    #                                                 feed_dict={batch_ph: x_batch,
    #                                                            target_ph: y_batch,
    #                                                            seq_len_ph: seq_len,
    #                                                            keep_prob_ph: KEEP_PROB})
    #             accuracy_train += acc
    #             loss_train = loss_tr * DELTA + loss_train * (1 - DELTA)
    #             train_writer.add_summary(summary, b + num_batches * epoch)
    #         accuracy_train /= num_batches
    #
    #         # Testing
    #         num_batches = word_seq_test.shape[0] // BATCH_SIZE
    #         assert num_batches>0
    #         for b in tqdm(range(num_batches)):
    #             x_batch, y_batch = next(test_batch_generator)
    #             seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
    #             loss_test_batch, acc, summary = sess.run([loss, accuracy, merged],
    #                                                      feed_dict={batch_ph: x_batch,
    #                                                                 target_ph: y_batch,
    #                                                                 seq_len_ph: seq_len,
    #                                                                 keep_prob_ph: 1.0})
    #             accuracy_test += acc
    #             loss_test += loss_test_batch
    #             test_writer.add_summary(summary, b + num_batches * epoch)
    #         accuracy_test /= num_batches
    #         loss_test /= num_batches
    #
    #         print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
    #             loss_train, loss_test, accuracy_train, accuracy_test
    #         ))
    #     train_writer.close()
    #     test_writer.close()
    #     saver.save(sess, r'./logdir/mymodel/')
    #     print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")
    #
    # return None

def deepCNN():

    #https://github.com/GINK03/keras-cnn-text-classify/tree/master/keras2-star-predictor-r2

    def CBRD(inputs, filters=64, kernel_size=3, droprate=0.5):
      x = Conv1D(filters, kernel_size, padding='same',
                kernel_initializer='random_normal')(inputs)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      return x


    def DBRD(inputs, units=4096, droprate=0.35):
      x = Dense(units)(inputs)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = Dropout(droprate)(x)
      return x

    input_tensor = Input( shape=(100, 3793) )

    x = Dense(3000, activation='relu')(input_tensor)
    x = CBRD(x, 16)
    x = CBRD(x, 16)
    x = MaxPool1D()(x)

    x = CBRD(x, 32)
    x = CBRD(x, 32)
    x = MaxPool1D()(x)

    x = CBRD(x, 64)
    x = CBRD(x, 64)
    x = MaxPool1D()(x)

    x = CBRD(x, 128)
    x = CBRD(x, 128)
    x = CBRD(x, 128)
    x = MaxPool1D()(x)

    x = CBRD(x, 128)
    x = CBRD(x, 128)
    x = CBRD(x, 128)
    x = MaxPool1D()(x)

    x = Flatten()(x)
    x = Dense(1, activation='linear')(x)

    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss='mae', optimizer='sgd')
    model.fit(data, y_train, epochs=10, batch_size=64)



def SimpleHATT(X_in, Y_in,Params):#,X_in_test = None,Y_in_test=None):

    MAX_SENT_LENGTH = Params['max_sentence_words']
    MAX_SENTS = Params['max_document_sentences']
    # training params
    batch_size = Params['Algorithm'][1]['batch']
    num_epochs = Params['Algorithm'][1]['epochs']
    #embedding_matrix = Params['W_embedding_matrix']
    MAX_NB_WORDS = Params['max_unique_words']
    EMBEDDING_DIM = Params['Algorithm'][1]['embedding_dim']
    GRU_UNITS = Params['Algorithm'][1]['rnn_units']

    #-----------------------------------------------

    converter = lambda x: data_to_tensor(x,MAX_SENTS,MAX_SENT_LENGTH)
    data = converter(X_in[0])

    doc_lst = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            sent = []
            for k in range(data.shape[2]):
                if data[i][j][k]!=0:
                    sent.append(str(data[i][j][k]))
                else:
                    break
            doc_lst.append(sent)

    # data = np.zeros((len(X_in[0]),MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    # # keep the MAX_NB_WORDS most frequent words and replace the rest with 'UNK'
    # # truncate to the first MAX_SENTS sentences per doc and MAX_SENT_LENGTH words per sentence
    # reviews = X_in[0]
    # for i, sentences in enumerate(reviews):
    #     for j, sent in enumerate(sentences):
    #         for k,word in enumerate(sent):
    #             data[i, j,k] = reviews[i][j][k]
    #         doc_lst.append([str(x) for x in sent])

    import gensim

    # use skip-gram
    word2vec_model = gensim.models.Word2Vec(doc_lst, min_count=1,size=EMBEDDING_DIM,sg=1,workers=os.cpu_count())
    embeddings_index = {}
    for word in word2vec_model.wv.vocab:
        coefs = np.asarray(word2vec_model.wv[word], dtype='float32')
        embeddings_index[int(word)] = coefs
    # Initial embedding
    doc_lst=None

    embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM),dtype=np.float32)

    unknown_count=0
    for word in range(MAX_NB_WORDS):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[word] = embedding_vector
        else:
            unknown_count += 1

    assert unknown_count==1,'should have only 1 unknown word (0=padding index)!'

    np.random.seed(0)

    y_train = Y_in  # train_df[label_names].values

    K.clear_session()

    REG_PARAM = 1e-13
    l2_reg = regularizers.l2(REG_PARAM)

    embedding_layer = Embedding(MAX_NB_WORDS,
                                EMBEDDING_DIM,
                                input_length=MAX_SENT_LENGTH,
                                trainable=True,
                                mask_zero=True,
                                embeddings_regularizer=l2_reg,
                                weights=[embedding_matrix])
                                #embeddings_initializer=keras.initializers.RandomUniform(minval=-1, maxval=1,seed=None))
                                #weights=[embedding_matrix])

    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    #l_lstm = (GRU(30,return_sequences=True, kernel_regularizer=l2_reg))(embedded_sequences)  # Bidirectional
    #l_lstm = NonMasking()(embedded_sequences)
    #l_att = Attention()(embedded_sequences)
    #l_att = GlobalAveragePooling1D()(l_lstm)#Attention(regularizer=l2_reg)(l_lstm)
    sentEncoder = Model(sentence_input, l_att)
    #print('Sentence encoder model:')
    #sentEncoder.summary()

    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    review_encoder = SpatialDropout1D(0.30)(review_encoder)

    l_lstm_sent = CuDNNLSTM(GRU_UNITS, return_sequences=True)(review_encoder) # Bidirectional
    l_att_sent = Attention()(l_lstm_sent)
    #preds = Dense(n_classes, activation='softmax', kernel_regularizer=l2_reg)(l_att_sent)
    preds = Dense(1, activation='sigmoid')(l_att_sent)
    model = Model(review_input, preds)

    model.compile(loss='mse', optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, clipnorm=10), metrics=['mse'])
    #print('Document encoder and regressor model:')
    #model.summary()

    mem = Utils.get_model_memory_usage(batch_size, model)
    if mem>6:
        print('!!!!! Model requires %f gigabytes of memory !!!!!!!' % mem)

    hist = model.fit(data, y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.0, shuffle=True, verbose=2)  # ,callbacks=callbacks)

    #plt.plot(hist.history['loss'])

    #print('step1')
    #mse_train = np.mean((y_train-model.predict(data))**2)
    #data = None
    #mse_null = np.mean((y_train-np.mean(y_train))**2)

    #print('step2')
    #fun = lambda x: model.predict(converter(x))

    fun = model.predict
    model.predict = lambda x: fun(converter(x[0]))

    #print('step3')
    #mse_train1 = np.mean((y_train - model.predict(X_in)) ** 2)

    #print('step4')
    #print('training mse=%f, another training mse=%f, null model mse=%f' % (mse_train,mse_train1,mse_null))

    return model
