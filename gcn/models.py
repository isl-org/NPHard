from layers import *
from metrics import *
from layers import _LAYER_UIDS

flags = tf.app.flags
FLAGS = flags.FLAGS

def lrelu(x):
    return tf.maximum(x*0.2,x)

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.outputs_softmax = None
        self.pred = None
        self.output_dim = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        layer_id = 0
        self.activations.append(self.inputs)
        for layer in self.layers:
            if self.name == 'gcn_deep' and layer_id % 2 == 0 and layer_id > 0 and layer_id < len(self.layers)-1:
                hidden = layer(self.activations[-1])
                self.activations.append(tf.nn.relu(hidden+self.activations[-2]))
                layer_id = layer_id + 1
            elif layer_id < len(self.layers)-1:
                hidden = tf.nn.relu(layer(self.activations[-1]))
                self.activations.append(hidden)
                layer_id = layer_id + 1
            else:
                hidden = layer(self.activations[-1])
                self.activations.append(hidden)
                layer_id = layer_id + 1
        self.outputs = self.activations[-1]
        if self.name != 'gcn_dqn':
            self.outputs_softmax = tf.nn.softmax(self.outputs[:,0:2])
        if self.name == 'gcn_deep_diver':
            for out_id in range(1, FLAGS.diver_num):
                self.outputs_softmax = tf.concat([self.outputs_softmax, tf.nn.softmax(self.outputs[:,self.output_dim*out_id:self.output_dim*(out_id+1)])], axis=1)
        if self.name == 'gcn_dqn':
            self.pred = tf.argmax(self.outputs)
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _loss_reg(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += my_softmax_cross_entropy(self.outputs, self.placeholders['labels'])
    def _loss_reg(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        self.loss += tf.reduce_mean(tf.square(self.outputs-self.placeholders['labels']))

    def _accuracy(self):
        self.accuracy = my_accuracy(self.outputs, self.placeholders['labels'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

class GCN_DEEP_DIVER(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN_DEEP_DIVER, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # 32 outputs
        diver_loss = my_softmax_cross_entropy(self.outputs[:,0:self.output_dim], self.placeholders['labels'])
        for i in range(1,FLAGS.diver_num):
            diver_loss = tf.reduce_min([diver_loss, my_softmax_cross_entropy(self.outputs[:, 2*i:2*i + self.output_dim], self.placeholders['labels'])])
        self.loss += diver_loss

    def _loss_reg(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # regression loss
        self.loss += tf.reduce_mean(tf.abs(self.outputs-self.placeholders['labels']))

    def _accuracy(self):
        # 32 outputs
        acc = my_accuracy(self.outputs[:,0:self.output_dim], self.placeholders['labels'])
        for i in range(1,FLAGS.diver_num):
            acc = tf.reduce_max([acc, my_accuracy(self.outputs[:,2*i:2*i+self.output_dim], self.placeholders['labels'])])
        self.accuracy = acc

    def _build(self):

        _LAYER_UIDS['graphconvolution'] = 0
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))
        for i in range(FLAGS.num_layer-2):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=2*FLAGS.diver_num,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)