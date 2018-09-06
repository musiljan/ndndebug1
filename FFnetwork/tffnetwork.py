"""Temporal FFNetwork"""

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np

#from .layer import Layer
#from .layer import ConvLayer
#from .layer import SepLayer
#from .layer import ConvSepLayer
#from .layer import AddLayer
#from .layer import SpikeHistoryLayer
#from .layer import BiConvLayer
#from .tlayer import TLayer
#from .tlayer import CaTentLayer
#from .tlayer import NoRollCaTentLayer

from .layer import *
from .tlayer import *



class TFFnetwork(FFNetwork):
    """Implementation of simple fully-connected feed-forward neural network.
    These networks can be composed to create much more complex network
    architectures using the NDN class.

    Attributes:
        input_dims (list of ints): three-element list containing the dimensions
            of the input to the network, in the form
            [num_lags, num_x_pix, num_y_pix]. If the input does not have
            spatial or temporal structure, this should be [1, num_inputs, 1].
        scope (str): name scope for network
        num_layers (int): number of layers in network (not including input)
        layer_types (list of strs): a string for each layer in the network that
            specifies its type.
            'normal' | 'sep' | 'conv' | 'convsep' | 'biconv' | 'add' | 'spike_history'
        layers (list of `Layer` objects): layers of network
        log (bool): use tf summary writers in layer activations

    """

    def __init__(self,
                 scope=None,
                 input_dims=None,
                 params_dict=None,
                 batch_size=None,
                 time_spread=None):
        """Constructor for TFFnetwork class"""

        super(TFFnetwork, self).__init__(
            scope=scope,
            input_dims=input_dims,
            params_dict=params_dict)

        self.batch_size = batch_size
        self.time_spread = time_spread

    # END TFFnetwork.__init

    def _define_network(self, network_params):

        layer_sizes = [self.input_dims] + network_params['layer_sizes']
        self.layers = []
        #print(self.scope, layer_sizes)

        for nn in range(self.num_layers):
            if self.layer_types[nn] == 'normal':
                self.layers.append(Layer(
                    scope='layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    output_dims=layer_sizes[nn+1],
                    activation_func=network_params['activation_funcs'][nn],
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

            elif self.layer_types[nn] == 'sep':
                self.layers.append(SepLayer(
                    scope='sep_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    output_dims=layer_sizes[nn+1],
                    activation_func=network_params['activation_funcs'][nn],
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

            elif self.layer_types[nn] == 'add':
                self.layers.append(AddLayer(
                    scope='add_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    output_dims=layer_sizes[nn+1],
                    activation_func=network_params['activation_funcs'][nn],
                    normalize_weights=network_params['normalize_weights'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

            elif self.layer_types[nn] == 'spike_history':
                self.layers.append(SpikeHistoryLayer(
                    scope='spike_history_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    output_dims=layer_sizes[nn+1],
                    activation_func=network_params['activation_funcs'][nn],
                    normalize_weights=network_params['normalize_weights'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

            elif self.layer_types[nn] == 'conv':
                if network_params['conv_filter_widths'][nn] is None:
                    conv_filter_size = layer_sizes[nn]
                else:
                    conv_filter_size = [
                        layer_sizes[nn][0],
                        network_params['conv_filter_widths'][nn], 1]
                    if layer_sizes[nn][2] > 1:
                        conv_filter_size[2] = \
                            network_params['conv_filter_widths'][nn]

                self.layers.append(ConvLayer(
                    scope='conv_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn+1],
                    filter_dims=conv_filter_size,
                    shift_spacing=network_params['shift_spacing'][nn],
                    activation_func=network_params['activation_funcs'][nn],
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

                # Modify output size to take into account shifts
                if nn < self.num_layers:
                    layer_sizes[nn+1] = self.layers[nn].output_dims

            elif self.layer_types[nn] == 'convsep':
                if network_params['conv_filter_widths'][nn] is None:
                    conv_filter_size = layer_sizes[nn]
                else:
                    conv_filter_size = [
                        layer_sizes[nn][0],
                        network_params['conv_filter_widths'][nn], 1]
                    if layer_sizes[nn][2] > 1:
                        conv_filter_size[2] = \
                            network_params['conv_filter_widths'][nn]

                self.layers.append(ConvSepLayer(
                    scope='sepconv_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn+1],
                    filter_dims=conv_filter_size,
                    shift_spacing=network_params['shift_spacing'][nn],
                    activation_func=network_params['activation_funcs'][nn],
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

                # Modify output size to take into account shifts
                if nn < self.num_layers:
                    layer_sizes[nn+1] = self.layers[nn].output_dims

            elif self.layer_types[nn] == 'biconv':
                if network_params['conv_filter_widths'][nn] is None:
                    conv_filter_size = layer_sizes[nn]
                else:
                    conv_filter_size = [
                        layer_sizes[nn][0],
                        network_params['conv_filter_widths'][nn], 1]
                    if layer_sizes[nn][2] > 1:
                        conv_filter_size[2] = \
                            network_params['conv_filter_widths'][nn]

                self.layers.append(BiConvLayer(
                    scope='conv_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn+1],
                    filter_dims=conv_filter_size,
                    shift_spacing=network_params['shift_spacing'][nn],
                    activation_func=network_params['activation_funcs'][nn],
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

                # Modify output size to take into account shifts
                if nn < self.num_layers:
                    layer_sizes[nn+1] = self.layers[nn].output_dims

            elif self.layer_types[nn] == 'temporal':
                self.layers.append(TLayer(
                    scope='temporal_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    output_dims=self.batch_size,
                    num_filters=layer_sizes[nn + 1],
                    batch_size=self.batch_size,
                    time_spread=self.time_spread,
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

                # Modify output size to take into account shifts
                if nn < self.num_layers:
                    layer_sizes[nn + 1] = self.layers[nn].output_dims

            elif self.layer_types[nn] == 'ca_tent':
                self.layers.append(CaTentLayer(
                    scope='ca_tent_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    output_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn + 1],
                    filter_width=network_params['ca_tent_widths'][nn],
                    batch_size=self.batch_size,
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

                # Modify output size to take into account shifts
                if nn < self.num_layers:
                    layer_sizes[nn + 1] = self.layers[nn].output_dims

            elif self.layer_types[nn] == 'ca_tent_no_roll':
                self.layers.append(NoRollCaTentLayer(
                    scope='ca_tent_layer_%i' % nn,
                    input_dims=layer_sizes[nn],
                    output_dims=layer_sizes[nn],
                    num_filters=layer_sizes[nn + 1],
                    filter_width=network_params['ca_tent_widths'][nn],
                    batch_size=self.batch_size,
                    normalize_weights=network_params['normalize_weights'][nn],
                    weights_initializer=network_params['weights_initializers'][nn],
                    biases_initializer=network_params['biases_initializers'][nn],
                    reg_initializer=network_params['reg_initializers'][nn],
                    num_inh=network_params['num_inh'][nn],
                    pos_constraint=network_params['pos_constraints'][nn],
                    log_activations=network_params['log_activations']))

                # Modify output size to take into account shifts
                if nn < self.num_layers:
                    layer_sizes[nn + 1] = self.layers[nn].output_dims

            else:
                raise TypeError('Layer type %i not defined.' % nn)

    # END TFFnetwork._define_network

    def build_graph(self, inputs, params_dict=None):
        """Build tensorflow graph for this network"""

        with tf.name_scope(self.scope):
            for layer in range(self.num_layers):
                # input_prime
                self.layers[layer].build_graph(inputs, params_dict)
                inputs = self.layers[layer].outputs
    # END TFFNetwork._build_graph