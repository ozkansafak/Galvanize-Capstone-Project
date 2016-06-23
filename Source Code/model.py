import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, RecurrentLayer, LSTMLayer, DenseLayer
from lasagne.nonlinearities import softmax
from lasagne.objectives import binary_crossentropy, aggregate






def build_rnn(sequence_length=8, num_units=512):
	'''
	INPUT:
	OUTPUT: 
	'''	
	
	# input layer
	# generate the data to pass into this
	
	l_in = InputLayer(shape=(None, sequence_length, 128))	
	# LSTM Layer
	l_LSTM = LSTMLayer(l_in, num_units=num_units, grad_clipping=100, only_return_final=True)
	# output layer
	l_out = DenseLayer(l_LSTM, num_units=128, nonlinearity=softmax)

	return l_out



def total_cost(predictions, target):
	note_cost = binary_crossentropy(predictions, target).mean()
	note_predictions = predictions[predictions > .5]
	
	chord_predictions = get_chord(note_predictions)
	chord_target = get_chord(target)
	chord_cost = cosine_distance(chord_predictions, chord_target)
	
	return note_cost + chord_cost



def build_trainer(l_out):
	'''
	INPUT:
	OUTPUT: 
	'''
	
	output = lasagne.layers.get_output(l_out)
	target = T.ivector('target_output')
	
	# Make a Cost Func
	note_cost = binary_crossentropy(output, target).mean()

	# Will use nolearn
	
	





