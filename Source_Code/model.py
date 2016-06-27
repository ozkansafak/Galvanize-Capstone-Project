import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, RecurrentLayer, LSTMLayer, DenseLayer, adagrad, get_all_params
from lasagne.nonlinearities import softmax
from lasagne.objectives import binary_crossentropy, aggregate



NUM_FEATURES = pitch_matrix.shape[1] # 73
SEQUENCE_LENGTH = 8
BATCH_SIZE = 5
num_epochs = 10

def iterate_minibatches(pitch_matrix, batch_size=BATCH_SIZE):
	for i in range(pitch_matrix.shape[0] / batch_size):
		i1 = i*batch_size
		i2 = (i+1)*batch_size
		yield pitch_matrix[i1:i2]
		

'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || 
'''
def build_rnn(sequence_length=SEQUENCE_LENGTH, num_units=512):

	# input layer
	# generate the data to pass into this

	l_in = InputLayer(shape=(None, sequence_length, NUM_FEATURES))	
	# LSTM Layer
	l_LSTM = LSTMLayer(l_in, num_units=num_units, grad_clipping=100, only_return_final=True)
	# output layer
	l_out = DenseLayer(l_LSTM, num_units=NUM_FEATURES, nonlinearity=softmax)

	return l_out

def total_cost(predictions, target):
	'''
	INPUT np.array of size (73,), np.array of size (73,)
	OUTPUT: 
	'''
	note_cost = binary_crossentropy(predictions, target).mean()
	
	chord_target = get_chord(target)
	note_predictions = predictions[predictions > .5]
	chord_predictions = get_chord(note_predictions)
	chord_cost = cosine_distance(chord_predictions, chord_target)
	
	return note_cost + chord_cost


def build_trainer(l_out):

	x_sym = T.tensor3()
	y_sym = T.imatrix()
	hid_init_sym = T.matrix()
	
	
	hid_out, prob_out = lasagne.layers.get_output([l_LSTM, l_out], \
												  {l_input: x_sym, \
												   l_input_hid: hid_init_sym
												  })
	hid_out = hid_out[:, -1]
	
	''' '''	
	''' '''	
	''' '''	
	
	output = lasagne.layers.get_output(l_out)
	target = T.ivector('target_output') # vector of ints
	
	# Make a Cost Func
	total_cost(predictions, target)
	
	# Will use nolearn
	cost = total_cost(predictions, target)
	params = get_all_params(l_out, trainable=True)
	updates = adagrad(cost,\
	 				  params, learning_rate=0.01, momentum=0.9)
	
	
	train_fn = theano.function([input_var, target_var], cost, updates=updates)
	
	for epoch in range(num_epochs):
	    for batch in iterate_minibatches(pitch_matrix):
	        inputs, targets = batch
	        train_fn(inputs, targets)
	
	
	
	
	
	
	
	
	
	
	
	
	