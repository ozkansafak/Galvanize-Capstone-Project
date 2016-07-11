import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, RecurrentLayer, LSTMLayer, DenseLayer, SliceLayer, get_all_params
from lasagne.updates import adagrad
from lasagne.nonlinearities import sigmoid, tanh
from lasagne.objectives import binary_crossentropy, aggregate
import numpy as np
import pickle
import time
from helpers_to_main import *

# Declare Global Variables
NUM_FEATURES = None
X = None
Y = None
SEQUENCE_LENGTH = 16
BATCH_SIZE = 48
LEARNING_RATE = .01
NUM_UNITS = 512
NUM_EPOCHS = 5000
PRINT_FREQ = 100
GRAD_CLIPPING = 10
THRESHOLD = .5
CLIP_PITCH_MATRIX = 4800
data_size = None
# 
# 
t0 = time.time()
print "\nLoading pitch_matrix ..."
sh = 0
pitch_matrix = pickle.load(open('training_data/pitch_matrix_'+str(96/4)+'ticks_sh'+"%02d" % (sh,)+'.p', 'rb'))

# patch the individual pitch_matrices of individual fugues
# into a single 2D numpy array and transpose it.
pitch_matrix = flatten_pitch_matrix(pitch_matrix)
NUM_FEATURES = pitch_matrix.shape[1]  # 61
# 
# extract the lowest notes  only
pitch_matrix = make_monophonic(pitch_matrix) 
# clip pitch_matrix for testing purposes.
pitch_matrix = pitch_matrix[:CLIP_PITCH_MATRIX+SEQUENCE_LENGTH] 
#
'''   '''
# xtra = (pitch_matrix.shape[0] - SEQUENCE_LENGTH) % BATCH_SIZE
# pitch_matrix = pitch_matrix[ 0 : pitch_matrix.shape[0] - xtra]
'''   '''
data_size = pitch_matrix.shape[0] - SEQUENCE_LENGTH
pitch_matrix, data_size = adjust_pitch_matrix(pitch_matrix, data_size, BATCH_SIZE, NUM_FEATURES)
#
print '\n\t-----------------------------'
print '\tSEQUENCE_LENGTH = {}'.format(SEQUENCE_LENGTH)
print '\tBATCH_SIZE = {}'.format(BATCH_SIZE) 
print '\tNUM_UNITS = {}'.format(NUM_UNITS)
print '\tLEARNING_RATE = {}'.format(LEARNING_RATE)
print '\tCLIP_PITCH_MATRIX = {}'.format(CLIP_PITCH_MATRIX)
print '\tpitch_matrix.shape[0] = {}'.format(pitch_matrix.shape[0])
print '\tdata_size = {}'.format(data_size)
print '\tPRINT_FREQ = {}'.format(PRINT_FREQ)
print '\tNUM_FEATURES = {}'.format(NUM_FEATURES)
print '\t-----------------------------\n'

'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def generate_a_fugue(epoch, cost, N=16*32):
	'''
	INPUT:  epoch and cost: Passed just for naming of the output files.	
			N: How many steps the fugue will be generated for. 
			96/4 is the number of 16th notes in one bar 
	'''
	#
	# Advance the RNN model for N steps at its current state.
	fugue = np.zeros((N, NUM_FEATURES))
	#
	x, _ = make_batch(0, X, Y, data_size, 1)
	print "epoch={}".format(epoch)
	for i in range(N):
		'''
		Pick a single note that got assigned the highest probability.
		Will be modified to accommodate polyphonic music creation
		'''
		#
		predict = probabilities(x) # predict.shape = (1, 61)
		ix = np.argmax(predict)
		#
		fugue[i][ix] = 1 
		x[0, :-1, :] = x[:, 1:, :] 
		x[0, -1, :] = fugue[i, :] 
		#
		# # # # # # # # # # # # # # # # # # 
	#
	print_epoch = "%04d" % (epoch,)
	print_data_size = "%05d" % (data_size,)
	print_SEQL = "%02d" % (SEQUENCE_LENGTH,)
	filename =  'fugue_dsize' + print_data_size + \
					'_SEQL' + print_SEQL + \
					'_epoch' + print_epoch
					
	dump_dict = {'NUM_FEATURES' : NUM_FEATURES,
				'SEQUENCE_LENGTH' : SEQUENCE_LENGTH,
				'BATCH_SIZE' : BATCH_SIZE,
				'LEARNING_RATE' : LEARNING_RATE,
				'NUM_UNITS' : NUM_UNITS,
				'NUM_EPOCHS' : NUM_EPOCHS,
				'PRINT_FREQ' : PRINT_FREQ,
				'THRESHOLD' : THRESHOLD,
				'data_size' : data_size,
				'GRAD_CLIPPING' : GRAD_CLIPPING,
				'CLIP_PITCH_MATRIX' : CLIP_PITCH_MATRIX,
				'predict' : predict, 
				'cost' : cost,
				'fugue' : fugue}
	#
	pickle.dump(dump_dict, open('Synthesized_Fugues/'+filename+'.p', 'wb'))
	time_series = pitch_matrix_TO_time_series_legato(fugue, sh=24)
	#
	print 'len(time_series)={}'.format(len(time_series))
	if len(time_series) > 0:
		time_series_TO_midi_file(time_series, 'Synthesized_Fugues/'+filename+'.mid')
	elif len(time_series) == 0:
		print 'ZERO NOTES PREDICTED'
	return
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def build_rnn(sequence_length=SEQUENCE_LENGTH, num_units=NUM_UNITS):
	# input layer
	# generate the data to pass into this
	l_in = InputLayer(shape=(None, sequence_length, NUM_FEATURES))
	# LSTM Layer
	l_LSTM1 = LSTMLayer(l_in, num_units=num_units, grad_clipping=GRAD_CLIPPING, nonlinearity=tanh)
	# 2nd LSTM Layer
	'''l_LSTM2 = LSTMLayer(l_LSTM1, num_units=num_units, grad_clipping=GRAD_CLIPPING, nonlinearity=tanh)'''
	# The l_forward layer creates an output of dimension (batch_size, SEQ_LENGTH, N_HIDDEN)
	# Since we are only interested in the final prediction, we isolate that quantity and feed it to the next layer. 
	# The output of the sliced layer will then be of size (batch_size, N_HIDDEN)
	'''l_slice = SliceLayer(l_LSTM2, -1, 1)'''
	# output layer
	l_out = DenseLayer(l_LSTM1, num_units=NUM_FEATURES, nonlinearity=sigmoid)
	return l_in, l_out
	
'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def total_cost(predictions, target):
	'''
	INPUT np.array of size (61,), np.array of size (61,)
	OUTPUT: 
	'''
	note_cost = binary_crossentropy(predictions, target).mean()
	return note_cost 
	
'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

print "Building network ..."
l_in, l_out = build_rnn()
target_values = T.imatrix('target_output')
network_output = lasagne.layers.get_output(l_out)

# The cost function of predictions and target.
cost = total_cost(network_output, target_values)

# Retrieve all parameters from the network
all_params = get_all_params(l_out, trainable=True)

# Compute AdaGrad updates for training
print "Computing updates for training w/ adagrad..."
updates = adagrad(loss_or_grads=cost, params=all_params, learning_rate=LEARNING_RATE)

# Theano functions for training and computing cost
print "Compiling functions ..."
train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)
	
# Need probability distribution of the next notes given 
# the state of the network and the input_seed.
# Compile a theano function called probabilities to produce 
# the probability distribution of the prediction.

# `probabilities` is analogous to .predict() function
probabilities = theano.function([l_in.input_var], network_output, allow_input_downcast=True)

# generation_phrase = input_seed(pitch_matrix)

''' ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||
    \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  
		T  R  A  I  N  I  N  G  --  T  R  A  I  N  I  N  G  --  T  R  A  I  N  I  N  G												
'''
# make the training data tensors X and Y, and train the model on them
print "Training ..."
print 'pitch_matrix.shape', pitch_matrix.shape
X, Y = make_X_Y(pitch_matrix, data_size, SEQUENCE_LENGTH, NUM_FEATURES)
print 'pitch_matrix.shape', pitch_matrix.shape

p, avg_cost, dummy_previous, cost = 0, 0, -1, []

for it in xrange(data_size * NUM_EPOCHS / BATCH_SIZE):
	epoch = float(it) * float(BATCH_SIZE) / data_size
	print "it:", it, "--- epoch:", round(epoch,3), "   \r",
	
	x, y = make_batch(p, X, Y, data_size, BATCH_SIZE)
	avg_cost = avg_cost + train(x, y)
	
	if epoch % PRINT_FREQ < dummy_previous:
		print " "*100,"\r",
		generate_a_fugue(epoch=epoch, cost=cost) 
		t5 = time.time()
		# Generate a fugue starting from a segment of length 
		# SEQUENCE_LENGTH starting at the p-th note of pitch_matrix 
		# print "fugue generation completed in {} sec".format(round(t5-t4, 2))
		print "Epoch {}, at run time={} sec, average loss = {}\n".format(epoch, round(t5-t0), avg_cost)
	pass
	
	p = p + BATCH_SIZE
	dummy_previous = epoch % PRINT_FREQ
	
	if p > data_size - 1:
		# print '\n\nEpoch completed. epoch={}, it={}, p={}'.format(epoch, it, p)
		cost.append(avg_cost)
		p = 0
		avg_cost = 0
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''

