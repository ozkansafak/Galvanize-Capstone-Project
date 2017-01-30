import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, RecurrentLayer, LSTMLayer, DenseLayer, SliceLayer, get_all_params
from lasagne.updates import adagrad
from lasagne.nonlinearities import sigmoid, tanh, softmax
from lasagne.objectives import binary_crossentropy, categorical_crossentropy
import numpy as np
import pickle
import time
from helpers_to_main import *

t0 = time.time()
#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))

# Declare Global Variables
NUM_FEATURES = None
X = None
Y = None
SEQ_LENGTH = 64
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_UNITS = 256
NUM_EPOCHS = 2000
PRINT_FREQ = 25
GRAD_CLIPPING = 100 
data_size = BATCH_SIZE*220

# 
print "\n\n\n\n\n\n==================================================="
print "Loading pitch_matrix ..."
sh = 0
pitch_matrix = pickle.load(open('training_data/pitch_matrix_'+str(96/4)+'ticks_sh'+"%02d" % (sh,)+'.p', 'rb'))
# at this point, pitch_matrix is a list of 2D lists. 
# pitch_matrix[i] is a fugue in the training set, for each i.

# patch the individual pitch_matrices of individual fugues
# into a single 2D numpy array and transpose it.
pitch_matrix = flatten_pitch_matrix(pitch_matrix)
NUM_FEATURES = pitch_matrix.shape[1]  # 61
# 
# extract the lowest notes  only
pitch_matrix = make_monophonic(pitch_matrix) 
# clip pitch_matrix for testing purposes.
if data_size < pitch_matrix.shape[0] - SEQ_LENGTH:
	clipped_pitch_matrix_length = data_size + SEQ_LENGTH 
	pitch_matrix = pitch_matrix[:clipped_pitch_matrix_length]
	print 'pitch_matrix CLIPPED, pitch_matrix.shape={}'.format(pitch_matrix.shape)
else:
	print 'pitch_matrix NOT clipped, pitch_matrix.shape={}'.format(pitch_matrix.shape)
# print 'pitch_matrix.shape={}'.format(pitch_matrix.shape)
# print 'data_size={}'.format(data_size)
# print 'SEQ_LENGTH={}'.format(SEQ_LENGTH)

#pitch_matrix, data_size = adjust_pitch_matrix(pitch_matrix, data_size, BATCH_SIZE, NUM_FEATURES, SEQ_LENGTH)

print '\n\t-----------------------------'
print '\tSEQ_LENGTH = {}'.format(SEQ_LENGTH)
print '\tBATCH_SIZE = {}'.format(BATCH_SIZE) 
print '\tNUM_UNITS = {}'.format(NUM_UNITS)
print '\tLEARNING_RATE = {}'.format(LEARNING_RATE)
print '\tpitch_matrix.shape[0] = {}'.format(pitch_matrix.shape[0])
print '\tdata_size = {}'.format(data_size)
print '\tPRINT_FREQ = {}'.format(PRINT_FREQ)
print '\tNUM_FEATURES = {}'.format(NUM_FEATURES)
print '\t-----------------------------\n'

'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''
def return_x_seed():
	x_seed = np.zeros((1,SEQ_LENGTH,NUM_FEATURES))
	for q in xrange(SEQ_LENGTH/2):
		f_ind = 24 + np.random.choice(range(12))
		x_seed[0,2*q,f_ind] = 1
		x_seed[0,2*q+1,f_ind] = 1
	return x_seed


def generate_a_fugue(epoch, cost, N=16*32):
	'''
	INPUT:  epoch and cost: Passed just to name the output files.	
			N: How many steps the fugue is to be generated for. 
	'''
	
	# Advance the RNN model for N steps at its current state.
	fugue = np.zeros((N, NUM_FEATURES))
	
	# x, _ = make_batch(1000, X, Y, data_size, 1) # x.shape = (1, 32, 61)
	x = return_x_seed()

	# fugue[:SEQ_LENGTH,:] = x[0,:,:]
	print "epoch={}".format(epoch)
	for i in xrange(N):
		# Pick the note w/ highest probability.
		predict = probabilities(x) # predict.shape = (1, 61)
		ix = np.argmax(predict)
		
		fugue[i][ix] = 1 
		x[0, :-1, :] = x[:, 1:, :] 
		x[0, -1, :] = fugue[i, :] 
		
		# # # # # # # # # # # # # # # # # # 
	
	print_epoch = "%04d" % (epoch,)
	print_data_size = "%05d" % (data_size,)
	print_SEQL = "%02d" % (SEQ_LENGTH,)
	filename =  'fugue_dsize' + print_data_size + \
					'_SEQL' + print_SEQL + \
					'_epoch' + print_epoch
					
	dump_dict = {'NUM_FEATURES' : NUM_FEATURES,
				'SEQ_LENGTH' : SEQ_LENGTH,
				'BATCH_SIZE' : BATCH_SIZE,
				'LEARNING_RATE' : LEARNING_RATE,
				'NUM_UNITS' : NUM_UNITS,
				'NUM_EPOCHS' : NUM_EPOCHS,
				'PRINT_FREQ' : PRINT_FREQ,
				'GRAD_CLIPPING' : GRAD_CLIPPING,
				'data_size' : data_size,
				'predict' : predict, 
				'cost' : cost,
				'fugue' : fugue}
				
	pickle.dump(dump_dict, open('Synthesized_Fugues/'+filename+'.p', 'wb'))
	time_series = pitch_matrix_TO_time_series_legato(fugue, sh=24)
	
	print 'len(time_series)={}'.format(len(time_series))
	if len(time_series) > 0:
		time_series_TO_midi_file(time_series, 'Synthesized_Fugues/'+filename+'.mid')
	elif len(time_series) == 0:
		print 'ZERO NOTES PREDICTED'
	return
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def build_rnn(sequence_length=SEQ_LENGTH, num_units=NUM_UNITS):
	# input layer
	# generate the data to pass into this
	l_in = InputLayer(shape=(None, sequence_length, NUM_FEATURES))

	l_LSTM1 = LSTMLayer(l_in, num_units=num_units, 
			grad_clipping=GRAD_CLIPPING, nonlinearity=tanh)

	l_LSTM2 = LSTMLayer(l_LSTM1, num_units=num_units, 
			grad_clipping=GRAD_CLIPPING, nonlinearity=tanh) 

	# The l_forward layer creates an output of dimension (batch_size, SEQ_LENGTH, N_HIDDEN)
	# Since we are only interested in the final prediction, we isolate that quantity and feed it to the next layer. 
	# The output of the sliced layer will then be of size (batch_size, N_HIDDEN)
	'''l_slice = SliceLayer(l_LSTM2, -1, 1)''' # don't use this SliceLayer
	
	# output layer
	l_out = DenseLayer(l_LSTM2, num_units=NUM_FEATURES, W=lasagne.init.Normal(), nonlinearity=softmax)
	
	return l_in, l_out
	
'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

print "Building network ..."
l_in, l_out = build_rnn()
target_values = T.imatrix('target_output')
network_output = lasagne.layers.get_output(l_out)

# The cost function of predictions and target.
cost = categorical_crossentropy(network_output, target_values).mean()
'''cost = T.nnet.categorical_crossentropy(network_output,target_values).mean()'''

# Retrieve all parameters from the network
all_params = get_all_params(l_out, trainable=True)

# Compute AdaGrad updates for training
print "Computing updates for training w/ adagrad..."
updates = adagrad(loss_or_grads=cost, params=all_params, learning_rate=LEARNING_RATE)

# Theano functions for training and computing cost
print "Compiling functions ..."
train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)
	
# Need probability distribution of the next note given 
# the state of the network and the input_seed.
# Compile a theano function called probabilities to produce 
# the probability distribution of the prediction.

# `probabilities` is analogous to .predict() function
probabilities = theano.function([l_in.input_var], network_output, allow_input_downcast=True)

# generation_phrase = input_seed(pitch_matrix)

'''  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || 
     \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/   
	   T  R  A  I  N  I  N  G  --  T  R  A  I  N  I  N  G  --  T  R  A  I  N  I  N  G												
'''
# make the training data tensors X and Y, and train the model on them
print "Training ..."
X, Y = make_X_Y(pitch_matrix, data_size, SEQ_LENGTH, NUM_FEATURES)
print 'X.shape, Y.shape = {} ---- {}'.format(X.shape, Y.shape)

p, avg_cost, dummy_previous, cost_p = 0, 0, -1, []


for it in xrange(data_size * NUM_EPOCHS / BATCH_SIZE + 1):
	epoch = float(it) * float(BATCH_SIZE) / data_size
	print "it:", it, "--- epoch:", round(epoch,3), "   \r",
	
	x, y = make_batch(p, X, Y, data_size, BATCH_SIZE)
	avg_cost = avg_cost + train(x, y)
	
	if epoch % PRINT_FREQ < dummy_previous:
		print " "*100,"\r",
		
		generate_a_fugue(epoch=epoch, cost=cost_p) 
		t5 = time.time()
		# Generate a fugue starting from a segment of length 
		# SEQ_LENGTH starting at the p-th note of pitch_matrix 
		# print "fugue generation completed in {} sec".format(round(t5-t4, 2))
		print "Epoch {}, at run time={} sec, average loss = {}\n".format(epoch, round(t5-t0), avg_cost)
	pass
	
	p = p + BATCH_SIZE
	dummy_previous = epoch % PRINT_FREQ
	
	if p > data_size - 1:
		# print '\n\nEpoch completed. epoch={}, it={}, p={}'.format(epoch, it, p)
		cost_p.append(avg_cost)
		p = 0
		avg_cost = 0
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''

