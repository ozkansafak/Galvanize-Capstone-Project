import theano, sys
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, RecurrentLayer, LSTMLayer, DenseLayer, SliceLayer, get_all_params
from lasagne.updates import adagrad, adam
from lasagne.nonlinearities import sigmoid, tanh, softmax
from lasagne.objectives import binary_crossentropy, categorical_crossentropy
import numpy as np
import pickle
import time
from helpers_to_main import *

# ------------------------- #
# Declare Global Variables
NUM_FEATURES = None
X = None
Y = None
SEQ_LENGTH = 32
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2000
PRINT_FUGUES = 20
GRAD_CLIPPING = 100
NUM_NEURONS = 256
NUM_LAYERS = 1
data_size = BATCH_SIZE*(228/19)
# ------------------------- #

t0 = time.time()
# Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))
np.random.seed(0)

pitch_matrix = load_n_flatten_pitch_matrix()
NUM_FEATURES = pitch_matrix.shape[1]  # 61
full_data_size = pitch_matrix.shape[0]-SEQ_LENGTH

# extract only lowest notes
pitch_matrix = make_monophonic(pitch_matrix)
# clip pitch_matrix to an integer multiple of BATCH_SIZE
if data_size < pitch_matrix.shape[0] - SEQ_LENGTH:
	clipped_pitch_matrix_length = data_size + SEQ_LENGTH 
	print 'pitch_matrix original, pitch_matrix.shape = {}'.format(pitch_matrix.shape)
	pitch_matrix = pitch_matrix[:clipped_pitch_matrix_length]
	print 'pitch_matrix CLIPPED,  pitch_matrix.shape = {}'.format(pitch_matrix.shape)
else:
	print '!!!!!!pitch_matrix NOT clipped, pitch_matrix.shape = {}'.format(pitch_matrix.shape)

print_inputs(NUM_LAYERS, NUM_NEURONS, NUM_FEATURES,
			SEQ_LENGTH, BATCH_SIZE, LEARNING_RATE,
			PRINT_FUGUES, data_size, pitch_matrix,
			full_data_size)
			
'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def pick_from_pdf(pdf, how_many=SEQ_LENGTH):
	# take top 4 notes only. 
	ind = np.argsort(pdf)[-how_many:]
	#normalize
	top_4_pdf = pdf[ind]/np.sum(pdf[ind])
	# sample from uniform distribution
	p = np.random.rand()
	# map this uniform random variable to adjusted pdf of top 4 notes.
	cdf = np.cumsum(top_4_pdf)
	return ind[len(cdf[cdf < p])]
	

def next_note(x):
	pdf = probabilities(x)
	# pdf.shape = (1,NUM_FEATURES)
	ix = pick_from_pdf(pdf[0,:])
	# ix = np.argmax(pdf)
	note = np.zeros((1,NUM_FEATURES))
	note[0,ix] = 1
	return note

def generate_a_fugue(epoch, loss, time_delta, N=16*32):
	'''
	INPUT:  epoch and loss: Passed just to name the output files.	
	N: How many steps the fugue is to be generated for. 
	'''
	
	# Advance the RNN model for N steps.
	fugue = np.zeros((N, NUM_FEATURES))
	
	x = np.zeros_like(x_seed)
	x[:] = x_seed[:]
	
	for i in xrange(N):		
		fugue[i] = next_note(x)
		x[0, :-1, :] = x[:, 1:, :] 
		x[0, -1, :] = fugue[i, :] 

	fugue = np.vstack((x_seed[0],fugue))
	# # # # # # # # # # # # # # # # # # 
	
	print_epoch = "%04d" % (epoch,)
	print_SEQ = "%02d" % (SEQ_LENGTH,)
	print_LSTM = "%01d" % (NUM_LAYERS,)+"x"+"%03d" % (NUM_NEURONS,)
	
	filename = 'fugue' + \
			'_SEQ' + print_SEQ + \
			'_LSTM' + print_LSTM + \
			'_epoch' + print_epoch

	dump_dict = {'NUM_FEATURES' : NUM_FEATURES,
				'SEQ_LENGTH' : SEQ_LENGTH,
				'BATCH_SIZE' : BATCH_SIZE,
				'LEARNING_RATE' : LEARNING_RATE,
				'NUM_EPOCHS' : NUM_EPOCHS,
				'GRAD_CLIPPING' : GRAD_CLIPPING,
				'NUM_NEURONS' : NUM_NEURONS,
				'NUM_LAYERS' : NUM_LAYERS,
				'data_size' : data_size,
				'loss' : loss,
				'time_delta' : time_delta,
				'fugue' : fugue}
	
	pickle.dump(dump_dict, open('Synthesized_Fugues/'+filename+'.p', 'wb'))
	time_series = pitch_matrix_TO_time_series_legato(fugue, sh=24)
	
	if len(time_series) > 0:
		time_series_TO_midi_file(time_series, 'Synthesized_Fugues/'+filename+'.mid')
	elif len(time_series) == 0:
		print 'ZERO NOTES PREDICTED'
	
	return
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def build_rnn(sequence_length=SEQ_LENGTH, num_units=NUM_NEURONS):
	# input layer
	l_in = InputLayer(shape=(None, sequence_length, NUM_FEATURES))
	l_LSTMs = [LSTMLayer(l_in, num_units=num_units, 
						grad_clipping=GRAD_CLIPPING, nonlinearity=tanh)]

	for i in range(1,NUM_LAYERS):
		l_LSTMs.append(LSTMLayer(l_LSTMs[i-1], num_units=num_units, 
		grad_clipping=GRAD_CLIPPING, nonlinearity=tanh))

	# The l_forward layer creates an output of dimension (batch_size, SEQ_LENGTH, N_HIDDEN)
	# Since we are only interested in the final prediction, we isolate that quantity and feed it to the next layer. 
	# The output of the sliced layer will then be of size (batch_size, N_HIDDEN)
	'''l_slice = SliceLayer(l_LSTM2, -1, 1)''' # don't use this SliceLayer
	
	# output layer
	l_out = DenseLayer(l_LSTMs[-1], num_units=NUM_FEATURES, W=lasagne.init.Normal(), nonlinearity=softmax)
	
	return l_in, l_out
	
'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

x_seed = np.zeros((1, SEQ_LENGTH, NUM_FEATURES))
for i, val in enumerate(pitch_matrix[:SEQ_LENGTH/2,:]):
	x_seed[0,i*2,:] = val
	x_seed[0,i*2+1,:] = val

# x_seed[0] = pitch_matrix[:SEQ_LENGTH]


print "Building network ..."
num_of_params = NUM_FEATURES * NUM_NEURONS*2 + (NUM_NEURONS + NUM_FEATURES) \
					+ (NUM_LAYERS-1) * (NUM_NEURONS**2) + (NUM_NEURONS*(NUM_LAYERS-1)) \
					+ NUM_LAYERS * NUM_NEURONS * (NUM_NEURONS+1)
print 'total no of params to be learned: {}'.format(num_of_params)

l_in, l_out = build_rnn()
target_values = T.imatrix('target_output')
network_output = lasagne.layers.get_output(l_out)

# The loss function of predictions and target.
loss = T.mean(categorical_crossentropy(network_output, target_values))

# Retrieve all parameters from the network
all_params = get_all_params(l_out, trainable=True)

# Compute AdaGrad updates for training
print "Computing updates for training w/ adam..."
updates = adam(loss_or_grads=loss, params=all_params, learning_rate=LEARNING_RATE)

# Theano functions for training and computing loss
print "Compiling functions ..."
train_function = theano.function([l_in.input_var, target_values], loss, updates=updates, allow_input_downcast=True)

# Need probability distribution of the next note given 
# the state of the network and the input_seed.
# Compile a theano function called probabilities to produce 
# the probability distribution of the prediction.

# `probabilities` is analogous to calling .predict()
probabilities = theano.function([l_in.input_var], network_output, allow_input_downcast=True)

'''  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || 
     \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/   
	   T  R  A  I  N  I  N  G  --  T  R  A  I  N  I  N  G  --  T  R  A  I  N  I  N  G
'''
# make the training data tensors X and Y, and train the model on them
print "Training ..."
X, Y = make_X_Y(pitch_matrix, data_size, SEQ_LENGTH, NUM_FEATURES)

p, avg_loss, loss_p, time_delta = 0, 0, [0], [0]
epoch, bs0 = 0, BATCH_SIZE
t_before = np.asarray(time.time())

index = np.arange(data_size)
np.random.shuffle(index)



while epoch < NUM_EPOCHS:	
	x, y = make_batch(p, X, Y, data_size, BATCH_SIZE,index)
	avg_loss += train_function(x, y)
	
	p = p + BATCH_SIZE
	epoch, BATCH_SIZE = compute_epoch(epoch, BATCH_SIZE, data_size, bs0, e0=200)
	
	if epoch == 1.0:
		print "At epoch:", round(epoch,3),", avg_loss=", avg_loss

	if np.abs(np.float(epoch) - np.round(epoch)) < 1e-8:
		# Another epoch completed.
		sys.stdout.flush()
		print "At epoch:", round(epoch,3),", avg_loss=", avg_loss," "*99,"\r",
		loss_p, time_delta ,t_before = append_to_loss_and_time_delta(loss_p,time_delta,avg_loss,t_before)
		
		if epoch % PRINT_FUGUES == 0.0:
			# generate a fugue every PRINT_FUGUES epochs
			generate_a_fugue(epoch=epoch, loss=loss_p, time_delta=time_delta) 
			print "At epoch:{}, avg_loss={}, .mid & .p files saved, t={}sec.".format(round(epoch,3), avg_loss, round(time.time()-t0))
	
		p, avg_loss = 0, 0

