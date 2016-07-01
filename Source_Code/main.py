import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, RecurrentLayer, LSTMLayer, DenseLayer, get_all_params
from lasagne.updates import adagrad
from lasagne.nonlinearities import sigmoid
from lasagne.objectives import binary_crossentropy, aggregate

import numpy as np
import pickle
import time
from helpers_to_main import *
							

# Declare Global Variables
NUM_FEATURES = None
X = None
Y = None
SEQUENCE_LENGTH = 48
BATCH_SIZE = 48
LEARNING_RATE = .01
NUM_EPOCHS = 10000
PRINT_FREQ = 1
THRESHOLD = .5
# D = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
data_size = None
# # # # # # # # # # # # # # 

t0 = time.time()
print "\nLoading pitch_matrix ..."

pitch_matrix = pickle.load(open('training_data/pitch_matrix_'+str(96/4)+'ticks_sh'+str(0)+'.p', 'rb'))

# patch the individual pitch_matrices of individual fugues
# into a single 2D numpy array and transpose it.
pitch_matrix = flatten_pitch_matrix(pitch_matrix)
NUM_FEATURES = pitch_matrix.shape[1]  # 63

# extract the lowest notes  only
# pitch_matrix = make_monophonic(pitch_matrix) 

pitch_matrix = pitch_matrix[:4800] # clip pitch_matrix to test the code.

# def pad_pitch_matrix(pitch_matrix):
# 	# make data_size % batch_size = 0
# 	N = pitch_matrix.shape[0]
# 	add = BATCH_SIZE - ((N-SEQUENCE_LENGTH) % BATCH_SIZE)
# 	add = add % BATCH_SIZE
# 	pitch_matrix = np.concatenate((pitch_matrix, np.zeros((add, NUM_FEATURES))), axis=0)
# 	
# 	return pitch_matrix
# pitch_matrix = pad_pitch_matrix(pitch_matrix)

'''   '''
# xtra = (pitch_matrix.shape[0] - SEQUENCE_LENGTH) % BATCH_SIZE
# pitch_matrix = pitch_matrix[ 0 : pitch_matrix.shape[0] - xtra]
'''   '''

data_size = pitch_matrix.shape[0] - SEQUENCE_LENGTH
print '\n\t-----------------------------'
print '\tpitch_matrix.shape[0] = {}'.format(pitch_matrix.shape[0])
print '\tdata_size = {}'.format(data_size)
print '\tSEQUENCE_LENGTH = {}'.format(SEQUENCE_LENGTH)
print '\tBATCH_SIZE = {}'.format(BATCH_SIZE) 
print '\tPRINT_FREQ = {}'.format(PRINT_FREQ)
print '\tNUM_FEATURES = {}'.format(NUM_FEATURES)
print '\tNUM_EPOCHS = {}'.format(NUM_EPOCHS)
print '\t-----------------------------\n'

'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def generate_a_fugue(epoch, cost, N=96/4*16*2):
	'''
	INPUT:  epoch and cost: Passed just for naming of the output files.	
			N: How many steps the fugue will be generated for.
			96/4 is the number of 16th notes in one bar
	'''
	
	# Advance the RNN model for N steps.
	fugue = np.zeros((N, NUM_FEATURES))
	
	# x, _ = make_batch(p, pitch_matrix, 1)
	x, _ = make_batch(2000, X, Y, 1)
	print "generate_a_fugue(epoch={})".format(epoch)
	for i in range(N):
		'''
		Pick the single note that got assigned the highest probability.
		Will be modified to accommodate polyphonic music creation
		'''
			
		predict = probs(x)#.ravel()
		ix = np.argmax(predict, axis=1)
		fugue[i][ix] = 1 
		x[0, 0:SEQUENCE_LENGTH-1, :] = x[:, 1:, :] 
		x[0, SEQUENCE_LENGTH-1, :] = 0 
		x[0, SEQUENCE_LENGTH-1, :] = fugue[i, :] 

		# probs(x).shape:  (1, 73)
		# print 'probs(x).shape: ', probs(x).shape 
		# print 'predict.shape={}'.format(predict.shape)
		# print 'x.shape={}'.format(x.shape)
		# print "i={}, ix={}".format(i,ix)

		# ... for polyphony:
		# predict = probs(x).ravel()
		# ix = predict > THRESHOLD

		# # # # # # # # # # # # # # # # # # 
		# fugue[i][ix] = 1 
		# x[:, 0:SEQUENCE_LENGTH-1, :] = x[:, 1:, :] 
		# x[:, SEQUENCE_LENGTH-1, :] = 0 
		# x[0, SEQUENCE_LENGTH-1, :] = fugue[i, :] 
		
	print_epoch = "%04d" % (epoch,)
	filename_f =  'fugue_N' + str(N)+ '_epoch' + print_epoch
	filename_t =  'train_N' + str(N)+ '_epoch' + print_epoch
	dump_dict = {'NUM_FEATURES' : NUM_FEATURES,
				'SEQUENCE_LENGTH' : SEQUENCE_LENGTH,
				'BATCH_SIZE' : BATCH_SIZE,
				'LEARNING_RATE' : LEARNING_RATE,
				'NUM_EPOCHS' : NUM_EPOCHS,
				'PRINT_FREQ' : PRINT_FREQ,
				'THRESHOLD' : THRESHOLD,
				'cost' : cost,
				'fugue' : fugue}
				
	pickle.dump(dump_dict, open('Synthesized_Fugues/'+filename_f+'.p', 'wb'))
	time_series = pitch_matrix_TO_time_series(fugue)
	time_series = time_series_legato(time_series)
	
	if len(time_series) > 0:
		print 'len(time_series)={}, N={} 16th notes'.format(len(time_series),N)

		time_series = time_series_legato(time_series)
		time_series_TO_midi_file(time_series, 'Synthesized_Fugues/'+filename_f+'.mid')
		# pickle.dump(train,     open('Synthesized_Fugues/'+filename_t+'.p', 'wb'))
	else:
		print 'NO NOTE PREDICTED'
	return
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''
# def input_seed(pitch_matrix):
# 	# starting from the middle index of the pitch matrix
# 	# extract a segment of length SEQUENCE_LENGTH
# 	# then strecth the notes to 2*SEQUENCE_LENGTH
# 	# eg. A, C#, C#, D => A, A, C#, C#, C#, C#, D, D
# 	
# 	mid = pitch_matrix.shape[0]/2
# 	extract = pitch_matrix[mid:mid+SEQUENCE_LENGTH]
# 	
# 	generation_phrase = np.zeros((SEQUENCE_LENGTH*2,pitch_matrix.shape[1]))
# 	generation_phrase[range(0,SEQUENCE_LENGTH*2,2)] = extract
# 	generation_phrase[range(1,SEQUENCE_LENGTH*2+1,2)] = extract
# 	
# 	return generation_phrase
# 	
# '''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
#    ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
#    \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def make_X_Y(pitch_matrix):

	X = np.zeros((data_size, SEQUENCE_LENGTH, NUM_FEATURES))
	Y = np.zeros((data_size, NUM_FEATURES))
	
	for i in range(data_size):
		X[i, : , :] = pitch_matrix[i : i+SEQUENCE_LENGTH, :]
		Y[i, :] = pitch_matrix[i+SEQUENCE_LENGTH]
		
	return X, Y

def make_batch(p, X, Y, batch_size=BATCH_SIZE):
	
	x = X[p : p+batch_size]
	y = Y[p : p+batch_size]
	return x, y


# def __make_batch_old(p, pitch_matrix=pitch_matrix, batch_size=BATCH_SIZE):
# 	'''Deprecated. 
# 	   use make_batch() instead
# 	'''
# 	x = np.zeros((batch_size, SEQUENCE_LENGTH, NUM_FEATURES))
# 	y = np.zeros((batch_size, NUM_FEATURES))
# 	for n in range(batch_size):
# 		x[n, : , :] = pitch_matrix[p+n:p+n+SEQUENCE_LENGTH, :]
# 		y[n, :] = pitch_matrix[p+n+SEQUENCE_LENGTH]
# 		return x, y
		
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def build_rnn(sequence_length=SEQUENCE_LENGTH, num_units=512):
	# input layer
	# generate the data to pass into this
	l_in = InputLayer(shape=(None, sequence_length, NUM_FEATURES))
	# LSTM Layer
	l_LSTM = LSTMLayer(l_in, num_units=num_units, grad_clipping=100, only_return_final=True)
	# output layer
	l_out = DenseLayer(l_LSTM, num_units=NUM_FEATURES, nonlinearity=sigmoid)
	return l_in, l_out
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def total_cost(predictions, target):
	'''
	INPUT np.array of size (73,), np.array of size (73,)
	OUTPUT: 
	'''
	note_cost = binary_crossentropy(predictions, target).mean()
	
	# chord_target = get_chord(target)
	# note_predictions = predictions[predictions > .5]
	# chord_predictions = get_chord(note_predictions)
	# chord_cost = cosine_distance(chord_predictions, chord_target)
	
	chord_cost = 0
	return note_cost + chord_cost
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

data_size = pitch_matrix.shape[0] - SEQUENCE_LENGTH

print "Building network ..."
l_in, l_out = build_rnn()
target_values = T.imatrix('target_output')
network_output = lasagne.layers.get_output(l_out)

# The loss function of predictions and target.
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
# Compile a theano function called probs to produce 
# the probability distribution of the prediction.

# `probs` is the .predict() function
probs = theano.function([l_in.input_var], network_output, allow_input_downcast=True)

# generation_phrase = input_seed(pitch_matrix)

''' ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||
    \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  
		T  R  A  I  N  I  N  G  --  T  R  A  I  N  I  N  G  --  T  R  A  I  N  I  N  G												
'''
# make training data tensors X and Y, and train the model on them
X, Y = make_X_Y(pitch_matrix)

print "Training ..."
cost = []
p = 0
avg_cost = 0
dummy_previous = -1

for it in xrange(data_size * NUM_EPOCHS / BATCH_SIZE):
	epoch = float(it)*float(BATCH_SIZE)/data_size
	if p  > data_size - 1:
		print '\nEpoch completed. epoch={}, it={}, p={}'.format(epoch,it, p)
		cost.append(avg_cost)
		p = 0
		avg_cost = 0
	pass
	
	print "it: ", it, "epoch: ", round(epoch,3), "\r",
		
	# print "Pick a new batch at p/data_size = {}/{}".format(p, data_size)
	x, y = make_batch(p, X, Y)
	avg_cost = avg_cost + train(x, y)
	
	if epoch % PRINT_FREQ < dummy_previous:
		'''
		End of PRINT_FREQ many epochs (typically at every 30 epoch).
		GENERATE FUGUE AND WRITE VARIABLES TO PICKLE FILE
		'''
		t4 = time.time()
		generate_a_fugue(epoch=epoch, cost=cost) 
		t5 = time.time()
		# Generate a fugue starting from a segment of length 
		# SEQUENCE_LENGTH starting at the p-th note of pitch_matrix 
		print "fugue generation completed in {} sec".format(round(t5-t4, 2))
		print "\t\t\t\tTOTAL RUN TIME: {} sec".format(round(t5-t0, 2))
		print("Epoch {} average loss = {}".format(epoch, avg_cost / PRINT_FREQ))
	pass
	
	p = p + BATCH_SIZE 			
	dummy_previous = epoch % PRINT_FREQ
	

'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''





