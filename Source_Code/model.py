import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, RecurrentLayer, LSTMLayer, DenseLayer, get_all_params
from lasagne.updates import adagrad
from lasagne.nonlinearities import sigmoid
from lasagne.objectives import binary_crossentropy, aggregate
import pickle
import time
from helpers_to_model import make_time_series, make_monophonic, \
							 flatten_pitch_matrix, time_series_legato, \
							 time_series_to_MIDI_file


t0 = time.time()
print "\nLoading pitch_matrix ..."
pitch_matrix = pickle.load(open('training_data/pitch_matrix_'+str(96/4)+'ticks_sh'+str(0)+'.p', 'rb'))

# patch the individual pitch_matrices of raw MIDI files into
# a single 2D array and transpose it.
pitch_matrix = flatten_pitch_matrix(pitch_matrix)

# extract the lowest notes  only
pitch_matrix = make_monophonic(pitch_matrix)

# clip it to get a shorter data set
print "(Retain only first 1600 notes in pitch_matrix)"
pitch_matrix = pitch_matrix[:4800]

pitch_matrix = make_monophonic(pitch_matrix)

t1 = time.time()
print "\t\ttime: ", round(t1-t0, 4), "sec"

NUM_FEATURES = pitch_matrix.shape[1] # 73
SEQUENCE_LENGTH = 48
BATCH_SIZE = 48
LEARNING_RATE = .01
NUM_EPOCHS = 10
PRINT_FREQ = 1000
THRESHOLD = .5

D = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def generate_a_fugue(epoch, N=96/4*16*2):
	# Advance the RNN model for N steps, generating a note at each step. 
	# generate N sequential notes. 	
	fugue = np.zeros((N, NUM_FEATURES))
	
	x, _ = make_batch(p, pitch_matrix, 1)
	print "generate_a_fugue(epoch={})".format(epoch)
	counter = 0
	for i in range(N):
		# Pick only a single note that got assigned the highest probability.
		# Will be modified to accommodate polyphonic music creation
			
		predict = probs(x)#.ravel()
		# probs(x).shape:  (1, 73)
		# print 'probs(x).shape: ', probs(x).shape 
		# print 'predict.shape={}'.format(predict.shape)
		# print 'x.shape={}'.format(x.shape)
		ix = np.argmax(predict, axis=1)
		# print "i={}, ix={}".format(i,ix)

		#    ... for polyphony:
		# predict = probs(x).ravel()
		# ix = predict > THRESHOLD

		# # # # # # # # # # # # # # # # # # 
		fugue[i][ix] = 1 
		x[:, 0:SEQUENCE_LENGTH-1, :] = x[:, 1:, :] 
		x[:, SEQUENCE_LENGTH-1, :] = 0 
		x[0, SEQUENCE_LENGTH-1, :] = fugue[i, :] 
		
	filename =  'fugue_N' + str(N) + '_epoch' + str(int(epoch))
	pickle.dump(fugue, open('Synthesized_Fugues/'+filename+'.p', 'wb'))
	time_series = make_time_series(fugue)
	time_series = time_series_legato(time_series)
	
	if len(time_series) > 0:
		print 'len(time_series)', len(time_series)

		time_series = time_series_legato(time_series)
		
		time_series_to_MIDI_file(time_series, 'Synthesized_Fugues/'+filename+'.mid')
	else:
		print 'NO NOTE PREDICTED'
	print 'counter =', counter
	return
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''
def input_seed(pitch_matrix):
	# starting from the middle index of the pitch matrix
	# extract a segment of length SEQUENCE_LENGTH
	# then strecth the notes to 2*SEQUENCE_LENGTH
	# eg. A, C#, C#, D => A, A, C#, C#, C#, C#, D, D
	
	
	mid = pitch_matrix.shape[0]/2
	extract = pitch_matrix[mid:mid+SEQUENCE_LENGTH]
	
	generation_phrase = np.zeros((SEQUENCE_LENGTH*2,pitch_matrix.shape[1]))
	generation_phrase[range(0,SEQUENCE_LENGTH*2,2)] = extract
	generation_phrase[range(1,SEQUENCE_LENGTH*2+1,2)] = extract
	
	return generation_phrase
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def make_batch(p, pitch_matrix=pitch_matrix, batch_size=BATCH_SIZE):
	x = np.zeros((batch_size, SEQUENCE_LENGTH, NUM_FEATURES))
	y = np.zeros((batch_size, NUM_FEATURES))
	for n in range(batch_size):
		x[n, : , :] = pitch_matrix[p+n:p+n+SEQUENCE_LENGTH, :]
		y[n, :] = pitch_matrix[p+n+SEQUENCE_LENGTH]
		return x, y
		
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

data_size = pitch_matrix.shape[0]

print "Building network ..."
l_in, l_out = build_rnn()
target_values = T.imatrix('target_output')
network_output = lasagne.layers.get_output(l_out)

# The loss function of predictions and target.
cost = total_cost(network_output, target_values)

# Retrieve all parameters from the network
all_params = get_all_params(l_out, trainable=True)

# Compute AdaGrad updates for training
print "Computing updates w/ adagrad..."
updates = adagrad(loss_or_grads=cost, params=all_params, learning_rate=LEARNING_RATE)
t2 = time.time()
print "\t\ttime: ", round(t2-t0, 4), "sec"

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

t3 = time.time()
print "\t\ttime: ", round(t3-t0, 4), "sec"

generation_phrase = input_seed(pitch_matrix)

''' ................................ '''	
''' ................................ '''

print "Training ..."
p = 0
for it in xrange(data_size * NUM_EPOCHS / BATCH_SIZE):
	epoch = float(it)*PRINT_FREQ*BATCH_SIZE/data_size
	print "it: {}".format(it)
	t4 = time.time()
	# Generate a fugue starting from a SEQUENCE_LENGTH long 
	# segment starting at the p-th note of pitch_matrix
	generate_a_fugue(epoch=epoch) 
	t5 = time.time()
	print "\t\tTOTAL TIME: ", round(t5-t0, 4), "sec"
	
	avg_cost = 0;
	for _ in range(PRINT_FREQ):
		x,y = make_batch(p, pitch_matrix)
		
		
		p = p + BATCH_SIZE 
		if(p+BATCH_SIZE+SEQUENCE_LENGTH >= data_size):
			# print('Carriage Return')
			p = 0;
			
		avg_cost += train(x, y)
		
	print("Epoch {} average loss = {}".format(epoch, avg_cost / PRINT_FREQ))
	

'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''
