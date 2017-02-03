import numpy as np
import midi
import pickle 
import itertools
import time

def append_to_loss_and_time_delta(loss_p,time_delta,avg_loss,t_before):
	loss_p.append(avg_loss)
	t = time.time()
	time_delta = np.append(time_delta, t-t_before)
	t_before = t

	return loss_p, time_delta, t_before


def compute_prime_factors(n):
	n = int(n)
	i = 2
	prime_factors = []
	while i * i <= n:
		if n % i:
			i += 1
		else:
			n //= i
			prime_factors.append(i)
	if n > 1:
		prime_factors.append(n)
	return prime_factors



def compute_factors(data_size, bs0):
	prime_factors = compute_prime_factors(data_size)
	factors_repeated = []
	
	for L in range(1, len(prime_factors)+1):
	  for subset in itertools.combinations(prime_factors, L):
	     factors_repeated.append(reduce(lambda x,y: x*y, subset))

	factors_repeated.sort(reverse=False) # in-place sort. Ascending

	factors = [factors_repeated[0]]
	for val in factors_repeated:
		if factors[-1] != val:
			factors.append(val)

	factors = np.asarray(factors)

	return factors[factors>=bs0]
	
	
def compute_epoch(epoch, BATCH_SIZE, data_size, bs0, e0=200):

	# Advance the epoch with current BATCH_SIZE
	epoch += float(BATCH_SIZE) / data_size
	if np.abs(np.float(epoch) - np.round(epoch)) < 1e-8: 
		epoch = round(epoch)
		# we're starting a new epoch
		factors = compute_factors(data_size, bs0)
	
		# compute adaptive BATCH_SIZE
		# At e0 = 200.0, BATCH_SIZE = data_size
		bs = (data_size-bs0) * (np.floor(epoch)/float(e0))**2 + bs0 
		
		bs_candidate = factors[factors <= bs].max()
		if bs_candidate != BATCH_SIZE:
		 	BATCH_SIZE = bs_candidate
			print "At epoch:", round(epoch,3),", BATCH_SIZE=", BATCH_SIZE, " "*99
		
	
	return epoch, BATCH_SIZE


def load_n_flatten_pitch_matrix():
	print "\n\n-------------------------------"
	print "Loading pitch_matrix ..."
	sh = 0
	pitch_matrix = pickle.load(open('training_data/pitch_matrix_'+str(96/4)+'ticks_sh'+"%02d" % (sh,)+'.p', 'rb'))
	# at this point, pitch_matrix is a list of 2D lists. 
	# pitch_matrix[i] is a fugue in the training set, for each i.

	# flatten pitch_matrix into a single 2D ndarray
	pitch_matrix = flatten_pitch_matrix(pitch_matrix)
	
	return pitch_matrix
	
	
def print_inputs(NUM_LAYERS, NUM_NEURONS, NUM_FEATURES,
				SEQ_LENGTH, BATCH_SIZE, LEARNING_RATE,
				PRINT_FUGUES, data_size, pitch_matrix, 
				full_data_size):
	# print the network architecture
	print '\n\t-----------------------------'
	print '\t Network Architecture:\n'
	for _ in range(3):
		print ' '*12 + '       ' + 'O      '*NUM_LAYERS + ' '
	for _ in range(4):
		print ' '*12 + 'O      ' + 'O      '*NUM_LAYERS + 'O'
	for _ in range(3):
		print ' '*12 + '       ' + 'O      '*NUM_LAYERS + ' '
	print ' '*12 + " "
	unit_no = "%3d" % (NUM_NEURONS,) + " "*4 
	print ' '*12+"%2d" % (NUM_FEATURES,) + " "*4 + unit_no*NUM_LAYERS + " %2d" % (NUM_FEATURES,)  

	print '\n\t-----------------------------'
	print '\tSEQ_LENGTH = {}'.format(SEQ_LENGTH)
	print '\tBATCH_SIZE = {}'.format(BATCH_SIZE) 
	print '\tLEARNING_RATE = {}'.format(LEARNING_RATE)
	print '\tdata_size = {} ({}% of available BWV data)'.format(data_size,round(100.*data_size/full_data_size,2))
	print '\tPRINT_FUGUES = {}'.format(PRINT_FUGUES)
	print '\t-----------------------------\n'
	
	return

'''def adjust_pitch_matrix(pitch_matrix, data_size, batch_size, num_features, sequence_length):
	p = (data_size / batch_size) * batch_size
	leftOver = (p+batch_size) % (data_size)
	if leftOver > 0:
		print "adjust_pitch_matrix(): Broken batch. leftOver =", leftOver
		# clip pitch_matrix so batch_size is an integer multiple of  
		pitch_matrix = pitch_matrix[:p+sequence_length] 
		data_size = pitch_matrix.shape[0] - sequence_length
		
		# pitch_matrix = np.concatenate((pitch_matrix,np.zeros((leftOver,num_features))), axis=0)
		print 'adjust_pitch_matrix reports: pitch_matrix.shape =', pitch_matrix.shape

	return pitch_matrix, data_size'''

def make_X_Y(pitch_matrix, data_size, sequence_length, num_features):
	'''pitch_matrix.shape '''
	X = np.zeros((data_size, sequence_length, num_features))
	Y = np.zeros((data_size, num_features))
	
	for i in xrange(data_size):
		X[i, : , :] = pitch_matrix[i : i+sequence_length, :]
		Y[i, :] = pitch_matrix[i+sequence_length]

	return X, Y
	

def make_batch(p, X, Y, data_size, batch_size,index):
	# random shuffle the training data X,Y,
	# Then take batch_size many
	x = X[index[p : p+batch_size]]
	y = Y[index[p : p+batch_size]]
	return x, y

	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def flatten_pitch_matrix(pitch_matrix):
	'''
	INPUT: list of 2d np.arrays 
	OUTPUT: 2d np.array 
	
	# patch the individual pitch_matrices of individual fugues
	# into a single 2D numpy array.
	# Transpose it, then return it.
	
	'''
	# flatten pitch_matrix
	pitch_matrix_long = pitch_matrix[0]
	for n in xrange(len(pitch_matrix)-1):
		pitch_matrix_long = np.column_stack((pitch_matrix_long, pitch_matrix[n+1]))

	# transpose it to swap the axes. 
	# pitch_matrix.shape = (N, NUM_FEATURES)
	
	return np.transpose(pitch_matrix_long)

'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''

def make_monophonic(pitch_matrix):
	'''
	INPUT: 2D numpy array -- flattened pitch matrix sth-by-NUM_FEATURES
	OUTPUT: 2D numpy array -- same shape
	This is a preprocess step that's invoked in model.py on AWS
	'''
	# pitch_matrix is already flattened upon being loaded in model.py
	# 		pitch_matrix.shape[1] = 73
	c=0
	for vec in pitch_matrix:
		c+=1
		for i, key in enumerate(vec):
			if key == 1: 
				break
		vec[i+1:] = 0
	
	return pitch_matrix

'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''

def pitch_matrix_TO_time_series_legato(pitch_matrix, bar=96/4, sh=0):
	'''
	INPUT: 2D numpy array (flattened. ie shape=sth-by-NUM_FEATURES)
	OUTPUT: list of tuples [(time, pitch, duration), ...]
	'''
	""" This seems to work right """
	time_series = []
	for pitch, row in enumerate(np.transpose(pitch_matrix)):
		i = 0	
		while i < len(row):
			if row[i] == 1:
				c = 0
				while i < len(row) and row[i] == 1:
					c+=1
					i+=1
				time_series.append(((i-c)*bar, pitch+sh, c*bar))
				i -= 1
			i += 1
	
	time_series.sort(key=lambda x: x[0])
	return time_series	

'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''

def time_series_TO_midi_file(time_series, filename="Synthesized_Fugues/out.mid"):
	'''
	INPUT: time_series LIST [(time, pitch, duration), ...]
	OUTPUT: track midi.Track()
	
	This is going to be used to convert RNN generated fugues back to MIDI
	'''
	track = midi.Track()
	
	# U list of noteEvents [('on', time, pitch), ...]
	U = []
	for (time, pitch, duration) in time_series:
		U.append(('on', time, pitch))
		U.append(('off', time+duration, pitch))
	
	#order the list of events	
	U.sort(key=lambda x: x[1])
	cursor = 0
	for (typ, time, pitch) in U:
		tick = time - cursor
		if typ == 'on':
			o = midi.NoteOnEvent(tick=tick, velocity=127, pitch=pitch)
		else:
			o = midi.NoteOffEvent(tick=tick, velocity=127, pitch=pitch)
		track.append(o)
		cursor = time

	# Add end of track event
	eot = midi.EndOfTrackEvent(tick=1)
	track.append(eot)
	
	pattern = midi.Pattern()
	pattern.append(track)
	midi.write_midifile(filename, pattern)
	
	return


'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/ '''



if __name__ == '__main__':
	# time_series_list = time_series_list_builder(filepath)
	pitch_matrix = time_series_list_TO_pitch_matrix(time_series_list)
	time_series = pitch_matrix_TO_time_series_legato(np.transpose(pitch_matrix))
	time_series_TO_midi_file(time_series)
	







