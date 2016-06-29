import numpy as np	
import midi
D = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''
class chord(object):
	def __init__(self, name, lst):
		# weight constants
		e = .125 # ordinary notes in the scale

		root = 10*e
		second = 3*e
		third = 7*e
		forth = 3*e
		fifth = 7*e
		sixth = 4*e
		seventh = 7*e
		
		self.notes = np.array(lst)
 		self.name = name
		
		# probability of chromatic tones
		self.wt = np.array([-e for i in range(12)]) 
		self.wt[lst[0]] = root
		self.wt[lst[1]] = second
		self.wt[lst[2]] = third
		self.wt[lst[3]] = forth
		self.wt[lst[4]] = fifth
		self.wt[lst[5]] = sixth
		self.wt[lst[6]] = seventh
		
		# Basic Notion: * punish wrong 3rd and 7ths
		#               * reward color tones
		if self.name == 'minor_diminished':
			self.wt[lst[-1]] = seventh
		
		# normalize
		self.wt = self.wt/np.linalg.norm(self.wt)

def get_chords_vocabulary():
	chords_vocabulary = []
	# Am
	chords_vocabulary.append(chord('minor', [0, 2, 3, 5, 7, 8, 10]))
	# Am phrygian 
	chords_vocabulary.append(chord('minor_phrygian', [0, 1, 3, 5, 7, 8, 10]))
	# Gm Maj7
	chords_vocabulary.append(chord('minor_harmonic', [0, 2, 3, 5, 7, 8, 11]))
	# Gm Maj6
	chords_vocabulary.append(chord('minor_melodic', [0, 2, 3, 5, 7, 9, 11]))
	# Gm dim WARNING: This is an OCTATONIC Scale
	chords_vocabulary.append(chord('minor_diminished', [0, 2, 3, 5, 6, 8, 9, 11]))
	# Gm b5
	chords_vocabulary.append(chord('minor_half_diminished', [0, 2, 3, 5, 6, 8, 10]))
	# GMaj
	chords_vocabulary.append(chord('major', [0, 2, 4, 5, 7, 9, 11]))
	# AMaj aug
	chords_vocabulary.append(chord('major_augmented', [0, 2, 4, 6, 7, 9, 11]))
	# G7
	chords_vocabulary.append(chord('dominant', [0, 2, 4, 5, 7, 9, 10]))
	# A7#9, A7b9, A7b13
	chords_vocabulary.append(chord('dominant_altered', [0, 1, 3, 4, 6, 8, 10]))
	# A7#11
	chords_vocabulary.append(chord('dominant_sharp_11', [0, 2, 4, 6, 7, 9, 10]))
	
	return chords_vocabulary


def fold_down_target(input):
	'''
	INPUT: n-by-1 numpy array, n=73 keys
	OUTPUT:12-by-1 numpy array,
	'''
	n = input.shape[0]
	output = np.zeros(12)

	for octave in range(n/12):
		# print octave*12 + np.arange(12)
		output = output + input[octave*12 + np.arange(12)]

	# take mean of all octaves aggregated
	output = output/float(octave+1)

	# last remaining octave
	octave = octave + 1
	for i in range(12):
		j = octave*12 + i
		if j <= n-1:
			output[i] = output[i] + input[octave*12 + i]
			# float(octave)/float(octave+1) term adjusts for the mean  
			# already calculated above
			output[i] = output[i] * float(octave) / float(octave+1) 
			
	return output
	

def transpose_target_up_1_fret(target_folded):
	return np.hstack((target_folded[-1], target_folded[0:-1]))


def	get_chord_similarity(target, chords_vocabulary):
	target_folded = fold_down_target(target)
	# NB should rename chords_vocabulary as canonical_chords_vocabulary 
	
	# initialize
	chord_similarity = np.zeros((12, len(chords_vocabulary)))
	for sh in range(12): 
		if sh != 0:
			target_folded = transpose_target_up_1_fret(target_folded)
		for i, ch in enumerate(chords_vocabulary):
			chord_similarity[sh,i] = target_folded.dot(ch.wt)
			# print cosine_distance(ch.wt, target_folded), ch.name
	
	return chord_similarity


def get_chord(target):
	
	chords_vocabulary = get_chords_vocabulary()
	chord_similarity = get_chord_similarity(target, chords_vocabulary)
	
	# most_similar_chord_across_all_roots.shape = (12,)
	# indices.shape = (12,)
	# chord_similarity.shape = (12, 11) 
	# for each root (C, C#, D ..,B) the closest chord type 
	most_similar_chord_across_all_roots = chord_similarity.max(axis=1)
	indices = chord_similarity.argmax(axis=1)
	
	# (root, chord_type_index) is the closest chord to target.
	root = most_similar_chord_across_all_roots.argmax()
	chord_type_index = indices[root]
	print '{} {}'.format(D[root], chords_vocabulary[chord_type_index].name)
	
	
	transpose_target_up_1_fret(target_folded)
	out = chords_vocabulary[chord_type_index]
	
	for i in range(root):
		out = transpose_target_up_1_fret(out)
	return out


'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''

def flatten_pitch_matrix(pitch_matrix):
	'''
	INPUT: list of 2d np.arrays 
	OUTPUT: 2d np.array 
	'''
	# flatten pitch_matrix
	pitch_matrix_long = pitch_matrix[0]
	for n in range(len(pitch_matrix)-1):
		pitch_matrix_long = np.column_stack((pitch_matrix_long, pitch_matrix[n+1]))

	# transpose it to swap the axes. 
	# pitch_matrix.shape = (N, NUM_FEATURES)
	pitch_matrix = np.transpose(pitch_matrix_long)	
	return pitch_matrix

'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def make_monophonic(pitch_matrix):
	'''
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
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def make_time_series(pitch_matrix, lowest_pitch=24):
	
	bar = 96/4
	dir = 'Synthesized_Fugues/'
	time_series = []
	for t_step, vec in enumerate(pitch_matrix):
		for pitch, key in enumerate(vec):
			if key == 1:
				time_series.append((t_step*bar, pitch+lowest_pitch, bar))
		
	return time_series

'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def time_series_legato(time_series):
	'''
	a Post-processing function
	time series is a single list of tuples 
	
	This only works for monophonic time_series
	'''
	
	pitch_previous = -1
	new_time_series = []
	
	for (time, pitch, duration) in time_series:
		if pitch == pitch_previous:
			new_time_series.pop()
			new_time_series.append((time_previous, pitch, duration_previous + duration))
			duration_previous = duration_previous+duration
		else:
			new_time_series.append((time, pitch, duration))
			duration_previous = duration
			pitch_previous = pitch
			time_previous = time

	return new_time_series

'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def time_series_to_MIDI_file(time_series, filename="Synthesized_Fugues/out.mid"):
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


if __name__ == '__main__':
	time_series = make_time_series(pitch_matrix)
	time_series = time_series_legato(time_series)
	time_series_to_MIDI_file(time_series)
	




















