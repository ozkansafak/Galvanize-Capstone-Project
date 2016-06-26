import numpy as np	
D = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def build_chords_vocabulary_vectorized():
	'''
	OUTPUT: chords_vocabulary (12 by 1) LIST of chord objects, 
			canonical_chord_vectors (n by 12) NP.ARRAY of 
			vector representation of chords 
	'''
	chords_vocabulary = get_chords_vocabulary()
	# initialize and compute canonical_chord_vectors
	canonical_chord_vectors = np.zeros((len(chords_vocabulary), 12), dtype=float)
	for i, el in enumerate(chords_vocabulary):
		canonical_chord_vectors[i][el.notes] = np.round(el.wt[el.notes],3)
	
	return chords_vocabulary, canonical_chord_vectors
	
	
	
def cosine_similarity(x,y):
	num = x.dot(y)
	denom = np.linalg.norm(x) * np.linalg.norm(y)
	return num/float(denom)



def cosine_distance(x,y):
	sim = cosine_similarity(x, y)
	return np.arccos(sim) / np.pi*180 # angle in degrees


def get_chord_from_gr_pitches(gr_pitches):
	'''
	INPUT: NDARRAY
	OUTPUT: (STR, STR)
	'''

	out = []
	for sh in range(12): 
		# transpose up by sh semitones
		gr_notes = pitches_to_notes(gr_pitches + sh)
		dict_ = predict_chord_type(gr_notes)
		dict_['shift'] = sh
		out.append(dict_)

	out.sort(key=lambda x: x['dist'], reverse=False) # reverse=False: ascending order
	sh = out[0]['shift'] # smallest distance

	root = D[-out[0]['shift']]
	chord_type = out[0]['chord_type']
	return root, chord_type, out


'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  
'''
# class chord(object):
# 	def __init__(self, name, lst):
# 		# weight constants
# 		e = .125 # ordinary notes in the scale
# 
# 		root = 10*e
# 		second = 3*e
# 		third = 7*e
# 		forth = 3*e
# 		fifth = 7*e
# 		sixth = 4*e
# 		seventh = 7*e
# 		
# 		self.notes = np.array(lst)
#  		self.name = name
# 		
# 		# probability of chromatic tones
# 		self.wt = np.array([-e for i in range(12)]) 
# 		self.wt[lst[0]] = root
# 		self.wt[lst[1]] = second
# 		self.wt[lst[2]] = third
# 		self.wt[lst[3]] = forth
# 		self.wt[lst[4]] = fifth
# 		self.wt[lst[5]] = sixth
# 		self.wt[lst[6]] = seventh
# 		
# 		# Basic Notion: * punish wrong 3rd and 7ths
# 		#               * reward color tones
# 		if self.name == 'minor_diminished':
# 			self.wt[lst[-1]] = seventh
# 		
# 		# normalize
# 		self.wt = self.wt/np.linalg.norm(self.wt)
# 
# def get_chords_vocabulary():
# 	chords_vocabulary = []
# 	# Am
# 	chords_vocabulary.append(chord('minor', [0, 2, 3, 5, 7, 8, 10]))
# 	# Am phrygian 
# 	chords_vocabulary.append(chord('minor_phrygian', [0, 1, 3, 5, 7, 8, 10]))
# 	# Gm Maj7
# 	chords_vocabulary.append(chord('minor_harmonic', [0, 2, 3, 5, 7, 8, 11]))
# 	# Gm Maj6
# 	chords_vocabulary.append(chord('minor_melodic', [0, 2, 3, 5, 7, 9, 11]))
# 	# Gm dim WARNING: This is an OCTATONIC Scale
# 	chords_vocabulary.append(chord('minor_diminished', [0, 2, 3, 5, 6, 8, 9, 11]))
# 	# Gm b5
# 	chords_vocabulary.append(chord('minor_half_diminished', [0, 2, 3, 5, 6, 8, 10]))
# 	# GMaj
# 	chords_vocabulary.append(chord('major', [0, 2, 4, 5, 7, 9, 11]))
# 	# AMaj aug
# 	chords_vocabulary.append(chord('major_augmented', [0, 2, 4, 6, 7, 9, 11]))
# 	# G7
# 	chords_vocabulary.append(chord('dominant', [0, 2, 4, 5, 7, 9, 10]))
# 	# A7#9, A7b9, A7b13
# 	chords_vocabulary.append(chord('dominant_altered', [0, 1, 3, 4, 6, 8, 10]))
# 	# A7#11
# 	chords_vocabulary.append(chord('dominant_sharp_11', [0, 2, 4, 6, 7, 9, 10]))
# 	
# 	return chords_vocabulary
# 
# 
# def fold_down_target(input):
# 	'''
# 	INPUT: n-by-1 numpy array, n=73 keys
# 	OUTPUT:12-by-1 numpy array,
# 	'''
# 	n = input.shape[0]
# 	output = np.zeros(12)
# 
# 	for octave in range(n/12):
# 		# print octave*12 + np.arange(12)
# 		output = output + input[octave*12 + np.arange(12)]
# 
# 	# take mean of all octaves aggregated
# 	output = output/float(octave+1)
# 
# 	# last remaining octave
# 	octave = octave + 1
# 	for i in range(12):
# 		j = octave*12 + i
# 		if j <= n-1:
# 			output[i] = output[i] + input[octave*12 + i]
# 			# float(octave)/float(octave+1) term adjusts for the mean  
# 			# already calculated above
# 			output[i] = output[i] * float(octave) / float(octave+1) 
# 			
# 	return output
# 	
# 
# def transpose_target_up_1_fret(target_folded):
# 	return np.hstack((target_folded[-1], target_folded[0:-1]))
# 	
# 
# def	get_chord_similarity(target, chords_vocabulary):
# 	target_folded = fold_down_target(target)
# 	# NB should rename chords_vocabulary as canonical_chords_vocabulary 
# 	
# 	# initialize
# 	chord_similarity = np.zeros((12, len(chords_vocabulary)))
# 	for sh in range(12): 
# 		if sh != 0:
# 			target_folded = transpose_target_up_1_fret(target_folded)
# 		for i, ch in enumerate(chords_vocabulary):
# 			chord_similarity[sh,i] = target_folded.dot(ch.wt)
# 			# print cosine_distance(ch.wt, target_folded), ch.name
# 	
# 	return chord_similarity
# 
# 
# def get_chord(target):
# 	
# 	chords_vocabulary = get_chords_vocabulary()
# 	chord_similarity = get_chord_similarity(target, chords_vocabulary)
# 	
# 	# most_similar_chord_across_all_roots.shape = (12,)
# 	# indices.shape = (12,)
# 	# chord_similarity.shape = (12, 11) 
# 	# for each root (C, C#, D ..,B) the closest chord type 
# 	most_similar_chord_across_all_roots = chord_similarity.max(axis=1)
# 	indices = chord_similarity.argmax(axis=1)
# 	
# 	# (root, chord_type_index) is the closest chord to target.
# 	root = most_similar_chord_across_all_roots.argmax()
# 	chord_type_index = indices[root]
# 	print '{} {}'.format(D[root], chords_vocabulary[chord_type_index].name)
# 	
# 	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || 
'''
	
def notes_to_vector(gr_notes, chords_vocabulary):
	# a helper func
	'''
	INPUT: gr_notes: 1-by-sth NP.ARRAY with distinct ascending values in 0,..11
		   chords_vocabulary: (12 by 1) LIST of chord objects, 
		   
	OUTPUT: N-by-12 NP.ARRAY
	'''
	vector = np.zeros((len(chords_vocabulary), 12), dtype=float)
	for i, el in enumerate(chords_vocabulary):
		vector[i, gr_notes] += el.wt[gr_notes]
		# normalize just for the heck of it.
		vector[i] = vector[i]/np.linalg.norm(vector[i])
		
	return vector
	
	

def predict_chord_type(gr_notes):
	'''
	INPUT: 1-by-sth NP.ARRAY with distinct ascending values in 0,..11 
	OUTPUT: (INT, STR) 
			gr_notes is a sorted ascending list. eg [1, 4, 6, 8]
			1st output variable: the index in chords_vocabulary
	chord_type is the type of chord (e.g. 'major','dominant_sharp_11')
	'''
	# print '\n\ngr_notes =', gr_notes
	chords_vocabulary, canonical_chord_vectors = build_chords_vocabulary_vectorized()
	
	# vector: a vector representation of gr_notes. 
	# 		  each entry is weighted with the 
	# 		  corresponding chord weights
	
	vector = notes_to_vector(gr_notes, chords_vocabulary)

	# compute cosine_similarity and pick the closest chord
	for id in range(len(vector)):
		# arccos(cos_sim(x,y)) maps (-1,1) to (pi,0)
		
		sim = cosine_similarity(vector[id], canonical_chord_vectors[id])
		dist = cosine_distance(vector[id], canonical_chord_vectors[id])

		if id == 0 or dist < memory['dist']: # keep the smallest distance and according id.
			memory = {'dist': dist, 'chord_id': id}
		# print 'distance: {} -- {},'.format(round(dist,3), chords_vocabulary[id].name)
		
	chord_type = chords_vocabulary[memory['chord_id']].name
	
	return {'chord_type': chord_type,\
	 		'chord_id': memory['chord_id'], \
			'dist': memory['dist']}

	
	 
def pitches_to_notes(gr_pitches):
	# a helper func
	'''
	INPUT: NP.ARRAY sth-by-2
	OUTPUT: NP.ARRAY 1-by-sth
	
	'''
	gr_notes = list(set([p%12 for p in gr_pitches]))
	gr_notes = np.array(sorted(gr_notes))
	return gr_notes
	

	

def pitches_to_vector(gr_pitches):	
	
	gr_notes = pitches_to_notes(gr_pitches)
	out = np.zeros((12))
	out[gr_pitches] = 1
	
	return out
	
	
	






	