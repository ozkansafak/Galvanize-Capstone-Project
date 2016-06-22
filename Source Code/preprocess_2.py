import numpy as np	
D = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def cosine_similarity(x,y):
	
	num = x.dot(y)
	denom = np.linalg.norm(x) * np.linalg.norm(y)
	return num/float(denom)



def note_name(pitch):
	'''
	INPUT: pitch INT
	OUTPUT: note STR
	'''
	
	octave = pitch/12
	note = D[pitch%12]
	
	return octave, note


def build_chords_vocabulary():
	'''
	OUTPUT: chords_vocabulary (12 by 1) LIST of chord objects, 
			canonical_chord_vectors (n by 12) NP.ARRAY of 
			vector representation of chords 
	
	'''

	class chord(object):
		def __init__(self, name, lst):
			# weight constants
			e = .15 # ordinary notes in the scale
			root = 1
			third = 5*e
			forth = 2*e
			fifth = 5*e
			seventh = 2*e
			
			self.notes = np.array(lst)
	 		self.name = name
			
			# probability of chromatic notes
			self.wt = np.array([-e/2 for i in range(12)]) 
			self.wt[lst[0]] = root
			self.wt[lst[1]] = e
			self.wt[lst[2]] = third
			self.wt[lst[3]] = forth
			self.wt[lst[4]] = fifth
			self.wt[lst[5]] = e
			self.wt[lst[6]] = seventh
			
			# Basic Notion: * punish wrong 3rd and 7ths
			#               * reward color tones
			if self.name[0:5] == 'minor':
				self.wt[4] = -10*e # punish wrong 3rd
			elif self.name[0:5] == 'major':
				self.wt[3] = -10*e # punish wrong 3rd
				self.wt[10] = -10*e # punish wrong 7th
			elif self.name == 'dominant':
				self.wt[3] = -10*e # minor 3rd
				self.wt[10] = -10*e # punish wrong 7th
			elif self.name == 'dominant_altered': 
				# [0, 1, 3, 4, 6, 8, 10]
				self.wt[5] = -10*e # punish 4th

				self.wt[lst[3]] = third # swap minor 3rd and major 3rd
				self.wt[lst[2]] = forth # ...
				self.wt[lst[2]] = 2*e # minor 3rd. raise value over an ordinary note
				self.wt[lst[5]] = 2*e # sharp 5. put it back
				self.wt[1] = 2*e # flat 9th
				self.wt[11] = -10*e
			elif self.name == 'dominant_sharp_11':
				self.wt[3] = -10*e
				self.wt[5] = -10*e # punish 4th
				self.wt[11] = -10*e
				self.wt[6] += e # add to #11
				self.wt[lst[1]] -= e/2 # subtract back what you added
				self.wt[lst[5]] -= e/2
				
			if self.name == 'minor_diminished':
				self.wt[11] += e
			if self.name =='minor_harmonic':
				self.wt[11] = 2*e
			if self.name =='major_augmented':
				self.wt[8] += 4*e 
				self.wt[7] =  2*e 				
			if self.name =='minor_melodic':
				self.wt[11] = 3*e
				self.wt[9] = 3*e
				
			# normalize
			self.wt = self.wt/np.linalg.norm(self.wt)
			
	# e.g.
	# self.notes = [0, 2, 3, 5, 7, 8, 10]
	# self.wt = [1, -0.1, 0.1, 0.75, -0.5, 0.3, -0.1, 0.75, 0.1, -0.1, 0.3, -0.1]
	# self.name = 'minor'
		
	chords_vocabulary = []
	# Am
	chords_vocabulary.append(chord('minor', [0, 2, 3, 5, 7, 8, 10]))
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

	# # # # # 
	# initialize and compute canonical_chord_vectors
	canonical_chord_vectors = np.zeros((len(chords_vocabulary), 12), dtype=float)
	for i, el in enumerate(chords_vocabulary):
		canonical_chord_vectors[i][el.notes] = el.wt[el.notes]
	
	return chords_vocabulary, canonical_chord_vectors

	
	
	
def find_chord(set_of_notes):
	'''
	INPUT: set_of_notes SET
	OUTPUT: (INT, STR) 
			1st output variable: the index in chords_vocabulary
	chord_type is the type of chord (e.g. 'major','dominant_sharp_11')
	'''
	
	chords_vocabulary, canonical_chord_vectors = build_chords_vocabulary()
	
	# vector: a vector representation of set_of_notes. 
	# 		  each entry is weighted with the 
	# 		  corresponding chord weights
	vector = np.zeros((len(chords_vocabulary), 12), dtype=float)
	for i, el in enumerate(chords_vocabulary):
		vector[i, set_of_notes] += el.wt[set_of_notes]
	
	# compute cosine_similarity and pick the closest chord
	for i in range(len(vector)):
		val = cosine_similarity(vector[i], canonical_chord_vectors[i])
		if i == 0 or val > memory['val']:
			memory = {'val': val, 'i':i}
		print 'likeliness: {} -- {},'.format(round(val,3), chords_vocabulary[i].name)
		
	chord_type = chords_vocabulary[memory['i']].name
	
	return {'chord_type':chord_type,\
	 		'i': memory['i'], \
			'val': memory['val']}

def find_chord_w_transpose(set_of_pitches):
	# N.B. 'set_of_pitches' could be accompanied by a corresponding 'set_of_duration'

	out = []
	for i in range(12):
		set_of_notes = np.array([p%12 for p in np.array(set_of_pitches) + i])
		set_of_notes = np.array(list(set(set_of_notes)))
		out.append(find_chord(set_of_notes))


	
	out.sort(key=lambda x: x['val'], reverse=True)
	i = out[0]['i']
	set_of_notes = np.array([p%12 for p in np.array(set_of_pitches) + i])
	set_of_notes = np.array(list(set(set_of_notes)))

	root = out[0]['i']
	chord_typeout[0]['chord_type']
	return chordroot,  







	