import numpy as np	
D = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def cosine_similarity(x,y):
	
	num = x.dot(y)
	denom = np.sqrt(x.dot(x)) * np.sqrt(y.dot(y))
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
	INPUT: set SET, chord STR
	OUPUT: distance FLOAT
	'''

	class chord(object):
		def __init__(self, name, lst):
			# weight constants
			root = 1
			third = .75
			fourth = .3
			fifth = .75
			seventh = .3
			e = .1 # all other notes in the scale
			
			self.notes = np.array(lst)
	 		self.name = name
			
			self.wt = np.array([-e for i in range(12)])
			self.wt[lst[0]] = root
			self.wt[lst[1]] = e
			self.wt[lst[2]] = third
			self.wt[lst[3]] = fourth
			self.wt[lst[4]] = fifth
			self.wt[lst[5]] = e
			self.wt[lst[6]] = seventh
			
			# punish the wrong thirds by -10*e
			if self.name[0:5] == 'minor':
				self.wt[4] = -10*e
			elif self.name[0:5] == 'major':
				self.wt[3] = -10*e
			elif (self.name[0:8] == 'dominant') | (self.name != 'dominant_altered'):
				self.wt[3] = -10*e
				
			if self.name != 'dominant_altered':
				# [0, 1, 3, 4, 6, 8, 10]
				self.wt[lst[2]] = e # minor 3rd
				self.wt[lst[3]] = third # raise the value of major 3rd
				self.wt[lst[4]] = e # cut down fifth
				
		# e.g.
		# self.notes = [0, 2, 3, 5, 7, 8, 10]
		# self.wt = [1, -0.1, 0.1, 0.75, -0.5, 0.3, -0.1, 0.75, 0.1, -0.1, 0.3, -0.1]
		# self.name = 'minor'
		
	chords_vocabulary = []
	# Am
	chords_vocabulary.append(chord('minor', [0, 2, 3, 5, 7, 8, 10]))
	# Gm Maj7
	chords_vocabulary.append(chord('minor_harmonic', [0, 2, 3, 5, 7, 8, 11]))
	# Gm dim WARNING:  This is an OCTATONIC Scale ()
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

	##### 
	canonical_chord_vector = np.zeros((len(chords_vocabulary), 12), dtype=float)
	for i, el in enumerate(chords_vocabulary):
		canonical_chord_vector[i][el.notes] = el.wt[el.notes]
	
	return chords_vocabulary, canonical_chord_vector
	
	
def find_chord(set_of_pitches):
	'''
	INPUT: set_of_pitches SET
	OUTPUT: 
	'''
	
	chords_vocabulary, canonical_chord_vector = build_chords_vocabulary()
	
	# N.B. set_of_pitches must be accompanied by a corresponding set_of_duration
	set_of_notes = np.array([p%12 for p in set_of_pitches])
	
	vector = np.zeros((len(chords_vocabulary), 12), dtype=float)
	for i, el in enumerate(chords_vocabulary):
		vector[i, set_of_notes] += el.wt[set_of_notes]
	print vector
	
		
	# now do cosine_similarity btw 'vector' & 'canonical_chord_vector'
	cs = np.empty((len(vector), len(canonical_chord_vector)))
	memory = {'val': 0, 'i': 0, 'j': 0}
	for i, v1 in enumerate(vector):
		for j, v2 in enumerate(canonical_chord_vector):
			val = cosine_similarity(v1, v2)
			if val > memory['val']:
				memory['val'] = val
				memory['i'] = i
				memory['j'] = j
			cs[i,j] = val
	
	cs = np.empty((len(vector), len(canonical_chord_vector)))
	memory = {'val': 0, 'i': 0}
	for i in range(len(vector)):
		val = cosine_similarity(vector[i], canonical_chord_vector[i])
		if val > memory['val']:
			memory['val'] = val
			memory['i'] = i
			memory['j'] = j
		cs[i,j] = val
	
	
	