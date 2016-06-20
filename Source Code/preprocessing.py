import midi
import re

 
#input_MIDI = '../MIDI files/jsbach.net/jsbach.net Preludes and Fugues/bwv867.mid'
input_MIDI = './Python MIDI/mary.mid'
out = midi.read_midifile(input_MIDI)

def note_extractor(m):
	"""
	INPUT: m, match object
	OUTPUT: (tick INT, data (INT, INT))
	"""
	
	tick_p = m.group(1)
	tick = int(re.findall(r'\d+', tick_p)[0])
	data_p = m.group(3)
	data = [int(elem) for elem in  re.findall(r'\D*(\d*)\D+', data_p)]
	
	return tick, data
	
note_on = []
note_off = []
for track in out[1]:
	line = str(track)
	print line
	m_Track = None
	
	# if not m_Track:
	# 	m_Track = re.match(r'midi\.Track\(', line)
	# else:
	r_on = r'.*midi\.NoteOnEvent.*tick=(\d*).*channel=(\d*).*data=(.*)'
	r_off = r'.*midi\.NoteOffEvent.*tick=(\d*).*channel=(\d*).*data=(.*)'
	m_note_on = re.match(r_on, line)
	m_note_off = re.match(r_off, line)
	
	if m_note_on:
		tup = note_extractor(m_note_on)
		if tup[1][1] == 0:
			# velocity = 0 means it's a NoteOffEvent
			note_off.append(tup)
		else:
			note_on.append(tup)
	elif m_note_off:
		note_off.append(note_extractor(m_note_off))
		
print 'note_on = ', note_on
