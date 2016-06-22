import midi
import os

"""	Stage 1: MIDI to time_series conversion
"""

def build_single_pattern(source, pattern):
	'''
	INPUT: track and pattern
	OUTPUT: pattern
	
			Append the source to pattern and return pattern
	'''
	# initialize track object
	track = midi.Track()
	
	# append to pattern object
	pattern.append(track)
	for line in source[0]:
			track.append(line)

	return pattern
	
def merger_t1s(filename='bwv733.mid'):
	'''
	INPUT: rootfilename STR (eg 'bwv733.mid')
	
		   merges the tracks found under, for instance,
		   'bwv733_t1.mid', 'bwv733_t2.mid' .. etc
		   into 'bwv733_io.mid'
	'''
	print 'Input filename: {}'.format(filename)
	rootfilename = filename[:-4]
	
	ls = os.listdir('.')
	lst = []
	for ifile in ls:
		if (ifile[0:len(rootfilename)] == rootfilename) and (len(ifile) > len(rootfilename)+4):
			lst.append(ifile)
			print '\tAbleton-exported files found: {}'.format(ifile)
	# for example, lst = ['bwv733_t1.mid', 'bwv733_t2.mid', 'bwv733_t3.mid']
	
	# initialize pattern object
	pattern = midi.Pattern()
	for item in lst:
		source = midi.read_midifile(item)
		pattern = build_single_pattern(source, pattern)
		
	# write pattern to file
	output_file = rootfilename + '_io.mid'
	midi.write_midifile(output_file, pattern)
	
	print 'Output file created: {}'.format(output_file)

'''===========================================================================
==============================================================================
==============================================================================
'''

def time_series_to_MIDI_track(time_series):
	'''
	INPUT: time_series LIST [(time, pitch, duration), ...]
	OUTPUT: track midi.Track()
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
	
	return track

if __name__=='__main__':
	# Instantiate a MIDI Pattern (contains a list of tracks)
	pattern = midi.Pattern()
	for time_series in time_series_list:
		track = time_series_to_MIDI_track(time_series)
		# append to pattern
		pattern.append(track)

	
	print pattern
	midi.write_midifile("track.mid", pattern)








