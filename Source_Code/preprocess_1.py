import pyaudio
import wave
import midi
import os
import pickle

"""	Stage 1: MIDI to time_series conversion
"""

def build_MIDI_pattern(source, pattern):
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
	
	
	
def merge_tracks_one_file(rootfilename):
	'''
	INPUT: rootfilename STR (eg 'bwv733')
	
			N.B. The Ableton exported tracks
			'bwv733_t1.mid', 'bwv733_t2.mid', 'bwv733_t3.mid'
			have to created and existing in the same folder
			prior to running merge_tracks('bwv733.mid').
			
			merges the tracks found under, for instance,
			'bwv733_t1.mid', 'bwv733_t2.mid' .. etc
			into 'bwv733_io.mid'
			
	OUTPUT: None
			The rootfilename_io.mid is generated in the same folder.
	'''
	
	print 'Input rootfilename: {}'.format(rootfilename)
	
	dir = os.listdir('.')
	lst = []
	for f in dir:
		if (f[0:len(rootfilename)+2] == ''.join([rootfilename, '_t'])) and (f[-4:] == '.mid'):
			lst.append(f)
			print '\t Ableton-exported files found: {}'.format(f)
	# for example, lst = ['bwv733_t1.mid', 'bwv733_t2.mid', 'bwv733_t3.mid']
	
	# initialize pattern object
	pattern = midi.Pattern()
	for item in lst:
		source = midi.read_midifile(item)
		pattern = build_MIDI_pattern(source, pattern)
		
	# write pattern to file
	output_file = rootfilename + '_io.mid'
	midi.write_midifile(output_file, pattern)
	under = '-' * len(output_file)
	print '\t                               ' + under
	
	print 'Output file created by merge_tracks(): {}\n'.format(output_file)





def main_merge_tracks():
	'''
	merges all corresponding '[filename]t1.mid' files found in the same directory.
	into  '[filename]_io.mid'
	'''
	dir = os.listdir('.')
	t_files = []
	for f in dir:
		if (f[-4:] == '.mid') and (f[-7:-5] == '_t'):
			t_files.append(f)
	t_files = list(set(t_files))
	
	for f in t_files:
		rootfilename = f[:-7]
		merge_tracks_one_file(rootfilename)


'''===========================================================================
==============================================================================
==============================================================================
'''


def mp3_player(songpath):
	chunk = 1024
	wf = wave.open(songpath, 'rb')
	p = pyaudio.PyAudio()

	stream = p.open(
		format = p.get_format_from_width(wf.getsampwidth()),
		channels = wf.getnchannels(),
		rate = wf.getframerate(),
		output = True)
	data = wf.readframes(chunk)

	print "... playing music. \n    Ctrl+C to break"
	while data != '':
		stream.write(data)
		data = wf.readframes(chunk)
	

	stream.close()
	p.terminate()

	
if __name__=='__main__':
	main_merge_tracks()









