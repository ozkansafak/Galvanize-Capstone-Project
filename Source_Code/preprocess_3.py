import midi
import re
import os
import matplotlib.pylab as plt


def get_bad_files():
	r=[]
	bad_files = []
	for f in retrieve_all_io_files(): 
		time_series_list = time_series_list_builder('../MIDI_data/io_files/io_files/' + f)
	
		l = note_value(time_series_list)

		# compute ratio of all quarter and shorter notes played 
		no_of_notes = reduce(lambda x, y: x+y, [e[1] for e in l])
		no_of_96_and_less = \
		reduce(lambda x, y: x+y, [e[1] for e in l if ((e[0] <= 96) and (e[0]%12 == 0))])
		ratio_96 = float(no_of_96_and_less) / no_of_notes
		r.append(ratio_96)
	
		print '\n', f[:-7], '--', ratio_96
		print 'ticks\t  count'
		for k, v in l:
			print ' ', k, '\t:' , v
		
		small_notes = []
		for k, v in l:
			if (k <= 96) and (k%6 ==0):
				small_notes.append((k,v))
		
		if sum([elem[1] for elem in small_notes]) / float(no_of_notes)>.5:
			# 4/4 Time Signature
			pass
		else:
			bad_files.append(f)

	return bad_files