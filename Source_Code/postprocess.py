import matplotlib.pylab as plt	
import os, sys, pickle, numpy as np
from preprocess_0 import plot_pitch_matrix
from helpers_to_main import pitch_matrix_TO_time_series_legato, time_series_TO_midi_file

'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''

def plot_loss(loss, title=''):
	
	fig = plt.figure(figsize=(9,5))
	x_axis = np.linspace(1,len(loss),len(loss))
	ax = fig.gca()
	ax.semilogy(x_axis, loss, label='loss', linewidth=2.0)
	ax.grid('on')
	
	plt.title(title)
	plt.xlabel('epochs')
	plt.legend()
	plt.ion()
	plt.show()
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''
	
def extract_from_pickle(filepath):
	dict = pickle.load(open(filepath, 'rb'))
	loss = dict['loss']
	fugue = dict['fugue']
	
	return dict, loss, fugue
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''
	
def sorted_ls(dirpath):
    mtime = lambda f: os.stat(os.path.join(dirpath, f)).st_mtime
    return list(sorted(os.listdir(dirpath), key=mtime))


def get_fugue_pickle_filepaths(dirpath):
	dir = sorted_ls(dirpath)
	p_files = []
	for f in dir:
	    if f.startswith('fugue') and f.endswith('.p'):
			# all .p files
			p_files.append(f)
	return p_files
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''

def plot_predicts():
	dirpath = '/Users/ozkansafak/Bach2.0/Source_Code/Synthesized_Fugues/new/'
	p_files = get_fugue_pickle_filepaths(dirpath)

	fig = plt.figure(figsize=(13,6))
	ax = fig.gca()
	ax.grid('on')
	plt.show()
	plt.ion()
	c = ['ko', 'b', 'r*','c.']

	#get all the epochs
	epoch =[]
	for fname in p_files:
		epoch.append(int(fname.split('epoch')[1].split('.')[0]))

	for i in range(len(p_files)):
		ind = [j for j, elem in enumerate(epoch) if elem == epoch[i]][0]
		filepath = dirpath + p_files[ind]
		dict, loss, fugue = extract_from_pickle(filepath)
		plt.plot(dict['predict'][0], c[i%4])
	
	plt.title(str(len(epoch)) + ' epochs')


'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''

if __name__ == '__main__':
	
	dirpath = '/Users/ozkansafak/Bach2.0/Source_Code/Synthesized_Fugues/new/'
		
	p_files = get_fugue_pickle_filepaths(dirpath)
	# get all the epochs
	epoch =[]
	for fname in p_files:
		epoch.append(int(fname.split('epoch')[1].split('.')[0]))
	
	if len(sys.argv) > 1:
		ind = [i for i,v in enumerate(epoch) if v == int(sys.argv[1])]
		if len(ind) > 1:
			print '{} files found'.format(len(ind))
			for q in ind:
				print p_files[q]
			ind = ind[-1]
			print 'picking: {}'.format(p_files[ind])
		else:
			ind = ind[0]
	else:
		ind = len(p_files)-1

	filepath = dirpath + p_files[ind]
	dict, loss, fugue = extract_from_pickle(filepath)
	
	plot_pitch_matrix(np.transpose(fugue), title=p_files[ind], xlabel='16th note time-step', ylabel='pitch')
	plot_loss(loss, title=p_files[ind])
	
	print "\nfilepath = {}\n{}\n".format(filepath,'-'*(len(dirpath)+10))
	print 
	for k in dict.keys():
		if (type(dict[k]) == int) or (type(dict[k]) == float):
			print k, ':', dict[k]
	
	#
	# time_series = pitch_matrix_TO_time_series_legato(fugue, sh=24)
	# time_series_TO_midi_file(time_series)
