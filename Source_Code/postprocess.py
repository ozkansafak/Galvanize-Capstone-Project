import matplotlib.pylab as plt	
import os, pickle, numpy as np
from preprocess_0 import plot_pitch_matrix
from helpers_to_main import pitch_matrix_TO_time_series_legato, time_series_TO_midi_file

'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''

def plot_cost(cost, title=''):
	
	fig = plt.figure(figsize=(9,5))
	x_axis = np.linspace(1,len(cost),len(cost))
	ax = fig.gca()
	ax.semilogy(x_axis, cost, label='cost', linewidth=2.0)
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
	cost = dict['cost']
	fugue = dict['fugue']
	
	return dict, cost, fugue
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''
	
def get_fugue_pickle_filepaths(dirpath):
	dir = os.listdir(dirpath)
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
		dict, cost, fugue = extract_from_pickle(filepath)
		plt.plot(dict['predict'][0], c[i%4])
	
	plt.title(str(len(epoch)) + ' epochs')


'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''

if __name__ == '__main__':
	dirpath = '/Users/ozkansafak/Bach2.0/Source_Code/Synthesized_Fugues/new/'
		
	p_files = get_fugue_pickle_filepaths(dirpath)
	#get all the epochs
	epoch =[]
	for fname in p_files:
		epoch.append(int(fname.split('epoch')[1].split('.')[0]))
		
	ind = np.argmax(epoch)
	ind = [i for i,elem in enumerate(epoch) if elem == 1][0]
	filepath = dirpath + p_files[ind]
	dict, cost, fugue = extract_from_pickle(filepath)
	
	
	plot_pitch_matrix(np.transpose(fugue), title=p_files[ind], xlabel='16th note time-step', ylabel='pitch')
	plot_cost(cost,title=p_files[ind])
	
	print "\nfilepath = {}\n{}\n".format(filepath,'-'*(len(dirpath)+10))
	print 
	for k in dict.keys():
		if (type(dict[k]) == int) or (type(dict[k]) == float):
			print k, ':', dict[k]
	
	#
	#
	# time_series = pitch_matrix_TO_time_series_legato(fugue, sh=24)
	# time_series_TO_midi_file(time_series)










