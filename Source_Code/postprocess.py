import matplotlib.pylab as plt	
import os, pickle, numpy as np
from preprocess_0 import plot_pitch_matrix


'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''

def plot_cost(cost):
	plt.plot(cost)
	

	fig = plt.figure(figsize=(6,3))
	x_axis = np.linspace(1,len(cost),len(cost))
	ax = fig.gca()
	ax.plot(x_axis, cost, label='cost', linewidth=2.0)
	ax.grid('on')

	plt.legend()
	plt.ion()
	plt.show()
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''
	
def extract_from_pickle(filepath):
	dict = pickle.load(open(filepath, 'rb'))
	cost = dict['cost']
	fugue = dict['fugue']
	
	return cost, fugue
	
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
	
if __name__ == '__main__':
	dirpath = '/Users/ozkansafak/Bach2.0/Source_Code/Synthesized_Fugues/new/'
	
	print "\n\ndirpath = {}\n{}\n".format(dirpath,'-'*(len(dirpath)+10))
	
	p_files = get_fugue_pickle_filepaths(dirpath)
	epoch =[]
	for fname in p_files:
		epoch.append(int(fname.split('epoch')[1].split('data')[0]))
		
	ind = np.argmax(epoch)
	last_epoch_filepath = dirpath + p_files[ind]
	cost, fugue = extract_from_pickle(last_epoch_filepath)
	fugue = np.transpose(fugue)
	plot_cost(cost)
	plot_pitch_matrix(fugue, title='epoch=' + str(epoch[ind]), \
							xlabel='16th note time-step', 
							ylabel='pitch')
	#
	#
	time_series = pitch_matrix_TO_time_series_legato(fugue, sh=24)
	time_series_TO_midi_file(time_series)



