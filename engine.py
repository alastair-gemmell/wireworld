import numpy as np
import wireworld as ww
import sys

class WireworldEngine:
	"""
	A wireworld state engine based on numpy arrays
	
	"""
	def load_state(self, fname):
		"""
		Load a state from a CSV fname. 
		
		Note we need to rotate by 270 degrees to account for the ways that
		numpy.genfromtxt reads to an array and how pygame displays arrays
		
		"""
		try:
			array = np.genfromtxt(fname, delimiter=',')
		except IOError:
			print("Could not read file:", fname, file=sys.stderr)
			sys.exit(1)
			
		return np.rot90(array, 3)


	def save_state(self, state, fname):
		"""
		Save a state from a CSV fname. 
		
		Note we need to rotate by 90 degrees to account for the ways that
		numpy.savetxt writes to a fname and how pygame displays arrays
		
		"""
		try:
			np.savetxt(fname, np.rot90(state), delimiter=',', fmt='%i')
		except FileNotFoundError:
			print("Could not save file to non-existent location:", fname, file=sys.stderr)
			sys.exit(1)

	
	def get_display_array_from_state(self, state, scale_factor):
		"""
    	Return a RGB array for display of the state (e.g. in pygame). 
    	
    	This is a 3D numpy array of the same shape as the state array 
    	but with a 3rd dimension present of size 3 for the Red, Green and Blue values.
    	
    	In addition the display array is scaled out so that each grid square is
    	not constrained to being only one pixel in size
    	
    	"""
		# Create an RGB array by adding extra dimension of size 3 and filling with RGB values
		rgb_array = np.repeat(state[:, :, np.newaxis], 3, axis=2)
		for ix, iy in np.ndindex(state.shape):
			rgb_array[ix, iy] = ww.VALUE_TO_RGB[state[ix, iy]]
	
		# scale_factor out the array so we don't just have one pixel per cell using
		# https://stackoverflow.com/questions/7525214/how-to-scale_factor-a-numpy-array
		return np.kron(rgb_array, np.ones((scale_factor, scale_factor, 1)))
	
	
	def get_next_state(self, current_state):
		"""
    	Use the wireworld state evolution rules in :func:`_get_next_value` 
    	to return the next state of the world from the current one.
    	
    	"""
		next_state = np.zeros_like(current_state)
		for coord, val in np.ndenumerate(next_state):
			next_state[coord] = self._get_next_value(coord, current_state)
		return next_state
	
	
	def _get_next_value(self, coord, state_array):
		"""
    	Use the wireworld state evolution rules to return next cell value 
    	from the current one.
    	
    	"""
		next_val = ww.EMPTY
		if state_array[coord] == ww.HEAD:
			next_val = ww.TAIL
		elif state_array[coord] == ww.TAIL:
			next_val = ww.CONDUCTOR
		elif state_array[coord] == ww.CONDUCTOR:
			# If a CONDUCTOR has exactly one or two HEAD neighbours then -> HEAD
			if 1 <= len(self._get_neighbours(coord, state_array, ww.HEAD)) <= 2:
				next_val = ww.HEAD
			else:
				next_val = ww.CONDUCTOR
	
		return next_val
	
	
	def _get_neighbours(self, coord, array, condition_value):
		"""
    	Helper function to return a list of array coordinates that are immediate neighbours
    	(including diagonally) of a particular coordinate AND which are equal to a
    	particular value
    	
    	"""
		neighbours = []
		rows = len(array)
		cols = len(array[0]) if rows else 0
		for i in range(max(0, coord[0] - 1), min(rows, coord[0] + 2)):
			for j in range(max(0, coord[1] - 1), min(cols, coord[1] + 2)):
				if (i, j) != coord and array[i][j] == condition_value:
					neighbours.append((i, j))
		return neighbours