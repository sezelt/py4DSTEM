import numpy as np
from tqdm import tqdm

from pdb import set_trace

class LibraryClassificaton:
	"""
	LibraryClassification finds the best match between experimental patterns and a 
	library of ideal diffraction patterns. Both are specified as PointListArrays.
	"""

	def __init__(self, library, braggvectors, x0,y0, cost_function='default'):
		pass