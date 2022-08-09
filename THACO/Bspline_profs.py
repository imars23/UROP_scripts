'''
Bspline_profs.py is a script that loads fitted ion temperature, Ti, and toroidal rotation, Omega, 
measured by the HIREX Sr diagonstic on C-Mod. These profiles where computed in THACO using the 
W_HIREXSR_OMFIT and W_HIREXSR_TIFIT widgets which perform a B-spline fit to the inverted
Ti and Omega profiles.

created: cjperks

'''

# Loads modules
import numpy as np
import matplotlib.pyplot as plt
from MDSplus import *

