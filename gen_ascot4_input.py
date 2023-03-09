'''
Generates ASCOT4 input files from magnetic equilibrium and plasma parameter profiles

imars, Jan 27, 2023
'''

import numpy as np
import matplotlib.pyplot as plt
import xarray
from omfit_classes import omfit_eqdsk
from scipy.interpolate import interp1d
import os

g_file_path = "/home/imars23/Desktop/Data/TRANSP_013/g1140221013.01000"

geqdsk = omfit_eqdsk.OMFITgeqdsk(g_file_path)

print(geqdsk.keys())

ga_code_path = "/home/imars23/Desktop/Data/GA_Codes/input_t0.8s.gacode"

# Retrieve data from GA Code
f = open(ga_code_path)
data_list = []
line = f.readline()
while line != "":
    if line[0] == "#":
        try: data_list.append(lines)
        except:
            pass
        lines = []
    lines.append(line[:-1])
    line = f.readline()

# Organize data
gacode = {}
for section in data_list:
    
    if len(section) > 1:
        key = section.pop(0)
        key = key.split(' ')[1]
        
        if len(section) == 1:
            section = section[0].split(' ')
            
            if key not in ['shot', 'name', 'type']:
                for i, item in enumerate(section):
                    if item != '':
                        section[i] = float(item)
        else:
            for i, row in enumerate(section):
                row_list = row.split(' ')
                row_list = [item for item in row_list if item != '']
                if len(row_list) != 2:
                    section[i] = [float(item) for item in row_list[1:]]
                else:
                    section[i] = float(row_list[1])
                        
        gacode[key] = section

print(gacode['ni'])
