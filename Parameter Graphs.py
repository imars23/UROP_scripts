import MDSplus
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.optimize as opt

"""
This script generates plots of different plasma parameters, especially in comparison to the brightness change over the pumpout phenomemon. Most
plots are made to replicate figures from J. E. Rice et al 2022 Nucl. Fusion 62 086009
"""

class plasma_shot:

    def __init__(self, shot_number):
        self.shot_num = shot_number
        self.HHD = 0
        self.W = 0
        self.br_chng = 0

    def set_br_data(self):
        #Get Z-Line Integrated Brightness
        specTree = MDSplus.Tree('spectroscopy', self.shot_num)
        br_branch = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREX_SR.ANALYSIS.Z:INT')
        self.br_data = br_branch.data()[0] #Location 2-250
        self.br_dim = br_branch.dim_of(0)

    def set_HHD_data(self):
        #Get H/H+D Data
        specTree = MDSplus.Tree('spectroscopy',  self.shot_num)
        HD_branch = specTree.getNode(r'\SPECTROSCOPY::BALMER_H_TO_D')
        HD_data = np.array(HD_branch.data())
        self.HHD_data = HD_data / (1 + HD_data)
        self.HHD_dim = HD_branch.dim_of(0)
        self.HHD_average = np.average(self.HHD_data)
        self.HHD_std = np.std(self.HHD_data)

    def set_WMHD(self):
        anTree = MDSplus.Tree('analysis', self.shot_num)
        WMHD_node = anTree.getNode(r'\analysis::EFIT_AEQDSK:wplasm')
        WMHD_data = WMHD_node.data()
        WMHD_dim = WMHD_node.dim_of(0)
        WMHD_max = np.max(WMHD_data)

        self.WMHD_dim = WMHD_dim
        self.WMHD_data = WMHD_data
        self.WMHD_max = WMHD_max

    def set_RF(self):
        RFTree = MDSplus.Tree('rf', self.shot_num)
        RFbranch = RFTree.getNode(r'\rf::RF_POWEr_net')
        RFdata = RFbranch.data()
        self.RF = RFdata
    
    def set_ne(self):
        electronTree = MDSplus.Tree('electrons', self.shot_num)
        ne_branch = electronTree.getNode(r'\ELECTRONS::top.tci.results:nL_04')
        ne_data = np.array(ne_branch.data())/0.6
        ne_dim = ne_branch.dim_of(0)
        self.ne_data = ne_data
        self.ne_dim = ne_dim

    def set_br_chng(self, br_chng):
        self.br_chng = br_chng

    def get_shot_num(self):
        return self.shot_num
    
    def get_HHD_average(self):
        return self.HHD_average
    
    def get_HHD_std(self):
        return self.HHD_std

    def get_br_chng(self):
        return self.br_chng
    
    def get_br_data(self):
        return (self.br_dim, self.br_data)
    
    def get_WMHD_max(self):
        return self.WMHD_max

    def get_ne(self):
        return self.ne_dim, self.ne_data

def average_over(x, y, x_range):
    """
    (Not in use) Returns an average of elements over a given range

    Parameters:
    x (Array): List of independent variable values
    y (Array): List of dependent variable values
    x_range (Array): (start, stop) Values over which to average

    Returns: Tuple (average, standard deviation)
    """

    copy_y = []
    for i in range(len(x)):
        if x[i] >= x_range[0] and x[i] <= x_range[1]:
            copy_y.append(y[i])

    copy_y = np.array(copy_y)/10**19
    average = np.average(copy_y)
    std = np.std(copy_y)
    average = average * 10**19
    std = std * 10**19
    return average, std

def read_shots(brchange = False, HHD = False):
    """
    Reads the shot numbers and generates a list of shot objects with Shot Number, Brightness Change, and HHD data set. Depends on Brightness Changes.txt and Shot Numbers.txt

    Parameters
    brchange (bool, default: False): If True, load Brightness Change data for each shot
    HHD (bool, default: False), If True, load HHD ratio data for each shot

    Returns: (array) List of plasma_shot objects
    """

    br = open('Brightness Changes.txt')
    sh = open('Shot Numbers.txt')

    prefix = 0
    line = sh.readline()
    shot = 0
    plasma_shots = []

    while line != '':
        if line[0] == 'A':
            # Date of shots
            prefix = int(line[1:-1])*100
        else:
            # Actual shot number given/known
            if line[0] == 'B':
                # Individual shot
                shot_num = int(line[1:-1])
            else:
                # Shot from multiple in one day
                shot_num = prefix + int(line[:-1])

            s = plasma_shot(shot_num)
            plasma_shots.append(s)

            if brchange:
                br_change = float(br.readline()[:-1])
                s.set_br_chng(br_change)

            if HHD:
                s.set_HHD_data()

        line = sh.readline()
    
    return plasma_shots

def HHD_plot(plasma_shots, W_color = False, T_E_color = False, chosen = False):
    """
    Create a plot of H/H+D ratio vs. Brightness Change. Depends on files Brightness Changes.txt, Shot Numbers.txt, and T_E_color.

    Parameters
    plasma_shots (array): List of plasma shot objects
    W_color (bool, default: False): If True, color-code the plot by max W_MHD
    T_E_color (bool, default: False): If True, color-code the plot by T_E
    chosen (bool, default: False): If True, color-code the plot based on the shots listed

    Returns: None
    """
    
    HHDs = []
    br_chngs = []
    W = []

    for s in plasma_shots:
        br_chngs.append(s.get_br_chng())
        HHDs.append(s.get_HHD_average())
        if W_color:
            s.set_WMHD()
            W.append(s.get_WMHD_max())

    fig = plt.figure()
    plt.xlabel('H/H+D Ratio')
    plt.ylabel('Brightness Change')

    if W_color:
        W = np.array(W)/1000
        c_day = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 
        6, 6, 6, 7, 8, 8, 8, 9])/9
        plt.scatter(HHDs, br_chngs, c = c_day, cmap = 'Set1')
        # plt.colorbar(label = 'Color Based on day', cmap = 'Set1')
        # plt.scatter(HHDs, br_chngs, c = W, cmap = 'gist_rainbow')
        # plt.colorbar(label = 'W_MHD (kJ)', cmap = 'gist_rainbow')
    elif T_E_color:
        T_E_file = open('T_E.txt')
        line = T_E_file.readline()
        T_E = []
        while line != '':
            T_E.append(float(line[:-1]))
            line = T_E_file.readline()
        
        plt.scatter(HHDs, br_chngs, c = T_E, cmap = 'gist_rainbow')
        plt.colorbar(label = 'Energy Confinement Time', cmap = 'gist_rainbow')
        
    else:
        # plt.plot(HHDs, br_chngs, '.')
        c = []
        chosen = [1160420024, 1140221007, 1120510027, 1140212015, 1140213022,
            1140213022, 1140213022, 1140221009, 1120510028, 1140213027, 1120523016, 1140213013]
        for s in plasma_shots:
            shot_num = s.get_shot_num()
            if shot_num in chosen:
                c.append(1)
            else:
                c.append(0)

        plt.scatter(HHDs, br_chngs, c=c, cmap = 'Set1')
    
    plt.show()

def density_plot():

    bright_chg = []
    HHD = []
    plasma_shots = read_shots(brchange = True, HHD = True)
   
    for shot in plasma_shots:
        change = shot.get_br_chng()
        HHD_ratio = shot.get_HHD_average()
        bright_chg.append(change)
        HHD.append(HHD_ratio)
    
    density = read_avg_dens()
    print(density)
    print(np.max(density))
    print(np.min(density))

    c_day = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 
        6, 6, 6, 7, 8, 8, 8, 9])/9

    HHD_diff = np.abs(np.array(HHD) - 0.31)/0.31

    plt.scatter(density, bright_chg, c=HHD_diff, cmap = 'gist_rainbow')
    plt.colorbar(label = 'Percent Difference of H/H+D Ratio from 0.31', cmap = 'gist_rainbow')
    plt.show()

def auto_avg_dens(shot):
    """
    Automatically get the average density for a shot from electron density data over range 0.5 < t < 1.5. Not the best method for most/all shots
    """

    shot.set_ne()
    x, y = shot.get_ne()    # Get time and density information for the shot in question

    shot_num = str(shot.get_shot_num())

    if shot_num[2] == '2':  # Shot is from 2012. These have fewer data points so need to treat differently
        avg_density, std = average_over(x, y, [0.5, 1.5]) # Should correspond approximately to the time range 0.5 < t < 1.5

        # try:
        #     if not 0.4 < x[215] < 0.6 or not 1.4 < x[615] < 1.6:
        #         raise Exception('Time window for shot {0} out of bounds 0.5 < t < 1.5'.format(shot_num))
        # except IndexError:
        #     raise Exception('Index Error for shot {0}. Time length is {1}'.format(shot_num, str(len(x))))
        
        # avg_density = np.average(y_windowed)
    else:
        y_windowed = y[1077:3077]   # Should correspond approximately to the time range 0.5 < t < 1.5
        
        if not 0.4 < x[1077] < 0.6 or not 1.4 < x[3077] < 1.6:
            raise Exception('Time window for shot {0} out of bounds 0.5 < t < 1.5'.format(shot_num))

        avg_density = np.average(y_windowed)

    if not np.isfinite(avg_density):
        raise Exception('Average density for shot {0} is not finite'.format(shot_num))
    
    return avg_density

def read_avg_dens():
    """
    Returns a list of average densities as stored in n_e.txt
    """
    n_e = open('n_e.txt')
    densities = []
    line = n_e.readline()

    while line != '':
        densities.append(float(line[:-1]))
        line = n_e.readline()
    
    return densities

# shot = plasma_shot(1140221012)
# shot.set_br_data()
# br_dim, br_data = shot.get_br_data()
# # print(br_dim)
# average_over(br_dim, br_data, [0.8, 1])
# # print(average, std)
# # plt.plot(br_dim, br_data)
# # plt.show()

plasma_shots = read_shots(brchange = True, HHD  = True) # Generate a list of shot objects that have brightness change and H/H+D ratio loaded
HHD_plot(plasma_shots, W_color = True)
# density_plot()

# shot_num = 1150522022

# shot = plasma_shot(shot_num)
# shot.set_ne()
# times, density = shot.get_ne()

# start = 0.6
# stop = 1.5
# average, std = average_over(times, density, (start, stop))
# print(average)