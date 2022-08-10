import MDSplus
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.optimize as opt
import plasma_shot as ps
import pandas as pd

"""
This script generates plots of different plasma parameters, especially in comparison to the brightness change over the pumpout phenomemon. Most
plots are made to replicate figures from J. E. Rice et al 2022 Nucl. Fusion 62 086009
"""
def flatten(nested_list):
    new_list = []
    for item in nested_list:
        if isinstance(item, list):
            new_list += flatten(item)
        else:
            new_list.append(item)
    return new_list

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

def read_shots(brchange = False, HHD = False, n_e = False):
    """
    Reads the shot numbers and generates a list of shot objects with Shot Number, Brightness Change, and HHD data set. Depends on Brightness Changes.txt and Shot Numbers.txt

    Parameters
    brchange (bool, default: False): If True, load Brightness Change data for each shot
    HHD (bool, default: False), If True, load HHD ratio data for each shot

    Returns: (array) List of plasma_shot objects
    """
    print('Loading Plasma Shot Data \n')
    path = r'/home/imars23/Desktop/Data/'

    columns = 'A'

    if brchange:
        columns += ', R'
    if n_e:
        columns += ', T'
    data = pd.read_excel(path + r'Pumpout_Shots_Database.xlsx', engine = 'openpyxl', usecols = columns, nrows = 51 )
    shot_numbers = pd.DataFrame(data, columns = ['Shot Number']).values.tolist()
    shot_numbers = flatten(shot_numbers)

    if brchange:
        br_changes = flatten(pd.DataFrame(data, columns = ['Br Change']).values.tolist())
    if n_e:
        densities = flatten(pd.DataFrame(data, columns = ['Average n_e']).values.tolist())

    plasma_shots = []
    print('Reading Plasma Shot Data \n')
    for i, shot_num in enumerate(shot_numbers):
        s = ps.shot(int(shot_num))

        if brchange:
            s.set_br_chng(br_changes[i])
        if n_e:
            s.set_n_e(densities[i])
        if HHD:
            s.set_HHD_data()
        
        plasma_shots.append(s)
    print('Done \n')

    return plasma_shots


def HHD_plot(plasma_shots, plot_W = False, plot_T_E = False, plot_chosen = False, plot_date = False):
    """
    Create a plot of H/H+D ratio vs. Brightness Change. Depends on files Brightness Changes.txt, Shot Numbers.txt, and T_E_color.

    Parameters
    plasma_shots (array): List of plasma shot objects
    plot_W (bool, default: False): If True, color-code the plot by max W_MHD
    plot_T_E (bool, default: False): If True, color-code the plot by T_E
    plot_chosen (bool, default: False): If True, color-code the plot based on the shots listed
    plot_date (bool, default: False): If True, color-code the plot based on date of shot

    Returns:
    fig, ax: Figure and axis objects
    """
    
    HHDs = []
    br_chngs = []
    W = []

    for s in plasma_shots:
        br_chngs.append(s.get_br_chng())
        HHDs.append(s.get_HHD_average())
        if plot_W:
            s.set_WMHD()
            W.append(s.get_WMHD_max())

    fig, ax = plt.subplots()
    plt.xlabel('H/H+D Ratio')
    plt.ylabel('Brightness Change')

    if plot_W:
        # Color based on W
        plt.scatter(HHDs, br_chngs, c = np.array(W)/1000, cmap = 'gist_rainbow')
        plt.colorbar(label = 'W_MHD (kJ)', cmap = 'gist_rainbow')

    elif plot_T_E:
        # Get list of T_E data
        path = r'/home/imars23/Desktop/Data/'
        data = pd.read_excel(path + r'Pumpout_Shots_Database.xlsx', engine = 'openpyxl', usecols = 'O', nrows = 51 )
        T_E = flatten(data.values.tolist())

        # Color plot based on T_E
        plt.scatter(HHDs, br_chngs, c = T_E, cmap = 'gist_rainbow')
        plt.colorbar(label = 'Energy Confinement Time', cmap = 'gist_rainbow')
        
    elif plot_chosen:
        # Color the plot differently for shots chosen as good representatives of trend
        # plt.plot(HHDs, br_chngs, '.')
        c = []
        chosen_shots = [1160420024, 1140221007, 1120510027, 1140212015, 1140213022,
            1140213022, 1140213022, 1140221009, 1120510028, 1140213027, 1120523016, 1140213013]
        for s in plasma_shots:
            shot_num = s.get_shot_num()
            if shot_num in chosen_shots:
                c.append(1)
            else:
                c.append(0)

        plt.scatter(HHDs, br_chngs, c=c, cmap = 'Set1')
    
    elif plot_date:
        # Color the plot according to the date of the shot
        W = np.array(W)/1000
        c_day = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 
        6, 6, 6, 7, 8, 8, 8, 9])/9
        plt.scatter(HHDs, br_chngs, c = c_day, cmap = 'Set1')
    
    else:
        plt.plot(HHDs, br_chngs, 'r.')

    return fig, ax

def density_plot(plasma_shots, plot_HHD_diff = False):
    """
    Generates a plot of the H/H+D ratio of plasma shots versus their brightness change

    Parameters:
    plasma_shots (Array): List of shot objects to plot

    Returns:
    fig, ax: Figure and axis objects for plot
    """
    bright_chg = []
    HHD = []
    density = []
   
    for shot in plasma_shots:
        change = shot.get_br_chng()
        if plot_HHD_diff:
            HHD_ratio = shot.get_HHD_average()
            HHD.append(HHD_ratio)
        density.append(shot.get_n_e())
        bright_chg.append(change)
        
    # print("Max Density: ", np.max(density))
    # print("Min Density: ", np.min(density))

    # c_day = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 
    #     6, 6, 6, 7, 8, 8, 8, 9])/9
    fig, ax = plt.subplots()

    if plot_HHD_diff:
        HHD_diff = np.abs(np.array(HHD) - 0.31)/0.31

        ax.scatter(np.array(density)*1e-19, bright_chg, c=HHD_diff, cmap = 'gist_rainbow')
        ax.colorbar(label = 'Percent Difference of H/H+D Ratio from 0.31', cmap = 'gist_rainbow')
    
    else:
        xerr = np.array(density)*1e-19*0.15
        yerr = np.array(bright_chg)*0.15
        ax.errorbar(np.array(density)*1e-19, bright_chg, xerr = xerr, yerr = yerr, linewidth=1, ls='none', fmt='o', color='red', capsize=3)

    ax.set_xlabel("Density (e19 m^-3)")
    ax.set_ylabel("Brightness Change")
    return fig, ax

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

def dens_plot_const_HHD(plasma_shots, HHD_range = [0, 1]):
    """
    Plots density versus brightness change for shots within a specified ratio of H/H+D ratio

    Parameters:
    plasma_shots (Array): List of shot objects to plot
    HHD_range (Array): Range over which to plot
    """

    shots_in_range = []
    for shot in plasma_shots:
        if HHD_range[0] <= shot.get_HHD_average() <= HHD_range[1]:
            shots_in_range.append(shot)
    
    fig, ax = density_plot(shots_in_range)
    ax.set_title('Density plot over H/H+D range ({0} - {1})'.format(HHD_range[0], HHD_range[1]))
    ax.set_ylim([0, 1])
    ax.set_xlim([9, 19])
    plt.savefig('/home/imars23/Desktop/Data/Density_Constant_HHD_0.15Errorbar_{0}_{1}.png'.format(HHD_range[0], HHD_range[1]))
    plt.show()

def plot_same_day(plasma_shots, date, density = True, HHD =  True):
    """
    Make a plot of density or HHD versus brightness change over a specific day
    """

    shots_on_date = []
    for shot in plasma_shots:
        shot.set_date()
        if shot.get_date() == date:
            shots_on_date.append(shot)
    
    if density:
        fig, ax = density_plot(shots_on_date)
        ax.set_title('Density plot on date {0}/{1}/20{2}'.format(date[2:4], date[4:], date[0:2]))
        ax.set_ylim([0, 1])
        ax.set_xlim([9, 19])
        plt.savefig('/home/imars23/Desktop/Data/Density_Errobrar_Plot_ {0}.{1}.20{2}.png'.format(date[2:4], date[4:], date[0:2]))
        plt.show()

    if HHD:
        fig, ax = HHD_plot(shots_on_date)
        ax.set_title('HHD plot on date {0}/{1}/20{2}'.format(date[2:4], date[4:], date[0:2]))
        plt.show()

plasma_shots = read_shots(brchange = True, HHD  = True, n_e = True) # Generate list of plasma shots with brightness change and H/H+D ratio loaded
# HHD_plot(plasma_shots, plot_W = True)  # Generate plot of H/H+D ratio versus brightness change
# density_plot()    # Generate plot of density versus brightness change
# plot_same_day(plasma_shots, '140221') # Generate plot of density versus brightness change for a specific day
for date in ['140213', '140221']:
    plot_same_day(plasma_shots, date, HHD = False)