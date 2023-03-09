'''

scenario_param.s.py is a script meant to give an overview of various important plasma
parameters for input files stored in tofu_sparc/background_plasma

cjperks, Jan 25, 2023

'''

# Modules
import numpy as np
import matplotlib.pyplot as plt
from omfit_classes import omfit_eqdsk, omfit_gapy
from scipy.interpolate import interp1d
import os

# List of scenario names in tofu_sparc/background_plasma to consider
scns = [
    'PRD_plasma',
    'TRANSP_scans_about_PRD_1',
    'TRANSP_scans_about _PRD_2',
    '12.2T_L-mode_BC.0.5keV',
    '12.2T_L-mode_BC.1keV',
    '12.2T_L-mode_BC.1.5keV',
]

# Path to input files
in_path = '/home/cjperks/tofu_sparc/background_plasma/'

# Loop over scenarios
for scn in scns:
    # Determines the runs within a scenerio
    runs = next(os.walk(in_path+scn))[1]

    # Initializes arrays to store scalar values vs. run#
    run_num = [] # run number
    Ip = [] # [MA]; plasma current
    Bt = [] # [T]; toroidal B-field
    P_ICRF = [] # [MW]; ICRF power
    fuel_mass = [] # [amu]; fuel mass
    f_W = [] # [frac]; concentration of W
    Zeff = [] # Z effective
    Te_avg = [] # volume-avg electron temp
    Ti_avg = [] # volume-avg ion temp
    f_G = [] # Greenwald fraction
    P_fus = [] # [MW]; fusion power
    Q = [] # Q
    f_rad = [] # [frac]; radiated power fraction
    P_rad = [] # [MW]; radiated power
    P_rad_W = [] # [MW]; radiated power from tungsten

    # Loop over runs
    for run in runs:
        # Reads input files
        gacode = omfit_gapy.OMFITgacode(in_path+scn+'/'+run+'/input.gacode')
        geqdsk = omfit_eqdsk.OMFITgeqdsk(in_path+scn+'/'+run+'/input.geq')

        # Obtains kinetic profiles
        rhop = np.sqrt(gacode['polflux']/gacode['polflux'][-1]) # dim(rhop,); sq. norm. pol flux
        ne = gacode['ne']*1e13 # dim(rhop,); [cm^-3]; electron density
        Te = gacode['Te']*1e3 # dim(rhop,); [eV]; electron temperature
        Ti = gacode['Ti_1']*1e3 # dim(rhop,); [eV]; ion temperatures
        ni = {} # dictionary to store ion density
        for n_ion in gacode['IONS']:
            ni[gacode['IONS'][n_ion][0]] = gacode['ni_'+str(n_ion)]*1e13 # dim(rhop,); [cm^-3]; ion density

        # Obtains plasma volume
        vol_geq = geqdsk['fluxSurfaces']['geo']['vol']  # [m^-3]
        rhop_geq = np.sqrt(geqdsk['fluxSurfaces']['geo']['psin'])
        vol = interp1d(rhop_geq, vol_geq)(rhop) # dim(rhop,); [m^-3]; plasma volume

        # Stores the run number
        run_num.append(run[3:])

        # Stores the plasma current
        Ip.append(abs(gacode['IP_EXP']))

        # Stroes the magnetic field
        Bt.append(abs(gacode['BT_EXP']))

        # Stores the ICRF power, [MW]
        P_ICRF.append(np.trapz(gacode['qrfe']+gacode['qrfi'], vol))

        # Stores the fuel mass

        # Stores Greenwald fraction
        n_G = abs(gacode['IP_EXP'])/(np.pi * gacode['rmin'][-1]**2)*1e14 # [cm^-3]
        f_G.append(np.trapz(ne, vol)/(vol[-1]*n_G))

        # Stores volume-averged electron temperature, [keV]
        Te_avg.append(np.trapz(Te/1e3, vol)/vol[-1])

        # Stores volume-averged ion temperature, [keV]
        Ti_avg.append(np.trapz(Ti/1e3, vol)/vol[-1])




