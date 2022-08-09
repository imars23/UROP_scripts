# -*- coding: utf-8 -*-
"""
ion_plot.py is a script meant to plot inverted (but not fit) kinetic profiles outputted by THACO

"""

# Loads useful scripts
import numpy as np
import matplotlib.pyplot as plt
from MDSplus import *



# Function to load THACO data from the tree
class ThacoData:
    def __init__(self, node):

        # Loads data from the tree node
        proNode = node.getNode('PRO')
        rhoNode = node.getNode('RHO')
        perrNode = node.getNode('PROERR')

        # Reads data from the tree node
        rpro = proNode.data()
        rrho = rhoNode.data()
        rperr = perrNode.data()
        rtime = rhoNode.dim_of()

        goodTimes = (rtime > 0).sum()

        # Outputs data
        self.time = rtime.data()[:goodTimes]
        self.rho = rrho[0,:] # Assume unchanging rho bins
        self.pro = rpro[:,:goodTimes,:len(self.rho)]
        self.perr = rperr[:,:goodTimes,:len(self.rho)]


specTree = Tree('SPECTROSCOPY', 1120815024)

# Load the nodes associated with inverted profile data
nodeA = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HELIKE.PROFILES.Z')
nodeB = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HLIKE.PROFILES.LYA1')

# Load the actual data
dataA = ThacoData(nodeA)
dataB = ThacoData(nodeB)

t0 = 1.01
# Look for the index corresponding to the given time point
indexA = np.searchsorted(dataA.time, t0)
indexB = np.searchsorted(dataB.time, t0)

# Plot the emissivity
fig1, ax1 = plt.subplots()
ax1.errorbar(dataA.rho, dataA.pro[0,indexA,:], yerr=dataA.perr[0,indexA,:], label = 'Z')
ax1.errorbar(dataB.rho, dataB.pro[0,indexB,:], yerr=dataB.perr[0,indexB,:], label = 'LYA1')
ax1.set_ylabel('emis [arb]')
ax1.set_xlabel('normalized poloidal flux')
ax1.legend()
fig1.show()

# Plot the toroidal rotation frequency
fig2, ax2 = plt.subplots()
ax2.errorbar(dataA.rho, dataA.pro[1,indexA,:], yerr=dataA.perr[1,indexA,:], label = 'Z')
ax2.errorbar(dataB.rho, dataB.pro[1,indexB,:], yerr=dataB.perr[1,indexB,:], label = 'LYA1')
ax2.set_ylabel(r'$\omega_{tor}$ [kHz]')
ax2.set_xlabel('normalized poloidal flux')
ax2.legend()
fig2.show()

# Plot the poloidal velocity
fig3, ax3 = plt.subplots()
ax3.errorbar(dataA.rho, dataA.pro[2,indexA,:], yerr=dataA.perr[2,indexA,:], label = 'Z')
ax3.errorbar(dataB.rho, dataB.pro[2,indexB,:], yerr=dataB.perr[2,indexB,:], label = 'LYA1')
ax3.set_ylabel(r'$v_{pol}$ [not sure]')
ax3.set_xlabel('normalized poloidal flux')
ax3.legend()
fig3.show()

# Plot the Ti
fig4, ax4 = plt.subplots()
ax4.errorbar(dataA.rho, dataA.pro[3,indexA,:], yerr=dataA.perr[3,indexA,:], label = 'Z')
ax4.errorbar(dataB.rho, dataB.pro[3,indexB,:], yerr=dataB.perr[3,indexB,:], label = 'LYA1')
ax4.set_ylabel(r'$T_i$ [keV]')
ax4.set_xlabel('normalized poloidal flux')
ax4.legend()
fig4.show()