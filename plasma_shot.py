import MDSplus
import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy.optimize as opt

"""
Contains the plasma_shot class
"""

class shot:

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

    def set_date(self):
        shot_str = str(self.shot_num)
        date = shot_str[1:-3]
        self.date = date

    def set_n_e(self, n_e):
        self.n_e = n_e

    def get_n_e(self):
        return self.n_e

    def get_date(self):
        return self.date

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