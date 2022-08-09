import numpy as np
import matplotlib.pyplot as plt
import MDSplus
import os

shot = 1120815024
specTree = MDSplus.Tree('spectroscopy', shot)
rootPath = r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HELIKE.FITS.ZJK'

bNode = specTree.getNode(rootPath)

coefs = bNode.getNode(':COEFS').data()