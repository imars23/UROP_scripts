import MDSplus
import numpy as np

shots = np.array([0])
shots = shots + 1150522022

for shot in shots:
    print(shot)
    specTree = MDSplus.Tree('spectroscopy', shot)
    branchHe = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HELIKE.PROFILES.Z')
    branchH = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.ANALYSIS.HLIKE.PROFILES.LYA1')

    try:
        branchHe.data()
        print(shot, ': THACO not run for He-like')
    except:
        pass
    
    try:
        branchH.data()
        print(shot, ': THACO not run for H-like')
    except:
        pass

print('Done')