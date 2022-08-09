"""
load_bright.py is a script meant to load brightness data from THACO
"""

import numpy as np
import matplotlib.pyplot as plt
import MDSplus
import os

# location of local directory
loc_dir = os.path.dirname(os.path.realpath(__file__))

def load_xics_specbr(shot, line=2, tht=0, plot=False, plot_time_s=1.0, plot_ch=30):
    '''
    Function to load Hirex-Sr spectrum brightness data for CMOD. Assumes that rest wavelengths, ionization stages and
    atomic line names are given in a file provided as input.

    Parameters
    -----------------
    shot : int
        C-Mod shot number
    line : int
        THACO line numbers
            LINE = 0 Ar w @ 3.94912 [Ang]
            LINE = 1 Ar x @ 3.96581 [Ang]
            LINE = 2 Ar z @ 3.99417 [Ang]
            LINE = 3 Ar lya1 @ 3.73114 [Ang]
            LINE = 4 Mo 4d @ 3.73980 [Ang]
            LINE = 5 Ar J @ 3.77179 [Ang]
            LINE = 6 Ca w @ 3.17730 [Ang]
            LINE = 7 Ar lya1 @ 3.73114 [Ang] (on Branch A)
            LINE = 8 Mo 4d @ 3.73980 [Ang] (on Branch A)
            LINE = 9 Ca lya1 @ 3.01850 [Ang]
    tht : int
        THACO tree analysis identifier.
    plot : bool
        If True, plot spectrum for the chosen time and channel
    plot_time_s : float
        Time to plot spectrum at.
    plot_ch : int
        Hirex-Sr channel number to plot. NB: this is normally between 1 and 48 for He-like Ar, 1 and 16 for H-like Ar (i.e. no Python indexing)

    Results
    -----------
    lams_A : array('n_lams', 'n_ch')
         Lambda vector for the spectrum
    times : array (`n_t`)
         Time vector for the spectrum.
    spec_br : array (`n_lams`,`n_t`,`n_ch`)
         Spectral brightness as a function of wavelength, time and spatial channel.
    spec_br_unc : array (`n_lams`,`n_t`,`n_ch`)
         Uncertainty on spectral brightness as a function of wavelength, time and spatial channel.
    '''

    # Loads THACO node for THT
    specTree = MDSplus.Tree('spectroscopy', shot)
    ana = '.ANALYSIS'
    if tht > 0:
        ana += str(tht)

    # Defines path of THT tree
    rootPath = r'\SPECTROSCOPY::TOP.HIREXSR'+ana

    # If user is considering a line from the He-like Ar branch
    if line == 0 or line == 1 or line == 2:
        # Loads Branch A data
        branchNode = specTree.getNode(rootPath+'.HELIKE')

        # Loads the wavelengths imaged
        lam_all = branchNode.getNode('SPEC:LAM').data()

    # If user is considering a line from the H-like Ar branch
    elif line == 3 or line ==4 or line ==5:
           # Loads Branch B data
            branchNode = specTree.getNode(rootPath+'.HLIKE')

            # Loads the wavelengths imaged
            lam_all = branchNode.getNode('SPEC:LAM').data()

    # Indices are [lambda, time, channel]
    specBr_all = branchNode.getNode('SPEC:SPECBR').data()
    specBr_unc_all = branchNode.getNode('SPEC:SIG').data()  # uncertainties

    # load pos vector:
    #pos, pos_lam = hirexsr_pos(shot, tht, lam_bounds=lam_bounds, hirex_branch=hirex_branch)

    # Maximum number of channels, time bins
    maxChan = np.max(branchNode.getNode('BINNING:CHMAP').data())+1
    maxTime = np.max(branchNode.getNode('BINNING:TMAP').data())+1

    # get time basis
    all_times =np.asarray(branchNode.getNode('SPEC:SIG').dim_of(1))
    mask = all_times >-1

    # select only available time elements
    times = all_times[mask] # dim = (time)
    lams_A= lam_all[:,mask,:] # dim = (lambda, time, ch)
    spec_br = specBr_all[:,mask,:] # dim = (lambda, time, ch)
    spec_br_unc = specBr_unc_all[:,mask,:] # dim = (lambda, time, ch)

    # Take XICS wavelengths to be time independent
    lams_A = lams_A[:,0,:] # dim = (lambda, ch)

    # If we wish to plot the spectrum
    if plot:
        # Load all wavelength data
        with open(loc_dir+'/hirexsr_wavelengths.csv', 'r') as f:
            lineData = [s.strip().split(',') for s in f.readlines()] # all data
            lineLam = np.array([float(ld[1]) for ld in lineData[2:]]) # wavelengths
            lineZ = np.array([int(ld[2]) for ld in lineData[2:]]) # element
            lineName = np.array([ld[3] for ld in lineData[2:]]) # spectroscopic name

        # select only lines from Ar
        xics_lams = lineLam[lineZ==18]
        xics_names = lineName[lineZ==18]

        # plot spectrum at chosen time and channel, displaying known lines in database
        tidx = np.argmin(np.abs(times - plot_time_s))
        fig = plt.figure()
        fig.set_size_inches(10,7, forward=True)
        ax1 = plt.subplot2grid((10,1),(0,0),rowspan = 1, colspan = 1, fig=fig)
        ax2 = plt.subplot2grid((10,1),(1,0),rowspan = 9, colspan = 1, fig=fig, sharex=ax1)

        ax2.errorbar(lams_A[:,plot_ch-1], spec_br[:,tidx,plot_ch-1], spec_br_unc[:,tidx,plot_ch-1], fmt='.')
        ax2.set_xlabel(r'$\lambda$ [\AA]',fontsize=14)
        ax2.set_ylabel(r'Signal [A.U.]',fontsize=14)

        for ii,_line in enumerate(xics_lams):
            if _line>ax2.get_xlim()[0] and _line<ax2.get_xlim()[1]:
                ax2.axvline(_line, c='r', ls='--')
                ax1.axvline(_line, c='r', ls='--')

                ax1.text(_line, 0.5, xics_names[ii], rotation=90, fontdict={'fontsize':14})
        ax1.axis('off')
        fig.show()

    # Returns lambda array, time array, spectrum brightness and uncertainty
    return lams_A, times, spec_br, spec_br_unc

def load_xics_lintbr(shot, line=2, tht=0, plot=False, plot_time_s=1.0, plot_ch=30):
    '''
    Function to load Hirex-Sr line-integrated brightness data for CMOD. Assumes that rest wavelengths, ionization stages and
    atomic line names are given in a file provided as input.

    Parameters
    -----------------
    shot : int
        C-Mod shot number
    line : int
        THACO line numbers
            LINE = 0 Ar w @ 3.94912 [Ang]
            LINE = 1 Ar x @ 3.96581 [Ang]
            LINE = 2 Ar z @ 3.99417 [Ang]
            LINE = 3 Ar lya1 @ 3.73114 [Ang]
            LINE = 4 Mo 4d @ 3.73980 [Ang]
            LINE = 5 Ar J @ 3.77179 [Ang]
            LINE = 6 Ca w @ 3.17730 [Ang]
            LINE = 7 Ar lya1 @ 3.73114 [Ang] (on Branch A)
            LINE = 8 Mo 4d @ 3.73980 [Ang] (on Branch A)
            LINE = 9 Ca lya1 @ 3.01850 [Ang]
    tht : int
        THACO tree analysis identifier.
    plot : bool
        If True, plot spectrum for the chosen time and channel
    plot_time_s : float
        Time to plot spectrum at.
    plot_ch : int
        Hirex-Sr channel number to plot. NB: this is normally between 1 and 48 for He-like Ar, 1 and 16 for H-like Ar (i.e. no Python indexing)

    Results
    -----------
    psin : array ('n_t', 'n_ch')
         Radial array for the tangential rho = normalized poloidal flux
    times : array (`n_t`)
         Time array
    ch_num : array('n_ch')
         Channel array
    lint_br : array (`n_lams`,`n_t`,`n_ch`)
         line-averaged brightness for a specific line as a function of time and spatial channel.
    lint_br_unc : array (`n_lams`,`n_t`,`n_ch`)
         Uncertainty on line-averaged brightness for a specific line as a function of time and spatial channel.
    '''

    # Loads THACO node for THT
    specTree = MDSplus.Tree('spectroscopy', shot)
    ana = '.ANALYSIS'
    if tht > 0:
        ana += str(tht)

    # Defines path of THT tree
    rootPath = r'\SPECTROSCOPY::TOP.HIREXSR'+ana

    # Determines which line to load moments data from
    if line == 0:
        # Loads Branch A data
        branchNode = specTree.getNode(rootPath+'.HELIKE')

        # Loads moment data
        momNode = branchNode.getNode('MOMENTS.W:MOM')
        errNode = branchNode.getNode('MOMENTS.W:ERR')

    elif line == 1:
        # Loads Branch A data
        branchNode = specTree.getNode(rootPath+'.HELIKE')

        # Loads moment data
        momNode = branchNode.getNode('MOMENTS.X:MOM')
        errNode = branchNode.getNode('MOMENTS.X:ERR')

    elif line == 2:
        # Loads Branch A data
        branchNode = specTree.getNode(rootPath+'.HELIKE')

        # Loads moment data
        momNode = branchNode.getNode('MOMENTS.Z:MOM')
        errNode = branchNode.getNode('MOMENTS.Z:ERR')

    elif line == 3:
        # Loads Branch A data
        branchNode = specTree.getNode(rootPath+'.HLIKE')

        # Loads moment data
        momNode = branchNode.getNode('MOMENTS.LYA1:MOM')
        errNode = branchNode.getNode('MOMENTS.LYA1:ERR')

    elif line == 4:
        # Loads Branch A data
        branchNode = specTree.getNode(rootPath+'.HLIKE')

        # Loads moment data
        momNode = branchNode.getNode('MOMENTS.MO4D:MOM')
        errNode = branchNode.getNode('MOMENTS.MO4D:ERR')

    elif line == 5:
        # Loads Branch A data
        branchNode = specTree.getNode(rootPath+'.HLIKE')

        # Loads moment data
        momNode = branchNode.getNode('MOMENTS.J:MOM')
        errNode = branchNode.getNode('MOMENTS.J:ERR')
    
    # Loads line-integrated brightness, dim = (time, channel)
    lintBr_all = momNode.data()[0]
    lintBr_unc_all = errNode.data()[0]  # uncertainties

    # Maximum number of channels, time bins
    maxChan = np.max(branchNode.getNode('BINNING:CHMAP').data())+1
    maxTime = np.max(branchNode.getNode('BINNING:TMAP').data())+1

    # Loads array of channel bins
    ch_num = np.arange(maxChan)+1

    # Loads matrix of psi_norm values for tangential rho
    psin = momNode.dim_of(0).data() # dim = (time,ch)

    # get time basis
    all_times =np.asarray(momNode.dim_of(1))
    mask = all_times >-1

    # select only available time elements
    times = all_times[mask] # dim = (time)
    lint_br = lintBr_all[mask, :] # dim = (time, ch)
    lint_br_unc = lintBr_unc_all[mask, :] # dim = (time, ch)
    psin = psin[mask,:] # dim = (time,ch)

    # selects only available channel bins
    lint_br = lint_br[:,ch_num-1] # dim = (time,ch)
    lint_br_unc = lint_br_unc[:,ch_num-1] # dim = (time,ch)
    psin = psin[:,ch_num-1] # dim (time,ch)

    # If user wishes to plot output
    if plot:
        # plot spectrum at chosen time and channel, displaying known lines in database
        tidx = np.argmin(np.abs(times - plot_time_s))

        fig, ax = plt.subplots(3,1)

        # Plots brightness and uncertainty at fixed time as a function of channel
        ax[0].errorbar(ch_num, lint_br[tidx,:], lint_br_unc[tidx,:], fmt='.')
        ax[0].set_xlabel('CH #')
        ax[0].set_ylabel('Birghtness [arb]')
        ax[0].set_ylim([0,1.5])

        # Plots brightness and uncertainty at fixed time as a function of psi_norm
        ax[1].errorbar(np.sqrt(psin[tidx,:]), lint_br[tidx,:], lint_br_unc[tidx,:], fmt='.')
        ax[1].set_xlabel(r'$\rho=\sqrt{\Psi_n}$')
        ax[1].set_ylabel('Brightness [arb]')
        ax[1].set_ylim([0,1.5])

        # Plots brightness and uncertainty at fixed chennel as a function of time
        ax[2].errorbar(times, lint_br[:,plot_ch-1], lint_br_unc[:, plot_ch-1], fmt='.-')
        ax[2].set_xlabel('Time [sec]')
        ax[2].set_ylabel('Brightness [arb]')
        ax[2].set_ylim([0,1.5])
        fig.tight_layout(pad=1.0)
        fig.show()

    # Returns psi_norm array, time array, channel array, line-integrated brightness and uncertainty
    return psin, times, ch_num, lint_br, lint_br_unc



def load_hirexsr_pos(shot, line=2, tht=0, plot=False, check_with_tree=False, plot_time_s=1.0, plot_ch = -1):
    '''
    Get the POS vector as defined in the THACO manual, keeping POS as a function of wavelength.

    Parameters
    ------------------
        shot : int
        C-Mod shot number
    line : int
        THACO line numbers
            LINE = 0 Ar w @ 3.94912 [Ang]
            LINE = 1 Ar x @ 3.96581 [Ang]
            LINE = 2 Ar z @ 3.99417 [Ang]
            LINE = 3 Ar lya1 @ 3.73114 [Ang]
            LINE = 4 Mo 4d @ 3.73980 [Ang]
            LINE = 5 Ar J @ 3.77179 [Ang]
            LINE = 6 Ca w @ 3.17730 [Ang]
            LINE = 7 Ar lya1 @ 3.73114 [Ang] (on Branch A)
            LINE = 8 Mo 4d @ 3.73980 [Ang] (on Branch A)
            LINE = 9 Ca lya1 @ 3.01850 [Ang]
    tht : int
        THACO tree analysis identifier.
    plot : bool
        If True, plot spectrum for the chosen time and channel
    check_with_tree : bool
         If True, also return POS vector as it is computed on the MDS+ tree
    plot_time_s : float
         Time to plot spectrum at.
    plot_ch : int
         Hirex-Sr channel number to plot. NB: this is normally between 1 and 48 for He-like Ar, 1 and 16 for H-like Ar (i.e. no Python indexing)

    Returns
    ------------
    pos_ave : array (`n_ch`, 4)
         POS vector for each channel, required for line integration.
    pos_on_tree : array ('n_ch', 4)
         Only returned if check_with_tree=True, to compare to pos_ave. 
    xyz_srt : array('n_ch', 3)
         Cartesian (x,y,z)-coordinates of the LOS start point
    xyz_end : array('n_ch', 3)
         Cartesian (x,y,z)-coordinate of the LOS end point

    '''

    # Loads THACO node for THT
    specTree = MDSplus.Tree('spectroscopy', shot)
    ana = '.ANALYSIS'
    if tht > 0:
        ana += str(tht)

    # Defines path of THT tree
    rootPath = r'\SPECTROSCOPY::TOP.HIREXSR'+ana

    # If considering the He-like Ar spectrum
    if line == 0 or line == 1 or line == 2:
        # Loads Branch A data
        branchNode = specTree.getNode(rootPath+'.HELIKE')

        # pos vectors for detector modules 1-3 --- never used to look at Ca
        pos1 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD1:POS').data()
        pos2 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD2:POS').data()
        pos3 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD3:POS').data()

        # wavelengths for each module
        lam1 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD1:LAMBDA').data()
        lam2 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD2:LAMBDA').data()
        lam3 = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD3:LAMBDA').data()

        pos_tot = np.hstack([pos1,pos2,pos3])
        lam_tot = np.hstack([lam1,lam2,lam3])

    # If considering the H-like Ar spectrum
    elif line == 3 or line == 4 or line == 5:
        # Loads Branch A data
        branchNode = specTree.getNode(rootPath+'.HLIKE')

        # 1 detector module
        pos_tot = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD4:POS').data()

        # wavelength
        lam_tot = specTree.getNode(r'\SPECTROSCOPY::TOP.HIREXSR.CALIB.MOD4:LAMBDA').data()


    # mapping from pixels to chords (wavelength x space pixels, but wavelength axis is just padding)
    chmap = branchNode.getNode('BINNING:CHMAP').data()
    pixels_to_chords = chmap[0,:]

    #lam_all = branchNode.getNode('SPEC:LAM').data()

    # exclude empty chords
    #mask = lam_all[0,0,:]!=-1
    #lam_masked = lam_all[:,:,mask]
    #num_chords =  lam_masked.shape[2]

    lam_masked = lam_tot
    num_chords = np.max(pixels_to_chords)+1

    # Find the wavelength range to average POS vectors over
    if line == 0:
        # Rest wavelength
        lam0 = 3.94912 # [Ang]

        # Integrated wavelength range
        dlam = branchNode.getNode('MOMENTS.W:DLAM').data()/1000 #[Ang]
    elif line == 1:
        # Rest wavelength
        lam0 = 3.96581 # [Ang]

        # Integrated wavelength range
        dlam = branchNode.getNode('MOMENTS.X:DLAM').data()/1000 #[Ang]
    elif line == 2:
        # Rest wavelength
        lam0 = 3.99417 # [Ang]

        # Integrated wavelength range
        dlam = branchNode.getNode('MOMENTS.Z:DLAM').data()/1000 #[Ang]
    elif line == 3:
        # Rest wavelength
        lam0 = 3.73114 # [Ang]

        # Integrated wavelength range
        dlam = branchNode.getNode('MOMENTS.LYA1:DLAM').data()/1000 #[Ang]
    elif line == 4:
        # Rest wavelength
        lam0 = 3.73980 # [Ang]

        # Integrated wavelength range
        dlam = branchNode.getNode('MOMENTS.MO4D:DLAM').data()/1000 #[Ang]
    elif line == 5:
        # Rest wavelength
        lam0 = 3.77179 # [Ang]

        # Integrated wavelength range
        dlam = branchNode.getNode('MOMENTS.J:DLAM').data()/1000 #[Ang]

    # Bounds to integrate over
    lam_bounds = [lam0-dlam, lam0+dlam] # [Ang]

    # lambda vector should not change over time, so just use tbin=0
    w0 = np.zeros(num_chords, dtype=int);  w1 = np.zeros(num_chords, dtype=int)
    for chbin in np.arange(num_chords):
        #bb = np.searchsorted(lam_masked[:,0,chbin], lam_bounds)
        bb = np.searchsorted(lam_masked[:,0], lam_bounds)
        w0[chbin] = bb[0]; w1[chbin] = bb[1]

    # use POS vector corresponding to an average over the imaged wavelength range    
    pos_ave = np.zeros((num_chords, 4))
    pos_std = np.zeros((num_chords, 4))
    for chord in np.arange(num_chords):
        pos_ave[chord, :] = np.mean(pos_tot[w0[chord]:w1[chord],:,:][:, pixels_to_chords == chord,:], axis=(0,1))
        pos_std[chord, :] = np.std(pos_tot[w0[chord]:w1[chord],:,:][:, pixels_to_chords == chord,:], axis=(0,1))
        # checked: pos_std seems always small; but is it negligible?

    # issue with using a POS vector as a function of wavelength would be that each chord may correspond
    # to a slightly different wavelength range...Also, using a single POS vector per chord is much simpler...
    sel_lam = (lam_tot[:,0]>lam_bounds[0])&(lam_tot[:,0]<lam_bounds[1])
    pos_not_ave = np.zeros((np.sum(sel_lam), num_chords, 4))
    for chord in np.arange(num_chords):
        pos_not_ave[:,chord, :] = np.mean(pos_tot[sel_lam, :, :][:,pixels_to_chords == chord,:], axis=1)
        #pos_ave[:,chord,:] = np.mean(pos_tot[sel_lam, pixels_to_chords == chord,:], axis=1 )

    # Calculates the tangential Z-coordinate
    Z_T = pos_ave[:,1]- np.tan(pos_ave[:,3])*np.sqrt(pos_ave[:,0]**2-pos_ave[:,2]**2) # [m], dim = n_ch

    # Calculates the toroidal angle subtended to the tangency radius
    phi_T = np.arccos(pos_ave[:,2]/pos_ave[:,0]) # [rad], dim = n_ch

    # Creates matrix to store LOS start point in Cartesian (x,y,z)-coordinates
    xyz_srt = np.stack((pos_ave[:,0],0*pos_ave[:,0],pos_ave[:,1]), axis = 1) # dim = (n_ch, 3)

    # Creates matrix to store LOS end point in Cartesian (x,y,z)-coordinates
    X_T = pos_ave[:,2]*np.cos(phi_T)
    Y_T = pos_ave[:,2]*np.sin(phi_T)
    xyz_end = np.stack((X_T, Y_T, Z_T), axis =1) # dim = (n_ch, 3)

    if plot:
        # show each component of the pos vector separately
        fig,ax = plt.subplots(2,2)
        axx = ax.flatten()
        for i in [0,1,2,3]:
            pcm = axx[i].pcolormesh(pos_tot[:,:,i].T)
            axx[i].axis('equal')
            axx[i].set_xlabel('lambda pixel')
            axx[i].set_ylabel('spatial pixel')
            fig.colorbar(pcm, ax=axx[i])
        fig.show()

        import TRIPPy
        import eqtools

        # visualize chords
        efit_tree = eqtools.CModEFITTree(shot)
        tokamak = TRIPPy.plasma.Tokamak(efit_tree)

        #pos_ave[:,0]*=1.2
        # pos[:,3] indicate spacing between rays
        rays = [TRIPPy.beam.pos2Ray(p, tokamak) for p in pos_ave]   #pos_old]

        weights = TRIPPy.invert.fluxFourierSens(rays,efit_tree.rz2psinorm,tokamak.center, plot_time_s, np.linspace(0,1, 150),ds=1e-5)[0]

        from TRIPPy.plot.pyplot import plotTokamak, plotLine

        # Plots lines-of-sight over the tokamak
        f = plt.figure()
        a = f.add_subplot(1, 1, 1)
        # Only plot the tokamak if an axis was not provided:
        plotTokamak(tokamak) # Plots the tokamak wall

        # Plots the lines-of-sight
        if plot_ch == -1:
            for r in rays:
                plotLine(r, pargs='r',lw=1.0)
        else:
            plotLine(rays[plot_ch-1], pargs='r',lw=1.0)
            a.plot([xyz_srt[plot_ch-1,0], xyz_end[plot_ch-1,0]], [xyz_srt[plot_ch-1,2], xyz_end[plot_ch-1,2]], 'b-')

        # Finds the flux surfaces at the specified times
        i_flux = np.searchsorted(efit_tree.getTimeBase(), plot_time_s)

        # mask out coils, where flux is highest
        flux = efit_tree.getFluxGrid()[i_flux, :, :] # Loads the flux surfaces
        psiLCFS = efit_tree.__dict__['_psiLCFS'][i_flux] # pol. flux at LCFS
        psiAxis = efit_tree.__dict__['_psiAxis'][i_flux] # pol. flux on-axis
        flux = (flux-psiAxis)/(psiLCFS-psiAxis) # normalizes the (R,Z) grid of pol. flux
        flux[flux>1] = 1.05 # Ignores flux surfaces over a percentage
        #flux[:,efit_tree.getRGrid()>0.9] = np.nan # Ignores flux surfaces over a R-position

        cset = a.contour(
            efit_tree.getRGrid(),
            efit_tree.getZGrid(),
            flux,
            11
        ) # Plots contours of the flux surfaces
        f.colorbar(cset,ax=a, label = r'\Psi_n')
        a.set_ylabel('Z [m]')
        a.set_xlabel('X [m]')
        a.set_xlim([0.4,1.1])
        f.show()

    if check_with_tree:
        try:
            #pos_on_tree = branchNode.getNode('MOMENTS.{:s}:POS'.format(primary_line.upper())).data()
            pos_on_tree = branchNode.getNode('MOMENTS.Z:POS').data()
        except:
            pos_on_tree = branchNode.getNode('MOMENTS.LYA1:POS').data()
        return pos_ave, pos_on_tree
    else:
        return pos_ave, pos_not_ave, xyz_srt, xyz_end


def load_thaco_good(ch_num, shot, line=2, tht=0):
    '''
    Function to load which spatial channels were marked as GOOD during the THACO profiles inversion analysis

    Parameters
    -----------------
    ch_num : array('n_ch')
        Channel array
    shot : int
        C-Mod shot number
    line : int
        THACO line numbers
            LINE = 0 Ar w @ 3.94912 [Ang]
            LINE = 1 Ar x @ 3.96581 [Ang]
            LINE = 2 Ar z @ 3.99417 [Ang]
            LINE = 3 Ar lya1 @ 3.73114 [Ang]
            LINE = 4 Mo 4d @ 3.73980 [Ang]
            LINE = 5 Ar J @ 3.77179 [Ang]
            LINE = 6 Ca w @ 3.17730 [Ang]
            LINE = 7 Ar lya1 @ 3.73114 [Ang] (on Branch A)
            LINE = 8 Mo 4d @ 3.73980 [Ang] (on Branch A)
            LINE = 9 Ca lya1 @ 3.01850 [Ang]
    tht : int
        THACO tree analysis identifier.

    Results
    -----------
    ch_GOOD : array('n_ch')
        GOOD channel array
    '''

    # Loads THACO node for THT
    specTree = MDSplus.Tree('spectroscopy', shot)
    ana = '.ANALYSIS'
    if tht > 0:
        ana += str(tht)

    # Defines path of THT tree
    rootPath = r'\SPECTROSCOPY::TOP.HIREXSR'+ana

    # Determines which line to load moments data from
    if line == 0:
        # Loads GOOD data
        GOOD = specTree.getNode(rootPath+'.HELIKE.PROFILES.W.CONFIG:GOOD').data()

    elif line == 1:
        # Loads GOOD data
        GOOD = specTree.getNode(rootPath+'.HELIKE.PROFILES.X.CONFIG:GOOD').data()

    elif line == 2:
        # Loads GOOD data
        GOOD = specTree.getNode(rootPath+'.HELIKE.PROFILES.Z.CONFIG:GOOD').data()

    elif line == 3:
        # Loads GOOD data
        GOOD = specTree.getNode(rootPath+'.HLIKE.PROFILES.LYA1.CONFIG:GOOD').data()

    elif line == 4:
        # Loads GOOD data
        GOOD = specTree.getNode(rootPath+'.HLIKE.PROFILES.MO4D.CONFIG:GOOD').data()

    elif line == 5:
        # Loads GOOD data
        GOOD = specTree.getNode(rootPath+'.HLIKE.PROFILES.J.CONFIG:GOOD').data()

    # Assume that BAD channels were removed for all times
    GOOD = GOOD[0]

    # Removes BAD channels
    ch_GOOD = ch_num[list(map(bool, GOOD))]

    # Returns array of GOOD channels
    return ch_GOOD

