import h5py
import numpy as np
import sys
import pdb
import get_rho_interpolator as get_rho
import matplotlib.pyplot as plt
from readGeqdsk import geqdsk
import matplotlib as mpl
from shapely.geometry import Point, Polygon
from matplotlib.patches import Rectangle
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import sparc_processing as proc
from scipy import optimize
import scipy.constants             as constants
import physicslib_20220211         as physicslib
import map_zone_volumes            as MZ
import compute_flux_surface_volume as CF
import sds_utilities               as SU

electron_charge  = constants.elementary_charge
electron_mass    = constants.physical_constants["electron mass"][0]
light_speed      = constants.physical_constants["speed of light in vacuum"][0]
alpha_mass       = constants.physical_constants["alpha particle mass"][0]
proton_mass      = constants.physical_constants["proton mass"][0]

def extract_runids(gg):
    
    return [key for key in gg.keys()]


def construct_full_filename(filename_in):
    #pdb.set_trace()
    stub          = filename_in.split('_')[0]
    remainder     = filename_in.split('_')[1]
    remainder     = remainder.split('.')[0]
    
    this_filename ='~/ascot/ascot5-knl/ascot_run_output/' + stub + '_work_' + remainder + '/' + stub + '_' + remainder + '.h5'

    return this_filename

# ++++++++++++++++++++++++++++++++++++

def plot_lcfs(fn_geqdsk, eq_index, thick=1, color='r'):

    gg = geqdsk(fn_geqdsk)
    gg.readGfile()
    
    rlcfs    =  gg.equilibria[eq_index].rlcfs
    zlcfs    =  gg.equilibria[eq_index].zlcfs

    style = color + '-'
    plt.plot(rlcfs, zlcfs, style, linewidth=thick)

def get_ini_end(runid, groupname, nmax=0):
    """
    inputs
       runid      = name of h5 file, e.g. ascot_12345678.h5'
       groupname  = inistate or endstate (a string)
    returns:  a dictionary with various parameters
    """

    file_name = construct_full_filename(runid)

    try:
       print("   ... get_ini_end: first try to read from ASCOT output: ", file_name)
       ff = h5py.File(file_name, 'r')
    except:
       print("   ... get_ini_end: that file not found.  Looking now for local file: ", runid)
       ff = h5py.File(runid,'r')
    
    rr = ff['results']

    results_keys = rr.keys()
    for key in results_keys:
        run_id = key
    results = rr[run_id]

    results = ff['results']
    run_id_list = extract_runids(results)

    run_id = run_id_list[0]

    out = {}

    rr_temp  = results[run_id][groupname]
    key_list = rr_temp.keys()
    
    for key in key_list:
        array = np.array(rr_temp[key])
        if(nmax > 0):
           array = array[0:nmax]
        out[key] = array

    #  convert momentum to velocity
    

    out["btot"] = np.sqrt(out["bphi"]**2 + out["br"]**2 + out["bz"]**2)

    try:
        ppar     = out['ppar']
        pphiprt  = out['pphiprt']
        prprt    = out['prprt']
        pzprt    = out['pzprt']
        itype    = 1
        print("   ... using type=1")
    except:                       # 2/11/2022
        pphiprt  = out["pphi"]
        prprt    = out["pr"]
        pzprt    = out["pz"]
        ppar = (out["pphi"]*out["bphi"] + out["pr"]*out["br"] + out["pz"]*out["bz"])/btot
        itype    = 2
        print("   ... using type=2")
        
    mass     = out['mass']
    endcond  = out['endcond']
    
    amu_mass = 1.6605e-27

    vphi    = pphiprt / (mass*amu_mass)     # **NEW**
    vr      = prprt    / (mass*amu_mass)     # **NEW**
    vz      = pzprt      / (mass*amu_mass)     # **NEW**
        
    vpar    = ppar  / (mass*amu_mass)     # **NEW**


    vtot    = (vphi**2 + vr**2 + vz**2)**(0.5)
    ekev    = 0.001  * amu_mass * mass * vtot**2/ (2.*electron_charge)
    pitch   = vpar/(vtot+1.e-9)

    
    out['vperp']  = np.sqrt(vtot**2 - vpar**2)
    out['vpar']   = vpar
    out['vphi']   = vphi
    out['vtot']   = vtot
    out['ekev']   = ekev
    out['pitch']  = pitch
    out['mu_sds'] = (mass/4.)  * alpha_mass * out['vperp']**2/(2. *electron_charge *  out["btot"])

    # +++++++++++++++++++++++++++++++++++++++
    #  arrays of endconditions
    
    ii_emin       = (endcond == 2)
    ii_therm      = (endcond == 4)
    ii_survived   = (endcond == 2) ^ (endcond == 4) ^ (endcond == 1) ^ (endcond == 256)
    ii_simtime    = (endcond == 1)
    ii_cputime    = (endcond == 256)
    ii_abort      = (endcond == 0)
    ii_rhomax     = (endcond == 32)
    ii_wall       = (endcond == 8)
    ii_520        = (endcond == 520)
    ii_lost       = (endcond == 8) ^ (endcond == 32)  ^ (endcond == 520)

    if(nmax>0):
        ii_emin         = ii_emin[0:nmax] 
        ii_therm        = ii_therm[0:nmax]
        ii_survived     = ii_survived[0:nmax]  
        ii_simtime      = ii_simtime[0:nmax]
        ii_cputime      = ii_cputime[0:nmax]
        ii_abort        = ii_abort[0:nmax]     
        ii_rhomax       = ii_rhomax[0:nmax]  
        ii_wall         = ii_wall[0:nmax] 
        ii_520          = ii_520[0:nmax]    
        ii_lost         = ii_lost[0:nmax] 

        
    out["ii_emin"]     = ii_emin       
    out["ii_therm"]    = ii_therm      
    out["ii_survived"] = ii_survived   
    out["ii_simtime"]  = ii_simtime    
    out["ii_cputime"]  = ii_cputime    
    out["ii_abort"]    = ii_abort      
    out["ii_rhomax"]   = ii_rhomax     
    out["ii_wall"]     = ii_wall       
    out["ii_520"]      = ii_520        
    out["ii_lost"]     = ii_lost

    if(groupname == 'endstate'):
        
        print("\n  summed endstate weights \n")
        print("    wall       %6d  %7.4f  "%(out['weight'][ii_wall].size,     np.sum(out['weight'][ii_wall]     )))
        print("    rhomax      %6d  %7.4f  "%(out['weight'][ii_rhomax].size,   np.sum(out['weight'][ii_rhomax]   )))
        print("    emin        %6d  %7.4f  "%(out['weight'][ii_emin].size,     np.sum(out['weight'][ii_emin]     )))
        print("    therm       %6d  %7.4f  "%(out['weight'][ii_therm].size,    np.sum(out['weight'][ii_therm]    )))
        print("    simtime    %6d  %7.4f  "%(out['weight'][ii_simtime].size,  np.sum(out['weight'][ii_simtime]  )))
        print("    cputime     %6d  %7.4f  "%(out['weight'][ii_cputime].size,  np.sum(out['weight'][ii_cputime]  )))
        print("    abort       %6d  %7.4f  "%(out['weight'][ii_abort].size,    np.sum(out['weight'][ii_abort]    )))
        print("    520         %6d  %7.4f  "%(out['weight'][ii_520].size,      np.sum(out['weight'][ii_520]      )))
        print("\n    lost        %6d  %7.4f  "%(out['weight'][ii_lost].size,     np.sum(out['weight'][ii_lost]     )))
        print("    survived    %6d  %7.4f  "%(out['weight'][ii_survived].size, np.sum(out['weight'][ii_survived] )))
        print("    grand total%8d  %7.4f  "%(out['weight'].size,              np.sum(out["weight"]              )))
        print("")
        
    elif (groupname == "inistate"):
        print("\n  summed marker weights \n")
        print("    grand total %8d  %7.4f  "%(out['weight'].size,              np.sum(out["weight"]              )))

    return out

# ++++++++++++++++++++++++++++++++++++


def get_options(runid):
    """
    inputs
       runid      = name of h5 file, e.g. ascot_12345678.h5'
    returns:  a dictionary with various parameters
    """

    file_name = construct_full_filename(runid)

    try:
        print("   ... get_options: first try to read from ASCOT output directory: ", file_name)
        ff = h5py.File(file_name, 'r')
    except:
        print("   ... get_options: that file not found.  Try to read from local file: ", runid)
        ff = h5py.File(runid, 'r')

    options = ff['options']

    run_id_list = extract_runids(options)
    run_id = run_id_list[0]
    options = options[run_id]
    key_list = options.keys()
    
    out = {}
    

    for key in key_list:
        array = np.array(options[key])
        out[key] = array
    
    return out

# ++++++++++++++++++++++++++++++++++++


def get_mhd(runid,detail=0):
    """
    inputs
       runid      = name of h5 file, e.g. ascot_12345678.h5'
    returns:  a dictionary with various parameters
    """

    file_name = construct_full_filename(runid)

    print("   ... get_mhd:  will try to read from file: ", file_name)
    try:
        print("   ... get_mhd: first try to read from ASCOT output directory: ", file_name)
        ff = h5py.File(file_name, 'r')
    except:
        print("   ... get_mhd: that file not found.  Try to read from local file: ", runid)
        ff = h5py.File(runid, 'r')

    options = ff['mhd']

    run_id_list = extract_runids(options)
    run_id      = run_id_list[0]
    options     = options[run_id]
    key_list    = options.keys()
    
    out = {}
    
    
    for key in key_list:
        array    = np.array(options[key])
        out[key] = array

    fn_out = runid.split(".")[0] + "_mhd_details.pdf"
 
    if(detail !=0):
        
        print("\n   ... get_ascot.get_mhd: sizes of returned arrays: \n")
        for key in key_list:
           this_shape = np.array(out[key]).shape
           print("  variable: ", key, "          shape: ", this_shape)
           
        print("")
        print("   nmode        %5d   "%(out["nmode"]))
        print("   nrho         %5d   "%(out["nrho"]))
        print("   rhomax       %6.3f "%(out["rhomax"][0]))
        print("   rhomin       %6.3f "%(out["rhomin"][0]))
        print("")
        

        print("   ... about to start plots")
        with PdfPages(fn_out) as pdf:

            plt.close("all")
            plt.plot(out["mmodes"],"bo-",ms=3)
            my_title = runid.split(".")[0] + ": mmodes"
            plt.title(my_title)
            pdf.savefig()

            plt.close("all")
            plt.plot(out["nmodes"],"bo-",ms=3)
            my_title = runid.split(".")[0] + ": nnodes"
            plt.title(my_title)
            pdf.savefig()

            plt.close("all")
            plt.plot(out["amplitude"],"bo-",ms=3)
            my_title = runid.split(".")[0] + ": amplitudes"
            plt.title(my_title)
            pdf.savefig()

            plt.close("all")
            plt.plot(out["omega"],"bo-",ms=3)
            my_title = runid.split(".")[0] + ": omega"
            plt.title(my_title)
            pdf.savefig()
        
            plt.close("all")
            plt.plot(out["phase"],"bo-",ms=3)
            my_title = runid.split(".")[0] + ": phase"
            plt.title(my_title)
            pdf.savefig()

            print("   ... about to start plot of alpha")
            
            alpha_shape = out["alpha"].shape

            myc = ["k","r","orange","y","g","c","b","m", "purple","gold"]
            nc  = len(myc)
            #pdb.set_trace()

            
            plt.close("all")
            plt.figure(figsize=(7.5,5.5))

            # ++++++++++++++++++++++++++++++
            #  plot ALL alphas
            
            my_labels=[]
            nn1 = alpha_shape[0]
            for jk in range(nn1):
                mc = np.mod(jk,nc)
                this_label = " index %2d"%(jk)
                this_line, = plt.plot(out["alpha"][jk,:], "-", color=myc[mc], label=this_label)
                my_labels.append(this_line)
                #print("\n\n  have finished jk = ", jk)
                #pdb.set_trace()
            plt.legend(handles=my_labels, loc="upper right", fontsize=9)
            my_title = runid.split(".")[0] + ": alpha (ALL)"
            plt.title(my_title)
            plt.xlabel("rhoindex")
            pdf.savefig()

            # ++++++++++++++++++++++++++++++
            #  plot FIRST 10 alphas
            my_labels=[]
            nn1 = alpha_shape[0]

            nn10 = np.min((10,nn1))
            for jk in range(nn10):
                mc = np.mod(jk,nc)
                this_label = " index %2d"%(jk)
                this_line, = plt.plot(out["alpha"][jk,:], "-", color=myc[mc], label=this_label)
                my_labels.append(this_line)
                #print("\n\n  have finished jk = ", jk)
                #pdb.set_trace()
            plt.legend(handles=my_labels, loc="upper right", fontsize=9)
            my_title = runid.split(".")[0] + ": alpha (first 10)"
            plt.title(my_title)
            plt.xlabel("rhoindex")
            pdf.savefig()

            # ++++++++++++++++++++++++++++++
            #  plot ALL phis
            
            phi_shape = out["phi"].shape
            my_labels=[]
            nn1 = phi_shape[0]
            for jk in range(nn1):
                mc = np.mod(jk,nc)
                this_label = " index %2d"%(jk)
                this_line, = plt.plot(out["phi"][jk,:], "-", color=myc[mc], label=this_label)
                my_labels.append(this_line)
                #print("\n\n  have finished jk = ", jk)
                #pdb.set_trace()
            plt.legend(handles=my_labels, loc="upper right", fontsize=9)
            my_title = runid.split(".")[0] + ": phi (ALL)"
            plt.title(my_title)
            plt.xlabel("rhoindex")
            pdf.savefig()

            
            my_labels=[]
            nn10 = np.min((10,nn1))
            for jk in range(nn10):
                mc = np.mod(jk,nc)
                this_label = " index %2d"%(jk)
                this_line, = plt.plot(out["phi"][jk,:], "-", color=myc[mc], label=this_label)
                my_labels.append(this_line)
                #print("\n\n  have finished jk = ", jk)
                #pdb.set_trace()
            plt.legend(handles=my_labels, loc="upper right", fontsize=9)
            my_title = runid.split(".")[0] + ": phi FIRST 10"
            plt.title(my_title)
            plt.xlabel("rhoindex")
            pdf.savefig()
    print("\n   ... your plotfile is: ", fn_out)
    return out

# ++++++++++++++++++++++++++++++++++++


def get_boozer(runid,detail=0):
    """
    inputs
       runid      = name of h5 file, e.g. ascot_12345678.h5'
    returns:  a dictionary with various parameters
    """

    file_name = construct_full_filename(runid)

    print("   ... get_boozer:  will try to read from file: ", file_name)
    try:
        print("   ... get_boozer: first try to read from ASCOT output directory: ", file_name)
        ff = h5py.File(file_name, 'r')
    except:
        print("   ... get_boozer: that file not found.  Try to read from local file: ", runid)
        ff = h5py.File(runid, 'r')

    options = ff['boozer']

    run_id_list = extract_runids(options)
    run_id      = run_id_list[0]
    options     = options[run_id]
    key_list    = options.keys()
    
    out = {}
    
    
    for key in key_list:
        array    = np.array(options[key])
        out[key] = array

    fn_out = runid.split(".")[0] + "_boozer_details.pdf"
 
    if(detail !=0):
        
        print("\n   ... get_boozer.get_mhd: sizes of returned arrays: \n")
        for key in key_list:
           this_shape = np.array(out[key]).shape
           print("  variable: ", key, "          shape: ", this_shape)

        npsi      = np.array(options["npsi"])[0]
        nr        = np.array(options["nr"])[0]
        nz        = np.array(options["nz"])[0]
        nrzs      = np.array(options["nrzs"])[0]
        ntheta    = np.array(options["ntheta"])[0]
        nthetag   = np.array(options["nthetag"])[0]
        psi0      = np.array(options["psi0"])[0]
        psi1      = np.array(options["psi1"])[0]
        psimin    = np.array(options["psimin"])[0]        
        psimax    = np.array(options["psimax"])[0]
        r0        = np.array(options["r0"])[0]
        rmin      = np.array(options["rmin"])[0]
        rmax      = np.array(options["rmax"])[0]
        z0        = np.array(options["z0"])[0]
        zmin      = np.array(options["zmin"])[0]
        zmax      = np.array(options["zmax"])[0]

        rs                 = np.array(options["rs"])
        zs                 = np.array(options["zs"])
        nu_psitheta        = np.array(options["nu_psitheta"])
        theta_psithetageom = np.array(options["theta_psithetageom"])
        psi_rz             = np.array(options["psi_rz"])
        
        print("   ... npsi     %5d"%(npsi))
        print("   ... nr       %5d"%(nr))
        print("   ... nz       %5d"%(nz))
        print("   ... nrzs     %5d"%(nrzs))
        print("   ... ntheta   %5d"%(ntheta))
        print("   ... nthetag  %5d"%(nthetag))
        print("   ... psi0     %13.5e"%(psi0))
        print("   ... psi0     %13.5e"%(psi1))
        print("   ... psimin   %13.5e"%(psimin))        
        print("   ... psimax   %13.5e"%(psimax))
        print("   ... r0       %13.5e"%(r0))
        print("   ... rmin     %13.5e"%(rmin))
        print("   ... rmax     %13.5e"%(rmax))
        print("   ... z0       %13.5e"%(z0))
        print("   ... zmin     %13.5e"%(zmin))
        print("   ... zmax     %13.5e"%(zmax))

        #pdb.set_trace()

              
        # ['npsi', 'nr', 'nrzs', 'ntheta', 'nthetag', 'nu_psitheta', 'nz', 'psi0', 'psi1', 'psi_rz', 'psimax', 'psimin', 'r0', 'rmax', 'rmin', 'rs', 'theta_psithetageom', 'z0', 'zmax', 'zmin', 'zs']>


     
        print("   ... about to start plots")
        with PdfPages(fn_out) as pdf:

            plt.close("all")
            plt.plot(rs,"bo-",ms=3)
            my_title = runid.split(".")[0] + ": rs"
            plt.title(my_title)
            pdf.savefig()

            plt.close("all")
            plt.plot(zs,"bo-",ms=3)
            my_title = runid.split(".")[0] + ": zs"
            plt.title(my_title)
            pdf.savefig()

            # ++++++++++++++++++++++++++++++
            #  plot ALL nu_psitheta
            #   note:  I don't know what the rows and columns represent
  
            myc = ["k","r","orange","y","g","c","b","m", "purple","gold"]
            nc  = len(myc)
            
            my_shape = nu_psitheta.shape
            ncol     = my_shape[1]
            
            my_labels=[]
            plt.close("all")
            plt.figure(figsize=(7.5,5.5))
            #pdb.set_trace()
            for jk in range(ncol):
                mc = np.mod(jk,nc)
                this_label = " column %2d"%(jk)
                this_line, = plt.plot(nu_psitheta[:,jk], "-", color=myc[mc], label=this_label)
                my_labels.append(this_line)

            plt.legend(handles=my_labels, loc="upper right", fontsize=9)
            my_title = runid.split(".")[0] + ": nu_psitheta (ALL)"
            plt.title(my_title)
            plt.xlabel("")
            pdf.savefig()

            # ++++++++++++++++++++++++++++++
            #  plot ALL theta_psithetageom
            #   note:  I don't know what the rows and columns represent
  
            myc = ["k","r","orange","y","g","c","b","m", "purple","gold"]
            nc  = len(myc)
            
            my_shape = theta_psithetageom.shape
            ncol     = my_shape[1]
            
            my_labels=[]
            plt.close("all")
            plt.figure(figsize=(7.5,5.5))
            #pdb.set_trace()
            for jk in range(ncol):
                mc = np.mod(jk,nc)
                this_label = " column %2d"%(jk)
                this_line, = plt.plot(theta_psithetageom[:,jk], "-", color=myc[mc], label=this_label)
                my_labels.append(this_line)

            plt.legend(handles=my_labels, loc="lower right", fontsize=9)
            my_title = runid.split(".")[0] + ": theta_psithetageom (ALL)"
            plt.title(my_title)
            plt.xlabel("")
            pdf.savefig()

       
    print("\n   ... your plotfile is: ", fn_out)
    return out

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_orbit(runid,mass, nmax=0):
    """
    inputs
       runid   filename, e.g. ascot_12345678.h5
       mass    amu of markers, e.g. 4
    returns:  a dictionary containing the orbit data
 
    history
        1/15/2022:  fixed sorting by time.  must be done individually by markers
        1/16/2022:  sorting is much too slow. remove sorting by time.  so time will not be monotonic upon return
    """

    file_name = construct_full_filename(runid)
    try:
        print("   ... get_orbit: first try to read from ASCOT output directory: ", file_name)
        ff = h5py.File(file_name, 'r')
    except:
        print("   ... get_orbit: that file not found. try to read local file: ", runid)
        ff = h5py.File(runid, 'r') 

    rr = ff['results']

    results_keys = rr.keys()
    for key in results_keys:
        run_id = key
    results = rr[run_id]

    # +++++++++++++++++++++++++++++++
    #  read data for each key
    
    out = {}
    
    rr_temp  = results['orbit']
    key_list = rr_temp.keys()
    #pdb.set_trace(header="inside orbits")

    id_array = np.array(rr_temp["ids"])

    if(nmax == 0):
        for key in key_list:
            array    = np.array(rr_temp[key])
            out[key] = array
    else:
        id_array = np.array(rr_temp["ids"])
        ii_good  = (id_array <= nmax)
        for key in key_list:
            array    = np.array(rr_temp[key])[ii_good]
            out[key] = array
            
             
    
    # 1/15/2022:  time is not monotonic so sort all arrays to make them monotonic
    #time_fixed = out['time']
    #ids        = out['ids']
    #nmax_id = np.max(out["ids"])

    #for key in key_list:
    #    print("   ... processing key = ", key)
    #    for iparticle in range(1,nmax_id):
    #
    #        array_big     = out[key]                 # extract data for all markers for this key
    #        ii            = (ids == iparticle)       # get indices of this marker
    #        array_snippet = array_big[ii]            # get data for this key for this marker
    #       
    #       time_local = time_fixed[ii]              # get time for this marker
    #        jj         = np.argsort(time_local)      # get indices to put this array in increasing order
    #
    #        array_snippet_new = array_snippet[jj]    # get ordered data for this key for this marker
    #        array_big[ii]     = array_snippet_new    # insert ordered data for this snippet back into array
    #
    #    out[key] = array_big                         # overwrite data for this key (all markers) into dictionary
    
    true_mass = alpha_mass * (mass/4.)

    #  according to ascot documentation, the units of mu are eV/Tesla
    
    out["btot"]  = np.sqrt( out["br"]**2 + out["bz"]**2 + out["bphi"]**2)
    
    try:
        out["vpar"]  = out["ppar"] / true_mass                    ## assume all markers are alphas
    except:
        out["ppar"] = (out["pphi"]*out["bphi"] + out["pr"]*out["br"] + out["pz"]*out["bz"])/out["btot"]
        out["vpar"]  = out["ppar"] / true_mass
         
    #out["vperp"] = np.sqrt(2.* electron_charge * out["mu"] * out["btot"] / true_mass)
    #out["ekev"]  = 0.5 * true_mass * (out["vperp"]**2 + out["vpar"]**2)  / (1000. * electron_charge)
    try:
        energy      = physicslib.Ekin(                                    \
                          masskg=true_mass,                               \
                          mueVperT=out["mu"], ppar=out["ppar"],           \
                          BR=out["br"], Bphi=out["bphi"], Bz=out["bz"])
    except:
        energy      = physicslib.Ekin(masskg=true_mass,pphi=out["pphi"],pR=out["pr"],pz=out["pz"])

        energy_perp_eV = (energy - (true_mass*out["vpar"]**2) / 2.)/electron_charge
        out["mu"]      = energy_perp_eV / out["btot"]                              # eV per Tesla

    vtot = np.sqrt(2. * energy/true_mass)
        
    # pdb.set_trace(header="before")
    out["ekev"] = physicslib.J2eV(energy)/1000.
    #pdb.set_trace(header="after")
    #vtot    = (vphi**2 + vr**2 + vz**2)**(0.5)
    #ekev    = 0.001  * true_mass * vtot**2/ (2.*1.602e-19)
    
    # pdb.set_trace()
    time = out["time"]
    
    delta_time = time[1:] - time[0:-1]
    last_one   = np.array([delta_time[-1]])

    #  add fake point at end so all arrays have same length
    #  note that individual markers will have wrong value
    #  at beginning and end because data for all markers
    #  is concatenated
    #
    delta_time = np.concatenate((delta_time, last_one))

    # -------------------------------------------

    #out["vpar"]       = vpar
    out["delta_time"] = delta_time

    #  2/22/2022:  we need to save memory for the big 18 TF orbit simulations
    
    out["charge"]     = np.array((0.,0.))
    out["theta"]      = np.array((0.,0.))
    out["weight"]     = np.array((0.,0.))
    out["delta_time"] = np.array((0.,0.))
    out["btot"]       = np.array((0.,0.))
    #out["vpar"]       = np.array((0.,0.))

    out["vtot"] = vtot

    #pdb.set_trace(header="inside get_orbit")
    return out

# ------------------------------------------------------------------



def get_wall(file_name):

    file_name_original = file_name
    file_name = construct_full_filename(file_name)

    try:
        print("   ... get_wall:Try to read file from ASCOT output directory: ", file_name)
        ff = h5py.File(file_name, 'r')
    except:
        print("   ... get_wall:that file not found.  Try to read from local file: ", file_name_original)
        ff = h5py.File(file_name_original, 'r')
              
    ww        = ff['wall']
    wall_keys = ww.keys()
    
    for key in wall_keys:
        wwid = key   # assume there is just one
        
    wall = ww[wwid]

    out = {}
    key_list = wall.keys()
    for key in key_list:
        array = np.array(wall[key])
        out[key] = array


    return out

# -----------------------------------------------------------

def myread_hdf5(file_name,do_corr=0):

    file_name_original = file_name
              
    file_name = construct_full_filename(file_name)

    try:
        print("   ... myread_hdf5: ry to read from ASCOT output directory: ", file_name)
        ff = h5py.File(file_name, 'r')
    except:
        print("   ... myread_hdf5: that file not found.  try to read local file: ", file_name_original)
        ff = h5py.File(file_name_original, 'r')
              
    ww        = ff['wall']
    wall_keys = ww.keys()
    
    for key in wall_keys:
        wwid = key   # assume there is just one
        
    wall = ww[wwid]

    rr_wall = wall['r']
    zz_wall = wall['z']

    r_wall = np.array(rr_wall)
    z_wall = np.array(zz_wall)

    mm = ff['marker']
    marker_keys = mm.keys()
    for key in marker_keys:
        mmid = key            # assume there is just one
    markers = mm[mmid]
    #import pdb; pdb.set_trace()

    marker_rr   = np.transpose(np.array(markers['r']))
    marker_zz   = np.transpose(np.array(markers['z']))
    marker_phi  = np.transpose(np.array(markers['phi']))
    marker_id   = np.transpose(np.array(markers['id']))
    marker_vphi = np.transpose(np.array(markers['vphi']))
    marker_vz   = np.transpose(np.array(markers['vz']))
    marker_vr   = np.transpose(np.array(markers['vr']))

    marker_rr   = marker_rr[0,:]
    marker_zz   = marker_zz[0,:]
    marker_phi  = marker_phi[0,:]
    marker_id   = marker_id[0,:]
    marker_vphi = marker_vphi[0,:]
    marker_vz   = marker_vz[0,:]
    marker_vr   = marker_vr[0,:]

    # import pdb; pdb.set_trace()


    rr = ff['results']

    results_keys = rr.keys()
    for key in results_keys:
        run_id = key
    results = rr[run_id]

    results = ff['results']
    run_id_list = extract_runids(results)

    print('')
    print(' list of run_ids: ', run_id_list)

    run_id = run_id_list[0]

    print(' I will read from : ', run_id)

    h5_anum    = results[run_id]['endstate']['anum']
    h5_znum    = results[run_id]['endstate']['znum']
    h5_charge  = results[run_id]['endstate']['charge']

    h5_mass    = results[run_id]['endstate']['mass']
    h5_vpar    = results[run_id]['endstate']['vpar']
    h5_vphi    = results[run_id]['endstate']['vphi']
    h5_vr      = results[run_id]['endstate']['vr']
    h5_vz      = results[run_id]['endstate']['vz'] 
    h5_weight  = results[run_id]['endstate']['weight']
    h5_time    = results[run_id]['endstate']['time']
    h5_cputime = results[run_id]['endstate']['cputime']
    h5_endcond = results[run_id]['endstate']['endcond']
    h5_phi     = results[run_id]['endstate']['phi']
    h5_r       = results[run_id]['endstate']['r']
    h5_z       = results[run_id]['endstate']['z']
    h5_id      = results[run_id]['endstate']['id']
    h5_theta   = results[run_id]['endstate']['theta']

    h5_r_ini    = results[run_id]['inistate']['r']
    h5_z_ini    = results[run_id]['inistate']['z']
    h5_phi_ini  = results[run_id]['inistate']['phi']
    h5_id_ini   = results[run_id]['inistate']['id']
    h5_vphi_ini = results[run_id]['inistate']['vphi']
    h5_vr_ini   = results[run_id]['inistate']['vr']
    h5_vz_ini   = results[run_id]['inistate']['vz']
    h5_vpar_ini = results[run_id]['inistate']['vpar']

    anum      = np.array(h5_anum)
    znum      = np.array(h5_znum)
    charge    = np.array(h5_charge)
    mass      = np.array(h5_mass)
    vpar      = np.array(h5_vpar)
    vphi      = np.array(h5_vphi)
    vr        = np.array(h5_vr)
    vz        = np.array(h5_vz)
    weight    = np.array(h5_weight)
    time      = np.array(h5_time)
    cputime   = np.array(h5_cputime)
    endcond   = np.array(h5_endcond)
    phi_end   = np.array(h5_phi)
    r_end     = np.array(h5_r)
    z_end     = np.array(h5_z)
    theta_end = np.array(h5_theta)
    id_end    = np.array(h5_id)

    r_ini    = np.array(h5_r_ini)
    z_ini    = np.array(h5_z_ini)
    phi_ini  = np.array(h5_phi_ini)
    id_ini   = np.array(h5_id_ini)
    vphi_ini = np.array(h5_vphi_ini)
    vr_ini   = np.array(h5_vr_ini)
    vz_ini   = np.array(h5_vz_ini)
    vpar_ini = np.array(h5_vpar_ini)

    vtot        = (vphi**2 + vr**2 + vz**2)**(0.5)
    vtot_ini    = (vphi_ini**2 + vr_ini**2 + vz_ini**2)**(0.5)
    marker_vtot = (marker_vphi**2 + marker_vr**2 + marker_vz**2)**(0.5)

    pitch_ini        = vpar_ini/(vtot_ini+1.e-9)
    pitch_phi_ini    = vphi_ini/(vtot_ini+1.e-9)
    marker_pitch_phi = marker_vphi/marker_vtot

    #  correction due to improper weighting of pitch angles

    weight_parent = weight/np.sum(weight)

    if(do_corr == 1):
        weight = weight * np.sqrt(1 - pitch_ini**2)
        weight = weight / np.sum(weight)
    elif(do_corr == 2):
        weight = weight * np.sqrt(1 - pitch_ini**2) * r_ini
        weight = weight / np.sum(weight)
    amu_mass = 1.6605e-27
    q_e         = 1.602e-19
    #pdb.set_trace()   
    ekev        = 0.001  * amu_mass * mass * vtot**2/ (2.*1.602e-19)
    ekev_ini    = 0.001  * amu_mass * mass * vtot_ini**2/ (2.*1.602e-19)
    marker_ekev = 0.001  * amu_mass * mass * marker_vtot**2/ (2.*1.602e-19)

    out = {}

    out["weight_parent"] = weight_parent

    out["anum"]    = anum
    out["znum"]    = znum
    out["charge"]  = charge
    out["ekev"]    = ekev
    out["vtot"]    = vtot
    out["mass"]    = mass
    out["vpar"]    = vpar
    out["vphi"]    = vphi
    out["vr"]      = vr
    out["vz"]      = vz
    out["weight"]  = weight
    out["time"]    = time
    out["cputime"] = cputime
    out["endcond"] = endcond
    out["phi_end"] = phi_end
    out["r_end"]   = r_end
    out["z_end"]   = z_end
    out["id_end"]  = id_end
    out["r_ini"]   = r_ini
    out["z_ini"]   = z_ini
    out["phi_ini"] = phi_ini
    out["id_ini"]  = id_ini
    out["vphi_ini"] = vphi_ini
    out["vr_ini"]   = vr_ini
    out["vz_ini"]   = vz_ini
    out["vpar_ini"] = vpar_ini
    out["pitch_ini"] = pitch_ini
    out["pitch_phi_ini"] = pitch_phi_ini
    out["marker_pitch_phi"] = marker_pitch_phi

    out["marker_r"] = marker_rr
    out["marker_z"] = marker_zz
    out["marker_phi"] = marker_phi
    out["marker_id"] = marker_id
    out["marker_vphi"] = marker_vphi
    out["marker_vr"] = marker_vr
    out["marker_vz"] = marker_vz

    out["r_wall"]    = r_wall
    out["z_wall"]    = z_wall
    out["theta_end"] = theta_end

    out["vtot_ini"]    = vtot_ini
    out["marker_vtot"] = marker_vtot
    out["ekev_ini"]    = ekev_ini
    out["marker_ekev"] = marker_ekev

    return out

if __name__ == '__main__':

    mylen = len(sys.argv)
    if (mylen !=5):
        
        print(" You provided insufficient arguments.")
        print(" arguments: filename.h5  v1d.geq and string(frac_alpha_sim) and weight_pitch")
        exit()
    print("Number of arguments: ", mylen)
    print("arguments: ", sys.argv)
    file_name                 = sys.argv[1]
    geq_name                  = sys.argv[2]    #'v1c.geq'
    fraction_alphas_simulated = float(sys.argv[3])

    corr = sys.argv[4]
    if(corr == "weight_pitch"):
        do_corr = 1
    elif(corr == "weight_both"):
        do_corr = 2   
    elif (corr == "noweight_pitch"):
        do_corr = 0
    else:
        print("  4th argument must be weight_pitch or noweight_pitch")
        exit()

    eq_index  = 0
    print(" I am about to invoke print_summary")
    print_summary(file_name, geq_name, eq_index, fraction_alphas_simulated, do_corr)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_bfield(runid):
    """
    inputs
       runid      = name of h5 file, e.g. ascot_12345678.h5'
    returns:  a dictionary with B-field data
    """
              
    file_name = construct_full_filename(runid)

    try:
        print("   ... get_bfield: try to read from ASCOT output directory: ", file_name)
        ff = h5py.File(file_name, 'r')
    except:
        print("   ... get_bfield: that file not found.  Read from local directory: ", runid)
        ff = h5py.File(runid, 'r') 
              
    rr = ff['bfield']

    results_keys = rr.keys()
    for key in results_keys:
        run_id = key
    results = rr[run_id]
    
    out = {}

    key_list = results.keys()
    
    for key in key_list:
        array = np.array(results[key])
        out[key] = array

    
    return out

# +++++++++++++++++++++++++++++++++++++++++

def get_markers(runid, nmax=0):
    """
    inputs
       runid      = name of h5 file, e.g. ascot_12345678.h5'
    returns:  a dictionary with marker data
    """

    file_name = construct_full_filename(runid)
    try:
        print("   ... get_markers: trying to read from ASCOT output directory: ", file_name)
        ff = h5py.File(file_name, 'r')
    except:
        print("   ... get_markers: could not open file.  try local file: ", runid)
        ff = h5py.File(runid, 'r')
        
    rr = ff['marker']

    results_keys = rr.keys()
    for key in results_keys:
        run_id = key
    results = rr[run_id]
    
    out = {}

    key_list = results.keys()
    
    for key in key_list:
        array = np.transpose(np.array(results[key]))[0,:]
        if(nmax>0):
            array = array[0:nmax]
        out[key] = array

    vphi = out['vphi']
    vr   = out['vr']
    vz   = out['vz']
    mass = out['mass']
    
    vtot     = (vphi**2 + vr**2 + vz**2)**(0.5)
    amu_mass = 1.6605e-27
    q_e      = 1.602e-19
    ekev     = 0.001  * amu_mass * mass * vtot**2/ (2.*1.602e-19)

    out['vtot'] = vtot
    out['ekev'] = ekev

    print(" Number of markers:   %7d   "%vphi.size)
    print(" Total marker weight: %7.4f "%np.sum(out['weight']), "\n")
        
    return out

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_distrho5d(runid, fn_profile, fn_geq, mass=4, doplot=False):
    """
    inputs
       runid      = name of h5 file, e.g. ascot_12345678.h5'
       fn_profile = filename with profile information
       mass       = mass of marker (amu)
    returns:  a dictionary with 5d distribution function data
    """

    #  get plasma volume integrated out to a given zone_boundary radius
    #  based on actual equilibrium (not zone volumes from transp)

    rhos_zb     = np.linspace(0.,1.,51)
    rhos_zb[0]  = 0.01    # get an error if we try to compute volume exactly at rho=0
    volumes_zb  = CF.compute_zb_volumes(fn_geq, rhos_zb)

    if(doplot):
        stub     = fn_geq.split(".")[0]
        fn_local = 'volume_' + stub + ".pdf"
        my_title = "Integrated volume for " + fn_geq
        plt.title(my_title)
        plt.xlabel('rho_poloidal_zb')
        plt.ylabel('[m^3]')

        my_ylims = (0., 1.1*np.max(volumes_zb))
        plt.xlim((0.,1.))
        plt.ylim(my_ylims)
        plt.plot(rhos_zb, volumes_zb)
        my_graph_text = "get_distrho5d/"+fn_local
        SU.graph_label(my_graph_text)
        plt.savefig(fn_local)
        plt.close()
        print("   ...get_distrho5d: plot file with integrated volumes = ", fn_local)
        #pdb.set_trace()
        #sys.exit()

    # ++++++++++++++++++++++++++++++
    # get profiles and ASCOT data
    
    aa_profile = proc.read_sparc_profiles_new(fn_profile, nrho=101,skip_plots=True)
    # print('aa_profile_keys', aa_profile.keys())
  
    file_name = construct_full_filename(runid)
    try:
        print("   ... get_distrho5d: trying to read from ASCOT output directory: ", file_name)
        ff = h5py.File(file_name, 'r')
    except:
        print("   ... get_distrho5d: could not open file.  try local file: ", runid)
        ff = h5py.File(runid, 'r')
        
    rr = ff['results']

    results_keys = rr.keys()
    for key in results_keys:
        run_id = key
    results = rr[run_id]

    rr = results['distrho5d']

    yy      =  np.array(rr["ordinate"])    # has the usual extra pesky index at the front
    zz      =  yy[0,:,:,:,:,:,:,:]
    
    nrho    =  np.array(rr["abscissa_nbin_01"])[0]
    ntheta  =  np.array(rr["abscissa_nbin_02"])[0]
    nphi    =  np.array(rr["abscissa_nbin_03"])[0]
    nppar   =  np.array(rr["abscissa_nbin_04"])[0]
    npperp  =  np.array(rr["abscissa_nbin_05"])[0]
    ntime   =  np.array(rr["abscissa_nbin_06"])[0]
    ncharge =  np.array(rr["abscissa_nbin_07"])[0]

    rhos    =  np.array(rr["abscissa_vec_01"])
    thetas  =  np.array(rr["abscissa_vec_02"])
    phis    =  np.array(rr["abscissa_vec_03"])
    ppars   =  np.array(rr["abscissa_vec_04"])
    pperps  =  np.array(rr["abscissa_vec_05"])
    times   =  np.array(rr["abscissa_vec_06"])
    charges =  np.array(rr["abscissa_vec_07"])

    #  map integrated volume onto local rho_poloidal array
    
    volint_mapped = np.interp(rhos, rhos_zb, volumes_zb)
    
    #  all returned absciaae are "zone boundaries"
    #  so create "zc arrays" which will have size (n-1) along
    #  with the corresponding volume centered at that zc
    
    zone_volumes  = volint_mapped[1:] - volint_mapped[0:-1]
    
    rhos          =  0.5 * (  rhos[0:-1]  +   rhos[1:])
    thetas        =  0.5 * (thetas[0:-1]  + thetas[1:])
    phis          =  0.5 * (  phis[0:-1]  +   phis[1:])
    ppars         =  0.5 * ( ppars[0:-1]  +  ppars[1:])
    pperps        =  0.5 * (pperps[0:-1]  + pperps[1:])

    #pdb.set_trace(header="after some mapping")
    
    mproton = 1.67e-27

    vpars   = ppars  / (mass * mproton)
    vperps  = pperps / (mass * mproton)

    epars   = 0.5 * mass * mproton * vpars**2
    eperps  = 0.5 * mass * mproton * vperps**2

    nee = int(nppar/2)
    etots   = np.linspace(0., np.max(epars), nee)  # create array of total energy

    etots_kev     = etots * 0.001 / 1.602e-19       # etots_kev[nppar]
    etots_kev_max = np.max(etots_kev)
    
    #   zz = zz(rho,  theta,  phi, pphi,   ppar,  time,  charge)

    nparticle          = np.zeros(nrho)
    etotal             = np.zeros(nrho)
    particle_density   = np.zeros(nrho)
    energy_density     = np.zeros(nrho)
    nparticle_2        = np.zeros(nrho)
    particle_density_2 = np.zeros(nrho)
    
    for jrho in range(nrho):                             # particle number and density"
        nparticle[jrho]        = np.sum(aa_profile['alpha_source']) * np.sum(zz[jrho,:,:,:,:,:,:])
        particle_density[jrho] = nparticle[jrho] / zone_volumes[jrho]

    for jrho in range(nrho):                             # energy density
        for jpar in range(nppar):
            for kperp in range(npperp):
                
                etotal[jrho]       += np.sum(aa_profile['alpha_source']) * (epars[jpar] + eperps[kperp]) * np.sum(zz[jrho,:,:,jpar,kperp,:,:])
                nparticle_2[jrho]  += np.sum(aa_profile['alpha_source'])                                 * np.sum(zz[jrho,:,:,jpar,kperp,:,:])
                
        energy_density[jrho]     =      etotal[jrho] / zone_volumes[jrho]
        particle_density_2[jrho] = nparticle_2[jrho] / zone_volumes[jrho]

    W_integrated_MJ = np.cumsum(etotal)/1.e6

    #pdb.set_trace(header="at end in get_ascot")
    #  should check this arithmetic.
    #  this computes f(E) over entire plasma
    
    fE    =  np.zeros(nee)
    fE_14 =  np.zeros(nee)    # 0    < rho < 0.25
    fE_24 =  np.zeros(nee)    # 0.25 < rho < 0.50
    fE_34 =  np.zeros(nee)    # 0.50 < rho < 0.75
    fE_44 =  np.zeros(nee)    # 0.75 < rho < 1.00

    ii_14 = (rhos <0.25)
    ii_24 = (rhos>=0.25) & (rhos<0.50)
    ii_34 = (rhos>=0.50) & (rhos<0.75)
    ii_44 = (rhos>=0.75)
    
    for jpar in range(nppar):
        for kperp in range(npperp):
             cell_etot_kev = (epars[jpar] + eperps[kperp]) * 0.001 /1.602e-19
             ii = int ( nee * cell_etot_kev / etots_kev_max)
             if(ii <= (nee-1)):
                 
                fE[ii]     += np.sum(zz[:,    :,:,jpar,kperp,:,:])
                fE_14[ii]  += np.sum(zz[ii_14,:,:,jpar,kperp,:,:])
                fE_24[ii]  += np.sum(zz[ii_24,:,:,jpar,kperp,:,:])
                fE_34[ii]  += np.sum(zz[ii_34,:,:,jpar,kperp,:,:])
                fE_44[ii]  += np.sum(zz[ii_44,:,:,jpar,kperp,:,:])

                
    
    out = {}

    out["W_integrated_MJ"] = W_integrated_MJ
    out["dist5d"]    = zz
    out["nparticle"] = nparticle
    out["etotal"]    = etotal
    
    out["nrho"]      = nrho
    out["ntheta"]    = ntheta
    out["nppar"]     = nppar
    out["npperp"]    = npperp
    out["ntime"]     = ntime
    out["ncharge"]   = ncharge

    out["rho"]       = rhos
    out["theta"]     = thetas
    out["ppar"]      = ppars
    out["pperp"]     = pperps
    out["vpar"]      = vpars
    out["vperp"]     = vperps
    out["epar"]      = epars
    out["eperp"]     = eperps
    out["epar_kev"]  = epars  * 0.001  / 1.602e-19
    out["eperp_kev"] = eperps * 0.001  / 1.602e-19
    out["time"]      = times
    out["charge"]    = charges
    
    out["fE"]       = fE
    out["fE_14"]    = fE_14
    out["fE_24"]    = fE_24
    out["fE_34"]    = fE_34
    out["fE_44"]    = fE_44

    out["particle_density"]   = particle_density
    out["particle_density_2"] = particle_density_2
    out["energy_density"]     = energy_density
    out["zone_volumes"]       = zone_volumes
    out["etots_kev"]          = etots_kev

    return out



# +++++++++++++++++++++++++++++++++++++++++

def get_markers_GC(runid, nmax=0,detail=0):
    """
    inputs
       runid      = name of h5 file, e.g. ascot_12345678.h5'
    returns:  a dictionary with marker data
 
    typical contents: 'anum', 'charge', 'energy', 'id', 'mass', 'n', 'phi', 'pitch', 'r', 'time', 'weight', 'z', 'zeta', 'znum'
    """

    file_name = construct_full_filename(runid)
    try:
        print("   ... get_markers: trying to read from ASCOT output directory: ", file_name)
        ff = h5py.File(file_name, 'r')
    except:
        print("   ... get_markers: could not open file.  try local file: ", runid)
        ff = h5py.File(runid, 'r')
        
    rr = ff['marker']

    results_keys = rr.keys()
    for key in results_keys:
        run_id = key
    results = rr[run_id]
    
    out = {}

    key_list = results.keys()
    
    for key in key_list:
        array = np.transpose(np.array(results[key]))[0,:]
        if(nmax>0):
            array = array[0:nmax]
        out[key] = array



    ekev = out["energy"]/1000.
    out["ekev"] = ekev


    print(" Number of markers:   %7d   "%out["energy"].size)
    print(" Total marker weight: %7.4f "%np.sum(out['weight']), "\n")


    if(detail !=0):

        fn_out = runid.split(".")[0] + "_markers.pdf"
        with PdfPages(fn_out) as pdf:

            title_stub = runid.split(".")[0] + ": "

            plt.close("all")
            plt.figure(figsize=(7.5,5.5))
            plt.plot(out["r"],out["z"],"bo", ms=2)
            my_title = title_stub + "marker [R,Z]"
            plt.xlabel("Rmajor [m]")
            plt.ylabel("Z [m]")
            plt.title(my_title)
            pdf.savefig()

            plt.close("all")
            plt.figure(figsize=(7.5,5.5))
            plt.hist(out["ekev"],bins=50, histtype="step",color="b")
            mytitle = title_stub + "histogram of Ekev"
            plt.title(mytitle)
            pdf.savefig()

            plt.close("all")
            plt.figure(figsize=(7.5,5.5))
            plt.hist(out["pitch"],bins=50, histtype="step",color="b")
            mytitle = title_stub + "histogram of pitch"
            plt.title(mytitle)
            pdf.savefig()

            plt.close("all")
            plt.figure(figsize=(7.5,5.5))
            plt.hist(out["weight"],bins=50, histtype="step",color="b")
            mytitle = title_stub + "histogram of weight"
            plt.title(mytitle)
            pdf.savefig()

            plt.close("all")
            plt.figure(figsize=(7.5,5.5))
            plt.hist(out["phi"],bins=50, histtype="step",color="b")
            mytitle = title_stub + "histogram of phi"
            plt.title(mytitle)
            pdf.savefig()


            plt.close("all")
            plt.figure(figsize=(7.5,5.5))
            plt.plot(out["weight"],"bo", ms=2)
            mytitle = title_stub + "weight"
            plt.title(mytitle)
            pdf.savefig()

            plt.close("all")
            plt.figure(figsize=(7.5,5.5))
            plt.plot(out["phi"],"bo", ms=2)
            mytitle = title_stub + "phi"
            plt.title(mytitle)
            pdf.savefig()
            
            plt.close("all")
            plt.figure(figsize=(7.5,5.5))
            plt.plot(out["anum"],"bo", ms=2)
            mytitle = title_stub + "anum"
            plt.title(mytitle)
            pdf.savefig()

            plt.close("all")
            plt.figure(figsize=(7.5,5.5))
            plt.plot(out["znum"],"bo", ms=2)
            mytitle = title_stub + "znum"
            plt.title(mytitle)
            pdf.savefig()

            plt.close("all")
            plt.figure(figsize=(7.5,5.5))
            plt.plot(out["charge"],"bo", ms=2)
            mytitle = title_stub + "charge"
            plt.title(mytitle)
            pdf.savefig()
        
    return out

def driver():

    filename = input("   enter name of ASCOT output file, e.g. ascot_12345678.h5:   ")
    mass     = float(input("   enter mass of markers, e.g. 4:   "))

    aa_markers = get_markers(filename)
    aa_ini     = get_ini_end(filename,"inistate")
    aa_end     = get_ini_end(filename,"endstate")
    aa_orbit   = get_orbit(filename, mass)
    aa_bfield  = get_bfield(filename)
    aa_wall    = get_wall(filename)

    print("")
    print(" the dictionaries are: \n")
    print("      aa_bfield")
    print("      aa_wall")
    print("      aa_markers")
    print("      aa_ini")
    print("      aa_end")
    print("      aa_orbit \n")

    print(" the keys to these dictionaries are: \n")
                     
    print("   aa_bfield:  ",aa_bfield.keys(), "\n")
    print("   aa_wall:    ",aa_wall.keys(), "\n")
    print("   aa_markers: ",aa_markers.keys(), "\n")
    print("   aa_ini:     ",aa_ini.keys(), "\n")
    print("   aa_end:     ",aa_end.keys(), "\n")
    print("   aa_orbit:   ",aa_orbit.keys(), "\n")

    print("")
    #pdb.set_trace()


