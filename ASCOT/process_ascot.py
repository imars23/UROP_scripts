import h5py
import numpy as np
import sys
import pdb
import get_rho_interpolator as get_rho
import matplotlib.pyplot as plt
from   matplotlib import cm

from readGeqdsk import geqdsk
import matplotlib as mpl
from shapely.geometry import Point, Polygon
from matplotlib.patches import Rectangle
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import sparc_processing as proc
from scipy import optimize
import read_any_file as myread
import write_a_file as mywrite
import triangulate_torus as tri
import get_process_parameters_simple as get
import time as mytime
import read_parameter_file as my
import list_to_string as my_list
import time as clock

def construct_full_filename(filename_in):

    stub          = filename_in.split('_')[0]
    remainder     = filename_in.split('_')[1]
    remainder     = remainder.split('.')[0]
    
    this_filename = '/project/projectdirs/m3195/ascot/ascot_run_output/' + stub + '_work_' + remainder + '/' + stub + '_' + remainder + '.h5'

    return this_filename

# +++++++++++++++++++++++++++++++++++++
def compute_phi_offset(phi_lost, rend_lost, zend_lost, width_limiter, height_limiter, rinner_wall):
    
    width_half_limiter = width_limiter/2.
    
    mm = phi_lost.size

    phi_offset = np.copy(phi_lost)

    for jk in range(mm):

        phi_offset[jk] = np.mod((phi_offset[jk]+width_half_limiter),20)

    ii_limiter =     (phi_offset <= width_limiter)      \
                  & (zend_lost > -1.*height_limiter)    \
                  & (zend_lost <  height_limiter)       \
                  & (rend_lost > rinner_wall)
        
    return phi_offset, ii_limiter


def old_or_new(fn_hdf5):

    #  returns 0 if file is old-style, returns 1 if new-style
    full_filename = construct_full_filename(fn_hdf5)
    ff = h5py.File(full_filename,'r')
    results = ff['results']
    run_id_list = extract_runids(results)
    run_id = run_id_list[0]

    new_or_old = 0
    
    try:
        check = results[run_id]['endstate']['ppar']
        new_or_old = 1
    except:
        xx_dummy = 0.

    print("   ... new_or_old = ", new_or_old)

    return new_or_old

# ---------------------------------------------------------------

def rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2):

    cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_list_1, linewidths=0.6, colors=color_1,zorder=10)
    plt.clabel(cs, fontsize=12)
    
    cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_list_2, linewidths=0.6, colors=color_2,zorder=10)
    # plt.clabel(cs, fontsize=12)

    return

# --------------------------------------

def my_powerlaw(x,a,b):
    return a*(x**b)

def extract_runids(gg):
    
    return [key for key in gg.keys()]

def extract_subgroups(endcond, ekev, weight, index):

    endcond = np.array(endcond)
    ekev    = np.array(ekev)
    weight  = np.array(weight)

    ii = (endcond == index)

    dummy = endcond[ii]

    nseg = 4       # divide ensemble into sub-ensembles
    nn = dummy.size

    if (nn > 0):

        out_ekev    = ekev[ii]
        out_weight  = weight[ii]
        out_product = np.multiply(out_ekev, out_weight)
        out_total   = np.sum(out_product)
        out_particle_total = np.sum(out_weight)
        
        if(nn > 2*nseg):

            qq = ((endcond.size)//nseg) - 1

            endcond1     = endcond[0:qq]
            endcond2     = endcond[qq+1:2*qq]
            endcond3     = endcond[2*qq+1:3*qq]
            endcond4     = endcond[3*qq+1:]

            weight1     = weight[0:qq]
            weight2     = weight[qq+1:2*qq]
            weight3     = weight[2*qq+1:3*qq]
            weight4     = weight[3*qq+1:]

            ekev1     = ekev[0:qq]
            ekev2     = ekev[qq+1:2*qq]
            ekev3     = ekev[2*qq+1:3*qq]
            ekev4     = ekev[3*qq+1:]

            ii1       = (endcond1 == index)
            ii2       = (endcond2 == index)
            ii3       = (endcond3 == index)
            ii4       = (endcond4 == index)

            out_ekev1    = ekev1[ii1]
            out_ekev2    = ekev2[ii2]
            out_ekev3    = ekev3[ii3]
            out_ekev4    = ekev4[ii4]

            out_weight1    = weight1[ii1]
            out_weight2    = weight2[ii2]
            out_weight3    = weight3[ii3]
            out_weight4    = weight4[ii4]
                        
            out_product1 = np.multiply(out_ekev1, out_weight1)
            out_product2 = np.multiply(out_ekev2, out_weight2)
            out_product3 = np.multiply(out_ekev3, out_weight3)
            out_product4 = np.multiply(out_ekev4, out_weight4)
            
            out_total1   = np.sum(out_product1)
            out_total2   = np.sum(out_product2)
            out_total3   = np.sum(out_product3)
            out_total4   = np.sum(out_product4)
            
            out_particle_total1 = np.sum(out_weight1)
            out_particle_total2 = np.sum(out_weight2)
            out_particle_total3 = np.sum(out_weight3)
            out_particle_total4 = np.sum(out_weight4)

            nn1 = out_ekev1.size
            nn2 = out_ekev2.size
            nn3 = out_ekev3.size
            nn4 = out_ekev4.size
            
        else:

            nn1 = 0
            nn2 = 0
            nn3 = 0
            nn4 = 0
        
            out_product1 = 0.
            out_product2 = 0.
            out_product3 = 0.
            out_product4 = 0.
            
            out_total1   = 0.
            out_total2   = 0.
            out_total3   = 0.
            out_total4   = 0.
            
            out_particle_total1 = 0.
            out_particle_total2 = 0.
            out_particle_total3 = 0.
            out_particle_total4 = 0.
            
    else:
            
        out_ekev           = 0.
        out_weight         = 0.
        out_product        = 0.
        out_particle_total = 0.
        out_total          = 0.

        nn1 = 0
        nn2 = 0
        nn3 = 0
        nn4 = 0
        
        out_product1 = 0.
        out_product2 = 0.
        out_product3 = 0.
        out_product4 = 0.
            
        out_total1   = 0.
        out_total2   = 0.
        out_total3   = 0.
        out_total4   = 0.
            
        out_particle_total1 = 0.
        out_particle_total2 = 0.
        out_particle_total3 = 0.
        out_particle_total4 = 0.
        
    out={}

    out["nn"]             = nn
    out["ekev"]           = out_ekev
    out["weight"]         = out_weight
    out["product"]        = out_product
    out["total"]          = out_total
    out["particle_total"] = out_particle_total
    
    out["product1"]        = out_product1
    out["total1"]          = out_total1
    out["particle_total1"] = out_particle_total1

    out["product2"]        = out_product2
    out["total2"]          = out_total2
    out["particle_total2"] = out_particle_total2

    out["product3"]        = out_product3
    out["total3"]          = out_total3
    out["particle_total3"] = out_particle_total3

    out["product4"]        = out_product4
    out["total4"]          = out_total4
    out["particle_total4"] = out_particle_total4

    out["nn1"] = nn1
    out["nn2"] = nn2
    out["nn3"] = nn3
    out["nn4"] = nn4
    
    return out

def surface_power_density_2(ploss_wall_kw, rloss, zloss, phiend_lost, weight_energy_lost, ishape, pdf, time_lost, fn_parameter="none"):

    # the following allows us to look into whether delayed losses have a
    # different spatial distribution than earlier losses
    
    original_total_weight_energy_lost = np.sum(weight_energy_lost)
    
    time_minimum = 0.   # need to set this manually

    ii = ( time_lost>= time_minimum)

    rloss              = rloss[ii]
    zloss              = zloss[ii]
    phiend_lost        = phiend_lost[ii]
    weight_energy_lost = weight_energy_lost[ii]

    new_total_energy_lost = np.sum(weight_energy_lost)

    ploss_wall_kw = ploss_wall_kw * new_total_energy_lost / original_total_weight_energy_lost
    
    #  colormaps:  viridis, cividis, plamsa, YlOrRd,
    #              spring, summer, hot, Blues, gnuplot2

    print("   ... starting surface_power_density_2 \n")
          
    my_colormap = cm.get_cmap('plasma')   # Blues
    
    philoss = phiend_lost * np.pi / 180.     # convert to radians
    
    #  get wall shape
    
    aa_inner = myread.read_any_file('/project/projectdirs/m3195/ascot/mypython/wall_inner_new.txt')  # sparc_pfc_inner_simple_culled3.txt
    aa_outer = myread.read_any_file('/project/projectdirs/m3195/ascot/mypython/wall_outer_new.txt')  # sparc_pfc_outer_simple.txt

    rr_inner = aa_inner[:,0]
    zz_inner = aa_inner[:,1]

    rr_outer = aa_outer[:,0]
    zz_outer = aa_outer[:,1]

    rmax        = np.max(rr_outer)   # 2.4411
    rmin_outer  = np.min(rr_outer)   # minimum major radius of outer wall

    zwall_max       = np.max(zz_outer) - 0.0001 
    zwall_inner_max = np.max(zz_inner) - 0.0001
    
    # specify sizes of sub-elements
    if(fn_parameter == "none"):
        
        ntf             = 18
        tall_rf_antenna = 1.00   # changed from 0.50   
        tall_limiter    = 1.00   # changed from 0.50
        hmax_limiter    = 0.01   # 0.08  0.01
        hmin_limiter    = 0.008  # 0.035  0.008
        taper_limiter   = 0.10
        y50_limiter     = 0.65    # height is 65% of (hmax-hmin) at 50% of toroidal extent        
        tall_tbl        = 0.10
        taper_tbl       = 0.02
        hmin_tbl        = 0.008   # 0.035 0.008
        hmax_tbl        = 0.01    # 0.08  0.01
        y50_tbl         = 0.65
 
        aminor          = 0.57
        kappa           = 1.75
        height_limiter  = 2. * tall_limiter
        height_rf       = 2. * tall_rf_antenna   
        width_rf_sector_limiter = 0.305
        width_rf_antenna        = (2. * np.pi * rmax/ntf) - width_rf_sector_limiter
        phi_sector_size         = 2. * np.pi              / ntf
        phi_rf_size             = width_rf_antenna        / rmax  
        phi_limiter_size        = width_rf_sector_limiter / rmax
        
        #  shape 6 = shape 5 but 10 cm resolution vertically on the poloidal limiters
        #  shape 7 = shape 6 but 7.5 cm resolution in the toroidal direction (most like spiral)
      
        # ishape          1    2    3   4   5  6    7   8
      
        nnphi_limiter  = [5,   5,  20, 10,  5, 5,   4,  5] 
        nnz_limiter    = [6,   6,  20, 10,  6, 10, 10, 15]
        nnphi_rf       = [5,   3,  20, 10,  3, 3,   3,  3]
        nnz_rf         = [6,   4,  20, 10,  6, 6,   6,  6]    #  was [6,4,2,10] until 10/3
        nnphi_tbl      = [72, 36, 180, 72, 72, 72, 72, 72]
        nnz_tbl        = [8,   8,  20,  8,  8, 8,   8,  8]
        nnphi_wall_1   = [72, 36,  90, 90, 72, 72, 72, 72]   # was (incorrectly) 72,6,90,90 until 10/3
        nnz_wall_1     = [6,  6,  12,  12,  6, 6,   6,  6]
        nnphi_wall_2   = [72, 36, 90,  72, 72, 72, 72, 72]
        nnz_wall_2     = [20, 20, 40,  20, 12, 12, 12, 12]
        nnr_sides      = [2,  2,   4,   4,  2, 2,  2,   2]
        nnz_sides      = [6,  6,  12,   6,  6, 6,  6,   6]
        nnphi_tbl_sides= [5,  5,  10,   5,  5, 5,  5,   5]
        nnz_tbl_sides  = [2,  2,   4,   2 , 2, 5,  5,   5]
        
        nphi_limiter   = nnphi_limiter[ishape-1]
        nz_limiter     = nnz_limiter[ishape-1]
        nphi_rf        = nnphi_rf[ishape-1]
        nz_rf          = nnz_rf[ishape-1] 
        nphi_tbl       = nnphi_tbl[ishape-1] 
        nz_tbl         = nnz_tbl[ishape-1]
        nphi_wall_1    = nnphi_wall_1[ishape-1]  
        nz_wall_1      = nnz_wall_1[ishape-1]
        nphi_wall_2    = nnphi_wall_2[ishape-1]
        nz_wall_2      = nnz_wall_2[ishape-1]
        nr_sides       = nnr_sides[ishape-1]
        nz_sides       = nnz_sides[ishape-1]
        nphi_tbl_sides = nnphi_tbl_sides[ishape-1]
        nz_tbl_sides   = nnz_tbl_sides[ishape-1]

    else:
        
        ff = open(fn_parameter,"r")
        
        suppress_rf             = my.read_next_line('suppress_rf',          ff) 
        suppress_tf_limiters    = my.read_next_line("suppress_tf_limiters", ff)  
        suppress_tbl            = my.read_next_line("suppress_tbl",         ff)       
        suppress_outer_walls    = my.read_next_line("suppress_outer_walls", ff)
        suppress_inner_wall     = my.read_next_line("suppress_inner_wall",  ff)   
        suppress_tbl_sides      = my.read_next_line("suppress_tbl_sides",   ff)  
        suppress_lim_sides      = my.read_next_line("suppress_lim_sides",   ff) 
        ishape                  = my.read_next_line("ishape",               ff)                   
        nphi_limiter            = my.read_next_line("nphi_limiter",         ff)     
        nz_limiter              = my.read_next_line("nz_limiter",           ff)        
        nphi_rf                 = my.read_next_line("nphi_rf", ff)           
        nz_rf                   = my.read_next_line("nz_rf", ff)              
        nphi_tbl                = my.read_next_line("nphi_tbl", ff)           
        nz_tbl                  = my.read_next_line("nz_tbl", ff)              
        nphi_wall_1             = my.read_next_line("nphi_wall_1", ff)        
        nz_wall_1               = my.read_next_line("nz_wall_1", ff)          
        nphi_wall_2             = my.read_next_line("nphi_wall_2", ff)        
        nz_wall_2               = my.read_next_line("nz_wall_2", ff)          
        nr_sides                = my.read_next_line("nr_sides", ff)            
        nz_sides                = my.read_next_line("nz_sides", ff)            
        nphi_tbl_sides          = my.read_next_line("nphi_tbl_sides", ff)      
        nz_tbl_sides            = my.read_next_line("nz_tbl_sides", ff)      
        tall_rf_antenna         = my.read_next_line("tall_rf_antenna", ff)     
        tall_limiter            = my.read_next_line("tall_limiter", ff)        
        hmin_limiter            = my.read_next_line("hmin_limiter", ff)         
        hmax_limiter            = my.read_next_line("hmax_limiter", ff)        
        taper_limiter           = my.read_next_line("taper_limiter", ff)       
        y50_limiter             = my.read_next_line("y50_limiter", ff)          
        tall_tbl                = my.read_next_line("tall_tbl", ff)            
        taper_tbl               = my.read_next_line("taper_tbl", ff)            
        hmin_tbl                = my.read_next_line("hmin_tbl", ff)            
        hmax_tbl                = my.read_next_line("hmax_tbl", ff)             
        y50_tbl                 = my.read_next_line("y50_tbl", ff)              
        width_rf_sector_limiter = my.read_next_line("width_rf_sector_limiter", ff)  
        width_rf_antenna        = my.read_next_line("width_rf_antenna", ff)          
        phi_sector_size         = my.read_next_line("phi_sector_size", ff)          
        phi_rf_size             = my.read_next_line("phi_rf_size", ff)            
        phi_limiter_size        = my.read_next_line("phi_limiter_size", ff)          
        rcrit                   = my.read_next_line("rcrit", ff)                     
        ntf                     = my.read_next_line("ntf", ff)                          
        aminor                  = my.read_next_line("aminor", ff)              
        kappa                   = my.read_next_line("kappa", ff)             
        height_limiter          = my.read_next_line("height_limiter", ff)   
        height_rf               = my.read_next_line("height_rf", ff)       
        proud_limiter_z         = my.read_next_line("proud_limiter_z", ff)      
        proud_rf_z              = my.read_next_line("proud_rf_z", ff)       
        proud_lim_phi           = my.read_next_line("proud_lim_phi", ff)      
        proud_rf_phi            = my.read_next_line("proud_rf_phi", ff)      
        proud_tbl_phi           = my.read_next_line("proud_tbl_phi", ff)

        ff.close()

        tall_tbl = 0.
        
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    tall_wall       = zwall_max - tall_limiter - tall_tbl 
    
    phistart_limiter = np.zeros((ntf, nphi_limiter, nz_limiter))
    phiend_limiter   = np.zeros((ntf, nphi_limiter, nz_limiter))
    zstart_limiter   = np.zeros((ntf, nphi_limiter, nz_limiter))
    zend_limiter     = np.zeros((ntf, nphi_limiter, nz_limiter))
    area_limiter     = np.zeros((ntf, nphi_limiter, nz_limiter))
    power_limiter    = np.zeros((ntf, nphi_limiter, nz_limiter))   # power to area elements, kW
    spd_limiter      = np.zeros((ntf, nphi_limiter, nz_limiter))   # surface power density to area elements, kW/m^2
    hits_limiter     = np.zeros((ntf, nphi_limiter, nz_limiter))   # hits within rectancle
    hpd_limiter      = np.zeros((ntf, nphi_limiter, nz_limiter))   #  hit density 
    
    phistart_rf      = np.zeros((ntf, nphi_rf, nz_rf))
    phiend_rf        = np.zeros((ntf, nphi_rf, nz_rf))
    zstart_rf        = np.zeros((ntf, nphi_rf, nz_rf))
    zend_rf          = np.zeros((ntf, nphi_rf, nz_rf))
    area_rf          = np.zeros((ntf, nphi_rf, nz_rf))
    power_rf         = np.zeros((ntf, nphi_rf, nz_rf))   # power to area elements, kW
    spd_rf           = np.zeros((ntf, nphi_rf, nz_rf))   # surface power density to area elements, kW/m^2
    hits_rf          = np.zeros((ntf, nphi_rf, nz_rf))   # hits within rectancle
    hpd_rf           = np.zeros((ntf, nphi_rf, nz_rf))   #  hit density     
    
    #phistart_tbl_upper = np.zeros((nphi_tbl, nz_tbl))
    #phiend_tbl_upper   = np.zeros((nphi_tbl, nz_tbl))
    #zstart_tbl_upper   = np.zeros((nphi_tbl, nz_tbl))
    #zend_tbl_upper     = np.zeros((nphi_tbl, nz_tbl))
    #area_tbl_upper     = np.zeros((nphi_tbl, nz_tbl))
    #power_tbl_upper    = np.zeros((nphi_tbl, nz_tbl))   # power to area elements, kW
    #spd_tbl_upper      = np.zeros((nphi_tbl, nz_tbl))   # surface power density to area elements, kW/m^2 
    #hits_tbl_upper     = np.zeros((nphi_tbl, nz_tbl))   # hits within rectancle
    #hpd_tbl_upper      = np.zeros((nphi_tbl, nz_tbl))   #  hit density 

    
    #phistart_tbl_lower = np.zeros((nphi_tbl, nz_tbl))
    #phiend_tbl_lower   = np.zeros((nphi_tbl, nz_tbl))
    #zstart_tbl_lower   = np.zeros((nphi_tbl, nz_tbl))
    #zend_tbl_lower     = np.zeros((nphi_tbl, nz_tbl))
    #area_tbl_lower     = np.zeros((nphi_tbl, nz_tbl))
    #power_tbl_lower    = np.zeros((nphi_tbl, nz_tbl))   # power to area elements, kW
    #spd_tbl_lower      = np.zeros((nphi_tbl, nz_tbl))   # surface power density to area elements, kW/m^2
    #hits_tbl_lower     = np.zeros((nphi_tbl, nz_tbl))   # hits within rectancle
    #hpd_tbl_lower      = np.zeros((nphi_tbl, nz_tbl))   #  hit density 
    
    phistart_wall_upper = np.zeros((nphi_wall_1, nz_wall_1))
    phiend_wall_upper   = np.zeros((nphi_wall_1, nz_wall_1))
    zstart_wall_upper   = np.zeros((nphi_wall_1, nz_wall_1))
    zend_wall_upper     = np.zeros((nphi_wall_1, nz_wall_1))
    area_wall_upper     = np.zeros((nphi_wall_1, nz_wall_1))
    power_wall_upper    = np.zeros((nphi_wall_1, nz_wall_1))   # power to area elements, kW
    spd_wall_upper      = np.zeros((nphi_wall_1, nz_wall_1))   # surface power density to area elements, kW/m^2 
    hits_wall_upper     = np.zeros((nphi_wall_1, nz_wall_1))   # hits within rectancle
    hpd_wall_upper      = np.zeros((nphi_wall_1, nz_wall_1))   #  hit density 

    
    phistart_wall_lower = np.zeros((nphi_wall_1, nz_wall_1))
    phiend_wall_lower   = np.zeros((nphi_wall_1, nz_wall_1))
    zstart_wall_lower   = np.zeros((nphi_wall_1, nz_wall_1))
    zend_wall_lower     = np.zeros((nphi_wall_1, nz_wall_1))
    area_wall_lower     = np.zeros((nphi_wall_1, nz_wall_1))
    power_wall_lower    = np.zeros((nphi_wall_1, nz_wall_1))   # power to area elements, kW
    spd_wall_lower      = np.zeros((nphi_wall_1, nz_wall_1))   # surface power density to area elements, kW/m^2 
    hits_wall_lower     = np.zeros((nphi_wall_1, nz_wall_1))   # hits within rectancle
    hpd_wall_lower      = np.zeros((nphi_wall_1, nz_wall_1))   #  hit density 

    # inner wall
    
    phistart_iwall = np.zeros((nphi_wall_2, nz_wall_2))
    phiend_iwall   = np.zeros((nphi_wall_2, nz_wall_2))
    zstart_iwall   = np.zeros((nphi_wall_2, nz_wall_2))
    zend_iwall     = np.zeros((nphi_wall_2, nz_wall_2))
    area_iwall     = np.zeros((nphi_wall_2, nz_wall_2))
    power_iwall    = np.zeros((nphi_wall_2, nz_wall_2))   # power to area elements, kW
    spd_iwall      = np.zeros((nphi_wall_2, nz_wall_2))   # surface power density to area elements, kW/m^2 
    hits_iwall     = np.zeros((nphi_wall_2, nz_wall_2))   # hits within rectancle
    hpd_iwall      = np.zeros((nphi_wall_2, nz_wall_2))   #  hit density
    
    phistart_limiter_edge = np.zeros(ntf)
    phiend_limiter_edge   = np.zeros(ntf)
    zstart_limiter_edge   = np.zeros(ntf)
    zend_limiter_edge     = np.zeros(ntf)
    
    phistart_rf_edge = np.zeros(ntf)
    phiend_rf_edge   = np.zeros(ntf)
    zstart_rf_edge   = np.zeros(ntf)
    zend_rf_edge     = np.zeros(ntf)

    delta_phi_limiter     = phi_limiter_size /nphi_limiter
    delta_phi_tbl         = 2. * np.pi       / nphi_tbl
    delta_phi_wall        = 2. * np.pi       / nphi_wall_1
    delta_phi_rf          = phi_rf_size      / nphi_rf

    delta_z_limiter       = 2. * tall_limiter    / nz_limiter
    delta_z_tbl           = tall_tbl             / nz_tbl
    delta_z_wall          = tall_wall            / nz_wall_1
    delta_z_rf            = 2. * tall_rf_antenna / nz_rf

    #
    total_weight_energy_lost = np.sum(weight_energy_lost)
    weight_eloss_norm        = weight_energy_lost/total_weight_energy_lost
    
    # toroidal belt limiters (removed) 

    #  outer wall
    print("   ... about to start outer walls")

    for jphi in range(nphi_wall_1):
        for jz in range(nz_wall_1):
            
            phistart_wall_lower[jphi,jz] = 2. * np.pi *  jphi    / nphi_wall_1
            phiend_wall_lower[jphi,jz]   = 2. * np.pi * (jphi+1) / nphi_wall_1

            phistart_wall_upper[jphi,jz] = 2. * np.pi *  jphi    / nphi_wall_1
            phiend_wall_upper[jphi,jz]   = 2. * np.pi * (jphi+1) / nphi_wall_1

            zstart_wall_upper[jphi,jz] =  tall_limiter + tall_tbl + tall_wall *  jz    / nz_wall_1
            zend_wall_upper[jphi,jz]   =  tall_limiter + tall_tbl + tall_wall * (jz+1) / nz_wall_1

            zstart_wall_lower[jphi,jz] =  -1. * zwall_max + tall_wall *  jz    / nz_wall_1
            zend_wall_lower[jphi,jz]   =  -1. * zwall_max + tall_wall * (jz+1) / nz_wall_1

            zmid_upper = (zstart_wall_upper[jphi,jz] + zend_wall_upper[jphi,jz]) / 2.
            zmid_lower = (zstart_wall_lower[jphi,jz] + zend_wall_lower[jphi,jz]) / 2.

            rmid_upper = tri.interpolate_rmajor_wall(rr_outer, zz_outer, zmid_upper)
            rmid_lower = tri.interpolate_rmajor_wall(rr_outer, zz_outer, zmid_lower)
            
            delta_z_lower   = zend_wall_lower[jphi,jz]   -   zstart_wall_lower[jphi,jz]
            delta_phi_lower = phiend_wall_lower[jphi,jz] - phistart_wall_lower[jphi,jz]

            delta_z_upper   = zend_wall_upper[jphi,jz]   -   zstart_wall_upper[jphi,jz]
            delta_phi_upper = phiend_wall_upper[jphi,jz] - phistart_wall_upper[jphi,jz]

            area_wall_upper[jphi,jz] = rmid_upper * delta_phi_upper * delta_z_upper
            area_wall_lower[jphi,jz] = rmid_lower * delta_phi_lower * delta_z_lower
            
            ii_upper = (rloss > rmin_outer) & (zloss > zstart_wall_upper[jphi,jz]) & (zloss < zend_wall_upper[jphi,jz]) \
                         & (philoss > phistart_wall_upper[jphi,jz]) & (philoss < phiend_wall_upper[jphi,jz])

            ii_lower = (rloss > rmin_outer) & (zloss > zstart_wall_lower[jphi,jz]) & (zloss < zend_wall_lower[jphi,jz]) \
                         & (philoss > phistart_wall_lower[jphi,jz]) & (philoss < phiend_wall_lower[jphi,jz])

            power_wall_upper[jphi,jz] = np.sum(weight_eloss_norm[ii_upper]) * ploss_wall_kw
            power_wall_lower[jphi,jz] = np.sum(weight_eloss_norm[ii_lower]) * ploss_wall_kw

            spd_wall_upper[jphi,jz] = power_wall_upper[jphi,jz] / area_wall_upper[jphi,jz]
            spd_wall_lower[jphi,jz] = power_wall_lower[jphi,jz] / area_wall_lower[jphi,jz]

            hits_wall_upper[jphi,jz] = np.sum(ii_upper)
            hits_wall_lower[jphi,jz] = np.sum(ii_lower)

            hpd_wall_upper[jphi,jz] = hits_wall_upper[jphi,jz] / area_wall_upper[jphi,jz]
            hpd_wall_lower[jphi,jz] = hits_wall_lower[jphi,jz] / area_wall_lower[jphi,jz]

    #  inner wall
    
    print("   ... about to start inner wall")
    for jphi in range(nphi_wall_2):
        for jz in range(nz_wall_2):
            
            phistart_iwall[jphi,jz] = 2. * np.pi *  jphi    / nphi_wall_2
            phiend_iwall[jphi,jz]   = 2. * np.pi * (jphi+1) / nphi_wall_2

            zstart_iwall[jphi,jz] =  -1. * zwall_inner_max + 2. * zwall_inner_max *  jz    / nz_wall_2
            zend_iwall[jphi,jz]   =  -1. * zwall_inner_max + 2. * zwall_inner_max * (jz+1) / nz_wall_2

            zmid = (zstart_iwall[jphi,jz] + zend_iwall[jphi,jz]) / 2.

            rmid = tri.interpolate_rmajor_wall(rr_inner, zz_inner, zmid)
            
            delta_z   = zend_iwall[jphi,jz]   -   zstart_iwall[jphi,jz]
            delta_phi = phiend_iwall[jphi,jz] - phistart_iwall[jphi,jz]

            area_iwall[jphi,jz] = rmid * delta_phi * delta_z
            
            ii_inner = (rloss < rmin_outer) & (zloss > zstart_iwall[jphi,jz]) & (zloss < zend_iwall[jphi,jz]) \
                         & (philoss > phistart_iwall[jphi,jz]) & (philoss < phiend_iwall[jphi,jz])

            power_iwall[jphi,jz] = np.sum(weight_eloss_norm[ii_inner]) * ploss_wall_kw

            spd_iwall[jphi,jz] = power_iwall[jphi,jz] / area_iwall[jphi,jz]

            hits_iwall[jphi,jz] = np.sum(ii_inner)
        
            hpd_iwall[jphi,jz] = hits_iwall[jphi,jz] / area_iwall[jphi,jz]
            
    #  limiters
    print("  ... about to start limiters")
    for jtf in range(ntf):
        
        phistart_limiter_edge[jtf]      =  -phi_limiter_size/2 + jtf * phi_sector_size
        phiend_limiter_edge[jtf]        =  phistart_limiter_edge[jtf] + phi_limiter_size
        
        zstart_limiter_edge[jtf]        =  -tall_limiter
        zend_limiter_edge[jtf]          =  tall_limiter

        for jphi in range(nphi_limiter):
            
            for jz in range(nz_limiter):
                
                phistart_limiter[jtf, jphi,jz] = phistart_limiter_edge[jtf]  + (phiend_limiter_edge[jtf]- phistart_limiter_edge[jtf]) * jphi/ nphi_limiter
                phiend_limiter[jtf,jphi,jz]    = phistart_limiter[jtf, jphi,jz] + (phiend_limiter_edge[jtf]- phistart_limiter_edge[jtf])       / nphi_limiter

                zstart_limiter[jtf,jphi,jz] = zstart_limiter_edge[jtf] + (zend_limiter_edge[jtf]- zstart_limiter_edge[jtf]) * jz / nz_limiter
                zend_limiter[jtf,jphi,jz]   = zstart_limiter[jtf, jphi,jz] + (zend_limiter_edge[jtf]- zstart_limiter_edge[jtf])       / nz_limiter

                zmid  = (zstart_limiter[jtf, jphi,jz] + zend_limiter[jtf, jphi,jz]) / 2.
                rmid  = tri.interpolate_rmajor_wall(rr_outer, zz_outer, zmid)

                delta_z   = zend_limiter[jtf, jphi,jz]  -   zstart_limiter[jtf,jphi,jz]
                delta_phi = phiend_limiter[jtf,jphi,jz] - phistart_limiter[jtf,jphi,jz]

                area_limiter[jtf,jphi,jz] = rmid * delta_phi * delta_z
            
                ii = (rloss > rmin_outer) & (zloss > zstart_limiter[jtf,jphi,jz]) & (zloss < zend_limiter[jtf,jphi,jz]) \
                         & (philoss > phistart_limiter[jtf,jphi,jz] ) & (philoss < phiend_limiter[jtf,jphi,jz])

                power_limiter[jtf,jphi,jz] = np.sum(weight_eloss_norm[ii]) * ploss_wall_kw
                spd_limiter[jtf,jphi,jz] = power_limiter[jtf,jphi,jz] / area_limiter[jtf,jphi,jz]

                hits_limiter[jtf,jphi,jz] = np.sum(ii)
                hpd_limiter[jtf,jphi,jz] = hits_limiter[jtf,jphi,jz] / area_limiter[jtf,jphi,jz]
 
                
    # rf antennas
    print("   ... about to start rf antennas")
    for jtf in range(ntf):
        
        phistart_rf_edge[jtf]      =  phi_limiter_size/2 + jtf * phi_sector_size
        phiend_rf_edge[jtf]        =  phistart_rf_edge[jtf] + phi_rf_size
        
        zstart_rf_edge[jtf]        =  -tall_rf_antenna
        zend_rf_edge[jtf]          =  tall_rf_antenna

        for jphi in range(nphi_rf):
            
            for jz in range(nz_rf):
                
                phistart_rf[jtf, jphi,jz] = phistart_rf_edge[jtf]  + (phiend_rf_edge[jtf]- phistart_rf_edge[jtf]) * jphi/ nphi_rf
                phiend_rf[jtf,jphi,jz]    = phistart_rf[jtf, jphi,jz] + (phiend_rf_edge[jtf]- phistart_rf_edge[jtf])       / nphi_rf

                zstart_rf[jtf,jphi,jz] = zstart_rf_edge[jtf] + (zend_rf_edge[jtf]- zstart_rf_edge[jtf]) * jz / nz_rf
                zend_rf[jtf,jphi,jz]   = zstart_rf[jtf, jphi,jz] + (zend_rf_edge[jtf]- zstart_rf_edge[jtf])  / nz_rf

                zmid = (zstart_rf[jtf, jphi,jz] + zend_rf[jtf, jphi,jz]) / 2.
                rmid = tri.interpolate_rmajor_wall(rr_outer, zz_outer, zmid)

                delta_z   = zend_rf[jtf, jphi,jz]  -   zstart_rf[jtf,jphi,jz]
                delta_phi = phiend_rf[jtf,jphi,jz] - phistart_rf[jtf,jphi,jz]

                area_rf[jtf, jphi,jz] = rmid * delta_phi * delta_z
                #pdb.set_trace()
                ii = (rloss > rmin_outer) & (zloss > zstart_rf[jtf,jphi,jz]) & (zloss < zend_rf[jtf,jphi,jz]) \
                         & (philoss > phistart_rf[jtf,jphi,jz] ) & (philoss < phiend_rf[jtf,jphi,jz])

                power_rf[jtf,jphi,jz] = np.sum(weight_eloss_norm[ii]) * ploss_wall_kw
                spd_rf[jtf,jphi,jz] = power_rf[jtf,jphi,jz] / area_rf[jtf,jphi,jz]

                hits_rf[jtf,jphi,jz] = np.sum(ii)
                hpd_rf[jtf,jphi,jz] = hits_rf[jtf,jphi,jz] / area_rf[jtf,jphi,jz]

     
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #  check that our geometry of components is correct
    #pdb.set_trace()
    
    plt.close()
    
    plt.figure(figsize=(9.,6.))
    fig,ax=plt.subplots()
    plt.xlim(-0.1, 2.*np.pi+0.1)
    plt.ylim(-1.25, 1.25)



    # rf antennas
    
    for jtf in range(ntf):
        for jphi in range(nphi_rf):
            for jz in range(nz_rf):
                
                delta_phi = phiend_rf[jtf,jphi,jz] - phistart_rf[jtf,jphi,jz]
                delta_z   = zend_rf[jtf,jphi,jz]   - zstart_rf[jtf,jphi,jz]
                rect = Rectangle( (phistart_rf[jtf,jphi,jz],zstart_rf[jtf,jphi,jz]),delta_phi, delta_z, facecolor='none', edgecolor='g', linewidth=0.25)
                ax.add_patch(rect)

    # poloidal limiters
    
    for jtf in range(ntf):
        for jphi in range(nphi_limiter):
            for jz in range(nz_limiter):

                delta_phi = phiend_limiter[jtf,jphi,jz] - phistart_limiter[jtf,jphi,jz]
                delta_z   = zend_limiter[jtf,jphi,jz]   - zstart_limiter[jtf,jphi,jz]
                
                rect = Rectangle( (phistart_limiter[jtf,jphi,jz],zstart_limiter[jtf,jphi,jz]),delta_phi, delta_z, facecolor='none', edgecolor='r', linewidth=0.25)
                ax.add_patch(rect)
                
    #  tbl limiters (removed)
    
    # outer walls

    for jphi in range(nphi_wall_1):
        for jz in range(nz_wall_1):

            delta_phi_lower = phiend_wall_lower[jphi,jz] - phistart_wall_lower[jphi,jz]
            delta_z_lower   = zend_wall_lower[jphi,jz]   - zstart_wall_lower[jphi,jz]

            delta_phi_upper = phiend_wall_upper[jphi,jz] - phistart_wall_upper[jphi,jz]
            delta_z_upper   = zend_wall_upper[jphi,jz]   - zstart_wall_upper[jphi,jz]
            
            rect = Rectangle( (phistart_wall_lower[jphi,jz],zstart_wall_lower[jphi,jz]),delta_phi_lower, delta_z_lower, facecolor='none', edgecolor='k', linewidth=0.25)
            ax.add_patch(rect)

            rect = Rectangle( (phistart_wall_upper[jphi,jz],zstart_wall_upper[jphi,jz]),delta_phi_upper, delta_z_upper, facecolor='none', edgecolor='k', linewidth=0.25)
            ax.add_patch(rect) 

    plt.title(' limiters and walls (starting surface_power_density_2)', fontsize=10)
    plt.xlabel('phi [radians]')
    plt.ylabel('Z [m]')
    #plt.tight_layout(pad=padsize)
    pdf.savefig()
    #plt.savefig('validate_surface.pdf')
    plt.close()


    # ++++++++++++++++++++++++++++++++++++++++++++++++
    #

    mm_limiter = ntf * nphi_limiter * nz_limiter
    mm_rf      = ntf * nphi_rf      * nz_rf
    #mm_tbl     = nphi_tbl * nz_tbl
    mm_wall_1  = nphi_wall_1 * nz_wall_1
    mm_wall_2  = nphi_wall_2 * nz_wall_2

    spd_limiter_1d    = np.reshape(spd_limiter, mm_limiter)
    spd_rf_1d         = np.reshape(spd_rf,      mm_rf)
    #spd_tbl_upper_1d  = np.reshape(spd_tbl_upper, mm_tbl)
    #spd_tbl_lower_1d  = np.reshape(spd_tbl_lower, mm_tbl)
    spd_wall_upper_1d = np.reshape(spd_wall_upper, mm_wall_1)
    spd_wall_lower_1d = np.reshape(spd_wall_lower, mm_wall_1)
    spd_iwall_1d      = np.reshape(spd_iwall, mm_wall_2)

    spd_all_1d = np.concatenate((spd_limiter_1d, spd_rf_1d, 
                                 spd_wall_upper_1d, spd_wall_lower_1d, spd_iwall_1d))

    # -----------------------------------------------------------------
    #  histograms for surface power density on limiters
    
    plt.close()
    plt.figure(figsize=(8.,6.))
    plt.hist(spd_limiter_1d, bins=40, histtype='step', rwidth=1.,color='k',log=True)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Surface power density on poloidal limiters')
    plt.xlabel('kW/m2')
    plt.ylabel('')
    plt.ylim(bottom=0.5)
    plt.tight_layout(pad=2)
    pdf.savefig()
    plt.close()

    plt.close()
    plt.figure(figsize=(8.,6.))
    plt.hist(spd_limiter_1d, bins=40, histtype='step', rwidth=1.,color='r',log=True)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Surface power density on poloidal limiters')
    plt.xlabel('kW/m2')
    plt.ylabel('')
    plt.ylim((0.5,5000.))
    plt.xlim((0.,1500.))
    plt.tight_layout(pad=2)
    pdf.savefig()
    plt.close()

    # -----------------------------------------------------------------
    #  histograms for surface power density on rf antennas
    
    plt.close()
    plt.figure(figsize=(8.,6.))
    plt.hist(spd_rf_1d, bins=40, histtype='step', rwidth=1.,color='k',log=True)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Surface power density on RF antennas')
    plt.xlabel('kW/m2')
    plt.ylabel('')
    plt.ylim(bottom=0.5)
    plt.tight_layout(pad=2)
    pdf.savefig()
    plt.close()

    plt.close()
    plt.figure(figsize=(8.,6.))
    plt.hist(spd_rf_1d, bins=40, histtype='step', rwidth=1.,color='r',log=True)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Surface power density on RF antennas')
    plt.xlabel('kW/m2')
    plt.ylabel('')
    plt.ylim((0.5,5000.))
    plt.xlim((0.,100.))
    plt.tight_layout(pad=2)
    pdf.savefig()
    plt.close()

    
    # ++++++++++++++++++++++++++++++++++++++++++++++

    plt.close()
    plt.figure(figsize=(8.,6.))
    plt.hist(spd_wall_upper_1d, bins=40, histtype='step', rwidth=1.,color='k',log=True)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Surface power density on upper wall')
    plt.xlabel('kW/m2')
    plt.ylabel('')
    plt.ylim(bottom=0.5)
    plt.tight_layout(pad=2)
    pdf.savefig()
    plt.close()

    plt.close()
    plt.figure(figsize=(8.,6.))
    plt.hist(spd_all_1d, bins=50, histtype='step', rwidth=1.,color='k',log=True)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Surface power density on all surfaces')
    plt.xlabel('kW/m2')
    plt.ylabel('')
    plt.ylim(bottom=0.5)
    plt.tight_layout(pad=2)
    pdf.savefig()
    plt.close()
        
        
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #
    #    plot surface power density
    
    spd_limiter_max    = np.max(spd_limiter)
    spd_rf_max         = np.max(spd_rf)
    spd_wall_upper_max = np.max(spd_wall_upper)
    spd_wall_lower_max = np.max(spd_wall_lower)
    spd_iwall_max      = np.max(spd_iwall)

    kk_limiter    = np.where(spd_limiter    == np.amax(spd_limiter))
    kk_rf         = np.where(spd_rf         == np.amax(spd_rf))
    
    kk_wall_upper = np.where(spd_wall_upper == np.amax(spd_wall_upper))
    kk_wall_lower = np.where(spd_wall_lower == np.amax(spd_wall_lower))
    kk_iwall      = np.where(spd_iwall      == np.amax(spd_iwall))

    hits_limiter_max    = int(hits_limiter[kk_limiter][0])
    hits_rf_max         = int(hits_rf[kk_rf][0])

    
    hits_wall_upper_max = int(hits_wall_upper[kk_wall_upper][0])
    hits_wall_lower_max = int(hits_wall_lower[kk_wall_lower][0])
    hits_iwall_max      = int(hits_iwall[kk_iwall][0])
    
    error_limiter       = 100./np.sqrt(hits_limiter_max)
    error_rf            = 100./np.sqrt(hits_rf_max)
    
    if(hits_wall_upper_max > 0):
        error_wall_upper    = 100./np.sqrt(hits_wall_upper_max)
    else:
        error_wall_upper    = 100.
        
    if(hits_wall_lower_max > 0):
        error_wall_lower    = 100./np.sqrt(hits_wall_lower_max)
    else:
        error_wall_lower    = 0.
        
    if(hits_iwall_max > 0):
        error_iwall         = 100./np.sqrt(hits_iwall_max)
    else:
        error_iwall         = 100.
    
    
    spd_max = np.max([spd_limiter_max,spd_rf_max,  spd_wall_upper_max, spd_wall_lower_max, spd_iwall_max])

    print("\n ++++++++++++++++++++++++++++++ \n maximum surface power density \n")
    print("  limiter      %8.2f "%spd_limiter_max)
    print("  rf antennas  %8.2f "%spd_rf_max)
    
    
    print("  wall_upper   %8.2f "%spd_wall_upper_max)
    print("  wall_lower   %8.2f "%spd_wall_lower_max)
    print("  wall_inner   %8.2f "%spd_iwall_max)
    print("\n maximum surface power density: %6.1f  kW/m2"%spd_max)

    print("\n ++++++++++++++++++++++++++++++ \n hits at maximum surface power density and percent error \n")

    print("  limiter      %6d %5.1f"%(hits_limiter_max, error_limiter))
    print("  rf antennas  %6d %5.1f"%(hits_rf_max, error_rf))
    
    
    print("  wall_upper   %6d %5.1f"%(hits_wall_upper_max, error_wall_upper))
    print("  wall_lower   %6d %5.1f"%(hits_wall_lower_max, error_wall_lower))
    print("  wall_inner   %6d %5.1f"%(hits_iwall_max, error_iwall))


    #  4/27/2021:  consistent normalization
    
    spd_max_max = np.max([spd_limiter_max, spd_rf_max, spd_wall_upper_max, spd_wall_lower_max,spd_iwall_max  ])
    
    spd_limiter_norm      = spd_limiter    / spd_max_max  # (1.e-10 + spd_limiter_max)
    spd_rf_norm           = spd_rf         / spd_max_max  # (1.e-10 + spd_rf_max)
    spd_wall_upper_norm   = spd_wall_upper / spd_max_max  # (1.e-10 + spd_wall_upper_max)
    spd_wall_lower_norm   = spd_wall_lower / spd_max_max  # (1.e-10 + spd_wall_lower_max)
    spd_iwall_norm        = spd_iwall      / spd_max_max  # (1.e-10 + spd_iwall_max)
    
    plt.close()
    
    plt.figure(figsize=(9.,7.))
    fig,ax=plt.subplots()
    plt.xlim(-0.1, 2.*np.pi+0.1)
    plt.ylim(-1.25, 1.25)


    # rf antennas
    
    for jtf in range(ntf):
        for jphi in range(nphi_rf):
            for jz in range(nz_rf):
                
                delta_phi = phiend_rf[jtf,jphi,jz] - phistart_rf[jtf,jphi,jz]
                delta_z   = zend_rf[jtf,jphi,jz]   - zstart_rf[jtf,jphi,jz]
                rect = Rectangle( (phistart_rf[jtf,jphi,jz],zstart_rf[jtf,jphi,jz]),delta_phi, delta_z, facecolor=my_colormap(spd_rf_norm[jtf,jphi,jz]), edgecolor='none', linewidth=0.25)
                ax.add_patch(rect)

    # poloidal limiters
    
    for jtf in range(ntf):
        for jphi in range(nphi_limiter):
            for jz in range(nz_limiter):

                delta_phi = phiend_limiter[jtf,jphi,jz] - phistart_limiter[jtf,jphi,jz]
                delta_z   = zend_limiter[jtf,jphi,jz]   - zstart_limiter[jtf,jphi,jz]
                
                rect = Rectangle( (phistart_limiter[jtf,jphi,jz],zstart_limiter[jtf,jphi,jz]),delta_phi, delta_z, facecolor=my_colormap(spd_limiter_norm[jtf,jphi,jz]), edgecolor='none', linewidth=0.25)
                ax.add_patch(rect)
                
    #  tbl limiters (removed)
    


    # outer walls

    for jphi in range(nphi_wall_1):
        for jz in range(nz_wall_1):

            delta_phi_lower = phiend_wall_lower[jphi,jz] - phistart_wall_lower[jphi,jz]
            delta_z_lower   = zend_wall_lower[jphi,jz]   - zstart_wall_lower[jphi,jz]

            delta_phi_upper = phiend_wall_upper[jphi,jz] - phistart_wall_upper[jphi,jz]
            delta_z_upper   = zend_wall_upper[jphi,jz]   - zstart_wall_upper[jphi,jz]
            #pdb.set_trace()
            
            rect = Rectangle( (phistart_wall_lower[jphi,jz],zstart_wall_lower[jphi,jz]),delta_phi_lower, delta_z_lower, facecolor=my_colormap(spd_wall_lower_norm[jphi,jz]), edgecolor='none', linewidth=0.25)
            ax.add_patch(rect)

            #pdb.set_trace()
            rect = Rectangle( (phistart_wall_upper[jphi,jz],zstart_wall_upper[jphi,jz]),delta_phi_upper, delta_z_upper, facecolor=my_colormap(spd_wall_upper_norm[jphi,jz]), edgecolor='none', linewidth=0.25)
            ax.add_patch(rect) 

            for jtf in range(ntf):

                delta_phi_limiter = phiend_limiter_edge[jtf] - phistart_limiter_edge[jtf]
                delta_z_limiter   = zend_limiter_edge[jtf] - zstart_limiter_edge[jtf]
                rect = Rectangle( (phistart_limiter_edge[jtf],zstart_limiter_edge[jtf]),delta_phi_limiter, delta_z_limiter, facecolor='none', edgecolor='k', linewidth=0.25)
                ax.add_patch(rect)

                delta_phi_rf = phiend_rf_edge[jtf] - phistart_rf_edge[jtf]
                delta_z_rf   = zend_rf_edge[jtf] - zstart_rf_edge[jtf]
                rect = Rectangle( (phistart_rf_edge[jtf],zstart_rf_edge[jtf]),delta_phi_rf, delta_z_rf, facecolor='none', edgecolor='k', linewidth=0.25)
                ax.add_patch(rect)

            #rect = Rectangle( (0.,tall_limiter),2.*np.pi, tall_tbl, facecolor='none', edgecolor='k', linewidth=0.25)
            #ax.add_patch(rect)

            #rect = Rectangle( (0.,-1.*(tall_limiter + tall_tbl)),2.*np.pi, tall_tbl, facecolor='none', edgecolor='k', linewidth=0.25)
            #ax.add_patch(rect)

            rect = Rectangle( (0.,(tall_limiter+tall_tbl)),2.*np.pi, tall_wall, facecolor='none', edgecolor='k', linewidth=0.25)
            ax.add_patch(rect)

            rect = Rectangle( (0.,-1.*(tall_limiter+tall_tbl+tall_wall)),2.*np.pi, tall_wall, facecolor='none', edgecolor='k', linewidth=0.25)
            ax.add_patch(rect)
    
    my_title2 = "surface power density (max = %6.1f)  kw/m2"%(spd_max)                    
    plt.title(my_title2, fontsize=10)
    #plt.colorbar()
    plt.xlabel('phi [radians]')
    plt.ylabel('Z [m]')
    #plt.tight_layout(pad=padsize)
    pdf.savefig()
    #plt.savefig('surface_heating.pdf')
    plt.close()

    plt.figure(figsize=(8.,6.))
    mat = np.random.random((30,30))
    plt.imshow(mat, origin="lower", cmap='plasma', interpolation='nearest')
    plt.colorbar()
    pdf.savefig()

    # ++++++++++++++++++++++++++++++++++++
    #  return to per-component normalization

    spd_limiter_norm      = spd_limiter    / (1.e-10 + spd_limiter_max)
    spd_rf_norm           = spd_rf         / (1.e-10 + spd_rf_max)
    spd_wall_upper_norm   = spd_wall_upper / (1.e-10 + spd_wall_upper_max)
    spd_wall_lower_norm   = spd_wall_lower / (1.e-10 + spd_wall_lower_max)
    spd_iwall_norm        = spd_iwall      / (1.e-10 + spd_iwall_max)
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   individual poloidal limiters

    spd_limiter_maxs  = np.zeros(ntf)
    spd_limiter_peaks = np.zeros(ntf)
    spd_limiter_means = np.zeros(ntf)
    
    for jtf in range(ntf):

        delta_phi_limiter = phiend_limiter_edge[jtf] - phistart_limiter_edge[jtf]
        delta_z_limiter   = zend_limiter_edge[jtf] - zstart_limiter_edge[jtf]
        rect = Rectangle( (phistart_limiter_edge[jtf],zstart_limiter_edge[jtf]),delta_phi_limiter, delta_z_limiter, facecolor='none', edgecolor='k', linewidth=0.25)
        ax.add_patch(rect)
        
        spd_max_local = np.max(spd_limiter[jtf,:,:])
        spd_limiter_local_norm = spd_limiter[jtf,:,:] / spd_max_local  # no longer used
        kw_local = np.sum(power_limiter[jtf,:,:])
        peak_local = spd_max_local / np.mean(spd_limiter[jtf,:,:])

        spd_limiter_maxs[jtf]  = spd_max_local
        spd_limiter_peaks[jtf] = peak_local
        spd_limiter_means[jtf] = np.mean(spd_limiter[jtf,:,:])

        print("   ... poloidal limiter %2d : max surface power density = %f7.2"%(jtf, spd_max_local))
        plt.close('all')
        plt.figure(figsize=(9.,7.))
        fig,ax=plt.subplots()
        plt.ylim((-tall_limiter-0.05,tall_limiter+0.05))

        my_phimin = np.min(phistart_limiter[jtf,:,:]) * 180. / np.pi - 0.5
        my_phimax = np.max(phiend_limiter[jtf,:,:])   * 180. / np.pi + 0.5
        
        for jphi in range(nphi_limiter):
            for jz in range(nz_limiter):

                delta_phi = phiend_limiter[jtf,jphi,jz] - phistart_limiter[jtf,jphi,jz]
                delta_z   = zend_limiter[jtf,jphi,jz]   - zstart_limiter[jtf,jphi,jz]
                
                rect = Rectangle( (phistart_limiter[jtf,jphi,jz]*180./np.pi,zstart_limiter[jtf,jphi,jz]),delta_phi*180./np.pi, delta_z, facecolor=my_colormap(spd_limiter_norm[jtf,jphi,jz]), edgecolor='none', linewidth=0.25)
                ax.add_patch(rect)
                #pdb.set_trace()

                delta_phi_limiter = (phiend_limiter_edge[jtf] - phistart_limiter_edge[jtf])
                delta_z_limiter   = zend_limiter_edge[jtf] - zstart_limiter_edge[jtf]
                rect = Rectangle( (phistart_limiter_edge[jtf]*180./np.pi,zstart_limiter_edge[jtf]),delta_phi_limiter*180./np.pi, delta_z_limiter, facecolor='none', edgecolor='k', linewidth=0.25)
                ax.add_patch(rect)
                
        my_title3 = "limiter %2d  %6.1f kw spd-norm  (max spd = %6.1f) peak-f: %5.2f"%(jtf,kw_local,spd_max_local, peak_local)                    
        plt.title(my_title3, fontsize=10)
        #plt.colorbar()
        plt.xlim((my_phimin, my_phimax))
        plt.ylim((-tall_limiter-0.05,tall_limiter+0.05))
        plt.xlabel('phi [degrees]')
        plt.ylabel('Z [m]')
        plt.tight_layout(pad=1.5)
        pdf.savefig()
        plt.close('all')

        print("   maximum of spd_limiter_norm = ", np.max(spd_limiter_norm))

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   profiles of spd for poloidal limiters

    #  plot versus z along phi-slice for which spd reaches a maximum
    my_colors = ('r','orange','green','cyan','blue','violet','magenta','lime','teal')
    plt.close()
    plt.figure(figsize=(8.,6.))
    
    for jtf in range(ntf):

       spd_limiter_2d = spd_limiter[jtf,:,:]
       
       imax_phi       = np.argmax(spd_limiter_2d)//nz_limiter
       spd_slice_z    = spd_limiter_2d[imax_phi,:]
       zarray_local   = 0.5 * (zstart_limiter[jtf,imax_phi,:] + zend_limiter[jtf,imax_phi,:])

       icolor = jtf%(len(my_colors))
       this_color = my_colors[icolor]

       plt.plot(zarray_local, spd_slice_z, '-', color=this_color)

    plt.ylim(bottom=0.)
    plt.xlabel('Z [m]')
    plt.ylabel('kW/m2')
    plt.title('Limiter SPD along vertical slice at maximum spd')
    pdf.savefig()

    #  now as a function of phi

    plt.close()
    plt.figure(figsize=(8.,6.))
    
    for jtf in range(ntf):

       spd_limiter_2d = spd_limiter[jtf,:,:]
       
       imax_phi       = np.argmax(spd_limiter_2d)//nz_limiter
       imax_z         = np.argmax(spd_limiter_2d) - imax_phi * nz_limiter
       spd_slice_phi  = spd_limiter_2d[:,imax_z]
       phi_local      = (0.5 * (phistart_limiter[0,:,imax_z] + phiend_limiter[0,:,imax_z]))*180./np.pi

       icolor = jtf%(len(my_colors))
       this_color = my_colors[icolor]

       plt.plot(phi_local, spd_slice_phi, '-', color=this_color)
       #pdb.set_trace()
    plt.ylim(bottom=0.)
    plt.xlabel('phi [degrees]')
    plt.ylabel('kW/m2')
    plt.title('Limiter SPD along toroidal slice at maximum spd')
    pdf.savefig()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   individual RF antennas

    spd_rf_maxs  = np.zeros(ntf)
    spd_rf_peaks = np.zeros(ntf)
    spd_rf_means = np.zeros(ntf)


    for jtf in range(ntf):

        delta_phi_rf = phiend_rf_edge[jtf] - phistart_rf_edge[jtf]
        delta_z_rf   = zend_rf_edge[jtf] - zstart_rf_edge[jtf]
        rect = Rectangle( (phistart_rf_edge[jtf],zstart_rf_edge[jtf]),delta_phi_rf, delta_z_rf, facecolor='none', edgecolor='k', linewidth=0.25)
        ax.add_patch(rect)
        
        spd_max_local           = np.max(spd_rf[jtf,:,:])
        spd_rf_local_norm       = spd_rf[jtf,:,:] / spd_max_local  # no longer used
        kw_local                = np.sum(power_rf[jtf,:,:])
        peak_local              = spd_max_local / np.mean(spd_rf[jtf,:,:])

        spd_rf_maxs[jtf]  = spd_max_local
        spd_rf_peaks[jtf] = peak_local
        spd_rf_means[jtf] = np.mean(spd_rf[jtf,:,:])

        print("   ... rf antenna %2d : max surface power density = %f7.2"%(jtf, spd_max_local))
        plt.close('all')
        plt.figure(figsize=(9.,7.))
        fig,ax=plt.subplots()
        plt.ylim((-tall_limiter-0.05,tall_limiter+0.05))

        my_phimin = np.min(phistart_rf[jtf,:,:]) * 180. / np.pi - 0.5
        my_phimax = np.max(phiend_rf[jtf,:,:])   * 180. / np.pi + 0.5
        
        for jphi in range(nphi_rf):
            for jz in range(nz_rf):

                delta_phi = phiend_rf[jtf,jphi,jz] - phistart_rf[jtf,jphi,jz]
                delta_z   = zend_rf[jtf,jphi,jz]   - zstart_rf[jtf,jphi,jz]
                
                rect = Rectangle( (phistart_rf[jtf,jphi,jz]*180./np.pi,zstart_rf[jtf,jphi,jz]),delta_phi*180./np.pi, delta_z, facecolor=my_colormap(spd_rf_norm[jtf,jphi,jz]), edgecolor='none', linewidth=0.25)
                ax.add_patch(rect)
                #pdb.set_trace()

                delta_phi_rf = (phiend_rf_edge[jtf] - phistart_rf_edge[jtf])
                delta_z_rf   = zend_rf_edge[jtf] - zstart_rf_edge[jtf]
                rect = Rectangle( (phistart_rf_edge[jtf]*180./np.pi,zstart_rf_edge[jtf]),delta_phi_rf*180./np.pi, delta_z_rf, facecolor='none', edgecolor='k', linewidth=0.25)
                ax.add_patch(rect)
                
        my_title3 = "rf antenna %2d  %6.1f kw spd-norm  (max spd = %6.1f) peak-f: %5.2f"%(jtf,kw_local,spd_max_local, peak_local)                    
        plt.title(my_title3, fontsize=10)
        #plt.colorbar()
        plt.xlim((my_phimin, my_phimax))
        plt.ylim((-tall_limiter-0.05,tall_limiter+0.05))
        plt.xlabel('phi [degrees]')
        plt.ylabel('Z [m]')
        plt.tight_layout(pad=1.5)
        pdf.savefig()
        plt.close('all')

        print("   maximum of spd_rf_norm = ", np.max(spd_rf_norm))
                
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #  upper tbl (removed)

 
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #  hits density



    hpd_limiter_max    = np.max(hpd_limiter)
    hpd_rf_max         = np.max(hpd_rf)
    
    
    hpd_wall_upper_max = np.max(hpd_wall_upper)
    hpd_wall_lower_max = np.max(hpd_wall_lower)
    hpd_iwall_max      = np.max(hpd_iwall)
    
    hpd_max = np.max([hpd_limiter_max,hpd_rf_max,  hpd_wall_upper_max, hpd_wall_lower_max, hpd_iwall_max])

    print("\n ++++++++++++++++++++++++++++++ \n maximum hits density")
    print("  limiter      %8.2f "%hpd_limiter_max)
    print("  rf antennas  %8.2f "%hpd_rf_max)

    
    print("  wall_upper   %8.2f "%hpd_wall_upper_max)
    print("  wall_lower   %8.2f "%hpd_wall_lower_max)
    print("  wall_inner   %8.2f "%hpd_iwall_max)
    print("\n maximum hits density: %8.2f  /m2"%hpd_max)

    hpd_limiter_norm      = hpd_limiter    / hpd_max
    hpd_rf_norm           = hpd_rf         / hpd_max
    
    
    hpd_wall_upper_norm   = hpd_wall_upper / hpd_max
    hpd_wall_lower_norm   = hpd_wall_lower / hpd_max
    hpd_iwall_norm        = hpd_iwall      / hpd_max
        
    plt.close()
    
    plt.figure(figsize=(9.,7.))
    fig,ax=plt.subplots()
    plt.xlim(-0.1, 2.*np.pi+0.1)
    plt.ylim(-1.25, 1.25)


    # rf antennas
    
    for jtf in range(ntf):
        for jphi in range(nphi_rf):
            for jz in range(nz_rf):
                
                delta_phi = phiend_rf[jtf,jphi,jz] - phistart_rf[jtf,jphi,jz]
                delta_z   = zend_rf[jtf,jphi,jz]   - zstart_rf[jtf,jphi,jz]
                rect = Rectangle( (phistart_rf[jtf,jphi,jz],zstart_rf[jtf,jphi,jz]),delta_phi, delta_z, facecolor=my_colormap(hpd_rf_norm[jtf,jphi,jz]), edgecolor='none', linewidth=0.25)
                ax.add_patch(rect)

    # poloidal limiters
    
    for jtf in range(ntf):
        for jphi in range(nphi_limiter):
            for jz in range(nz_limiter):

                delta_phi = phiend_limiter[jtf,jphi,jz] - phistart_limiter[jtf,jphi,jz]
                delta_z   = zend_limiter[jtf,jphi,jz]   - zstart_limiter[jtf,jphi,jz]
                
                rect = Rectangle( (phistart_limiter[jtf,jphi,jz],zstart_limiter[jtf,jphi,jz]),delta_phi, delta_z, facecolor=my_colormap(hpd_limiter_norm[jtf,jphi,jz]), edgecolor='none', linewidth=0.25)
                ax.add_patch(rect)
                
    #  tbl limiters (removed)
    


    # outer walls

    for jphi in range(nphi_wall_1):
        for jz in range(nz_wall_1):

            delta_phi_lower = phiend_wall_lower[jphi,jz] - phistart_wall_lower[jphi,jz]
            delta_z_lower   = zend_wall_lower[jphi,jz]   - zstart_wall_lower[jphi,jz]

            delta_phi_upper = phiend_wall_upper[jphi,jz] - phistart_wall_upper[jphi,jz]
            delta_z_upper   = zend_wall_upper[jphi,jz]   - zstart_wall_upper[jphi,jz]
            
            rect = Rectangle( (phistart_wall_lower[jphi,jz],zstart_wall_lower[jphi,jz]),delta_phi_lower, delta_z_lower, facecolor=my_colormap(hpd_wall_lower_norm[jphi,jz]), edgecolor='none', linewidth=0.25)
            ax.add_patch(rect)

            rect = Rectangle( (phistart_wall_upper[jphi,jz],zstart_wall_upper[jphi,jz]),delta_phi_upper, delta_z_upper, facecolor=my_colormap(hpd_wall_upper_norm[jphi,jz]), edgecolor='none', linewidth=0.25)
            ax.add_patch(rect) 

            for jtf in range(ntf):

                delta_phi_limiter = phiend_limiter_edge[jtf] - phistart_limiter_edge[jtf]
                delta_z_limiter   = zend_limiter_edge[jtf] - zstart_limiter_edge[jtf]
                rect = Rectangle( (phistart_limiter_edge[jtf],zstart_limiter_edge[jtf]),delta_phi_limiter, delta_z_limiter, facecolor='none', edgecolor='k', linewidth=0.25)
                ax.add_patch(rect)

                delta_phi_rf = phiend_rf_edge[jtf] - phistart_rf_edge[jtf]
                delta_z_rf   = zend_rf_edge[jtf] - zstart_rf_edge[jtf]
                rect = Rectangle( (phistart_rf_edge[jtf],zstart_rf_edge[jtf]),delta_phi_rf, delta_z_rf, facecolor='none', edgecolor='k', linewidth=0.25)
                ax.add_patch(rect)



            rect = Rectangle( (0.,(tall_limiter+tall_tbl)),2.*np.pi, tall_wall, facecolor='none', edgecolor='k', linewidth=0.25)
            ax.add_patch(rect)

            rect = Rectangle( (0.,-1.*(tall_limiter+tall_tbl+tall_wall)),2.*np.pi, tall_wall, facecolor='none', edgecolor='k', linewidth=0.25)
            ax.add_patch(rect)
    my_title = " hits density (max = %10.2f)"%(hpd_max)                         
    plt.title(my_title, fontsize=10)
    #plt.colorbar()
    plt.xlabel('phi [radians]')
    plt.ylabel('Z [m]')
    #plt.tight_layout(pad=padsize)
    pdf.savefig()
    #plt.savefig('surface_heating.pdf')
    plt.close()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    xx = np.linspace(1,ntf,ntf)
    
    plt.close()
    plt.plot(xx,spd_limiter_maxs, 'bo', ms=2)
    plt.plot(xx, spd_limiter_maxs, 'b-', linewidth=1)
    plt.xlabel('TF sector')
    plt.ylabel('kw/m2')
    plt.title('Max spd on poloidal limiters')
    plt.xlim((0.,ntf+1))
    plt.ylim(bottom=0.)
    plt.grid(True)
    pdf.savefig()

    plt.close()
    plt.plot(xx,spd_limiter_means, 'bo', ms=2)
    plt.plot(xx, spd_limiter_means, 'b-', linewidth=1)
    plt.xlabel('TF sector')
    plt.ylabel('kw/m2')
    plt.title('Mean spd on poloidal limiters')
    plt.xlim((0.,ntf+1))
    plt.ylim(bottom=0.)
    plt.grid(True)
    pdf.savefig()
    
    plt.close()
    plt.plot(xx,spd_limiter_peaks, 'bo', ms=2)
    plt.plot(xx, spd_limiter_peaks, 'b-', linewidth=1)
    plt.xlabel('TF sector')
    plt.title('spd peaking factor on poloidal limiters')
    plt.xlim((0.,ntf+1))
    plt.ylim(bottom=0.)
    plt.grid(True)
    pdf.savefig()

    # repeat with fixed ymax

    plt.close()
    plt.plot(xx,spd_limiter_maxs, 'bo', ms=2)
    plt.plot(xx, spd_limiter_maxs, 'b-', linewidth=1)
    plt.xlabel('TF sector')
    plt.ylabel('kw/m2')
    plt.title('Max spd on poloidal limiters')
    plt.xlim((0.,ntf+1))
    plt.ylim((0.,1500.))
    plt.grid(True)
    pdf.savefig()

    plt.close()
    plt.plot(xx,spd_limiter_means, 'bo', ms=2)
    plt.plot(xx, spd_limiter_means, 'b-', linewidth=1)
    plt.xlabel('TF sector')
    plt.ylabel('kw/m2')
    plt.title('Mean spd on poloidal limiters')
    plt.xlim((0.,ntf+1))
    plt.ylim((0.,100.))
    plt.grid(True)
    pdf.savefig()
    
    plt.close()
    plt.plot(xx,spd_limiter_peaks, 'bo', ms=2)
    plt.plot(xx, spd_limiter_peaks, 'b-', linewidth=1)
    plt.xlabel('TF sector')
    plt.title('spd peaking factor on poloidal limiters')
    plt.xlim((0.,ntf+1))
    plt.ylim((0.,20.))
    plt.grid(True)
    pdf.savefig()
    
    spd_limiter_maxs = np.zeros(ntf)
    spd_limiter_peaks = np.zeros(ntf)
    
 # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #  compute total hits for a check

    hits_rf_total         = np.sum(hits_rf)
    hits_limiter_total    = np.sum(hits_limiter)
    
    
    hits_wall_upper_total = np.sum(hits_wall_upper)
    hits_wall_lower_total = np.sum(hits_wall_lower)
    hits_iwall_total      = np.sum(hits_iwall)

    hits_total = hits_rf_total + hits_limiter_total  \
                 + hits_wall_upper_total + hits_wall_lower_total + hits_iwall_total

    area_rf_total         = np.sum(area_rf)
    area_limiter_total    = np.sum(area_limiter)
    area_wall_upper_total = np.sum(area_wall_upper)
    area_wall_lower_total = np.sum(area_wall_lower)
    
    
    area_iwall_total      = np.sum(area_iwall)

    area_total = area_rf_total + area_limiter_total + area_wall_upper_total + area_wall_lower_total \
                  + area_iwall_total

    kw_limiter    = np.sum(np.multiply(area_limiter, spd_limiter))
    kw_rf         = np.sum(np.multiply(area_rf, spd_rf))


    kw_wall_upper  = np.sum(np.multiply(area_wall_upper, spd_wall_upper))
    kw_wall_lower  = np.sum(np.multiply(area_wall_lower, spd_wall_lower))
    kw_iwall       = np.sum(np.multiply(area_iwall, spd_iwall))

    kw_total       = kw_limiter + kw_rf  + kw_wall_upper + kw_wall_lower + kw_iwall

    kw_limiters     = np.zeros(ntf)
    kw_rfs          = np.zeros(ntf)
    
    
    kw_wall_uppers  = np.zeros(nphi_wall_1)
    kw_wall_lowers  = np.zeros(nphi_wall_1)
    kw_iwalls       = np.zeros(nphi_wall_2)

    print("\n ++++++++++++++++++++++++++++ \n  jtf  kw-lim  kw-rf \n")
    for jtf in range(ntf):
        kw_limiters[jtf] = np.sum(np.multiply(area_limiter[jtf,:,:], spd_limiter[jtf,:,:]))
        kw_rfs[jtf]      = np.sum(np.multiply(area_rf[jtf,:,:], spd_rf[jtf,:,:]))
        print(" %3d  %6.2f %6.2f "%(jtf, kw_limiters[jtf], kw_rfs[jtf]))

    peak_limiters = np.max(kw_limiters) / np.mean(kw_limiters)
    peak_rfs      = np.max(kw_rfs)      / np.mean(kw_rfs)
        
    print(" \n +++++++++++++++++++ toroidal-peaking factors  \n")
    print("   limiters   %6.3f  "%peak_limiters)
    print("   antennas   %6.3f  \n"%peak_rfs)
    
    kw_limiter_frac       = 100. * kw_limiter    / kw_total
    kw_rf_frac            = 100. * kw_rf         / kw_total
    
    
    kw_wall_upper_frac    = 100. * kw_wall_upper / kw_total
    kw_wall_lower_frac    = 100. * kw_wall_lower / kw_total
    kw_iwall_frac         = 100. * kw_iwall      / kw_total
    
    spd_limiter_avg     = kw_limiter    / area_limiter_total
    spd_rf_avg          = kw_rf         / area_rf_total
    
    
    spd_wall_upper_avg  = kw_wall_upper / area_wall_upper_total
    spd_wall_lower_avg  = kw_wall_lower / area_wall_lower_total
    spd_iwall_avg       = kw_iwall      / area_iwall_total

    if(spd_limiter_avg > 0):
        spd_limiter_ratio     = spd_limiter_max    / spd_limiter_avg
    else:
        spd_limiter_ratio     = 0.
       
    if(spd_rf_avg > 0):
        spd_rf_ratio          = spd_rf_max         / spd_rf_avg
    else:
        spd_rf_ratio = 0.

    if(spd_wall_upper_avg > 0.):
        spd_wall_upper_ratio  = spd_wall_upper_max / spd_wall_upper_avg
    else:
        spd_wall_upper_ratio = 0.

    if(spd_wall_lower_avg > 0.):
        spd_wall_lower_ratio  = spd_wall_lower_max / spd_wall_lower_avg
    else:
        spd_wall_lower_ratio  = 0.
        
    if(spd_iwall_avg > 0.):
        spd_iwall_ratio       = spd_iwall_max      / spd_iwall_avg
    else:
        spd_iwall_ratio = 0.

    
    print("\n +++++++++++++++++++++++++++++++++++++ \n kilowatts onto integrated components and percent of total")
    
    print("   kw_limiter:     %7.1f   %5.2f "%(kw_limiter,    kw_limiter_frac))
    print("   kw_rf:          %7.1f   %5.2f "%(kw_rf,         kw_rf_frac))
    

    print("   kw_wall_upper:  %7.1f   %5.2f "%(kw_wall_upper, kw_wall_upper_frac))
    print("   kw_wall_lower:  %7.1f   %5.2f "%(kw_wall_lower, kw_wall_lower_frac))
    print("   kw_inner_wall:  %7.1f   %5.2f "%(kw_iwall,      kw_iwall_frac))
    print("   kw (total):     %7.1f         "%(kw_total))

    print("\n +++++++++++++++++++++++++++++++++++++ \n average surface power density of integrated components")
    
    print("   limiter:     %8.2f  "%(spd_limiter_avg))
    print("   rf:          %8.2f  "%(spd_rf_avg))
    

    print("   wall_upper:  %8.2f  "%(spd_wall_upper_avg))
    print("   wall_lower:  %8.2f  "%(spd_wall_lower_avg))
    print("   inner_wall:  %8.2f  "%(spd_iwall_avg))

    print("\n +++++++++++++++++++++++++++++++++++++ \n max/average surface power density of integrated components")
    
    print("   limiter:     %8.2f  "%(spd_limiter_ratio))
    print("   rf:          %8.2f  "%(spd_rf_ratio))
    
    
    print("   wall_upper:  %8.2f  "%(spd_wall_upper_ratio))
    print("   wall_lower:  %8.2f  "%(spd_wall_lower_ratio))
    print("   inner_wall:  %8.2f  "%(spd_iwall_ratio))
    
    print("\n +++++++++++++++++++++++++++++++++++++ \n total hits on integrated components")
    
    print("   hits_limiter:     %7.0f  "%(hits_limiter_total))
    print("   hits_rf:          %7.0f  "%(hits_rf_total))
    
    
    print("   hits_wall_upper:  %7.0f  "%(hits_wall_upper_total))
    print("   hits_wall_lower:  %7.0f  "%(hits_wall_lower_total))
    print("   hits_inner_wall:  %7.0f  "%(hits_iwall_total))
    print("   hits (total):     %7.0f  "%(hits_total))
    
    print("\n +++++++++++++++++++++++++++++++++++++ \n total area of integrated components ")
    
    print("   area_limiter:      %7.3f "%(area_limiter_total))
    print("   area_rf:           %7.3f "%(area_rf_total))
    
    
    print("   area_wall_upper:   %7.3f "%(area_wall_upper_total))
    print("   area_wall_lower:   %7.3f "%(area_wall_lower_total))
    print("   area_inner_wall:   %7.3f "%(area_iwall_total))
    print("   area (total):      %7.3f "%(area_total))

    print("\n +++++++++++++++++++++++++++++++++++++ \n range of size of individual rectangles")

    print("   limiter:    %7.5f  %7.5f "%(np.min(area_limiter), np.max(area_limiter)))
    print("   antenna:    %7.5f  %7.5f "%(np.min(area_rf), np.max(area_rf)))
    
    
    print("   wall_upper: %7.5f  %7.5f "%(np.min(area_wall_upper), np.max(area_wall_upper)))
    print("   wall_lower: %7.5f  %7.5f "%(np.min(area_wall_lower), np.max(area_wall_lower)))
    print("   wall_inner: %7.5f  %7.5f "%(np.min(area_iwall), np.max(area_iwall)))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    print("\n +++++++++++++++++++++++++++++++++++++++++++++++ \n \n")

    top_limiter       = np.flip(np.sort(np.reshape(spd_limiter,-1)))
    
    
    top_wall_upper    = np.flip(np.sort(np.reshape(spd_wall_upper,-1)))
    top_wall_lower    = np.flip(np.sort(np.reshape(spd_wall_lower,-1)))
    top_wall_inner    = np.flip(np.sort(np.reshape(spd_iwall,-1)))
    top_rf            = np.flip(np.sort(np.reshape(spd_rf,-1)))

    print("  top 10 surface power densities \n")
    
    print(" n  limiter   wall-upper wall-lower wall-inner  antenna \n")
    for qq in range(10):
        print("%3d %9.2f %9.2f  %9.2f %9.2f %9.2f"%(qq,top_limiter[qq],    \
                                                               top_wall_upper[qq],  \
                                                               top_wall_lower[qq],top_wall_inner[qq], \
                                                               top_rf[qq]))
        
    
    print("\n +++++++++++++++++++++++++++++++++++++++++++++++ \n \n")

    print("                limiters     wall_upper    wall_lower    wall_inner    antenna \n")
    print(" kw       %12.1f %12.1f %12.1f  %12.1f %12.1f"%(kw_limiter,  kw_wall_upper, kw_wall_lower, kw_iwall, kw_rf))
    print(" kw-frac  %12.1f  %12.1f %12.1f %12.1f %12.1f"%(kw_limiter_frac,  kw_wall_upper_frac,
                                                                        kw_wall_lower_frac, kw_iwall_frac, kw_rf_frac))
    print("")
    print(" spd-max  %12.1f %12.1f %12.1f %12.1f  %12.1f"%(spd_limiter_max,  spd_wall_upper_max, spd_wall_lower_max, \
                                                                        spd_iwall_max, spd_rf_max))
    print(" spd-avg  %12.1f %12.1f %12.1f  %12.1f %12.1f"%(spd_limiter_avg,  spd_wall_upper_avg, spd_wall_lower_avg, \
                                                                        spd_iwall_avg, spd_rf_avg))
    print(" peak-f   %12.1f %12.1f %12.1f %12.1f %12.1f"%(spd_limiter_ratio,  spd_wall_upper_ratio, spd_wall_lower_ratio, \
                                                                        spd_iwall_ratio, spd_rf_ratio))
    print("")
    print(" max-hits %12d %12d %12d %12d %12d"%(hits_limiter_max,  hits_wall_upper_max, hits_wall_lower_max,       hits_iwall_max,   hits_rf_max))
    print(" error    %12.1f  %12.1f %12.1f  %12.1f %12.1f"%(   error_limiter,     error_wall_upper,    error_wall_lower,     error_iwall,      error_rf))
    print(" hits-tot %12d %12d %12d %12d %12d"%(hits_limiter_total,  hits_wall_upper_total, hits_wall_lower_total, hits_iwall_total, hits_rf_total))
    print("")
    print(" tpf-lim-rf: %9.2f %9.2f "%(peak_limiters, peak_rfs))
    
    print("\n +++++++++++++++++++++++++++++++++++++++++++\n")
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # pdb.set_trace()
    
def surface_power_density(ploss_wall_kw, rloss, zloss, phiend_lost, weight_energy_lost, ishape, pdf, time_lost, fn_parameter="none"):

    # the following allows us to look into whether delayed losses have a
    # different spatial distribution than earlier losses
    
    original_total_weight_energy_lost = np.sum(weight_energy_lost)
    
    time_minimum = 0.   # need to set this manually

    ii = ( time_lost>= time_minimum)

    rloss              = rloss[ii]
    zloss              = zloss[ii]
    phiend_lost        = phiend_lost[ii]
    weight_energy_lost = weight_energy_lost[ii]

    new_total_energy_lost = np.sum(weight_energy_lost)

    ploss_wall_kw = ploss_wall_kw * new_total_energy_lost / original_total_weight_energy_lost
    
    #  colormaps:  viridis, cividis, plamsa, YlOrRd,
    #              spring, summer, hot, Blues, gnuplot2

    my_colormap = cm.get_cmap('plasma')   # Blues
    
    philoss = phiend_lost * np.pi / 180.     # convert to radians
    
    #  get wall shape
    
    aa_inner = myread.read_any_file('/project/projectdirs/m3195/ascot/mypython/wall_inner_new.txt')  # sparc_pfc_inner_simple_culled3.txt
    aa_outer = myread.read_any_file('/project/projectdirs/m3195/ascot/mypython/wall_outer_new.txt')  # sparc_pfc_outer_simple.txt

    rr_inner = aa_inner[:,0]
    zz_inner = aa_inner[:,1]

    rr_outer = aa_outer[:,0]
    zz_outer = aa_outer[:,1]

    rmax        = np.max(rr_outer)   # 2.4411
    rmin_outer  = np.min(rr_outer)   # minimum major radius of outer wall

    zwall_max       = np.max(zz_outer) - 0.0001 
    zwall_inner_max = np.max(zz_inner) - 0.0001
    
    # specify sizes of sub-elements
    if(fn_parameter == "none"):
        
        ntf             = 18
        tall_rf_antenna = 0.50   # changed from 0.50  
        tall_limiter    = 0.50   # changed from 0.50
        hmax_limiter    = 0.01   # 0.08  0.01
        hmin_limiter    = 0.008  # 0.035  0.008
        taper_limiter   = 0.10
        y50_limiter     = 0.65    # height is 65% of (hmax-hmin) at 50% of toroidal extent        
        tall_tbl        = 0.10
        taper_tbl       = 0.02
        hmin_tbl        = 0.008   # 0.035 0.008
        hmax_tbl        = 0.01    # 0.08  0.01
        y50_tbl         = 0.65
 
        aminor          = 0.57
        kappa           = 1.75
        height_limiter  = 2. * tall_limiter
        height_rf       = 2. * tall_rf_antenna   
        width_rf_sector_limiter = 0.305
        width_rf_antenna        = (2. * np.pi * rmax/ntf) - width_rf_sector_limiter
        phi_sector_size         = 2. * np.pi              / ntf
        phi_rf_size             = width_rf_antenna        / rmax  
        phi_limiter_size        = width_rf_sector_limiter / rmax
        
        #  shape 6 = shape 5 but 10 cm resolution vertically on the poloidal limiters
        #  shape 7 = shape 6 but 7.5 cm resolution in the toroidal direction (most like spiral)
      
        # ishape          1    2    3   4   5  6    7   8
      
        nnphi_limiter  = [5,   5,  20, 10,  5, 5,   4,  5] 
        nnz_limiter    = [6,   6,  20, 10,  6, 10, 10, 15]
        nnphi_rf       = [5,   3,  20, 10,  3, 3,   3,  3]
        nnz_rf         = [6,   4,  20, 10,  6, 6,   6,  6]    #  was [6,4,2,10] until 10/3
        nnphi_tbl      = [72, 36, 180, 72, 72, 72, 72, 72]
        nnz_tbl        = [8,   8,  20,  8,  8, 8,   8,  8]
        nnphi_wall_1   = [72, 36,  90, 90, 72, 72, 72, 72]   # was (incorrectly) 72,6,90,90 until 10/3
        nnz_wall_1     = [6,  6,  12,  12,  6, 6,   6,  6]
        nnphi_wall_2   = [72, 36, 90,  72, 72, 72, 72, 72]
        nnz_wall_2     = [20, 20, 40,  20, 12, 12, 12, 12]
        nnr_sides      = [2,  2,   4,   4,  2, 2,  2,   2]
        nnz_sides      = [6,  6,  12,   6,  6, 6,  6,   6]
        nnphi_tbl_sides= [5,  5,  10,   5,  5, 5,  5,   5]
        nnz_tbl_sides  = [2,  2,   4,   2 , 2, 5,  5,   5]
        
        nphi_limiter   = nnphi_limiter[ishape-1]
        nz_limiter     = nnz_limiter[ishape-1]
        nphi_rf        = nnphi_rf[ishape-1]
        nz_rf          = nnz_rf[ishape-1] 
        nphi_tbl       = nnphi_tbl[ishape-1] 
        nz_tbl         = nnz_tbl[ishape-1]
        nphi_wall_1    = nnphi_wall_1[ishape-1]  
        nz_wall_1      = nnz_wall_1[ishape-1]
        nphi_wall_2    = nnphi_wall_2[ishape-1]
        nz_wall_2      = nnz_wall_2[ishape-1]
        nr_sides       = nnr_sides[ishape-1]
        nz_sides       = nnz_sides[ishape-1]
        nphi_tbl_sides = nnphi_tbl_sides[ishape-1]
        nz_tbl_sides   = nnz_tbl_sides[ishape-1]

    else:
        
        ff = open(fn_parameter,"r")
        
        suppress_rf             = my.read_next_line('suppress_rf',          ff) 
        suppress_tf_limiters    = my.read_next_line("suppress_tf_limiters", ff)  
        suppress_tbl            = my.read_next_line("suppress_tbl",         ff)       
        suppress_outer_walls    = my.read_next_line("suppress_outer_walls", ff)
        suppress_inner_wall     = my.read_next_line("suppress_inner_wall",  ff)   
        suppress_tbl_sides      = my.read_next_line("suppress_tbl_sides",   ff)  
        suppress_lim_sides      = my.read_next_line("suppress_lim_sides",   ff) 
        ishape                  = my.read_next_line("ishape",               ff)                   
        nphi_limiter            = my.read_next_line("nphi_limiter",         ff)     
        nz_limiter              = my.read_next_line("nz_limiter",           ff)        
        nphi_rf                 = my.read_next_line("nphi_rf", ff)           
        nz_rf                   = my.read_next_line("nz_rf", ff)              
        nphi_tbl                = my.read_next_line("nphi_tbl", ff)           
        nz_tbl                  = my.read_next_line("nz_tbl", ff)              
        nphi_wall_1             = my.read_next_line("nphi_wall_1", ff)        
        nz_wall_1               = my.read_next_line("nz_wall_1", ff)          
        nphi_wall_2             = my.read_next_line("nphi_wall_2", ff)        
        nz_wall_2               = my.read_next_line("nz_wall_2", ff)          
        nr_sides                = my.read_next_line("nr_sides", ff)            
        nz_sides                = my.read_next_line("nz_sides", ff)            
        nphi_tbl_sides          = my.read_next_line("nphi_tbl_sides", ff)      
        nz_tbl_sides            = my.read_next_line("nz_tbl_sides", ff)      
        tall_rf_antenna         = my.read_next_line("tall_rf_antenna", ff)     
        tall_limiter            = my.read_next_line("tall_limiter", ff)        
        hmin_limiter            = my.read_next_line("hmin_limiter", ff)         
        hmax_limiter            = my.read_next_line("hmax_limiter", ff)        
        taper_limiter           = my.read_next_line("taper_limiter", ff)       
        y50_limiter             = my.read_next_line("y50_limiter", ff)          
        tall_tbl                = my.read_next_line("tall_tbl", ff)            
        taper_tbl               = my.read_next_line("taper_tbl", ff)            
        hmin_tbl                = my.read_next_line("hmin_tbl", ff)            
        hmax_tbl                = my.read_next_line("hmax_tbl", ff)             
        y50_tbl                 = my.read_next_line("y50_tbl", ff)              
        width_rf_sector_limiter = my.read_next_line("width_rf_sector_limiter", ff)  
        width_rf_antenna        = my.read_next_line("width_rf_antenna", ff)          
        phi_sector_size         = my.read_next_line("phi_sector_size", ff)          
        phi_rf_size             = my.read_next_line("phi_rf_size", ff)            
        phi_limiter_size        = my.read_next_line("phi_limiter_size", ff)          
        rcrit                   = my.read_next_line("rcrit", ff)                     
        ntf                     = my.read_next_line("ntf", ff)                          
        aminor                  = my.read_next_line("aminor", ff)              
        kappa                   = my.read_next_line("kappa", ff)             
        height_limiter          = my.read_next_line("height_limiter", ff)   
        height_rf               = my.read_next_line("height_rf", ff)       
        proud_limiter_z         = my.read_next_line("proud_limiter_z", ff)      
        proud_rf_z              = my.read_next_line("proud_rf_z", ff)       
        proud_lim_phi           = my.read_next_line("proud_lim_phi", ff)      
        proud_rf_phi            = my.read_next_line("proud_rf_phi", ff)      
        proud_tbl_phi           = my.read_next_line("proud_tbl_phi", ff)

        ff.close()
        print(" ... surface_power_density_2:  have read parameter file \n")
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    tall_wall       = zwall_max - tall_limiter - tall_tbl 
    
    phistart_limiter = np.zeros((ntf, nphi_limiter, nz_limiter))
    phiend_limiter   = np.zeros((ntf, nphi_limiter, nz_limiter))
    zstart_limiter   = np.zeros((ntf, nphi_limiter, nz_limiter))
    zend_limiter     = np.zeros((ntf, nphi_limiter, nz_limiter))
    area_limiter     = np.zeros((ntf, nphi_limiter, nz_limiter))
    power_limiter    = np.zeros((ntf, nphi_limiter, nz_limiter))   # power to area elements, kW
    spd_limiter      = np.zeros((ntf, nphi_limiter, nz_limiter))   # surface power density to area elements, kW/m^2
    hits_limiter     = np.zeros((ntf, nphi_limiter, nz_limiter))   # hits within rectancle
    hpd_limiter      = np.zeros((ntf, nphi_limiter, nz_limiter))   #  hit density 
    
    phistart_rf      = np.zeros((ntf, nphi_rf, nz_rf))
    phiend_rf        = np.zeros((ntf, nphi_rf, nz_rf))
    zstart_rf        = np.zeros((ntf, nphi_rf, nz_rf))
    zend_rf          = np.zeros((ntf, nphi_rf, nz_rf))
    area_rf          = np.zeros((ntf, nphi_rf, nz_rf))
    power_rf         = np.zeros((ntf, nphi_rf, nz_rf))   # power to area elements, kW
    spd_rf           = np.zeros((ntf, nphi_rf, nz_rf))   # surface power density to area elements, kW/m^2
    hits_rf          = np.zeros((ntf, nphi_rf, nz_rf))   # hits within rectancle
    hpd_rf           = np.zeros((ntf, nphi_rf, nz_rf))   #  hit density     
    
    phistart_tbl_upper = np.zeros((nphi_tbl, nz_tbl))
    phiend_tbl_upper   = np.zeros((nphi_tbl, nz_tbl))
    zstart_tbl_upper   = np.zeros((nphi_tbl, nz_tbl))
    zend_tbl_upper     = np.zeros((nphi_tbl, nz_tbl))
    area_tbl_upper     = np.zeros((nphi_tbl, nz_tbl))
    power_tbl_upper    = np.zeros((nphi_tbl, nz_tbl))   # power to area elements, kW
    spd_tbl_upper      = np.zeros((nphi_tbl, nz_tbl))   # surface power density to area elements, kW/m^2 
    hits_tbl_upper     = np.zeros((nphi_tbl, nz_tbl))   # hits within rectancle
    hpd_tbl_upper      = np.zeros((nphi_tbl, nz_tbl))   #  hit density 

    
    phistart_tbl_lower = np.zeros((nphi_tbl, nz_tbl))
    phiend_tbl_lower   = np.zeros((nphi_tbl, nz_tbl))
    zstart_tbl_lower   = np.zeros((nphi_tbl, nz_tbl))
    zend_tbl_lower     = np.zeros((nphi_tbl, nz_tbl))
    area_tbl_lower     = np.zeros((nphi_tbl, nz_tbl))
    power_tbl_lower    = np.zeros((nphi_tbl, nz_tbl))   # power to area elements, kW
    spd_tbl_lower      = np.zeros((nphi_tbl, nz_tbl))   # surface power density to area elements, kW/m^2
    hits_tbl_lower     = np.zeros((nphi_tbl, nz_tbl))   # hits within rectancle
    hpd_tbl_lower      = np.zeros((nphi_tbl, nz_tbl))   #  hit density 
    
    phistart_wall_upper = np.zeros((nphi_wall_1, nz_wall_1))
    phiend_wall_upper   = np.zeros((nphi_wall_1, nz_wall_1))
    zstart_wall_upper   = np.zeros((nphi_wall_1, nz_wall_1))
    zend_wall_upper     = np.zeros((nphi_wall_1, nz_wall_1))
    area_wall_upper     = np.zeros((nphi_wall_1, nz_wall_1))
    power_wall_upper    = np.zeros((nphi_wall_1, nz_wall_1))   # power to area elements, kW
    spd_wall_upper      = np.zeros((nphi_wall_1, nz_wall_1))   # surface power density to area elements, kW/m^2 
    hits_wall_upper     = np.zeros((nphi_wall_1, nz_wall_1))   # hits within rectancle
    hpd_wall_upper      = np.zeros((nphi_wall_1, nz_wall_1))   #  hit density 

    
    phistart_wall_lower = np.zeros((nphi_wall_1, nz_wall_1))
    phiend_wall_lower   = np.zeros((nphi_wall_1, nz_wall_1))
    zstart_wall_lower   = np.zeros((nphi_wall_1, nz_wall_1))
    zend_wall_lower     = np.zeros((nphi_wall_1, nz_wall_1))
    area_wall_lower     = np.zeros((nphi_wall_1, nz_wall_1))
    power_wall_lower    = np.zeros((nphi_wall_1, nz_wall_1))   # power to area elements, kW
    spd_wall_lower      = np.zeros((nphi_wall_1, nz_wall_1))   # surface power density to area elements, kW/m^2 
    hits_wall_lower     = np.zeros((nphi_wall_1, nz_wall_1))   # hits within rectancle
    hpd_wall_lower      = np.zeros((nphi_wall_1, nz_wall_1))   #  hit density 

    # inner wall
    
    phistart_iwall = np.zeros((nphi_wall_2, nz_wall_2))
    phiend_iwall   = np.zeros((nphi_wall_2, nz_wall_2))
    zstart_iwall   = np.zeros((nphi_wall_2, nz_wall_2))
    zend_iwall     = np.zeros((nphi_wall_2, nz_wall_2))
    area_iwall     = np.zeros((nphi_wall_2, nz_wall_2))
    power_iwall    = np.zeros((nphi_wall_2, nz_wall_2))   # power to area elements, kW
    spd_iwall      = np.zeros((nphi_wall_2, nz_wall_2))   # surface power density to area elements, kW/m^2 
    hits_iwall     = np.zeros((nphi_wall_2, nz_wall_2))   # hits within rectancle
    hpd_iwall      = np.zeros((nphi_wall_2, nz_wall_2))   #  hit density
    
    phistart_limiter_edge = np.zeros(ntf)
    phiend_limiter_edge   = np.zeros(ntf)
    zstart_limiter_edge   = np.zeros(ntf)
    zend_limiter_edge     = np.zeros(ntf)
    
    phistart_rf_edge = np.zeros(ntf)
    phiend_rf_edge   = np.zeros(ntf)
    zstart_rf_edge   = np.zeros(ntf)
    zend_rf_edge     = np.zeros(ntf)

    delta_phi_limiter     = phi_limiter_size /nphi_limiter
    delta_phi_tbl         = 2. * np.pi       / nphi_tbl
    delta_phi_wall        = 2. * np.pi       / nphi_wall_1
    delta_phi_rf          = phi_rf_size      / nphi_rf

    delta_z_limiter       = 2. * tall_limiter    / nz_limiter
    delta_z_tbl           = tall_tbl             / nz_tbl
    delta_z_wall          = tall_wall            / nz_wall_1
    delta_z_rf            = 2. * tall_rf_antenna / nz_rf

    #
    total_weight_energy_lost = np.sum(weight_energy_lost)
    weight_eloss_norm        = weight_energy_lost/total_weight_energy_lost

    print("   ... position 1 in surface_power_density_2 \n")
          
    # toroidal belt limiters
    print("   ... about to start toroidal belt limiters")
    for jphi in range(nphi_tbl):
        for jz in range(nz_tbl):
            
            phistart_tbl_lower[jphi,jz] = 2. * np.pi *  jphi    / nphi_tbl
            phiend_tbl_lower[jphi,jz]   = 2. * np.pi * (jphi+1) / nphi_tbl

            phistart_tbl_upper[jphi,jz] = 2. * np.pi *  jphi    / nphi_tbl
            phiend_tbl_upper[jphi,jz]   = 2. * np.pi * (jphi+1) / nphi_tbl  

            zstart_tbl_lower[jphi,jz] = -1.*(tall_limiter + tall_tbl) + jz     * tall_tbl/nz_tbl
            zend_tbl_lower[jphi,jz]   = -1.*(tall_limiter + tall_tbl) + (jz+1) * tall_tbl/nz_tbl

            zstart_tbl_upper[jphi,jz] = tall_limiter +  jz    * tall_tbl/nz_tbl
            zend_tbl_upper[jphi,jz]   = tall_limiter  + (jz+1) * tall_tbl/nz_tbl

            #  compute area of this rectangle
            
            zmid_upper = (zstart_tbl_upper[jphi,jz] + zend_tbl_upper[jphi,jz]) / 2.
            zmid_lower = (zstart_tbl_lower[jphi,jz] + zend_tbl_lower[jphi,jz]) / 2.

            rmid_upper = tri.interpolate_rmajor_wall(rr_outer, zz_outer, zmid_upper)
            rmid_lower = tri.interpolate_rmajor_wall(rr_outer, zz_outer, zmid_lower)
            
            delta_z_lower   = zend_tbl_lower[jphi,jz]   -   zstart_tbl_lower[jphi,jz]
            delta_phi_lower = phiend_tbl_lower[jphi,jz] - phistart_tbl_lower[jphi,jz]

            delta_z_upper   = zend_tbl_upper[jphi,jz]   -   zstart_tbl_upper[jphi,jz]
            delta_phi_upper = phiend_tbl_upper[jphi,jz] - phistart_tbl_upper[jphi,jz]

            area_tbl_upper[jphi,jz] = rmid_upper * delta_phi_upper * delta_z_upper
            area_tbl_lower[jphi,jz] = rmid_lower * delta_phi_lower * delta_z_lower
            
            ii_upper = (rloss > rmin_outer) & (zloss > zstart_tbl_upper[jphi,jz]) & (zloss <zend_tbl_upper[jphi,jz]) \
                         & (philoss > phistart_tbl_upper[jphi,jz]) & (philoss < phiend_tbl_upper[jphi,jz])

            ii_lower = (rloss > rmin_outer) & (zloss > zstart_tbl_lower[jphi,jz]) & (zloss <zend_tbl_lower[jphi,jz]) \
                         & (philoss > phistart_tbl_lower[jphi,jz]) & (philoss < phiend_tbl_lower[jphi,jz])
            #pdb.set_trace()
            power_tbl_upper[jphi,jz] = np.sum(weight_eloss_norm[ii_upper]) * ploss_wall_kw
            power_tbl_lower[jphi,jz] = np.sum(weight_eloss_norm[ii_lower]) * ploss_wall_kw

            spd_tbl_upper[jphi,jz] = power_tbl_upper[jphi,jz] / area_tbl_upper[jphi,jz]
            spd_tbl_lower[jphi,jz] = power_tbl_lower[jphi,jz] / area_tbl_lower[jphi,jz]

            hits_tbl_upper[jphi,jz] = np.sum(ii_upper)
            hits_tbl_lower[jphi,jz] = np.sum(ii_lower)

            hpd_tbl_upper[jphi,jz] = hits_tbl_upper[jphi,jz] / area_tbl_upper[jphi,jz]
            hpd_tbl_lower[jphi,jz] = hits_tbl_lower[jphi,jz] / area_tbl_lower[jphi,jz]     

            #  outer wall
    print("   ... about to start outer walls")

    for jphi in range(nphi_wall_1):
        for jz in range(nz_wall_1):
            
            phistart_wall_lower[jphi,jz] = 2. * np.pi *  jphi    / nphi_wall_1
            phiend_wall_lower[jphi,jz]   = 2. * np.pi * (jphi+1) / nphi_wall_1

            phistart_wall_upper[jphi,jz] = 2. * np.pi *  jphi    / nphi_wall_1
            phiend_wall_upper[jphi,jz]   = 2. * np.pi * (jphi+1) / nphi_wall_1

            zstart_wall_upper[jphi,jz] =  tall_limiter + tall_tbl + tall_wall *  jz    / nz_wall_1
            zend_wall_upper[jphi,jz]   =  tall_limiter + tall_tbl + tall_wall * (jz+1) / nz_wall_1

            zstart_wall_lower[jphi,jz] =  -1. * zwall_max + tall_wall *  jz    / nz_wall_1
            zend_wall_lower[jphi,jz]   =  -1. * zwall_max + tall_wall * (jz+1) / nz_wall_1

            zmid_upper = (zstart_wall_upper[jphi,jz] + zend_wall_upper[jphi,jz]) / 2.
            zmid_lower = (zstart_wall_lower[jphi,jz] + zend_wall_lower[jphi,jz]) / 2.

            rmid_upper = tri.interpolate_rmajor_wall(rr_outer, zz_outer, zmid_upper)
            rmid_lower = tri.interpolate_rmajor_wall(rr_outer, zz_outer, zmid_lower)
            
            delta_z_lower   = zend_wall_lower[jphi,jz]   -   zstart_wall_lower[jphi,jz]
            delta_phi_lower = phiend_wall_lower[jphi,jz] - phistart_wall_lower[jphi,jz]

            delta_z_upper   = zend_wall_upper[jphi,jz]   -   zstart_wall_upper[jphi,jz]
            delta_phi_upper = phiend_wall_upper[jphi,jz] - phistart_wall_upper[jphi,jz]

            area_wall_upper[jphi,jz] = rmid_upper * delta_phi_upper * delta_z_upper
            area_wall_lower[jphi,jz] = rmid_lower * delta_phi_lower * delta_z_lower
            
            ii_upper = (rloss > rmin_outer) & (zloss > zstart_wall_upper[jphi,jz]) & (zloss < zend_wall_upper[jphi,jz]) \
                         & (philoss > phistart_wall_upper[jphi,jz]) & (philoss < phiend_wall_upper[jphi,jz])

            ii_lower = (rloss > rmin_outer) & (zloss > zstart_wall_lower[jphi,jz]) & (zloss < zend_wall_lower[jphi,jz]) \
                         & (philoss > phistart_wall_lower[jphi,jz]) & (philoss < phiend_wall_lower[jphi,jz])

            power_wall_upper[jphi,jz] = np.sum(weight_eloss_norm[ii_upper]) * ploss_wall_kw
            power_wall_lower[jphi,jz] = np.sum(weight_eloss_norm[ii_lower]) * ploss_wall_kw

            spd_wall_upper[jphi,jz] = power_wall_upper[jphi,jz] / area_wall_upper[jphi,jz]
            spd_wall_lower[jphi,jz] = power_wall_lower[jphi,jz] / area_wall_lower[jphi,jz]

            hits_wall_upper[jphi,jz] = np.sum(ii_upper)
            hits_wall_lower[jphi,jz] = np.sum(ii_lower)

            hpd_wall_upper[jphi,jz] = hits_wall_upper[jphi,jz] / area_wall_upper[jphi,jz]
            hpd_wall_lower[jphi,jz] = hits_wall_lower[jphi,jz] / area_wall_lower[jphi,jz]

    #  inner wall
    print("   ... about to start inner wall")
    for jphi in range(nphi_wall_2):
        for jz in range(nz_wall_2):
            
            phistart_iwall[jphi,jz] = 2. * np.pi *  jphi    / nphi_wall_2
            phiend_iwall[jphi,jz]   = 2. * np.pi * (jphi+1) / nphi_wall_2

            zstart_iwall[jphi,jz] =  -1. * zwall_inner_max + 2. * zwall_inner_max *  jz    / nz_wall_2
            zend_iwall[jphi,jz]   =  -1. * zwall_inner_max + 2. * zwall_inner_max * (jz+1) / nz_wall_2

            zmid = (zstart_iwall[jphi,jz] + zend_iwall[jphi,jz]) / 2.

            rmid = tri.interpolate_rmajor_wall(rr_inner, zz_inner, zmid)
            
            delta_z   = zend_iwall[jphi,jz]   -   zstart_iwall[jphi,jz]
            delta_phi = phiend_iwall[jphi,jz] - phistart_iwall[jphi,jz]

            area_iwall[jphi,jz] = rmid * delta_phi * delta_z
            
            ii_inner = (rloss < rmin_outer) & (zloss > zstart_iwall[jphi,jz]) & (zloss < zend_iwall[jphi,jz]) \
                         & (philoss > phistart_iwall[jphi,jz]) & (philoss < phiend_iwall[jphi,jz])

            power_iwall[jphi,jz] = np.sum(weight_eloss_norm[ii_inner]) * ploss_wall_kw

            spd_iwall[jphi,jz] = power_iwall[jphi,jz] / area_iwall[jphi,jz]

            hits_iwall[jphi,jz] = np.sum(ii_inner)
        
            hpd_iwall[jphi,jz] = hits_iwall[jphi,jz] / area_iwall[jphi,jz]
            
    #  limiters
    print("  ... about to start limiters")
    for jtf in range(ntf):
        
        phistart_limiter_edge[jtf]      =  -phi_limiter_size/2 + jtf * phi_sector_size
        phiend_limiter_edge[jtf]        =  phistart_limiter_edge[jtf] + phi_limiter_size
        
        zstart_limiter_edge[jtf]        =  -tall_limiter
        zend_limiter_edge[jtf]          =  tall_limiter

        for jphi in range(nphi_limiter):
            
            for jz in range(nz_limiter):
                
                phistart_limiter[jtf, jphi,jz] = phistart_limiter_edge[jtf]  + (phiend_limiter_edge[jtf]- phistart_limiter_edge[jtf]) * jphi/ nphi_limiter
                phiend_limiter[jtf,jphi,jz]    = phistart_limiter[jtf, jphi,jz] + (phiend_limiter_edge[jtf]- phistart_limiter_edge[jtf])       / nphi_limiter

                zstart_limiter[jtf,jphi,jz] = zstart_limiter_edge[jtf] + (zend_limiter_edge[jtf]- zstart_limiter_edge[jtf]) * jz / nz_limiter
                zend_limiter[jtf,jphi,jz]   = zstart_limiter[jtf, jphi,jz] + (zend_limiter_edge[jtf]- zstart_limiter_edge[jtf])       / nz_limiter

                zmid  = (zstart_limiter[jtf, jphi,jz] + zend_limiter[jtf, jphi,jz]) / 2.
                rmid  = tri.interpolate_rmajor_wall(rr_outer, zz_outer, zmid)

                delta_z   = zend_limiter[jtf, jphi,jz]  -   zstart_limiter[jtf,jphi,jz]
                delta_phi = phiend_limiter[jtf,jphi,jz] - phistart_limiter[jtf,jphi,jz]

                area_limiter[jtf,jphi,jz] = rmid * delta_phi * delta_z
            
                ii = (rloss > rmin_outer) & (zloss > zstart_limiter[jtf,jphi,jz]) & (zloss < zend_limiter[jtf,jphi,jz]) \
                         & (philoss > phistart_limiter[jtf,jphi,jz] ) & (philoss < phiend_limiter[jtf,jphi,jz])

                power_limiter[jtf,jphi,jz] = np.sum(weight_eloss_norm[ii]) * ploss_wall_kw
                spd_limiter[jtf,jphi,jz] = power_limiter[jtf,jphi,jz] / area_limiter[jtf,jphi,jz]

                hits_limiter[jtf,jphi,jz] = np.sum(ii)
                hpd_limiter[jtf,jphi,jz] = hits_limiter[jtf,jphi,jz] / area_limiter[jtf,jphi,jz]
 
                
    # rf antennas
    print("   ... about to start rf antennas")
    for jtf in range(ntf):
        
        phistart_rf_edge[jtf]      =  phi_limiter_size/2 + jtf * phi_sector_size
        phiend_rf_edge[jtf]        =  phistart_rf_edge[jtf] + phi_rf_size
        
        zstart_rf_edge[jtf]        =  -tall_rf_antenna
        zend_rf_edge[jtf]          =  tall_rf_antenna

        for jphi in range(nphi_rf):
            
            for jz in range(nz_rf):
                
                phistart_rf[jtf, jphi,jz] = phistart_rf_edge[jtf]  + (phiend_rf_edge[jtf]- phistart_rf_edge[jtf]) * jphi/ nphi_rf
                phiend_rf[jtf,jphi,jz]    = phistart_rf[jtf, jphi,jz] + (phiend_rf_edge[jtf]- phistart_rf_edge[jtf])       / nphi_rf

                zstart_rf[jtf,jphi,jz] = zstart_rf_edge[jtf] + (zend_rf_edge[jtf]- zstart_rf_edge[jtf]) * jz / nz_rf
                zend_rf[jtf,jphi,jz]   = zstart_rf[jtf, jphi,jz] + (zend_rf_edge[jtf]- zstart_rf_edge[jtf])  / nz_rf

                zmid = (zstart_rf[jtf, jphi,jz] + zend_rf[jtf, jphi,jz]) / 2.
                rmid = tri.interpolate_rmajor_wall(rr_outer, zz_outer, zmid)

                delta_z   = zend_rf[jtf, jphi,jz]  -   zstart_rf[jtf,jphi,jz]
                delta_phi = phiend_rf[jtf,jphi,jz] - phistart_rf[jtf,jphi,jz]

                area_rf[jtf, jphi,jz] = rmid * delta_phi * delta_z
                #pdb.set_trace()
                ii = (rloss > rmin_outer) & (zloss > zstart_rf[jtf,jphi,jz]) & (zloss < zend_rf[jtf,jphi,jz]) \
                         & (philoss > phistart_rf[jtf,jphi,jz] ) & (philoss < phiend_rf[jtf,jphi,jz])

                power_rf[jtf,jphi,jz] = np.sum(weight_eloss_norm[ii]) * ploss_wall_kw
                spd_rf[jtf,jphi,jz] = power_rf[jtf,jphi,jz] / area_rf[jtf,jphi,jz]

                hits_rf[jtf,jphi,jz] = np.sum(ii)
                hpd_rf[jtf,jphi,jz] = hits_rf[jtf,jphi,jz] / area_rf[jtf,jphi,jz]

     
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #  check that our geometry of components is correct
    #pdb.set_trace()
    
    plt.close()
    
    plt.figure(figsize=(9.,6.))
    fig,ax=plt.subplots()
    plt.xlim(-0.1, 2.*np.pi+0.1)
    plt.ylim(-1.25, 1.25)



    # rf antennas
    
    for jtf in range(ntf):
        for jphi in range(nphi_rf):
            for jz in range(nz_rf):
                
                delta_phi = phiend_rf[jtf,jphi,jz] - phistart_rf[jtf,jphi,jz]
                delta_z   = zend_rf[jtf,jphi,jz]   - zstart_rf[jtf,jphi,jz]
                rect = Rectangle( (phistart_rf[jtf,jphi,jz],zstart_rf[jtf,jphi,jz]),delta_phi, delta_z, facecolor='none', edgecolor='g', linewidth=0.25)
                ax.add_patch(rect)

    # poloidal limiters
    
    for jtf in range(ntf):
        for jphi in range(nphi_limiter):
            for jz in range(nz_limiter):

                delta_phi = phiend_limiter[jtf,jphi,jz] - phistart_limiter[jtf,jphi,jz]
                delta_z   = zend_limiter[jtf,jphi,jz]   - zstart_limiter[jtf,jphi,jz]
                
                rect = Rectangle( (phistart_limiter[jtf,jphi,jz],zstart_limiter[jtf,jphi,jz]),delta_phi, delta_z, facecolor='none', edgecolor='r', linewidth=0.25)
                ax.add_patch(rect)
                
    #  tbl limiters
    
    for jphi in range(nphi_tbl):
        for jz in range(nz_tbl):

           delta_phi_lower = phiend_tbl_lower[jphi,jz] - phistart_tbl_lower[jphi,jz]
           delta_z_lower   = zend_tbl_lower[jphi,jz]   - zstart_tbl_lower[jphi,jz]

           delta_phi_upper = phiend_tbl_upper[jphi,jz] - phistart_tbl_upper[jphi,jz]
           delta_z_upper   = zend_tbl_upper[jphi,jz]   - zstart_tbl_upper[jphi,jz]
           
           rect = Rectangle( (phistart_tbl_lower[jphi,jz],zstart_tbl_lower[jphi,jz]),delta_phi_lower, delta_z_lower, facecolor='none', edgecolor='b', linewidth=0.25)
           ax.add_patch(rect)
    
           rect = Rectangle( (phistart_tbl_upper[jphi,jz],zstart_tbl_upper[jphi,jz]),delta_phi_upper, delta_z_upper, facecolor='none', edgecolor='b', linewidth=0.25)
           ax.add_patch(rect)

    # outer walls

    for jphi in range(nphi_wall_1):
        for jz in range(nz_wall_1):

            delta_phi_lower = phiend_wall_lower[jphi,jz] - phistart_wall_lower[jphi,jz]
            delta_z_lower   = zend_wall_lower[jphi,jz]   - zstart_wall_lower[jphi,jz]

            delta_phi_upper = phiend_wall_upper[jphi,jz] - phistart_wall_upper[jphi,jz]
            delta_z_upper   = zend_wall_upper[jphi,jz]   - zstart_wall_upper[jphi,jz]
            
            rect = Rectangle( (phistart_wall_lower[jphi,jz],zstart_wall_lower[jphi,jz]),delta_phi_lower, delta_z_lower, facecolor='none', edgecolor='k', linewidth=0.25)
            ax.add_patch(rect)

            rect = Rectangle( (phistart_wall_upper[jphi,jz],zstart_wall_upper[jphi,jz]),delta_phi_upper, delta_z_upper, facecolor='none', edgecolor='k', linewidth=0.25)
            ax.add_patch(rect) 

    plt.title(' limiters and walls (starting surface power density)', fontsize=10)
    plt.xlabel('phi [radians]')
    plt.ylabel('Z [m]')
    #plt.tight_layout(pad=padsize)
    pdf.savefig()
    #plt.savefig('validate_surface.pdf')
    plt.close()


    # ++++++++++++++++++++++++++++++++++++++++++++++++
    #

    mm_limiter = ntf * nphi_limiter * nz_limiter
    mm_rf      = ntf * nphi_rf      * nz_rf
    mm_tbl     = nphi_tbl * nz_tbl
    mm_wall_1  = nphi_wall_1 * nz_wall_1
    mm_wall_2  = nphi_wall_2 * nz_wall_2

    spd_limiter_1d    = np.reshape(spd_limiter, mm_limiter)
    spd_rf_1d         = np.reshape(spd_rf,      mm_rf)
    spd_tbl_upper_1d  = np.reshape(spd_tbl_upper, mm_tbl)
    spd_tbl_lower_1d  = np.reshape(spd_tbl_lower, mm_tbl)
    spd_wall_upper_1d = np.reshape(spd_wall_upper, mm_wall_1)
    spd_wall_lower_1d = np.reshape(spd_wall_lower, mm_wall_1)
    spd_iwall_1d      = np.reshape(spd_iwall, mm_wall_2)

    spd_all_1d = np.concatenate((spd_limiter_1d, spd_rf_1d, spd_tbl_upper_1d, spd_tbl_lower_1d,
                                 spd_wall_upper_1d, spd_wall_lower_1d, spd_iwall_1d))

    # -----------------------------------------------------------------
        
    plt.close()
    plt.figure(figsize=(8.,6.))
    plt.hist(spd_limiter_1d, bins=40, histtype='step', rwidth=1.,color='k',log=True)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Surface power density on poloidal limiters')
    plt.xlabel('kW/m2')
    plt.ylabel('')
    plt.ylim(bottom=0.5)
    plt.tight_layout(pad=2)
    pdf.savefig()
    plt.close()

    plt.close()
    plt.figure(figsize=(8.,6.))
    plt.hist(spd_tbl_upper_1d, bins=40, histtype='step', rwidth=1.,color='k',log=True)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Surface power density on upper TBL')
    plt.xlabel('kW/m2')
    plt.ylabel('')
    plt.ylim(bottom=0.5)
    plt.tight_layout(pad=2)
    pdf.savefig()
    plt.close()

    plt.close()
    plt.figure(figsize=(8.,6.))
    plt.hist(spd_wall_upper_1d, bins=40, histtype='step', rwidth=1.,color='k',log=True)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Surface power density on upper wall')
    plt.xlabel('kW/m2')
    plt.ylabel('')
    plt.ylim(bottom=0.5)
    plt.tight_layout(pad=2)
    pdf.savefig()
    plt.close()

    plt.close()
    plt.figure(figsize=(8.,6.))
    plt.hist(spd_all_1d, bins=50, histtype='step', rwidth=1.,color='k',log=True)
    plt.grid(axis='y', alpha=0.75)
    plt.title('Surface power density on all surfaces')
    plt.xlabel('kW/m2')
    plt.ylabel('')
    plt.ylim(bottom=0.5)
    plt.tight_layout(pad=2)
    pdf.savefig()
    plt.close()
        
        
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #
    #    plot surface power density

    spd_tbl_upper_max  = np.max(spd_tbl_upper)
    spd_tbl_lower_max  = np.max(spd_tbl_lower)
    spd_limiter_max    = np.max(spd_limiter)
    spd_rf_max         = np.max(spd_rf)
    spd_tbl_upper_max  = np.max(spd_tbl_upper)
    spd_tbl_lower_max  = np.max(spd_tbl_lower)
    spd_wall_upper_max = np.max(spd_wall_upper)
    spd_wall_lower_max = np.max(spd_wall_lower)
    spd_iwall_max = np.max(spd_iwall)

    kk_limiter    = np.where(spd_limiter    == np.amax(spd_limiter))
    kk_rf         = np.where(spd_rf         == np.amax(spd_rf))
    kk_tbl_upper  = np.where(spd_tbl_upper  == np.amax(spd_tbl_upper))
    kk_tbl_lower  = np.where(spd_tbl_lower  == np.amax(spd_tbl_lower))
    kk_wall_upper = np.where(spd_wall_upper == np.amax(spd_wall_upper))
    kk_wall_lower = np.where(spd_wall_lower == np.amax(spd_wall_lower))
    kk_iwall      = np.where(spd_iwall      == np.amax(spd_iwall))

    hits_limiter_max    = int(hits_limiter[kk_limiter][0])
    hits_rf_max         = int(hits_rf[kk_rf][0])
    hits_tbl_upper_max  = int(hits_tbl_upper[kk_tbl_upper][0])
    hits_tbl_lower_max  = int(hits_tbl_lower[kk_tbl_lower][0])
    hits_wall_upper_max = int(hits_wall_upper[kk_wall_upper][0])
    hits_wall_lower_max = int(hits_wall_lower[kk_wall_lower][0])
    hits_iwall_max      = int(hits_iwall[kk_iwall][0])
    
    error_limiter       = 100./np.sqrt(hits_limiter_max)
    error_rf            = 100./np.sqrt(hits_rf_max)
    error_tbl_upper     = 100./np.sqrt(hits_tbl_upper_max)
    error_tbl_lower     = 100./np.sqrt(hits_tbl_lower_max)
    error_wall_upper    = 100./np.sqrt(hits_wall_upper_max)
    error_wall_lower    = 100./np.sqrt(hits_wall_lower_max)
    error_iwall         = 100./np.sqrt(hits_iwall_max)

    
    
    spd_max = np.max([spd_limiter_max,spd_rf_max, spd_tbl_upper_max, spd_tbl_lower_max, spd_wall_upper_max, spd_wall_lower_max, spd_iwall_max])

    print("\n ++++++++++++++++++++++++++++++ \n maximum surface power density \n")
    print("  limiter      %8.2f "%spd_limiter_max)
    print("  rf antennas  %8.2f "%spd_rf_max)
    print("  tbl_upper    %8.2f "%spd_tbl_upper_max)
    print("  tbl_lower    %8.2f "%spd_tbl_lower_max)
    print("  wall_upper   %8.2f "%spd_wall_upper_max)
    print("  wall_lower   %8.2f "%spd_wall_lower_max)
    print("  wall_inner   %8.2f "%spd_iwall_max)
    print("\n maximum surface power density: %6.1f  kW/m2"%spd_max)

    print("\n ++++++++++++++++++++++++++++++ \n hits at maximum surface power density and percent error \n")

    print("  limiter      %6d %5.1f"%(hits_limiter_max, error_limiter))
    print("  rf antennas  %6d %5.1f"%(hits_rf_max, error_rf))
    print("  tbl_upper    %6d %5.1f"%(hits_tbl_upper_max, error_tbl_upper))
    print("  tbl_lower    %6d %5.1f"%(hits_tbl_lower_max, error_tbl_lower))
    print("  wall_upper   %6d %5.1f"%(hits_wall_upper_max, error_wall_upper))
    print("  wall_lower   %6d %5.1f"%(hits_wall_lower_max, error_wall_lower))
    print("  wall_inner   %6d %5.1f"%(hits_iwall_max, error_iwall))


    spd_limiter_norm      = spd_limiter    / spd_limiter_max
    spd_rf_norm           = spd_rf         / spd_rf_max
    spd_wall_upper_norm   = spd_wall_upper / spd_wall_upper_max
    spd_wall_lower_norm   = spd_wall_lower / spd_wall_lower_max
    spd_iwall_norm        = spd_iwall      / spd_iwall_max
    spd_tbl_upper_norm    = spd_tbl_upper  / spd_tbl_upper_max
    spd_tbl_lower_norm    = spd_tbl_lower  / spd_tbl_lower_max

   #  4/27/2021:  consistent normalization
    
    spd_max_max = np.max([spd_limiter_max, spd_rf_max, spd_wall_upper_max, spd_wall_lower_max,spd_iwall_max  ])
    
    spd_limiter_norm      = spd_limiter    / spd_max_max  # (1.e-10 + spd_limiter_max)
    spd_rf_norm           = spd_rf         / spd_max_max  # (1.e-10 + spd_rf_max)
    spd_wall_upper_norm   = spd_wall_upper / spd_max_max  # (1.e-10 + spd_wall_upper_max)
    spd_wall_lower_norm   = spd_wall_lower / spd_max_max  # (1.e-10 + spd_wall_lower_max)
    spd_iwall_norm        = spd_iwall      / spd_max_max  # (1.e-10 + spd_iwall_max)
    spd_tbl_upper_norm    = spd_tbl_upper  / spd_max_max  # (1.e-10 + spd_tbl_upper_max)
    spd_tbl_lower_norm    = spd_tbl_lower  / spd_max_max  # (1.e-10 + spd_tbl_lower_max)

    plt.close()
    
    plt.figure(figsize=(9.,7.))
    fig,ax=plt.subplots()
    plt.xlim(-0.1, 2.*np.pi+0.1)
    plt.ylim(-1.25, 1.25)


    # rf antennas
    
    for jtf in range(ntf):
        for jphi in range(nphi_rf):
            for jz in range(nz_rf):
                
                delta_phi = phiend_rf[jtf,jphi,jz] - phistart_rf[jtf,jphi,jz]
                delta_z   = zend_rf[jtf,jphi,jz]   - zstart_rf[jtf,jphi,jz]
                rect = Rectangle( (phistart_rf[jtf,jphi,jz],zstart_rf[jtf,jphi,jz]),delta_phi, delta_z, facecolor=my_colormap(spd_rf_norm[jtf,jphi,jz]), edgecolor='none', linewidth=0.25)
                ax.add_patch(rect)

    # poloidal limiters
    
    for jtf in range(ntf):
        for jphi in range(nphi_limiter):
            for jz in range(nz_limiter):

                delta_phi = phiend_limiter[jtf,jphi,jz] - phistart_limiter[jtf,jphi,jz]
                delta_z   = zend_limiter[jtf,jphi,jz]   - zstart_limiter[jtf,jphi,jz]
                
                rect = Rectangle( (phistart_limiter[jtf,jphi,jz],zstart_limiter[jtf,jphi,jz]),delta_phi, delta_z, facecolor=my_colormap(spd_limiter_norm[jtf,jphi,jz]), edgecolor='none', linewidth=0.25)
                ax.add_patch(rect)
                
    #  tbl limiters
    
    for jphi in range(nphi_tbl):
        for jz in range(nz_tbl):

           delta_phi_lower = phiend_tbl_lower[jphi,jz] - phistart_tbl_lower[jphi,jz]
           delta_z_lower   = zend_tbl_lower[jphi,jz]   - zstart_tbl_lower[jphi,jz]

           delta_phi_upper = phiend_tbl_upper[jphi,jz] - phistart_tbl_upper[jphi,jz]
           delta_z_upper   = zend_tbl_upper[jphi,jz]   - zstart_tbl_upper[jphi,jz]
           
           rect = Rectangle( (phistart_tbl_lower[jphi,jz],zstart_tbl_lower[jphi,jz]),delta_phi_lower, delta_z_lower, facecolor=my_colormap(spd_tbl_lower_norm[jphi,jz]), edgecolor='none', linewidth=0.25)
           ax.add_patch(rect)
    
           rect = Rectangle( (phistart_tbl_upper[jphi,jz],zstart_tbl_upper[jphi,jz]),delta_phi_upper, delta_z_upper, facecolor=my_colormap(spd_tbl_upper_norm[jphi,jz]), edgecolor='none', linewidth=0.25)
           ax.add_patch(rect)

    # outer walls

    for jphi in range(nphi_wall_1):
        for jz in range(nz_wall_1):

            delta_phi_lower = phiend_wall_lower[jphi,jz] - phistart_wall_lower[jphi,jz]
            delta_z_lower   = zend_wall_lower[jphi,jz]   - zstart_wall_lower[jphi,jz]

            delta_phi_upper = phiend_wall_upper[jphi,jz] - phistart_wall_upper[jphi,jz]
            delta_z_upper   = zend_wall_upper[jphi,jz]   - zstart_wall_upper[jphi,jz]
            
            rect = Rectangle( (phistart_wall_lower[jphi,jz],zstart_wall_lower[jphi,jz]),delta_phi_lower, delta_z_lower, facecolor=my_colormap(spd_wall_lower_norm[jphi,jz]), edgecolor='none', linewidth=0.25)
            ax.add_patch(rect)

            rect = Rectangle( (phistart_wall_upper[jphi,jz],zstart_wall_upper[jphi,jz]),delta_phi_upper, delta_z_upper, facecolor=my_colormap(spd_wall_upper_norm[jphi,jz]), edgecolor='none', linewidth=0.25)
            ax.add_patch(rect) 

            for jtf in range(ntf):

                delta_phi_limiter = phiend_limiter_edge[jtf] - phistart_limiter_edge[jtf]
                delta_z_limiter   = zend_limiter_edge[jtf] - zstart_limiter_edge[jtf]
                rect = Rectangle( (phistart_limiter_edge[jtf],zstart_limiter_edge[jtf]),delta_phi_limiter, delta_z_limiter, facecolor='none', edgecolor='k', linewidth=0.25)
                ax.add_patch(rect)

                delta_phi_rf = phiend_rf_edge[jtf] - phistart_rf_edge[jtf]
                delta_z_rf   = zend_rf_edge[jtf] - zstart_rf_edge[jtf]
                rect = Rectangle( (phistart_rf_edge[jtf],zstart_rf_edge[jtf]),delta_phi_rf, delta_z_rf, facecolor='none', edgecolor='k', linewidth=0.25)
                ax.add_patch(rect)

            rect = Rectangle( (0.,tall_limiter),2.*np.pi, tall_tbl, facecolor='none', edgecolor='k', linewidth=0.25)
            ax.add_patch(rect)

            rect = Rectangle( (0.,-1.*(tall_limiter + tall_tbl)),2.*np.pi, tall_tbl, facecolor='none', edgecolor='k', linewidth=0.25)
            ax.add_patch(rect)

            rect = Rectangle( (0.,(tall_limiter+tall_tbl)),2.*np.pi, tall_wall, facecolor='none', edgecolor='k', linewidth=0.25)
            ax.add_patch(rect)

            rect = Rectangle( (0.,-1.*(tall_limiter+tall_tbl+tall_wall)),2.*np.pi, tall_wall, facecolor='none', edgecolor='k', linewidth=0.25)
            ax.add_patch(rect)
    
    my_title2 = "surface power density (max = %6.1f)  kw/m2"%(spd_max)                    
    plt.title(my_title2, fontsize=10)
    #plt.colorbar()
    plt.xlabel('phi [radians]')
    plt.ylabel('Z [m]')
    #plt.tight_layout(pad=padsize)
    pdf.savefig()
    #plt.savefig('surface_heating.pdf')
    plt.close()

    plt.figure(figsize=(8.,6.))
    mat = np.random.random((30,30))
    plt.imshow(mat, origin="lower", cmap='plasma', interpolation='nearest')
    plt.colorbar()
    pdf.savefig()

    # +++++++++++++++++++++++++++++++++++++++
    #  return to per-component normalization
    
    spd_limiter_norm      = spd_limiter    / (1.e-10 + spd_limiter_max)
    spd_rf_norm           = spd_rf         / (1.e-10 + spd_rf_max)
    spd_wall_upper_norm   = spd_wall_upper / (1.e-10 + spd_wall_upper_max)
    spd_wall_lower_norm   = spd_wall_lower / (1.e-10 + spd_wall_lower_max)
    spd_iwall_norm        = spd_iwall      / (1.e-10 + spd_iwall_max)
    spd_tbl_upper_norm    = spd_tbl_upper  / (1.e-10 + spd_tbl_upper_max)
    spd_tbl_lower_norm    = spd_tbl_lower  / (1.e-10 + spd_tbl_lower_max)
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   individual poloidal limiters
    
    spd_limiter_maxs = np.zeros(ntf)
    spd_limiter_peaks = np.zeros(ntf)
    spd_limiter_means = np.zeros(ntf)
    
    for jtf in range(ntf):

        delta_phi_limiter = phiend_limiter_edge[jtf] - phistart_limiter_edge[jtf]
        delta_z_limiter   = zend_limiter_edge[jtf] - zstart_limiter_edge[jtf]
        rect = Rectangle( (phistart_limiter_edge[jtf],zstart_limiter_edge[jtf]),delta_phi_limiter, delta_z_limiter, facecolor='none', edgecolor='k', linewidth=0.25)
        ax.add_patch(rect)
        
        spd_max_local = np.max(spd_limiter[jtf,:,:])
        spd_limiter_local_norm = spd_limiter[jtf,:,:] / spd_max_local  # no longer used
        kw_local = np.sum(power_limiter[jtf,:,:])
        peak_local = spd_max_local / np.mean(spd_limiter[jtf,:,:])

        spd_limiter_maxs[jtf]  = spd_max_local
        spd_limiter_peaks[jtf] = peak_local
        spd_limiter_means[jtf] = np.mean(spd_limiter[jtf,:,:])
        
        print("   ... poloidal limiter %2d : max surface power density = %f7.2"%(jtf, spd_max_local))
        plt.close('all')
        plt.figure(figsize=(9.,7.))
        fig,ax=plt.subplots()
        plt.ylim(-0.55, 0.55)

        my_phimin = np.min(phistart_limiter[jtf,:,:]) * 180. / np.pi - 0.5
        my_phimax = np.max(phiend_limiter[jtf,:,:])   * 180. / np.pi + 0.5
        
        for jphi in range(nphi_limiter):
            for jz in range(nz_limiter):

                delta_phi = phiend_limiter[jtf,jphi,jz] - phistart_limiter[jtf,jphi,jz]
                delta_z   = zend_limiter[jtf,jphi,jz]   - zstart_limiter[jtf,jphi,jz]
                
                rect = Rectangle( (phistart_limiter[jtf,jphi,jz]*180./np.pi,zstart_limiter[jtf,jphi,jz]),delta_phi*180./np.pi, delta_z, facecolor=my_colormap(spd_limiter_norm[jtf,jphi,jz]), edgecolor='none', linewidth=0.25)
                ax.add_patch(rect)
                #pdb.set_trace()

                delta_phi_limiter = (phiend_limiter_edge[jtf] - phistart_limiter_edge[jtf])
                delta_z_limiter   = zend_limiter_edge[jtf] - zstart_limiter_edge[jtf]
                rect = Rectangle( (phistart_limiter_edge[jtf]*180./np.pi,zstart_limiter_edge[jtf]),delta_phi_limiter*180./np.pi, delta_z_limiter, facecolor='none', edgecolor='k', linewidth=0.25)
                ax.add_patch(rect)
                
        my_title3 = "limiter %2d  %6.1f kw spd-norm  (max spd = %6.1f) peak-f: %5.2f"%(jtf,kw_local,spd_max_local, peak_local)                    
        plt.title(my_title3, fontsize=10)
        #plt.colorbar()
        plt.xlim((my_phimin, my_phimax))
        plt.ylim((-0.55,0.55))
        plt.xlabel('phi [degrees]')
        plt.ylabel('Z [m]')
        plt.tight_layout(pad=1.5)
        pdf.savefig()
        plt.close('all')
   # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   profiles of spd for poloidal limiters

    #  plot versus z along phi-slice for which spd reaches a maximum
    my_colors = ('r','orange','green','cyan','blue','violet','magenta','lime','teal')
    plt.close()
    plt.figure(figsize=(8.,6.))
    
    for jtf in range(ntf):

       spd_limiter_2d = spd_limiter[jtf,:,:]
       
       imax_phi       = np.argmax(spd_limiter_2d)//nz_limiter
       spd_slice_z    = spd_limiter_2d[imax_phi,:]
       zarray_local   = 0.5 * (zstart_limiter[jtf,imax_phi,:] + zend_limiter[jtf,imax_phi,:])

       icolor = jtf%(len(my_colors))
       this_color = my_colors[icolor]

       plt.plot(zarray_local, spd_slice_z, '-', color=this_color)

    plt.ylim(bottom=0.)
    plt.xlabel('Z [m]')
    plt.ylabel('kW/m2')
    plt.title('Limiter SPD along vertical slice at maximum spd')
    pdf.savefig()

    #  now as a function of phi

    plt.close()
    plt.figure(figsize=(8.,6.))
    
    for jtf in range(ntf):

       spd_limiter_2d = spd_limiter[jtf,:,:]
       
       imax_phi       = np.argmax(spd_limiter_2d)//nz_limiter
       imax_z         = np.argmax(spd_limiter_2d) - imax_phi * nz_limiter
       spd_slice_phi  = spd_limiter_2d[:,imax_z]
       phi_local      = (0.5 * (phistart_limiter[0,:,imax_z] + phiend_limiter[0,:,imax_z]))*180./np.pi

       icolor = jtf%(len(my_colors))
       this_color = my_colors[icolor]

       plt.plot(phi_local, spd_slice_phi, '-', color=this_color)
       #pdb.set_trace()
    plt.ylim(bottom=0.)
    plt.xlabel('phi [degrees]')
    plt.ylabel('kW/m2')
    plt.title('Limiter SPD along toroidal slice at maximum spd')
    pdf.savefig()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #   individual RF antennas

    spd_rf_maxs  = np.zeros(ntf)
    spd_rf_peaks = np.zeros(ntf)
    spd_rf_means = np.zeros(ntf)


    for jtf in range(ntf):

        delta_phi_rf = phiend_rf_edge[jtf] - phistart_rf_edge[jtf]
        delta_z_rf   = zend_rf_edge[jtf] - zstart_rf_edge[jtf]
        rect = Rectangle( (phistart_rf_edge[jtf],zstart_rf_edge[jtf]),delta_phi_rf, delta_z_rf, facecolor='none', edgecolor='k', linewidth=0.25)
        ax.add_patch(rect)
        
        spd_max_local           = np.max(spd_rf[jtf,:,:])
        spd_rf_local_norm       = spd_rf[jtf,:,:] / spd_max_local  # no longer used
        kw_local                = np.sum(power_rf[jtf,:,:])
        peak_local              = spd_max_local / np.mean(spd_rf[jtf,:,:])

        spd_rf_maxs[jtf]  = spd_max_local
        spd_rf_peaks[jtf] = peak_local
        spd_rf_means[jtf] = np.mean(spd_rf[jtf,:,:])

        print("   ... rf antenna %2d : max surface power density = %f7.2"%(jtf, spd_max_local))
        plt.close('all')
        plt.figure(figsize=(9.,7.))
        fig,ax=plt.subplots()
        plt.ylim((-tall_limiter-0.05,tall_limiter+0.05))

        my_phimin = np.min(phistart_rf[jtf,:,:]) * 180. / np.pi - 0.5
        my_phimax = np.max(phiend_rf[jtf,:,:])   * 180. / np.pi + 0.5
        
        for jphi in range(nphi_rf):
            for jz in range(nz_rf):

                delta_phi = phiend_rf[jtf,jphi,jz] - phistart_rf[jtf,jphi,jz]
                delta_z   = zend_rf[jtf,jphi,jz]   - zstart_rf[jtf,jphi,jz]
                
                rect = Rectangle( (phistart_rf[jtf,jphi,jz]*180./np.pi,zstart_rf[jtf,jphi,jz]),delta_phi*180./np.pi, delta_z, facecolor=my_colormap(spd_rf_norm[jtf,jphi,jz]), edgecolor='none', linewidth=0.25)
                ax.add_patch(rect)
                #pdb.set_trace()

                delta_phi_rf = (phiend_rf_edge[jtf] - phistart_rf_edge[jtf])
                delta_z_rf   = zend_rf_edge[jtf] - zstart_rf_edge[jtf]
                rect = Rectangle( (phistart_rf_edge[jtf]*180./np.pi,zstart_rf_edge[jtf]),delta_phi_rf*180./np.pi, delta_z_rf, facecolor='none', edgecolor='k', linewidth=0.25)
                ax.add_patch(rect)
                
        my_title3 = "rf antenna %2d  %6.1f kw spd-norm  (max spd = %6.1f) peak-f: %5.2f"%(jtf,kw_local,spd_max_local, peak_local)                    
        plt.title(my_title3, fontsize=10)
        #plt.colorbar()
        plt.xlim((my_phimin, my_phimax))
        plt.ylim((-tall_limiter-0.05,tall_limiter+0.05))
        plt.xlabel('phi [degrees]')
        plt.ylabel('Z [m]')
        plt.tight_layout(pad=1.5)
        pdf.savefig()
        plt.close('all')

        print("   maximum of spd_rf_norm = ", np.max(spd_rf_norm))
                
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #  upper tbl

    spd_max_local = np.max(spd_tbl_upper)

    plt.close()
    plt.figure(figsize=(9.,7.))
    fig,ax=plt.subplots()
    plt.xlim(-5, 365.)
    plt.ylim(0.48, 0.62)

    for jphi in range(nphi_tbl):
        for jz in range(nz_tbl):
    
           delta_phi_lower = phiend_tbl_lower[jphi,jz] - phistart_tbl_lower[jphi,jz]
           delta_z_lower   = zend_tbl_lower[jphi,jz]   - zstart_tbl_lower[jphi,jz]

           delta_phi_upper = phiend_tbl_upper[jphi,jz] - phistart_tbl_upper[jphi,jz]
           delta_z_upper   = zend_tbl_upper[jphi,jz]   - zstart_tbl_upper[jphi,jz]
        
           rect = Rectangle( (phistart_tbl_upper[jphi,jz]*180./np.pi,zstart_tbl_upper[jphi,jz]),delta_phi_upper*180./np.pi, delta_z_upper, facecolor=my_colormap(spd_tbl_upper_norm[jphi,jz]), edgecolor='none', linewidth=0.25)
           ax.add_patch(rect)

    rect = Rectangle( (0.,tall_limiter),360., tall_tbl, facecolor='none', edgecolor='k', linewidth=0.25)
    ax.add_patch(rect)
    #plt.colorbar()   
    my_title3 = "tbl  spd-norm  (max spd = %6.1f)"%(spd_max_local)                    
    plt.title(my_title3, fontsize=10)
    plt.xlabel('phi [degrees]')
    plt.ylabel('Z [m]')
    plt.tight_layout(pad=1.5)
    pdf.savefig()
    plt.close('all')
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #  hits density



    hpd_limiter_max    = np.max(hpd_limiter)
    hpd_rf_max         = np.max(hpd_rf)
    hpd_tbl_upper_max  = np.max(hpd_tbl_upper)
    hpd_tbl_lower_max  = np.max(hpd_tbl_lower)
    hpd_wall_upper_max = np.max(hpd_wall_upper)
    hpd_wall_lower_max = np.max(hpd_wall_lower)
    hpd_iwall_max      = np.max(hpd_iwall)
    
    hpd_max = np.max([hpd_limiter_max,hpd_rf_max, hpd_tbl_upper_max, hpd_tbl_lower_max, hpd_wall_upper_max, hpd_wall_lower_max, hpd_iwall_max])

    print("\n ++++++++++++++++++++++++++++++ \n maximum hits density")
    print("  limiter      %8.2f "%hpd_limiter_max)
    print("  rf antennas  %8.2f "%hpd_rf_max)
    print("  tbl_upper    %8.2f "%hpd_tbl_upper_max)
    print("  tbl_lower    %8.2f "%hpd_tbl_lower_max)
    print("  wall_upper   %8.2f "%hpd_wall_upper_max)
    print("  wall_lower   %8.2f "%hpd_wall_lower_max)
    print("  wall_inner   %8.2f "%hpd_iwall_max)
    print("\n maximum hits density: %8.2f  /m2"%hpd_max)

    hpd_limiter_norm      = hpd_limiter    / hpd_max
    hpd_rf_norm           = hpd_rf         / hpd_max
    hpd_tbl_upper_norm    = hpd_tbl_upper  / hpd_max
    hpd_tbl_lower_norm    = hpd_tbl_lower  / hpd_max
    hpd_wall_upper_norm   = hpd_wall_upper / hpd_max
    hpd_wall_lower_norm   = hpd_wall_lower / hpd_max
    hpd_iwall_norm        = hpd_iwall      / hpd_max
        
    plt.close()
    
    plt.figure(figsize=(9.,7.))
    fig,ax=plt.subplots()
    plt.xlim(-0.1, 2.*np.pi+0.1)
    plt.ylim(-1.25, 1.25)


    # rf antennas
    
    for jtf in range(ntf):
        for jphi in range(nphi_rf):
            for jz in range(nz_rf):
                
                delta_phi = phiend_rf[jtf,jphi,jz] - phistart_rf[jtf,jphi,jz]
                delta_z   = zend_rf[jtf,jphi,jz]   - zstart_rf[jtf,jphi,jz]
                rect = Rectangle( (phistart_rf[jtf,jphi,jz],zstart_rf[jtf,jphi,jz]),delta_phi, delta_z, facecolor=my_colormap(hpd_rf_norm[jtf,jphi,jz]), edgecolor='none', linewidth=0.25)
                ax.add_patch(rect)

    # poloidal limiters
    
    for jtf in range(ntf):
        for jphi in range(nphi_limiter):
            for jz in range(nz_limiter):

                delta_phi = phiend_limiter[jtf,jphi,jz] - phistart_limiter[jtf,jphi,jz]
                delta_z   = zend_limiter[jtf,jphi,jz]   - zstart_limiter[jtf,jphi,jz]
                
                rect = Rectangle( (phistart_limiter[jtf,jphi,jz],zstart_limiter[jtf,jphi,jz]),delta_phi, delta_z, facecolor=my_colormap(hpd_limiter_norm[jtf,jphi,jz]), edgecolor='none', linewidth=0.25)
                ax.add_patch(rect)
                
    #  tbl limiters
    
    for jphi in range(nphi_tbl):
        for jz in range(nz_tbl):

           delta_phi_lower = phiend_tbl_lower[jphi,jz] - phistart_tbl_lower[jphi,jz]
           delta_z_lower   = zend_tbl_lower[jphi,jz]   - zstart_tbl_lower[jphi,jz]

           delta_phi_upper = phiend_tbl_upper[jphi,jz] - phistart_tbl_upper[jphi,jz]
           delta_z_upper   = zend_tbl_upper[jphi,jz]   - zstart_tbl_upper[jphi,jz]
           
           rect = Rectangle( (phistart_tbl_lower[jphi,jz],zstart_tbl_lower[jphi,jz]),delta_phi_lower, delta_z_lower, facecolor=my_colormap(hpd_tbl_lower_norm[jphi,jz]), edgecolor='none', linewidth=0.25)
           ax.add_patch(rect)
    
           rect = Rectangle( (phistart_tbl_upper[jphi,jz],zstart_tbl_upper[jphi,jz]),delta_phi_upper, delta_z_upper, facecolor=my_colormap(hpd_tbl_upper_norm[jphi,jz]), edgecolor='none', linewidth=0.25)
           ax.add_patch(rect)

    # outer walls

    for jphi in range(nphi_wall_1):
        for jz in range(nz_wall_1):

            delta_phi_lower = phiend_wall_lower[jphi,jz] - phistart_wall_lower[jphi,jz]
            delta_z_lower   = zend_wall_lower[jphi,jz]   - zstart_wall_lower[jphi,jz]

            delta_phi_upper = phiend_wall_upper[jphi,jz] - phistart_wall_upper[jphi,jz]
            delta_z_upper   = zend_wall_upper[jphi,jz]   - zstart_wall_upper[jphi,jz]
            
            rect = Rectangle( (phistart_wall_lower[jphi,jz],zstart_wall_lower[jphi,jz]),delta_phi_lower, delta_z_lower, facecolor=my_colormap(hpd_wall_lower_norm[jphi,jz]), edgecolor='none', linewidth=0.25)
            ax.add_patch(rect)

            rect = Rectangle( (phistart_wall_upper[jphi,jz],zstart_wall_upper[jphi,jz]),delta_phi_upper, delta_z_upper, facecolor=my_colormap(hpd_wall_upper_norm[jphi,jz]), edgecolor='none', linewidth=0.25)
            ax.add_patch(rect) 

            for jtf in range(ntf):

                delta_phi_limiter = phiend_limiter_edge[jtf] - phistart_limiter_edge[jtf]
                delta_z_limiter   = zend_limiter_edge[jtf] - zstart_limiter_edge[jtf]
                rect = Rectangle( (phistart_limiter_edge[jtf],zstart_limiter_edge[jtf]),delta_phi_limiter, delta_z_limiter, facecolor='none', edgecolor='k', linewidth=0.25)
                ax.add_patch(rect)

                delta_phi_rf = phiend_rf_edge[jtf] - phistart_rf_edge[jtf]
                delta_z_rf   = zend_rf_edge[jtf] - zstart_rf_edge[jtf]
                rect = Rectangle( (phistart_rf_edge[jtf],zstart_rf_edge[jtf]),delta_phi_rf, delta_z_rf, facecolor='none', edgecolor='k', linewidth=0.25)
                ax.add_patch(rect)

            rect = Rectangle( (0.,tall_limiter),2.*np.pi, tall_tbl, facecolor='none', edgecolor='k', linewidth=0.25)
            ax.add_patch(rect)

            rect = Rectangle( (0.,-1.*(tall_limiter + tall_tbl)),2.*np.pi, tall_tbl, facecolor='none', edgecolor='k', linewidth=0.25)
            ax.add_patch(rect)

            rect = Rectangle( (0.,(tall_limiter+tall_tbl)),2.*np.pi, tall_wall, facecolor='none', edgecolor='k', linewidth=0.25)
            ax.add_patch(rect)

            rect = Rectangle( (0.,-1.*(tall_limiter+tall_tbl+tall_wall)),2.*np.pi, tall_wall, facecolor='none', edgecolor='k', linewidth=0.25)
            ax.add_patch(rect)
    my_title = " hits density (max = %10.2f)"%(hpd_max)                         
    plt.title(my_title, fontsize=10)
    #plt.colorbar()
    plt.xlabel('phi [radians]')
    plt.ylabel('Z [m]')
    #plt.tight_layout(pad=padsize)
    pdf.savefig()
    #plt.savefig('surface_heating.pdf')
    plt.close()

    #  +++++++++++++++++++++++++++++++++++++++
    #    plots copied from _2

      # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    xx = np.linspace(1,ntf,ntf)
    
    plt.close()
    plt.plot(xx,spd_limiter_maxs, 'bo', ms=2)
    plt.plot(xx, spd_limiter_maxs, 'b-', linewidth=1)
    plt.xlabel('TF sector')
    plt.ylabel('kw/m2')
    plt.title('Max spd on poloidal limiters')
    plt.xlim((0.,ntf+1))
    plt.ylim(bottom=0.)
    plt.grid(True)
    pdf.savefig()

    plt.close()
    plt.plot(xx,spd_limiter_means, 'bo', ms=2)
    plt.plot(xx, spd_limiter_means, 'b-', linewidth=1)
    plt.xlabel('TF sector')
    plt.ylabel('kw/m2')
    plt.title('Mean spd on poloidal limiters')
    plt.xlim((0.,ntf+1))
    plt.ylim(bottom=0.)
    plt.grid(True)
    pdf.savefig()
    
    plt.close()
    plt.plot(xx,spd_limiter_peaks, 'bo', ms=2)
    plt.plot(xx, spd_limiter_peaks, 'b-', linewidth=1)
    plt.xlabel('TF sector')
    plt.title('spd peaking factor on poloidal limiters')
    plt.xlim((0.,ntf+1))
    plt.ylim(bottom=0.)
    plt.grid(True)
    pdf.savefig()

    # repeat with fixed ymax

    plt.close()
    plt.plot(xx,spd_limiter_maxs, 'bo', ms=2)
    plt.plot(xx, spd_limiter_maxs, 'b-', linewidth=1)
    plt.xlabel('TF sector')
    plt.ylabel('kw/m2')
    plt.title('Max spd on poloidal limiters')
    plt.xlim((0.,ntf+1))
    plt.ylim((0.,1500.))
    plt.grid(True)
    pdf.savefig()

    plt.close()
    plt.plot(xx,spd_limiter_means, 'bo', ms=2)
    plt.plot(xx, spd_limiter_means, 'b-', linewidth=1)
    plt.xlabel('TF sector')
    plt.ylabel('kw/m2')
    plt.title('Mean spd on poloidal limiters')
    plt.xlim((0.,ntf+1))
    plt.ylim((0.,100.))
    plt.grid(True)
    pdf.savefig()
    
    plt.close()
    plt.plot(xx,spd_limiter_peaks, 'bo', ms=2)
    plt.plot(xx, spd_limiter_peaks, 'b-', linewidth=1)
    plt.xlabel('TF sector')
    plt.title('spd peaking factor on poloidal limiters')
    plt.xlim((0.,ntf+1))
    plt.ylim((0.,20.))
    plt.grid(True)
    pdf.savefig()
    
    spd_limiter_maxs  = np.zeros(ntf)
    spd_limiter_peaks = np.zeros(ntf)
    
 # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #  compute total hits for a check

    hits_rf_total         = np.sum(hits_rf)
    hits_limiter_total    = np.sum(hits_limiter)
    hits_tbl_upper_total  = np.sum(hits_tbl_upper)
    hits_tbl_lower_total  = np.sum(hits_tbl_lower)
    hits_wall_upper_total = np.sum(hits_wall_upper)
    hits_wall_lower_total = np.sum(hits_wall_lower)
    hits_iwall_total      = np.sum(hits_iwall)

    hits_total = hits_rf_total + hits_limiter_total + hits_tbl_upper_total + hits_tbl_lower_total \
                 + hits_wall_upper_total + hits_wall_lower_total + hits_iwall_total

    area_rf_total         = np.sum(area_rf)
    area_limiter_total    = np.sum(area_limiter)
    area_wall_upper_total = np.sum(area_wall_upper)
    area_wall_lower_total = np.sum(area_wall_lower)
    area_tbl_upper_total  = np.sum(area_tbl_upper)
    area_tbl_lower_total  = np.sum(area_tbl_lower)
    area_iwall_total      = np.sum(area_iwall)

    area_total = area_rf_total + area_limiter_total + area_wall_upper_total + area_wall_lower_total \
                 + area_tbl_upper_total + area_tbl_lower_total + area_iwall_total

    kw_limiter    = np.sum(np.multiply(area_limiter, spd_limiter))
    kw_rf         = np.sum(np.multiply(area_rf, spd_rf))
    kw_tbl_upper  = np.sum(np.multiply(area_tbl_upper, spd_tbl_upper))
    kw_tbl_lower  = np.sum(np.multiply(area_tbl_lower, spd_tbl_lower))
    kw_wall_upper  = np.sum(np.multiply(area_wall_upper, spd_wall_upper))
    kw_wall_lower  = np.sum(np.multiply(area_wall_lower, spd_wall_lower))
    kw_iwall       = np.sum(np.multiply(area_iwall, spd_iwall))

    kw_total       = kw_limiter + kw_rf + kw_tbl_upper + kw_tbl_lower + kw_wall_upper + kw_wall_lower + kw_iwall

    kw_limiters     = np.zeros(ntf)
    kw_rfs          = np.zeros(ntf)
    kw_tbl_uppers   = np.zeros(ntf)
    kw_tbl_lowers   = np.zeros(ntf)
    kw_wall_uppers  = np.zeros(nphi_wall_1)
    kw_wall_lowers  = np.zeros(nphi_wall_1)
    kw_iwalls       = np.zeros(nphi_wall_2)

    print("\n ++++++++++++++++++++++++++++ \n  jtf  kw-lim  kw-rf \n")
    for jtf in range(ntf):
        kw_limiters[jtf] = np.sum(np.multiply(area_limiter[jtf,:,:], spd_limiter[jtf,:,:]))
        kw_rfs[jtf]      = np.sum(np.multiply(area_rf[jtf,:,:], spd_rf[jtf,:,:]))
        print(" %3d  %6.2f %6.2f "%(jtf, kw_limiters[jtf], kw_rfs[jtf]))

    peak_limiters = np.max(kw_limiters) / np.mean(kw_limiters)
    peak_rfs      = np.max(kw_rfs)      / np.mean(kw_rfs)
        
    print(" \n +++++++++++++++++++ toroidal-peaking factors  \n")
    print("   limiters   %6.3f  "%peak_limiters)
    print("   antennas   %6.3f  \n"%peak_rfs)
    
    kw_limiter_frac       = 100. * kw_limiter    / kw_total
    kw_rf_frac            = 100. * kw_rf         / kw_total
    kw_tbl_upper_frac     = 100. * kw_tbl_upper  / kw_total
    kw_tbl_lower_frac     = 100. * kw_tbl_lower  / kw_total
    kw_wall_upper_frac    = 100. * kw_wall_upper / kw_total
    kw_wall_lower_frac    = 100. * kw_wall_lower / kw_total
    kw_iwall_frac         = 100. * kw_iwall      / kw_total
    
    spd_limiter_avg     = kw_limiter    / area_limiter_total
    spd_rf_avg          = kw_rf         / area_rf_total
    spd_tbl_upper_avg   = kw_tbl_upper  / area_tbl_upper_total
    spd_tbl_lower_avg   = kw_tbl_lower  / area_tbl_lower_total
    spd_wall_upper_avg  = kw_wall_upper / area_wall_upper_total
    spd_wall_lower_avg  = kw_wall_lower / area_wall_lower_total
    spd_iwall_avg       = kw_iwall      / area_iwall_total

    spd_limiter_ratio     = spd_limiter_max    / spd_limiter_avg
    spd_rf_ratio          = spd_rf_max         / spd_rf_avg
    spd_tbl_upper_ratio   = spd_tbl_upper_max  / spd_tbl_upper_avg
    spd_tbl_lower_ratio   = spd_tbl_lower_max  / spd_tbl_lower_avg
    spd_wall_upper_ratio  = spd_wall_upper_max / spd_wall_upper_avg
    spd_wall_lower_ratio  = spd_wall_lower_max / spd_wall_lower_avg
    spd_iwall_ratio       = spd_iwall_max      / spd_iwall_avg


    
    print("\n +++++++++++++++++++++++++++++++++++++ \n kilowatts onto integrated components and percent of total")
    
    print("   kw_limiter:     %7.1f   %5.2f "%(kw_limiter,    kw_limiter_frac))
    print("   kw_rf:          %7.1f   %5.2f "%(kw_rf,         kw_rf_frac))
    print("   kw_tbl_upper:   %7.1f   %5.2f "%(kw_tbl_upper,  kw_tbl_upper_frac))
    print("   kw_tbl_lower:   %7.1f   %5.2f "%(kw_tbl_lower,  kw_tbl_lower_frac))
    print("   kw_wall_upper:  %7.1f   %5.2f "%(kw_wall_upper, kw_wall_upper_frac))
    print("   kw_wall_lower:  %7.1f   %5.2f "%(kw_wall_lower, kw_wall_lower_frac))
    print("   kw_inner_wall:  %7.1f   %5.2f "%(kw_iwall,      kw_iwall_frac))
    print("   kw (total):     %7.1f         "%(kw_total))

    print("\n +++++++++++++++++++++++++++++++++++++ \n average surface power density of integrated components")
    
    print("   limiter:     %8.2f  "%(spd_limiter_avg))
    print("   rf:          %8.2f  "%(spd_rf_avg))
    print("   tbl_upper:   %8.2f  "%(spd_tbl_upper_avg))
    print("   tbl_lower:   %8.2f  "%(spd_tbl_lower_avg))
    print("   wall_upper:  %8.2f  "%(spd_wall_upper_avg))
    print("   wall_lower:  %8.2f  "%(spd_wall_lower_avg))
    print("   inner_wall:  %8.2f  "%(spd_iwall_avg))

    print("\n +++++++++++++++++++++++++++++++++++++ \n max/average surface power density of integrated components")
    
    print("   limiter:     %8.2f  "%(spd_limiter_ratio))
    print("   rf:          %8.2f  "%(spd_rf_ratio))
    print("   tbl_upper:   %8.2f  "%(spd_tbl_upper_ratio))
    print("   tbl_lower:   %8.2f  "%(spd_tbl_lower_ratio))
    print("   wall_upper:  %8.2f  "%(spd_wall_upper_ratio))
    print("   wall_lower:  %8.2f  "%(spd_wall_lower_ratio))
    print("   inner_wall:  %8.2f  "%(spd_iwall_ratio))
    
    print("\n +++++++++++++++++++++++++++++++++++++ \n total hits on integrated components")
    
    print("   hits_limiter:     %7.0f  "%(hits_limiter_total))
    print("   hits_rf:          %7.0f  "%(hits_rf_total))
    print("   hits_tbl_upper:   %7.0f  "%(hits_tbl_upper_total))
    print("   hits_tbl_lower:   %7.0f  "%(hits_tbl_lower_total))
    print("   hits_wall_upper:  %7.0f  "%(hits_wall_upper_total))
    print("   hits_wall_lower:  %7.0f  "%(hits_wall_lower_total))
    print("   hits_inner_wall:  %7.0f  "%(hits_iwall_total))
    print("   hits (total):     %7.0f  "%(hits_total))
    
    print("\n +++++++++++++++++++++++++++++++++++++ \n total area of integrated components ")
    
    print("   area_limiter:      %7.3f "%(area_limiter_total))
    print("   area_rf:           %7.3f "%(area_rf_total))
    print("   area_tbl_upper:    %7.3f "%(area_tbl_upper_total))
    print("   area_tbl_lower:    %7.3f "%(area_tbl_lower_total))
    print("   area_wall_upper:   %7.3f "%(area_wall_upper_total))
    print("   area_wall_lower:   %7.3f "%(area_wall_lower_total))
    print("   area_inner_wall:   %7.3f "%(area_iwall_total))
    print("   area (total):      %7.3f "%(area_total))

    print("\n +++++++++++++++++++++++++++++++++++++ \n range of size of individual rectangles")

    print("   limiter:    %7.5f  %7.5f "%(np.min(area_limiter), np.max(area_limiter)))
    print("   antenna:    %7.5f  %7.5f "%(np.min(area_rf), np.max(area_rf)))
    print("   tbl_upper:  %7.5f  %7.5f "%(np.min(area_tbl_upper), np.max(area_tbl_upper)))
    print("   tbl_lower:  %7.5f  %7.5f "%(np.min(area_tbl_lower), np.max(area_tbl_lower)))
    print("   wall_upper: %7.5f  %7.5f "%(np.min(area_wall_upper), np.max(area_wall_upper)))
    print("   wall_lower: %7.5f  %7.5f "%(np.min(area_wall_lower), np.max(area_wall_lower)))
    print("   wall_inner: %7.5f  %7.5f "%(np.min(area_iwall), np.max(area_iwall)))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    print("\n +++++++++++++++++++++++++++++++++++++++++++++++ \n \n")

    top_limiter       = np.flip(np.sort(np.reshape(spd_limiter,-1)))
    top_tbl_upper     = np.flip(np.sort(np.reshape(spd_tbl_upper,-1)))
    top_tbl_lower     = np.flip(np.sort(np.reshape(spd_tbl_lower,-1)))
    top_wall_upper    = np.flip(np.sort(np.reshape(spd_wall_upper,-1)))
    top_wall_lower    = np.flip(np.sort(np.reshape(spd_wall_lower,-1)))
    top_wall_inner    = np.flip(np.sort(np.reshape(spd_iwall,-1)))
    top_rf            = np.flip(np.sort(np.reshape(spd_rf,-1)))

    print("  top 10 surface power densities \n")
    
    print(" n  limiter  tbl-upper  tbl-lower  wall-upper wall-lower wall-inner  antenna \n")
    for qq in range(10):
        print("%3d %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f %9.2f"%(qq,top_limiter[qq], top_tbl_upper[qq],    \
                                                               top_tbl_lower[qq],top_wall_upper[qq],  \
                                                               top_wall_lower[qq],top_wall_inner[qq], \
                                                               top_rf[qq]))
        
    
    print("\n +++++++++++++++++++++++++++++++++++++++++++++++ \n \n")

    print("                limiters    tbl-upper    tbl_lower     wall_upper    wall_lower     wall_inner    antenna \n")
    print(" kw       %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f"%(kw_limiter, kw_tbl_upper, kw_tbl_lower, kw_wall_upper, kw_wall_lower, kw_iwall, kw_rf))
    print(" kw-frac  %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f"%(kw_limiter_frac, kw_tbl_upper_frac, kw_tbl_lower_frac, kw_wall_upper_frac,
                                                                        kw_wall_lower_frac, kw_iwall_frac, kw_rf_frac))
    print("")
    print(" spd-max  %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f"%(spd_limiter_max, spd_tbl_upper_max, spd_tbl_lower_max, spd_wall_upper_max, spd_wall_lower_max, \
                                                                        spd_iwall_max, spd_rf_max))
    print(" spd-avg  %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f"%(spd_limiter_avg, spd_tbl_upper_avg, spd_tbl_lower_avg, spd_wall_upper_avg, spd_wall_lower_avg, \
                                                                        spd_iwall_avg, spd_rf_avg))
    print(" peak-f   %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f"%(spd_limiter_ratio, spd_tbl_upper_ratio, spd_tbl_lower_ratio, spd_wall_upper_ratio, spd_wall_lower_ratio, \
                                                                        spd_iwall_ratio, spd_rf_ratio))
    print("")
    print(" max-hits %12d %12d %12d %12d %12d %12d %12d"%(hits_limiter_max, hits_tbl_upper_max, hits_tbl_lower_max, hits_wall_upper_max, hits_wall_lower_max, \
                                                                         hits_iwall_max, hits_rf_max))
    print(" error    %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f"%(error_limiter, error_tbl_upper, error_tbl_lower, error_wall_upper, error_wall_lower, error_iwall, error_rf))
    print(" hits-tot %12d %12d %12d %12d %12d %12d %12d"%(hits_limiter_total, hits_tbl_upper_total, hits_tbl_lower_total, hits_wall_upper_total, hits_wall_lower_total, \
                                                           hits_iwall_total, hits_rf_total))
    print("")
    print(" tpf-lim-rf: %9.2f %9.2f "%(peak_limiters, peak_rfs))
    
    print("\n +++++++++++++++++++++++++++++++++++++++++++\n")
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #pdb.set_trace()
    


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def print_summary(file_name, fn_geqdsk, eq_index, fraction_alphas_simulated,do_corr, fn_profiles, ploss_wall_kw, ishape, stub, suppress_tbl, max_markers, fn_parameter="none"):

    print("   ... print_summary starting at time: ", clock.ctime())
    print("         file_name:       ",file_name)
    print("         ploss_wall_kw:   ", ploss_wall_kw)
    print("         fn_parameter:    ", fn_parameter)
    print("         fn_geqdsk:       ", fn_geqdsk)
    print("         fn_profiles:     ", fn_profiles)
    print("         max_markers:     ", max_markers)
    print("         suppress_tbl:    ", suppress_tbl)

   
    
    time_first = mytime.time()
    time_last = mytime.time()
    
    do_rasterized   = True
    do_rasterized_2 = True
    
    mpl.rcParams['image.composite_image']=False
    mpl.rcParams['pdf.fonttype']=42
    mpl.rcParams['ps.fonttype']=42
    mpl.rcParams['xtick.direction']='in'
    mpl.rcParams['ytick.direction']='in'
    mpl.rcParams['xtick.major.size'] = 7
    mpl.rcParams['ytick.major.size'] = 7
    mpl.rcParams['xtick.minor.size'] = 3.5
    mpl.rcParams['ytick.minor.size'] = 3.5
    mpl.rcParams.update({'font.size':12})
    padsize=1.5
    plt.rcParams['xtick.minor.visible']=True
    plt.rcParams['ytick.minor.visible']=True

    print("   ... print_summary:  have defined all of the rcParams")
    
    size_factor = 4.5    # for nifty loss plot
    fontsize_2 = 12

    #  if you want to debug, you must enter "none" as the filename stub
    
    print("   ... print_summmary:  stub = ", stub, " at time: ",clock.ctime())
    
    if (stub != "none"):
        print("   ... further output will go to the stub file\n")
        fn_out = stub + '.txt'
        sys.stdout = open(fn_out,"w")
    else:
        print("   ... stub is none, so output will go to the screen\n")

    print("   ... print_summary:  about to get profile information")
    # -----------------------------
    #  get the profile information
    
    aa_profile = proc.read_sparc_profiles(fn_profiles)
    rho_tor    = aa_profile["rho_tor"]
    rho_pol    = aa_profile["rho_pol"]

    rho_pol_or_tor   = 1    # 1 = use poloidal rho.  2 = use toroidal rho
    
    # --------------------------------------
    #  get the equliibrium and [R,Z] shape of LCFS

    print("   ... print_summary:  about to get the equilbrium")
    
    gg = geqdsk(fn_geqdsk)
    gg.readGfile()
    
    rl =  gg.equilibria[eq_index].rlcfs
    zl =  gg.equilibria[eq_index].zlcfs
    
    print("   ... print_summary: I have read the equilibrium, rmin = ", np.min(rl))

    # ---------------------------------------------

    psi_rmin       =  gg.equilibria[eq_index].rmin
    psi_rmax       =  gg.equilibria[eq_index].rmin + gg.equilibria[eq_index].rdim
    psi_nr         =  gg.equilibria[eq_index].nw
    psi_zmin       =  gg.equilibria[eq_index].zmid - gg.equilibria[eq_index].zdim/2.
    psi_zmax       =  gg.equilibria[eq_index].zmid + gg.equilibria[eq_index].zdim/2.
    psi_nz         =  gg.equilibria[eq_index].nh
        
    psiPolSqrt     =  gg.equilibria[eq_index].psiPolSqrt

    geq_rarray = np.linspace(psi_rmin, psi_rmax, psi_nr)
    geq_zarray = np.linspace(psi_zmin, psi_zmax, psi_nz)

    # transpose so that we are on a grid = [R,z]. define a function
        
    rhogeq_transpose_2d = np.transpose(psiPolSqrt)   # psi_pol_sqrt = sqrt( (psi-psi0)/psi1-psi0))

    # ---------------------------------------------
    ##pdb.set_trace()

    #  get a function that we can use to map (R,z) --> rho

    # fraction_alphas_simulated = 1.0     #  0.166 for case 83
    rho_interpolator = get_rho.get_rho_interpolator(fn_geqdsk, eq_index)
    
    mpl.rcParams['image.composite_image'] = False

    #if isinstance(file_name,list):
    #    string_list = file_name
    #else:
    #    string_list = file_name.split('.')
    # stub = string_list[0]
    
    # pdb.set_trace()
          
    filename_text = stub + '_krs.txt'
    
    end_conditions={}
    
    end_conditions['0']   = 'abort'
    end_conditions['1']   = 'simtime'
    end_conditions['2']   = 'emin'
    end_conditions['8']   = 'wall'
    end_conditions['32']  = 'maxrho'
    end_conditions['256'] = 'cputime'
    end_conditions['40']  = 'unk40'
    end_conditions['4']   = 'therm'
    end_conditions['6']   = 'unk6'
    end_conditions['520'] = 'unk520'
    
    # end_conditions = ['abort', 'simtime', 'emin', 'unk', 'unk', 'unk', 'unk', 'unk', 'wall']

    # -----------------------------------------
    #  read either the old or new hdf5 format

    #  if a list, should be e.g.  'ascot_35529768.h5','ascot_35529771.h5' i.e. no brackets
    
    # fname_length = len(file_name)
    
    #print("  length of filename: ", len(file_name))
    #print(" file_name = ", file_name)

    #pdb.set_trace()

    # the next silly code needed because file_name can be either a
    # list or a string
    #  sds 1/11/2021
    
    
    if isinstance(file_name, list):
        file_name = my_list.list_to_string(file_name)
    
    
    file_list = file_name.split(",")
    

    print("The files I will read are: ", file_list)
    if (len(file_list) > 1):             

        aa = myread_hdf5_multiple(file_list, max_markers)
        
    else:
        
        new_or_old = old_or_new(file_list[0])
    
        if(new_or_old == 0):
            aa = myread_hdf5(file_list[0], max_markers)
        elif(new_or_old == 1):
            file_name = file_list[0]
            aa = myread_hdf5_new(file_list[0], max_markers)
    
    #  away we go ...
    print("   ... print_summary:  time to read in hdf5 data [sec]: ", mytime.time()-time_last)
    time_last = mytime.time()
    pitch_ini        = aa["pitch_ini"]

    print("   ... print_summary: total number of markers read in from .h5 file(s): ", len(pitch_ini))
    pitch_phi_ini    = aa["pitch_phi_ini"]
    pitch_phi_marker = aa["marker_pitch_phi"]
    
    mass      = aa["mass"]
    vtot      = aa["vtot"]
    ekev      = aa["ekev"]
    vpar      = aa["vpar"]
    vphi      = aa["vphi"]
    vr        = aa["vr"]
    vz        = aa["vz"]
    weight    = aa["weight"]

    weight    = weight / np.sum(weight)    # 11/15/2020 ... needed for multiple-file read
    weight_parent = aa["weight_parent"]

    weight_ratio = weight / weight_parent
    
    time      = aa["time"]
    cputime   = aa["cputime"]
    endcond   = aa["endcond"]
    phi_end   = aa["phi_end"]
    r_end     = aa["r_end"]
    z_end     = aa["z_end"]
    theta_end = aa["theta_end"]
    id_end    = aa["id_end"]

    r_wall  = aa['r_wall']
    z_wall  = aa['z_wall']

    rwall_max = np.max(r_wall)

    id_ini  = aa["id_ini"]
    z_ini   = aa["z_ini"]
    r_ini   = aa["r_ini"]
    phi_ini = aa["phi_ini"]
    vphi_ini = aa["vphi_ini"]
    vz_ini   = aa["vz_ini"]
    vr_ini   = aa["vr_ini"]
    vpar_ini = aa["vpar_ini"]
    vtot_ini = aa["vtot_ini"]
    ekev_ini = aa["ekev_ini"]

    r_marker    = aa["marker_r"]
    z_marker    = aa["marker_z"]
    phi_marker  = aa["marker_phi"]
    id_marker   = aa["marker_id"]
    vphi_marker = aa["marker_vphi"]
    vz_marker   = aa["marker_vr"]
    vr_marker   = aa["marker_vz"]
    vtot_marker = aa["marker_vtot"]
    marker_ekev = aa["marker_ekev"]    # was ekev_marker = until 12/31/2020

    rho_ini    = np.zeros(id_marker.size)
    rho_marker = np.zeros(id_marker.size)
    rho_end    = np.zeros(id_marker.size)

    #  fake marker ensemble for testing
    

    rmarker_fake = np.linspace(np.min(r_marker), np.max(r_marker), 100)
    zmarker_fake = np.linspace(np.min(z_marker), np.max(z_marker), 200)

    rho_marker_fake     = np.zeros(20000)
    weight_marker_fake  = np.zeros(20000)
    icc = 0
    
    for ir_fake in range(100):
        for iz_fake in range(200):
            weight_marker_fake[icc] = rmarker_fake[ir_fake]
            rho_marker_fake[icc]    = rho_interpolator(rmarker_fake[ir_fake], zmarker_fake[iz_fake])
            icc += 1

    # use rho-interpolator to get rho

    #   do a simple test

    test_rho_800 = rho_interpolator(2.2, 0.21)
    test_rho_900 = rho_interpolator(1.4, 0.41)

    print('   test_rho_800 = ', test_rho_800)
    print('   test_rho_900 = ', test_rho_900)

    my_rmajor_array = np.linspace(np.min(r_marker), np.max(r_marker), 500)
    my_z_array      = np.linspace(np.min(z_marker), np.max(z_marker), 500)
    my_rho_array    = np.zeros(500)
    my_rhoz_array   = np.zeros(500)
    for jj in range(500):
        my_rho_array[jj]  = rho_interpolator(my_rmajor_array[jj], 0.)
        my_rhoz_array[jj] = rho_interpolator(1.85, my_z_array[jj])

    print("   ... print_summary:  about to compute fraction of particles born outside given rho") 

    
    # ------------------------------------------------------------------
    #  compute and plot fraction of particles born outside of given rho

    for jj in range(id_marker.size):
        rho_ini[jj]    = rho_interpolator(r_ini[jj],    z_ini[jj])
        rho_marker[jj] = rho_interpolator(r_marker[jj], z_marker[jj])
        rho_end[jj]    = rho_interpolator(r_end[jj],    z_end[jj])
    print(" I have computed the rho-arrays")
    print("  zzz time to compute rho-arrays: ", mytime.time()-time_last)
    time_last = mytime.time()
    if(rho_pol_or_tor   == 2):     # sds 2/11/2020

        rho_ini_parent = np.copy(rho_ini)
        # 
        for jj in range(id_marker.size):

            #  ynew = np.interp(xnew, xparent_array, yparent_array)
            
            rho_ini[jj]    = np.interp(rho_ini[jj],    rho_pol, rho_tor)
            rho_marker[jj] = np.interp(rho_marker[jj], rho_pol, rho_tor)
            rho_end[jj]    = np.interp(rho_end[jj],    rho_pol, rho_tor)

    
    delta_rho = rho_end - rho_ini

    nn_ini    = rho_ini.size
    nn_marker = rho_marker.size

    ff_ini_outside    = np.zeros(101)
    ff_marker_outside = np.zeros(101)
    
    rho_array = np.linspace(0., 1., 101)

    for jkl in range(101):
        
        ini_outside         = (rho_ini >= rho_array[jkl])
        ff_ini_outside[jkl] = rho_ini[ini_outside].size

        marker_outside         = (rho_marker >= rho_array[jkl])
        ff_marker_outside[jkl] = rho_ini[ini_outside].size

    ff_ini_outside = ff_ini_outside / nn_ini
    ff_marker_outside = ff_marker_outside / nn_marker

    print("   ... starting plotting at time: ", clock.ctime())
    
    filename_multi = stub + '.pdf'
    with PdfPages(filename_multi) as pdf:

        plt.figure()
        plt.plot(my_rmajor_array, my_rho_array, 'b-', linewidth=1.5, rasterized=do_rasterized)
        plt.ylim(0.,1.1)
        plt.grid(True)
        plt.xlabel('Rmajor',fontsize=fontsize_2)
        plt.ylabel('rho',fontsize=fontsize_2)
        plt.title('rho versus Rmajor at Z=0', fontsize=fontsize_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.plot(my_z_array, my_rhoz_array, 'b-', linewidth=1.5, rasterized=do_rasterized)
        plt.ylim(0.,1.1)
        plt.grid(True)
        plt.xlabel('Z',fontsize=fontsize_2)
        plt.ylabel('rho',fontsize=fontsize_2)
        plt.title('rho versus Z at Rmajor=1.85', fontsize=fontsize_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()  
        
        if(rho_pol_or_tor== 2):
            plt.figure()
            plt.plot(rho_ini_parent, rho_ini, 'b-', linewidth=1.5, rasterized=do_rasterized)
            plt.ylim(0.,1.)
            plt.xlim(0.,1.)
            plt.grid(True)
            plt.xlabel('rho_pol',fontsize=fontsize_2)
            plt.ylabel('rho_tor',fontsize=fontsize_2)
            plt.title('rho_tor is new rho', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()          

        print("   zzz finished plot-set 1, time = ", mytime.time()-time_last)
        time_last = mytime.time()
        plt.figure()
        plt.plot(rho_array, ff_marker_outside, 'k-', linewidth=1.5, rasterized=do_rasterized)
        plt.plot(rho_array, ff_ini_outside,    'r-', linewidth=1.5, rasterized=do_rasterized)
        plt.ylim(0.,1.)
        plt.xlim(0.,1.)
        plt.grid(True)
        plt.xlabel('rho',fontsize=fontsize_2)
        plt.title('fraction born outside given rho (bl, re = marker, ini)', fontsize=fontsize_2)
        plot_filename = stub + '_born_outside_rho.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        print("   ... at position A at time: ", clock.ctime())
        
        ##pdb.set_trace()
       
        # --------------------------------------------------------------------
        mod_phi_marker = np.mod(phi_marker, 20.)
        mod_phi_ini    = np.mod(phi_ini,    20.)
    
        nn = len(time)
        #import pdb ; #pdb.set_trace()
        print("")
        print("  mark mass  end   endcond      time   cputime    vend     ekev     weight    r_mark   r_ini    z_mark   z_ini  phimark    phiini id_mark  id_ini  id_end")
        print("")
        #    for i in range(nn):
        #        print(" %4d %4.1f %4d %8s   %10.6f %6.1f %11.3e %7.1f  %8.4f  %7.3f  %7.3f  %7.3f  %7.3f  %7.1f  %7.1f    %7d    %7d   %7d "\
        #        % (i, mass[i], endcond[i], end_conditions[str(endcond[i])], time[i], cputime[i], vtot[i], ekev[i], weight[i], \
        #           r_marker[i], r_ini[i], z_marker[i], z_ini[i], mod_phi_marker[i], mod_phi_ini[i], id_marker[i], id_ini[i], id_end[i]))
        #        print("")

        print(" --------------------------------------------------------------------")
        print("")
        print("  now for only those markers that were aborted")
        print("")

        print("  mark mass  end   endcond      time   cputime    vend     ekev     weight    r_mark   z_mark  phimark  r_ini   z_ini  phiini r_end  z_end  id_mark  id_ini  id_end")

        mmm = nn
        if(mmm > 100):
            mmm = 100
        print("")
        for i in range(mmm):
            if endcond[i] == 0:
                print(" %4d %4.1f %4d %8s   %10.6f %6.1f %11.3e %7.1f  %8.4f  %7.3f  %7.3f  %7.3f  %7.3f  %7.3f %7.3f %7.3f  %7.3f %7d  %7d %7d "\
                % (i, mass[i], endcond[i], end_conditions[str(endcond[i])], time[i], cputime[i], vtot[i], ekev[i], weight[i], \
                   r_marker[i], z_marker[i], mod_phi_marker[i], r_ini[i],  z_ini[i],  mod_phi_ini[i], r_end[i], z_end[i], id_marker[i], id_ini[i], id_end[i]))

                print("")
        print(" --------------------------------------------------------------------")
        # -------------------------------------------------------------

        print(" --------------------------------------------------------------------")
        print("")
        print("  now for only those markers that terminated with maxrho")
        print("")

        print("  mark mass  end   endcond      time   cputime    vend     ekev     weight    r_mark   z_mark  phimark r_ini    z_ini  phiini id_mark  id_ini  id_end")

        print("")
        mmm = nn
        if(mmm > 100):
            mmm = 100
        for i in range(mmm):
            if endcond[i] == 32:
                print(" %4d %4.1f %4d %8s   %10.6f %6.1f %11.3e %7.1f  %8.4f  %7.3f  %7.3f  %7.3f  %7.3f  %7.1f % 7.1f %7d  %7d %7d "\
                % (i, mass[i], endcond[i], end_conditions[str(endcond[i])], time[i], cputime[i], vtot[i], ekev[i], weight[i], \
                   r_marker[i], z_marker[i], mod_phi_marker[i], r_ini[i],  z_ini[i], mod_phi_ini[i], id_marker[i], id_ini[i], id_end[i]))

                print("")
        print(" --------------------------------------------------------------------")
        # -------------------------------------------------------------

        print("")
        print("  now for only those markers that terminated with hit wall")
        print("")
        print("  mark mass  end   endcond      time   cputime    vend     ekev     weight    r_mark   z_mark  phimark r_ini     z_ini  phiini id_mark  id_ini  id_end")

        print("")
        #    for i in range(nn):
        #        if endcond[i] == 8:
        #            print(" %4d %4.1f %4d %8s   %10.6f %6.1f %11.3e %7.1f  %8.4f  %7.3f  %7.3f  %7.3f  %7.3f  %7.1f % 7.1f %7d  %7d %7d "\
        #            % (i, mass[i], endcond[i], end_conditions[str(endcond[i])], time[i], cputime[i], vtot[i], ekev[i], weight[i], \
        #               r_marker[i], z_marker[i], mod_phi_marker[i],r_ini[i], z_ini[i],  mod_phi_ini[i], id_marker[i], id_ini[i], id_end[i]))
        #            print("")

        print(" --------------------------------------------------------------------")
        # -------------------------------------------------------------
        print(" --------------------------------------------------------------------")
        # -------------------------------------------------------------

        print("")
        print("  now for only those markers that terminated with end condition 520")
        print("")

        print("  mark mass  end   endcond      time   cputime    vend     ekev     weight    r_mark   z_mark  phimark r_ini     z_ini  phi_ini id_mark  id_ini  id_end")

        print("")
        #    for i in range(nn):
        #        if endcond[i] == 520:
        #            print(" %4d %4.1f %4d %8s   %10.6f %6.1f %11.3e %7.1f  %8.4f  %7.3f  %7.3f  %7.3f  %7.3f  %7.1f % 7.1f %7d  %7d %7d "\
        #            % (i, mass[i], endcond[i], end_conditions[str(endcond[i])], time[i], cputime[i], vtot[i], ekev[i], weight[i], \
        #               r_marker[i], z_marker[i], mod_phi_marker[i],r_ini[i], z_ini[i],  mod_phi_ini[i], id_marker[i], id_ini[i], id_end[i]))
        #            print("")

        aa_abort      = extract_subgroups(endcond, ekev, weight, 0)
        aa_simtime    = extract_subgroups(endcond, ekev, weight, 1)
        aa_emin       = extract_subgroups(endcond, ekev, weight, 2)
        aa_therm      = extract_subgroups(endcond, ekev, weight, 4)
        aa_emin_th    = extract_subgroups(endcond, ekev, weight, 6)
        aa_wall       = extract_subgroups(endcond, ekev, weight, 8)
        aa_rhomin     = extract_subgroups(endcond, ekev, weight, 16)
        aa_rhomax     = extract_subgroups(endcond, ekev, weight, 32)
        aa_unk40      = extract_subgroups(endcond, ekev, weight, 40)
        aa_cputime    = extract_subgroups(endcond, ekev, weight, 256)
        aa_hybrid     = extract_subgroups(endcond, ekev, weight, 512)
        aa_hybrid_rho = extract_subgroups(endcond, ekev, weight, 544)
        aa_unk520     = extract_subgroups(endcond, ekev, weight, 520)
    
        print("")
        print("                       --------weighted------")
        print("  endcond         nn   particle    energy ")
        print(" ")
        print("  abort      %6d    %9.6f  %9.6f" % (aa_abort["nn"],      aa_abort["particle_total"],      aa_abort["total"]))
        print("  emin       %6d    %9.6f  %9.6f" % (aa_emin["nn"],       aa_emin["particle_total"],       aa_emin["total"]))
        print("  therm      %6d    %9.6f  %9.6f" % (aa_therm["nn"],      aa_therm["particle_total"],      aa_therm["total"]))
        print("  wall       %6d    %9.6f  %9.6f" % (aa_wall["nn"],       aa_wall["particle_total"],       aa_wall["total"]))
        print("  rhomin     %6d    %9.6f  %9.6f" % (aa_rhomin["nn"],     aa_rhomin["particle_total"],     aa_rhomin["total"]))
        print("  rhomax     %6d    %9.6f  %9.6f" % (aa_rhomax["nn"],     aa_rhomax["particle_total"],     aa_rhomax["total"]))
        print("  simtime    %6d    %9.6f  %9.6f" % (aa_simtime["nn"],    aa_simtime["particle_total"],    aa_simtime["total"]))
        print("  cputime    %6d    %9.6f  %9.6f" % (aa_cputime["nn"],    aa_cputime["particle_total"],    aa_cputime["total"]))
        print("  emin+th    %6d    %9.6f  %9.6f" % (aa_emin_th["nn"],    aa_emin_th["particle_total"],    aa_emin_th["total"]))
        print("  hybrid     %6d    %9.6f  %9.6f" % (aa_hybrid["nn"],     aa_hybrid["particle_total"],     aa_hybrid["total"]))
        print("  unk40      %6d    %9.6f  %9.6f" % (aa_unk40["nn"],      aa_unk40["particle_total"],      aa_unk40["total"]))
        print("  unk520     %6d    %9.6f  %9.6f" % (aa_unk520["nn"],     aa_unk520["particle_total"],     aa_unk520["total"]))
        print("  hybridrho  %6d    %9.6f  %9.6f" % (aa_hybrid_rho["nn"], aa_hybrid_rho["particle_total"], aa_hybrid_rho["total"]))
        print("")

        #print('')
        #print('  nwall:        ', aa_wall["nn"])
        #print('  wall ekev:    ', aa_wall["ekev"])
        #print('  wall weights: ', aa_wall["weight"])
        #print('  wall product: ', aa_wall["product"])
        #print('  wall total:   ', aa_wall["total"])
        #print('')

        nlaunched = r_end.size
        wlaunched = np.sum(weight)
        # elaunched = 3515. * np.sum(weight)                # fixed 4/13/2021
        # eaborted  = 3515  * np.sum(aa_abort["weight"])    # fixed 4/13/2021
        # pdb.set_trace()
        elaunched = np.sum(ekev_ini*weight)
        eaborted  = np.sum(aa_abort["ekev"]*aa_abort["weight"])   # until 9/10/2021 had been ekev_ini * aa_abort["weight"]

        wall_fraw_particles  = aa_wall["nn"]              / (nlaunched - aa_abort["nn"])
        wall_fparticles      = aa_wall["particle_total"]  / (wlaunched - aa_abort["particle_total"])
        wall_fenergy         = aa_wall["total"]           / (elaunched - eaborted)

        rhomax_fraw_particles  = aa_rhomax["nn"]              / (nlaunched - aa_abort["nn"])
        rhomax_fparticles      = aa_rhomax["particle_total"]  / (wlaunched - aa_abort["particle_total"])
        rhomax_fenergy         = aa_rhomax["total"]           / (elaunched - eaborted)

        s520_fraw_particles  = aa_unk520["nn"]              / (nlaunched - aa_abort["nn"])
        s520_fparticles      = aa_unk520["particle_total"]  / (wlaunched - aa_abort["particle_total"])
        s520_fenergy         = aa_unk520["total"]           / (elaunched - eaborted)

        hybrid_rho_fraw_particles  = aa_hybrid_rho["nn"]              / (nlaunched - aa_abort["nn"])
        hybrid_rho_fparticles      = aa_hybrid_rho["particle_total"]  / (wlaunched - aa_abort["particle_total"])
        hybrid_rho_fenergy         = aa_hybrid_rho["total"]           / (elaunched - eaborted)
        
        #  new 8/13/2020.  prior analyses may have underreported loases.  had excluded "hybrid"
        #  but i am not sure we ever got a nonzero number of this end condition

        hybrid_fraw_particles  = aa_hybrid["nn"]              / (nlaunched - aa_abort["nn"])
        hybrid_fparticles      = aa_hybrid["particle_total"]  / (wlaunched - aa_abort["particle_total"])
        hybrid_fenergy         = aa_hybrid["total"]           / (elaunched - eaborted)
        
        sum_fraw_particles =  wall_fraw_particles + rhomax_fraw_particles +  s520_fraw_particles + hybrid_rho_fraw_particles + hybrid_fraw_particles
        sum_fparticles     =  wall_fparticles     + rhomax_fparticles     +  s520_fparticles     + hybrid_rho_fparticles     + hybrid_fparticles
        sum_fenergy        =  wall_fenergy        + rhomax_fenergy        +  s520_fenergy        + hybrid_rho_fenergy        + hybrid_fenergy

        # ====================================================================
        #  compute uncertainties

        # unweighted particles
        
        wall_fraw_particles1       =       aa_wall["nn1"]
        rhomax_fraw_particles1     =     aa_rhomax["nn1"]          
        s520_fraw_particles1       =     aa_unk520["nn1"]
        hybrid_rho_fraw_particles1 = aa_hybrid_rho["nn1"]
        hybrid_fraw_particles1     =     aa_hybrid["nn1"]

        wall_fraw_particles2       =       aa_wall["nn2"]
        rhomax_fraw_particles2     =     aa_rhomax["nn2"]          
        s520_fraw_particles2       =     aa_unk520["nn2"]
        hybrid_rho_fraw_particles2 = aa_hybrid_rho["nn2"]
        hybrid_fraw_particles2     =     aa_hybrid["nn2"]
        
        wall_fraw_particles3      =        aa_wall["nn3"]
        rhomax_fraw_particles3    =      aa_rhomax["nn3"]          
        s520_fraw_particles3      =      aa_unk520["nn3"]
        hybrid_rho_fraw_particles3 = aa_hybrid_rho["nn3"]
        hybrid_fraw_particles3     =     aa_hybrid["nn3"]
        
        wall_fraw_particles4       =        aa_wall["nn4"]
        rhomax_fraw_particles4     =      aa_rhomax["nn4"]          
        s520_fraw_particles4       =      aa_unk520["nn4"]
        hybrid_rho_fraw_particles4 =  aa_hybrid_rho["nn4"]
        hybrid_fraw_particles4     =      aa_hybrid["nn4"]

        # weighted particles
        
        wall_fparticles1       =       aa_wall["particle_total1"]
        rhomax_fparticles1     =     aa_rhomax["particle_total1"]
        s520_fparticles1       =     aa_unk520["particle_total1"]
        hybrid_rho_fparticles1 = aa_hybrid_rho["particle_total1"]
        hybrid_fparticles1     =     aa_hybrid["particle_total1"]

        wall_fparticles2        =       aa_wall["particle_total2"]
        rhomax_fparticles2      =     aa_rhomax["particle_total2"]
        s520_fparticles2        =     aa_unk520["particle_total2"]
        hybrid_rho_fparticles2  = aa_hybrid_rho["particle_total2"]
        hybrid_fparticles2      =     aa_hybrid["particle_total2"]

        wall_fparticles3       =       aa_wall["particle_total3"]
        rhomax_fparticles3     =     aa_rhomax["particle_total3"]
        s520_fparticles3       =     aa_unk520["particle_total3"]
        hybrid_rho_fparticles3 = aa_hybrid_rho["particle_total3"]
        hybrid_fparticles3     =     aa_hybrid["particle_total3"]

        wall_fparticles4       =       aa_wall["particle_total4"]
        rhomax_fparticles4     =     aa_rhomax["particle_total4"]
        s520_fparticles4       =     aa_unk520["particle_total4"]
        hybrid_rho_fparticles4 = aa_hybrid_rho["particle_total4"]
        hybrid_fparticles4     =     aa_hybrid["particle_total4"]

        # weighted energy

        wall_fenergy1       =        aa_wall["total1"]
        rhomax_fenergy1      =     aa_rhomax["total1"]          
        s520_fenergy1        =     aa_unk520["total1"]
        hybrid_rho_fenergy1  = aa_hybrid_rho["total1"]
        hybrid_fenergy1      =     aa_hybrid["total1"]
        
        wall_fenergy2        =       aa_wall["total2"]
        rhomax_fenergy2      =     aa_rhomax["total2"]          
        s520_fenergy2        =     aa_unk520["total2"]
        hybrid_rho_fenergy2  = aa_hybrid_rho["total2"]
        hybrid_fenergy2      =     aa_hybrid["total2"]

        wall_fenergy3        =       aa_wall["total3"]
        rhomax_fenergy3      =     aa_rhomax["total3"]          
        s520_fenergy3        =     aa_unk520["total3"]
        hybrid_rho_fenergy3  = aa_hybrid_rho["total3"]
        hybrid_fenergy3      =     aa_hybrid["total3"]

        wall_fenergy4        =       aa_wall["total4"]
        rhomax_fenergy4      =     aa_rhomax["total4"]          
        s520_fenergy4        =     aa_unk520["total4"]
        hybrid_rho_fenergy4  = aa_hybrid_rho["total4"]
        hybrid_fenergy4      =     aa_hybrid["total4"]

        # now compute sums and statistics
        
        sum_fraw_particles1 =  wall_fraw_particles1 + rhomax_fraw_particles1 +  s520_fraw_particles1 + hybrid_rho_fraw_particles1 + hybrid_fraw_particles1
        sum_fparticles1     =  wall_fparticles1     + rhomax_fparticles1     +  s520_fparticles1     + hybrid_rho_fparticles1     + hybrid_fparticles1
        sum_fenergy1        =  wall_fenergy1        + rhomax_fenergy1        +  s520_fenergy1        + hybrid_rho_fenergy1        + hybrid_fenergy1

        sum_fraw_particles2 =  wall_fraw_particles2 + rhomax_fraw_particles2 +  s520_fraw_particles2 + hybrid_rho_fraw_particles2 + hybrid_fraw_particles2
        sum_fparticles2     =  wall_fparticles2     + rhomax_fparticles2     +  s520_fparticles2     + hybrid_rho_fparticles2     + hybrid_fparticles2
        sum_fenergy2        =  wall_fenergy2        + rhomax_fenergy2        +  s520_fenergy2        + hybrid_rho_fenergy2        + hybrid_fenergy2

        sum_fraw_particles3 =  wall_fraw_particles3 + rhomax_fraw_particles3 +  s520_fraw_particles3 + hybrid_rho_fraw_particles3 + hybrid_fraw_particles3
        sum_fparticles3     =  wall_fparticles3     + rhomax_fparticles3     +  s520_fparticles3     + hybrid_rho_fparticles3     + hybrid_fparticles3
        sum_fenergy3        =  wall_fenergy3        + rhomax_fenergy3        +  s520_fenergy3        + hybrid_rho_fenergy3        + hybrid_fenergy3

        sum_fraw_particles4 =  wall_fraw_particles4 + rhomax_fraw_particles4 +  s520_fraw_particles4 + hybrid_rho_fraw_particles4 + hybrid_fraw_particles4
        sum_fparticles4     =  wall_fparticles4     + rhomax_fparticles4     +  s520_fparticles4     + hybrid_rho_fparticles4     + hybrid_fparticles4
        sum_fenergy4        =  wall_fenergy4        + rhomax_fenergy4        +  s520_fenergy4        + hybrid_rho_fenergy4        + hybrid_fenergy4

        fraw_particles_array = np.array([sum_fraw_particles1, sum_fraw_particles2, sum_fraw_particles3, sum_fraw_particles4])
        fparticles_array     = np.array([sum_fparticles1,     sum_fparticles2,     sum_fparticles3,     sum_fparticles4])
        fenergy_array        = np.array([sum_fenergy1,        sum_fenergy2,        sum_fenergy3,        sum_fenergy4])
        
        
        fraw_particles_mean  = np.mean(fraw_particles_array)
        fparticles_mean      = np.mean(fparticles_array)
        fenergy_mean         = np.mean(fenergy_array)

        fraw_particles_std  = np.std(fraw_particles_array)
        fparticles_std      = np.std(fparticles_array)
        fenergy_std         = np.std(fenergy_array)

        fraw_particles_stderr = fraw_particles_std / np.sqrt(4.)
        fparticles_stderr     = fparticles_std     / np.sqrt(4.)
        fenergy_stderr        = fenergy_std        / np.sqrt(4.)

        fraw_particles_relerr = 0.
        fparticles_relerr     = 0. 
        fenergy_relerr        = 0.

        if (fraw_particles_mean > 0):
            fraw_particles_relerr = fraw_particles_stderr / fraw_particles_mean

        if(fparticles_mean > 0):
            fparticles_relerr    = fparticles_stderr      / fparticles_mean

        if(fenergy_stderr > 0):
            fenergy_relerr       = fenergy_stderr         / fenergy_mean


        #   those values are for an ensemble of markers that is four    
        #   times smaller than the full ensemble.  So our final
        #   values for the uncertainties must be divided by two.

        full_fraw_particles_relerr = fraw_particles_relerr / 2.
        full_fparticles_relerr     = fparticles_relerr     / 2. 
        full_fenergy_relerr        = fenergy_relerr        / 2.
        
        # statistics on raw number of particles is meaningless because by
        #  construction they are almost the same (broke all markers
        #  into nearly same-sized arrays)

        fraw_particles_relerr = -99.   # flag as meaningless
        
        print("\n ----------------------------------------------")
        print("\n Fractions for markers actually simulated\n")
        print(" endcond      particle    particle   energy\n")
        print("               (raw)      fraction   fraction\n")
        

        print("  wall1       %9.5f    %9.5f   %9.5f  " % (wall_fraw_particles1,   wall_fparticles1,   wall_fenergy1))
        print("  wall2       %9.5f    %9.5f   %9.5f  " % (wall_fraw_particles2,   wall_fparticles2,   wall_fenergy2))
        print("  wall3       %9.5f    %9.5f   %9.5f  " % (wall_fraw_particles3,   wall_fparticles3,   wall_fenergy3))
        print("  wall4       %9.5f    %9.5f   %9.5f  " % (wall_fraw_particles4,   wall_fparticles4,   wall_fenergy4))
        print("")
        print("  rhomax1     %9.5f    %9.5f   %9.5f  " % (rhomax_fraw_particles1, rhomax_fparticles1, rhomax_fenergy1))
        print("  rhomax2     %9.5f    %9.5f   %9.5f  " % (rhomax_fraw_particles2, rhomax_fparticles2, rhomax_fenergy2))
        print("  rhomax3     %9.5f    %9.5f   %9.5f  " % (rhomax_fraw_particles3, rhomax_fparticles3, rhomax_fenergy3))
        print("  rhomax4     %9.5f    %9.5f   %9.5f  " % (rhomax_fraw_particles4, rhomax_fparticles4, rhomax_fenergy4))
        print("")
        print("  unk5201     %9.5f    %9.5f   %9.5f  " % (s520_fraw_particles1,   s520_fparticles1,   s520_fenergy1))
        print("  unk5202     %9.5f    %9.5f   %9.5f  " % (s520_fraw_particles2,   s520_fparticles2,   s520_fenergy2))
        print("  unk5203     %9.5f    %9.5f   %9.5f  " % (s520_fraw_particles3,   s520_fparticles3,   s520_fenergy3))
        print("  unk5204     %9.5f    %9.5f   %9.5f  " % (s520_fraw_particles4,   s520_fparticles4,   s520_fenergy4))
        print("")
        print("  hybrid_rho1 %9.5f    %9.5f   %9.5f  " % (hybrid_rho_fraw_particles1,  hybrid_rho_fparticles1, hybrid_rho_fenergy1))
        print("  hybrid_rho2 %9.5f    %9.5f   %9.5f  " % (hybrid_rho_fraw_particles2,  hybrid_rho_fparticles2, hybrid_rho_fenergy2))
        print("  hybrid_rho3 %9.5f    %9.5f   %9.5f  " % (hybrid_rho_fraw_particles3,  hybrid_rho_fparticles3, hybrid_rho_fenergy3))
        print("  hybrid_rho4 %9.5f    %9.5f   %9.5f  " % (hybrid_rho_fraw_particles4,  hybrid_rho_fparticles4, hybrid_rho_fenergy4))
        print("")
        print("  hybrid1 %9.5f    %9.5f   %9.5f  " % (hybrid_fraw_particles1,  hybrid_fparticles1, hybrid_fenergy1))
        print("  hybrid2 %9.5f    %9.5f   %9.5f  " % (hybrid_fraw_particles2,  hybrid_fparticles2, hybrid_fenergy2))
        print("  hybrid3 %9.5f    %9.5f   %9.5f  " % (hybrid_fraw_particles3,  hybrid_fparticles3, hybrid_fenergy3))
        print("  hybrid4 %9.5f    %9.5f   %9.5f  " % (hybrid_fraw_particles4,  hybrid_fparticles4, hybrid_fenergy4))
        print("")
        print("  sum1 %9.5f    %9.5f   %9.5f  " % (sum_fraw_particles1,  sum_fparticles1, sum_fenergy1))
        print("  sum2 %9.5f    %9.5f   %9.5f  " % (sum_fraw_particles2,  sum_fparticles2, sum_fenergy2))
        print("  sum3 %9.5f    %9.5f   %9.5f  " % (sum_fraw_particles3,  sum_fparticles3, sum_fenergy3))
        print("  sum4 %9.5f    %9.5f   %9.5f  " % (sum_fraw_particles4,  sum_fparticles4, sum_fenergy4))
        print("")


        print(" mean %9.5f    %9.5f   %9.5f         " % (fraw_particles_mean,   fparticles_mean,   fenergy_mean))
        print("  std %9.5f    %9.5f   %9.5f         " % (fraw_particles_std,    fparticles_std,    fenergy_std))
        print(" uncertainty %9.5f    %9.5f   %9.5f  " % (fraw_particles_relerr, fparticles_relerr, fenergy_relerr))
        print("")
        print("")
        print("  wall       %9.5f    %9.5f   %9.5f  " % (wall_fraw_particles,   wall_fparticles,   wall_fenergy))
        print("  rhomax     %9.5f    %9.5f   %9.5f  " % (rhomax_fraw_particles, rhomax_fparticles, rhomax_fenergy))
        print("  unk520     %9.5f    %9.5f   %9.5f  " % (s520_fraw_particles,   s520_fparticles,   s520_fenergy))
        print("  hybrid_rho %9.5f    %9.5f   %9.5f  " % (hybrid_rho_fraw_particles,  hybrid_rho_fparticles, hybrid_rho_fenergy))
        print("")
        print("  sum        %9.5f    %9.5f   %9.5f  " % (sum_fraw_particles, sum_fparticles, sum_fenergy))
        print(" uncertainty %9.5f    %9.5f   %9.5f  " % (fraw_particles_relerr, fparticles_relerr, fenergy_relerr))
        print("")

        print("   zzz time to complete first set statistics: ", mytime.time() - time_last)
        time_last = mytime.time()

        
        print(" full-ensemble: uncertainty weighted lost particles: ", full_fparticles_relerr)
        print(" full-ensemble: uncertainty weighted lost energy:    ", full_fenergy_relerr)
        # ---------------------------------------------------------------------------

        print("\n ----------------------------------------------")
        print(" fraction alphas simulated: %8.4f" % (fraction_alphas_simulated))
        print("\n adjusted fractions\n")
        print(" endcond     particle   particle  energy\n")
        print("              (raw)     fraction  fraction\n")

        awall_fraw_particles   = wall_fraw_particles    * fraction_alphas_simulated
        awall_fparticles       = wall_fparticles        * fraction_alphas_simulated
        awall_fenergy          = wall_fenergy           * fraction_alphas_simulated

        arhomax_fraw_particles = rhomax_fraw_particles  * fraction_alphas_simulated
        arhomax_fparticles     = rhomax_fparticles      * fraction_alphas_simulated
        arhomax_fenergy        = rhomax_fenergy         * fraction_alphas_simulated

        as520_fraw_particles = s520_fraw_particles  * fraction_alphas_simulated
        as520_fparticles     = s520_fparticles      * fraction_alphas_simulated
        as520_fenergy        = s520_fenergy         * fraction_alphas_simulated

        asum_fraw_particles    = sum_fraw_particles     * fraction_alphas_simulated
        asum_fparticles        = sum_fparticles         * fraction_alphas_simulated
        asum_fenergy           = sum_fenergy            * fraction_alphas_simulated

        print("  wall     %8.4f    %8.4f   %8.4f  " % (awall_fraw_particles,   awall_fparticles,   awall_fenergy))
        print("  rhomax   %8.4f    %8.4f   %8.4f  " % (arhomax_fraw_particles, arhomax_fparticles, arhomax_fenergy))
        print("  unk520   %8.4f    %8.4f   %8.4f  " % (as520_fraw_particles, as520_fparticles, as520_fenergy))
        print("")
        print("  sum      %8.4f    %8.4f   %8.4f  " % (asum_fraw_particles, asum_fparticles, asum_fenergy))
        print("")

        #  my best guess  8/17/2020
        
        sds_energy_rel_energy_uncertainty = (fenergy_std/fenergy_mean)/2.  # e.g. 5000 --> 20000 particles
        
        print(' sds-best-rel-energy-uncertainty: ', sds_energy_rel_energy_uncertainty)

        
        print("  std %9.5f    %9.5f   %9.5f         " % (fraw_particles_std,    fparticles_std,    fenergy_std))
    
        # ---------------------------------------------------------------------------
        
        nn             = cputime.size    # total number markers simulated
    
        total_cpu_time = np.sum(cputime)
        total_sim_time = np.sum(time)

        avg_cpu_time   = total_cpu_time / nn
        avg_sim_time   = total_sim_time / nn

        max_cpu_time   = np.max(cputime)
        max_sim_time   = np.max(time)

        ratio_cpu_sim = avg_cpu_time / avg_sim_time

        print("")
        print("  across all particles (i.e. all end-conditions)")
        print("")
        print("   total sim time:  %11.3f" % (total_sim_time))
        print("   total cpu time:  %11.3f" % (total_cpu_time))
        print("")
        print("   avg sim time:   %10.5f" % (avg_sim_time))
        print("   max sim time:   %10.5f" % (max_sim_time))
        print("")
        print("   avg   cpu time:  %9.1f" % (avg_cpu_time))
        print("   max   cpu time:  %9.1f" % (max_cpu_time))

        print("   cpu / sim ratio: %9.1f" % (ratio_cpu_sim))
        print("")

        ii_emin       = (endcond == 2)
        ii_therm      = (endcond == 4)
        ii_survived   = (endcond == 2) ^ (endcond == 4) ^ (endcond == 1) ^ (endcond == 256)
        ii_simtime    = (endcond == 1)
        ii_cputime    = (endcond == 256)
        ii_abort     = ( endcond == 0)

        nn_cputime    = cputime[ii_cputime].size
        if(nn_cputime>0):
            simtime_min = np.min(time[ii_cputime])
            simtime_max = np.max(time[ii_cputime])
            print("")
            print(" cpu-time min/max of markers that reach max-cputime: %9.5f %9.5f"%(simtime_min,simtime_max))
            print("")
        # ---------------------------------------------------------------
 

        deltarho_emin_all = delta_rho[ii_emin]
        nn_emin_all       = deltarho_emin_all.size
        ii_positive       = (deltarho_emin_all >= 0)
        nn_emin_positive  = deltarho_emin_all[ii_positive].size
        ii_negative       = (deltarho_emin_all < 0)
        nn_emin_negative  = deltarho_emin_all[ii_negative].size

        deltarho_therm_all = delta_rho[ii_therm]
        nn_therm_all       = deltarho_therm_all.size
        ii_positive       = (deltarho_therm_all >= 0)
        nn_therm_positive  = deltarho_therm_all[ii_positive].size
        ii_negative       = (deltarho_therm_all < 0)
        nn_therm_negative  = deltarho_therm_all[ii_negative].size

        deltarho_simtime_all = delta_rho[ii_simtime]
        nn_simtime_all       = deltarho_simtime_all.size
        ii_positive       = (deltarho_simtime_all >= 0)
        nn_simtime_positive  = deltarho_simtime_all[ii_positive].size
        ii_negative       = (deltarho_simtime_all < 0)
        nn_simtime_negative  = deltarho_simtime_all[ii_negative].size

        deltarho_survived_all = delta_rho[ii_survived]
        nn_survived_all       = deltarho_survived_all.size
        ii_positive       = (deltarho_survived_all >= 0)
        nn_survived_positive  = deltarho_survived_all[ii_positive].size
        ii_negative       = (deltarho_survived_all < 0)
        nn_survived_negative  = deltarho_survived_all[ii_negative].size

        deltarho_cputime_all = delta_rho[ii_cputime]
        nn_cputime_all       = deltarho_cputime_all.size
        ii_positive          = (deltarho_cputime_all >= 0)
        nn_cputime_positive  = deltarho_cputime_all[ii_positive].size
        ii_negative          = (deltarho_cputime_all < 0)
        nn_cputime_negative  = deltarho_cputime_all[ii_negative].size
        
        nn_rho = 10
        
        deltarho_emin_mean = np.zeros(nn_rho)
        deltarho_emin_stdev = np.zeros(nn_rho)
        deltarho_emin_serr  = np.zeros(nn_rho)

        deltarho_simtime_mean  = np.zeros(nn_rho)
        deltarho_simtime_stdev = np.zeros(nn_rho)
        deltarho_simtime_serr = np.zeros(nn_rho)

        deltarho_therm_mean  = np.zeros(nn_rho)
        deltarho_therm_stdev = np.zeros(nn_rho)
        deltarho_therm_serr  = np.zeros(nn_rho)

        deltarho_survived_mean  = np.zeros(nn_rho)
        deltarho_survived_stdev = np.zeros(nn_rho)
        deltarho_survived_serr  = np.zeros(nn_rho)

        deltarho_cputime_mean  = np.zeros(nn_rho)
        deltarho_cputime_stdev = np.zeros(nn_rho)
        deltarho_cputime_serr  = np.zeros(nn_rho)

        my_drho = 1./nn_rho
        fake_rho_array = np.linspace(0.5/nn_rho, 1.-0.5/nn_rho, nn_rho)
        
        for mxx in range(nn_rho):
            
            jj_emin       = ( rho_ini[ii_emin]     >= mxx*my_drho)& (rho_ini[ii_emin]     <= (mxx+1)*my_drho) 
            jj_simtime    = ( rho_ini[ii_simtime]  >= mxx*my_drho)& (rho_ini[ii_simtime]  <= (mxx+1)*my_drho)
            jj_cputime    = ( rho_ini[ii_cputime]  >= mxx*my_drho)& (rho_ini[ii_cputime]  <= (mxx+1)*my_drho)
            jj_therm      = ( rho_ini[ii_therm]    >= mxx*my_drho) & (rho_ini[ii_therm]    <= (mxx+1)*my_drho) 
            jj_survived   = ( rho_ini[ii_survived] >= mxx*my_drho)& (rho_ini[ii_survived] <= (mxx+1)*my_drho) 

            
            if (  deltarho_emin_all[jj_emin].size >= 10):
                deltarho_emin_mean[mxx]  = np.mean(deltarho_emin_all[jj_emin])
                deltarho_emin_stdev[mxx] =  np.std(deltarho_emin_all[jj_emin])
                ##pdb.set_trace()
                deltarho_emin_serr[mxx]  =  np.std(deltarho_emin_all[jj_emin])/np.sqrt(deltarho_emin_all[jj_emin].size)

            if (  deltarho_cputime_all[jj_cputime].size >= 10):
                deltarho_cputime_mean[mxx]  = np.mean(deltarho_cputime_all[jj_cputime])
                deltarho_cputime_stdev[mxx] =  np.std(deltarho_cputime_all[jj_cputime])
                ##pdb.set_trace()
                deltarho_cputime_serr[mxx]  =  np.std(deltarho_cputime_all[jj_cputime])/np.sqrt(deltarho_cputime_all[jj_cputime].size)
                
            if (  deltarho_therm_all[jj_therm].size >= 10):
                deltarho_therm_mean[mxx]  = np.mean(deltarho_therm_all[jj_therm])
                deltarho_therm_stdev[mxx] =  np.std(deltarho_therm_all[jj_therm])
                deltarho_therm_serr[mxx]  =  np.std(deltarho_therm_all[jj_therm])/np.sqrt(deltarho_therm_all[jj_therm].size)

            if (  deltarho_simtime_all[jj_simtime].size >= 10):
                deltarho_simtime_mean[mxx]  = np.mean(deltarho_simtime_all[jj_simtime])
                deltarho_simtime_stdev[mxx] =  np.std(deltarho_simtime_all[jj_simtime])
                deltarho_simtime_serr[mxx]  =  np.std(deltarho_simtime_all[jj_simtime])/np.sqrt(deltarho_simtime_all[jj_simtime].size)

            if (  deltarho_survived_all[jj_survived].size >= 10):
                deltarho_survived_mean[mxx]  = np.mean(deltarho_survived_all[jj_survived])
                deltarho_survived_stdev[mxx] =  np.std(deltarho_survived_all[jj_survived])
                deltarho_survived_serr[mxx]  =  np.std(deltarho_survived_all[jj_survived])/np.sqrt(deltarho_survived_all[jj_survived].size)

        print("")
        print(" survived markers:  deltarho (end - ini)")
        print("   rho     mean     sigma     serr")
        for mxx in range(nn_rho):
            print(" %6.2f %8.4f  %8.4f  %8.4f"%( fake_rho_array[mxx], deltarho_survived_mean[mxx], deltarho_survived_stdev[mxx], deltarho_survived_serr[mxx]))

        print("")
        print(" simtime markers:  deltarho (end - ini)")
        print("   rho     mean     sigma     serr")
        for mxx in range(nn_rho):
            print(" %6.2f %8.4f  %8.4f  %8.4f"%( fake_rho_array[mxx], deltarho_simtime_mean[mxx], deltarho_simtime_stdev[mxx], deltarho_simtime_serr[mxx]))

        print("")
        print(" emin markers:  deltarho (end - ini)")
        print("   rho     mean     sigma     serr")
        for mxx in range(nn_rho):
            print(" %6.2f %8.4f  %8.4f  %8.4f"%( fake_rho_array[mxx], deltarho_emin_mean[mxx], deltarho_emin_stdev[mxx], deltarho_emin_serr[mxx]))

        print("")
        print(" therm markers:  deltarho (end - ini)")
        print("   rho     mean     sigma     serr")
        for mxx in range(nn_rho):
            print(" %6.2f %8.4f  %8.4f  %8.4f"%( fake_rho_array[mxx], deltarho_therm_mean[mxx], deltarho_therm_stdev[mxx], deltarho_therm_serr[mxx]))

        print("")
        print(" cputime markers:  deltarho (end - ini)")
        print("   rho     mean     sigma     serr")
        for mxx in range(nn_rho):
            print(" %6.2f %8.4f  %8.4f  %8.4f"%( fake_rho_array[mxx], deltarho_cputime_mean[mxx], deltarho_cputime_stdev[mxx], deltarho_cputime_serr[mxx]))

        print("   zzz time to complete second set of statistics: ", mytime.time()-time_last)
        time_last = mytime.time()

        print("   ... at position B at time: ", clock.ctime())
            
        xx_fake = np.linspace(0.,1.,2)
        yy_fake = 0. * xx_fake
        plt.close()
        plt.figure()
        plt.plot(xx_fake, yy_fake, 'r-', linewidth=1)
        plt.plot(fake_rho_array, deltarho_emin_mean, 'bo', linewidth=1, rasterized=do_rasterized)
        plt.errorbar(fake_rho_array, deltarho_emin_mean, yerr=deltarho_emin_stdev, linewidth=1, rasterized=do_rasterized)
        plt.errorbar(fake_rho_array, deltarho_emin_mean, yerr=deltarho_emin_serr,  linewidth=1, rasterized=do_rasterized, ecolor='r', fmt='none')
        plt.xlabel('rho', fontsize=fontsize_2)
        plt.xlim(0., 1.)
        plt.title('mean delta_rho [emin]', fontsize=fontsize_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.close()
        plt.figure()
        plt.plot(xx_fake, yy_fake, 'r-', linewidth=1)
        plt.plot(fake_rho_array, deltarho_therm_mean, 'bo', linewidth=1, rasterized=do_rasterized)
        plt.errorbar(fake_rho_array, deltarho_therm_mean, yerr=deltarho_therm_stdev,  linewidth=1, rasterized=do_rasterized)
        plt.errorbar(fake_rho_array, deltarho_therm_mean, yerr=deltarho_therm_serr,   linewidth=2, rasterized=do_rasterized,ecolor='r', fmt='none')
        plt.xlabel('rho', fontsize=fontsize_2)
        plt.xlim(0., 1.)
        plt.title('mean delta_rho [therm]', fontsize=fontsize_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.close()
        plt.figure()
        plt.plot(xx_fake, yy_fake, 'r-', linewidth=1)
        plt.plot(fake_rho_array, deltarho_simtime_mean, 'bo', linewidth=1, rasterized=do_rasterized)
        plt.errorbar(fake_rho_array, deltarho_simtime_mean, yerr=deltarho_simtime_stdev,  linewidth=1, rasterized=do_rasterized)
        plt.errorbar(fake_rho_array, deltarho_simtime_mean, yerr=deltarho_simtime_serr, linewidth=2, rasterized=do_rasterized,ecolor='r', fmt='none')
        plt.xlabel('rho', fontsize=fontsize_2)
        plt.xlim(0., 1.)
        plt.title('mean delta_rho [simtime]', fontsize=fontsize_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.close()
        plt.figure()
        plt.plot(xx_fake, yy_fake, 'r-', linewidth=1)
        plt.plot(fake_rho_array, deltarho_cputime_mean,  'bo-', linewidth=1, rasterized=do_rasterized)
        plt.errorbar(fake_rho_array, deltarho_cputime_mean, yerr=deltarho_cputime_stdev,  linewidth=1, rasterized=do_rasterized)
        plt.errorbar(fake_rho_array, deltarho_cputime_mean, yerr=deltarho_cputime_serr,   linewidth=2, rasterized=do_rasterized,ecolor='r', fmt='none')
        plt.xlabel('rho', fontsize=fontsize_2)
        plt.xlim(0., 1.)
        plt.title('mean delta_rho [cputime]', fontsize=fontsize_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.close()
        plt.figure()
        plt.plot(fake_rho_array, deltarho_survived_mean, 'bo', linewidth=1, rasterized=do_rasterized)
        plt.errorbar(fake_rho_array, deltarho_survived_mean, yerr=deltarho_survived_stdev,  linewidth=1, rasterized=do_rasterized)
        plt.errorbar(fake_rho_array, deltarho_survived_mean, yerr=deltarho_survived_serr, linewidth=2, rasterized=do_rasterized, ecolor='r', fmt='none')
        plt.xlabel('rho', fontsize=fontsize_2)
        plt.xlim(0., 1.)
        plt.title('mean delta_rho [survived]', fontsize=fontsize_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()
        
        
        print("")
        print("--------------------------------------------------")
        print("")
        print("   end_condition        delta_rho     >0      <0")
        print("")
        print("    emin      %15d  %10d    %d" % (nn_emin_all,     nn_emin_positive,     nn_emin_negative))
        print("    therm     %15d  %10d    %d" % (nn_therm_all,    nn_therm_positive,    nn_therm_negative))
        print("    simtime   %15d  %10d    %d" % (nn_simtime_all,  nn_simtime_positive,  nn_simtime_negative))
        print("    cputime   %15d  %10d    %d" % (nn_cputime_all,  nn_cputime_positive,  nn_cputime_negative))
        print("    combined  %15d  %10d    %d" % (nn_survived_all, nn_survived_positive, nn_survived_negative))
        print("")
    
        # -------------------------------------------------------------------
        
        nn   = cputime[ii_emin].size    # total number markers simulated

        if (nn > 0):

            total_cpu_time = np.sum(cputime[ii_emin])
            total_sim_time = np.sum(time[ii_emin])

            avg_cpu_time   = total_cpu_time / nn
            avg_sim_time   = total_sim_time / nn

            max_cpu_time   = np.max(cputime[ii_emin])
            max_sim_time   = np.max(time[ii_emin])

            ratio_cpu_sim = avg_cpu_time / avg_sim_time

            print("")
            print(" looking only at particles that reach emin")
            print("")
            print("   total sim time:  %9.1f" % (total_sim_time))
            print("   total cpu time:  %9.1f" % (total_cpu_time))
            print("")
            print("   avg sim time:   %9.4f" % (avg_sim_time))
            print("   max sim time:   %9.4f" % (max_sim_time))
            print("")
            print("   avg   cpu time:  %9.1f" % (avg_cpu_time))
            print("   max   cpu time:  %9.1f" % (max_cpu_time))

            print("   cpu / sim ratio: %9.1f" % (ratio_cpu_sim))
            print("")

        ii_no_abort = ( endcond != 0)

        rmarker_all = r_marker[ii_no_abort]
        zmarker_all = z_marker[ii_no_abort]
      
        ii_wall   = (endcond == 8)
        ii_rhomax = (endcond == 32) ^ (endcond == 544)
        ii_520    = (endcond == 520)
        
        ii_lost   =    (endcond ==  8)    \
                    ^  (endcond ==  16)   \
                    ^  (endcond ==  32)   \
                    ^  (endcond ==  40)   \
                    ^  (endcond == 512)   \
                    ^  (endcond == 520)   \
                    ^  (endcond == 544) 

        
        ekev_lost   = ekev[ii_lost]
        weight_lost = weight[ii_lost]
        nlost       = ekev_lost.size
        nall        = endcond.size

        weight_survived = weight[ii_survived]
        ekev_survived   =   ekev[ii_survived]
        nsurvived       = ekev_survived.size

        # -------------------------------------------
        #  write file for Becky
        
        myfile=open(filename_text,"w")
        for jjkk in range(nlost):
            myfile.write(" %d  %10.5f  %12.9f \n" % (jjkk, ekev_lost[jjkk], weight_lost[jjkk]))
        for jjkk in range(nsurvived):
            myfile.write(" %d %10.5f  %12.9f \n" % (jjkk+nlost, 0., weight_survived[jjkk]))                                 
        myfile.close()
                                              # --------------------------------------------
        #  compute uncertainty in average lost energy

        nlost       = ekev_lost.size
        nall        = endcond.size

        particle_loss_frac         = nlost/nall
        serr_particles             = np.sqrt(nall * particle_loss_frac * (1.-particle_loss_frac))
        fractional_error_particles = serr_particles / nlost
                                           
        average_lost_energy = np.sum(ekev_lost) / nall

        my_sum = (nall-nlost)*(average_lost_energy)*2   # confined alphas have zero lost energy

        for rr in range(nlost):
            my_sum = my_sum + (ekev_lost[rr] - average_lost_energy)**2

        standard_error_lost_energy = np.sqrt(my_sum / (nall*(nall-1))   )
        fractional_error           = standard_error_lost_energy / average_lost_energy

        print("")
        print("  KRS  ")
        print("   nall:                      ", nall)
        print("   lost:                      ", nlost)
        print("   avg_lost_energy:           ", average_lost_energy)
        print("   my_sum:                    ", my_sum)
        print("   serr_lost_energy:          ", standard_error_lost_energy)
        print("   fractional_error:          ", fractional_error)
        print("   fractional_particle_error: ", fractional_error_particles)
        print("")
        
        # ---------------------------------------
        phi_lost   = phi_end[ii_lost]   % 360.
        phi_wall   = phi_end[ii_wall]   % 360.
        phi_rhomax = phi_end[ii_rhomax] % 360.
        phi_520    = phi_end[ii_520]    % 360.

    
        phi_lost_18 = phi_lost % 20
        phi_lost_16 = phi_lost % 22.5
        phi_lost_14 = phi_lost % 25.7143
        phi_lost_12 = phi_lost % 30.

 
        weight_energy_lost = np.multiply(weight_lost, ekev_lost)
        pitch_ini_lost = vpar_ini[ii_lost] / vtot_ini[ii_lost]
        r_ini_lost     = r_ini[ii_lost]
        z_ini_lost     = z_ini[ii_lost]
        rho_ini_lost   = rho_ini[ii_lost]
        rho_ini_wall   = rho_ini[ii_wall]
        rho_ini_rhomax = rho_ini[ii_rhomax]
        rho_ini_520    = rho_ini[ii_520]

        rho_marker_lost    = rho_marker[ii_lost]
        weight_marker_lost = weight[ii_lost]
        energy_lost        = ekev[ii_lost]
        rho_marker_wall    = rho_marker[ii_wall]
        rho_marker_rhomax  = rho_marker[ii_rhomax]
        rho_marker_520     = rho_marker[ii_520]

        #  changed from 2.e-5 to 3.e-5 8:40 AM 5/13/2021
        simtime_lost = time[ii_lost]
        ii_prompt    = (simtime_lost < 3.e-5)
        ii_medium    = (simtime_lost >= 3.e-5) & (simtime_lost < 0.01)
        ii_long      = (simtime_lost >= 0.01)
        ii_nonprompt = (simtime_lost >= 3.e-5)

        total_weight_energy_born = np.sum(weight*ekev_ini)  # was 3515 until 4/13/2021
        floss_prompt    = np.sum(weight_energy_lost[ii_prompt])    / total_weight_energy_born
        floss_nonprompt = np.sum(weight_energy_lost[ii_nonprompt]) / total_weight_energy_born

        nloss_prompt = weight_energy_lost[ii_prompt].size
        nloss_nonprompt = weight_energy_lost[ii_nonprompt].size
        ntotal_all = time.size

        print("\n Number of total markers born:                  ", ntotal_all)
        print(" Number of markers prompt-lost (unweighted):    ", nloss_prompt)
        print(" Number of markers nonprompt-lost (unweighted): ", nloss_nonprompt)

        
        weight_total = np.sum(weight)
        weight_prompt = np.sum(weight_lost[ii_prompt])
        weight_nonprompt = np.sum(weight_lost[ii_nonprompt])

        print(" Total weight of all markers:      ", weight_total)
        print(" weight of prompt-lost markers:    ", weight_prompt)
        print(" weight of nonprompt-lost markers: ", weight_nonprompt, "\n")
                                             
        print(" fraction of initial energy that is prompt-lost (weighted):    ", floss_prompt)
        print(" fraction of initial energy that is nonprompt-lost (weighted): ", floss_nonprompt, "\n")
        
        total_energy_lost = np.sum(weight_energy_lost)
        
        eloss_prompt = np.sum(weight_energy_lost[ii_prompt]) / total_energy_lost

        eloss_nonprompt = np.sum(weight_energy_lost[ii_nonprompt]) / total_energy_lost
        
        eloss_medium = np.sum(weight_energy_lost[ii_medium]) / total_energy_lost
        eloss_long   = np.sum(weight_energy_lost[ii_long])   / total_energy_lost

        
        print("\n   fraction energy lost, time < 3.e-5:     ", eloss_prompt)
        print("   fraction energy lost, time > 3.e-5:     ", eloss_nonprompt)
        print("   fraction energy lost, 3.e-5 < t < 0.01: ", eloss_medium)
        print("   fraction energy lost, time > 0.01:      ", eloss_long, "\n")

        print("   ... at position C at time: ", clock.ctime())
        
        plt.close()
        plt.plot(simtime_lost, ekev_lost, 'ro', ms=1)
        plt.xlabel('time [sec]')
        plt.ylabel('[ekev]')
        plt.ylim(bottom=0.)
        plt.xlim(left=0.)
        plt.title('Marker energy at time of loss')
        plt.grid(axis='both', alpha=0.50)
        pdf.savefig()
        
        ii_zeroes    = (simtime_lost == 0.)
        simtime_lost[ii_zeroes] = 1.e-10

        theta_lost   = theta_end[ii_lost]   % 360.
        theta_wall   = theta_end[ii_wall]   % 360.
        theta_rhomax = theta_end[ii_rhomax] % 360.
        theta_520    = theta_end[ii_520]    % 360.

        theta_new      = np.arctan2(z_end, (r_end-1.90)) * 180./np.pi
        theta_lost_new = theta_new[ii_lost]

        pitch_lost   = vpar[ii_lost] / vtot[ii_lost]


        print("   zzz time to complete third set of statisticvs: ", mytime.time()-time_last)
        time_last = mytime.time()
        # --------------------------------------------------
        #   plot start and end positions of lost markers

        print("I am about to plot start and end positions of lost markers'")

        my_zmin =  1.1 * np.min(zl)
        my_zmax =  1.1 * np.max(zl)
        my_rmin =  0.9 * np.min(rl)
        my_rmax =  1.1 * np.max(rl)

        xsize = 4.
        ysize = xsize * (my_zmax-my_zmin)/(my_rmax-my_rmin)


        rho_list_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, ]
        rho_list_2 = [1.02, 1.04]
        color_1 = 'b'
        color_2 = 'gold'

        # ----------------------------------------------------------------

        plt.close()
        plt.figure(figsize=(9.,7.))
        plt.plot(phi_lost, theta_lost_new, 'ro', ms=0.07, rasterized=do_rasterized)
        plt.xlabel('phi')
        plt.ylabel('theta')
        plt.tight_layout(h_pad=1.5)
        plt.grid(axis='both', alpha=0.75)
        plt.title('theta vs phi of lost markers',fontsize=12)
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(9.,7.))
        plt.plot(phi_lost_18, theta_lost_new, 'ro', ms=0.05, rasterized=do_rasterized)
        plt.xlabel('phi mod(20)')
        plt.ylabel('theta')
        plt.tight_layout(h_pad=1.5)
        plt.grid(axis='both', alpha=0.75)
        plt.title('theta vs phi mod(20for 18T.F) of lost markers',fontsize=12)
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(9.,7.))
        plt.plot(phi_lost_16, theta_lost_new, 'ro', ms=0.10, rasterized=do_rasterized)
        plt.xlabel('phi mod(20)')
        plt.ylabel('theta')
        plt.tight_layout(h_pad=1.5)
        plt.grid(axis='both', alpha=0.75)
        plt.title('theta vs phi mod(22.5for 16TF) of lost markers',fontsize=12)
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(9.,7.))
        plt.plot(phi_lost_14, theta_lost_new, 'ro', ms=0.10, rasterized=do_rasterized)
        plt.xlabel('phi mod(20)')
        plt.ylabel('theta')
        plt.tight_layout(h_pad=1.5)
        plt.grid(axis='both', alpha=0.75)
        plt.title('theta vs phi mod(25.71 for 14TF) of lost markers',fontsize=12)
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(9.,7.))
        plt.plot(phi_lost_12, theta_lost_new, 'ro', ms=0.10, rasterized=do_rasterized)
        plt.xlabel('phi mod(20)')
        plt.ylabel('theta')
        plt.tight_layout(h_pad=1.5)
        plt.grid(axis='both', alpha=0.75)
        plt.title('theta vs phi mod(30 for 12TF) of lost markers',fontsize=12)
        pdf.savefig()
        
        # --------------------------------------------------------------

        qq_rhomarker_lost = rho_marker[ii_lost]
        qq_rhoini_lost    =    rho_ini[ii_lost]
        qq_rhoend_lost    =    rho_end[ii_lost]

        qq_ii = (qq_rhoini_lost > 1.)

        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(qq_rhoini_lost[qq_ii], qq_rhoend_lost[qq_ii], 'ro', ms=2.0, rasterized=do_rasterized)
        plt.xlabel('rho_ini')
        plt.ylabel('rho_end')
        plt.tight_layout(h_pad=1.5)
        plt.grid(axis='both', alpha=0.75)
        plt.title('markers with rho_in i> 1 : rho_end vs rho_ini > 1',fontsize=12)
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(qq_rhoini_lost[qq_ii], qq_rhoend_lost[qq_ii]-qq_rhoini_lost[qq_ii], 'ro', ms=2.0, rasterized=do_rasterized)
        plt.xlabel('rho_ini')
        
        plt.tight_layout(h_pad=1.5)
        plt.grid(axis='both', alpha=0.75)
        plt.title('markers with rho_in i> 1 : rho_end - rho_ini > 1',fontsize=12)
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(qq_rhomarker_lost[qq_ii], qq_rhoend_lost[qq_ii], 'ro', ms=2.0, rasterized=do_rasterized)
        plt.xlabel('rho_marker')
        plt.ylabel('rho_end')
        plt.tight_layout(h_pad=1.5)
        plt.grid(axis='both', alpha=0.75)
        plt.title('markers with rho_ini > 1: rho_end vs rho_marker > 1',fontsize=12)
        pdf.savefig()

        print("   ... at position D at time: ", clock.ctime())

        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(qq_rhomarker_lost[qq_ii], qq_rhoini_lost[qq_ii], 'ro', ms=2.0, rasterized=do_rasterized)
        plt.xlabel('rho_marker')
        plt.ylabel('rho_ini')
        plt.tight_layout(h_pad=1.5)
        plt.grid(axis='both', alpha=0.75)
        plt.title('markers with rho_ini > 1 : rho_ini vs rho_marker > 1',fontsize=12)
        pdf.savefig()
        
        # -------------------------------------------------------------

        plt.close()
        plt.figure(figsize=(xsize, ysize))
        plt.plot(rl, zl,           'g-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(h_pad=1.5)
        plt.grid(axis='both', alpha=0.75)
        plt.title('2D wall and LCFS',fontsize=12)
        pdf.savefig()
        
        plt.close()
        plt.figure(figsize=(xsize, ysize))
        plt.plot(rl, zl,           'g-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.xlim((np.min(r_wall)-0.05, np.min(r_wall)+0.05))
        plt.ylim((-0.05, 0.05))
        plt.tight_layout(h_pad=1.5)
        plt.grid(axis='both', alpha=0.75)
        plt.title('2D wall and LCFS',fontsize=12)
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(xsize, ysize))
        plt.plot(rl, zl,           'g-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.xlim((np.max(r_wall)-0.2, np.max(r_wall)+0.2))
        plt.ylim((-0.6, 0.6))
        plt.tight_layout(h_pad=1.5)
        plt.grid(axis='both', alpha=0.75)
        plt.title('2D wall and LCFS',fontsize=12)
        pdf.savefig()
        
        plt.close()
        plt.figure(figsize=(xsize, ysize))
        plt.plot(rl, zl,           'c-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_marker[ii_lost], z_marker[ii_lost], 'ro', ms=1.0, rasterized=do_rasterized)
        rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(h_pad=1.5)
        plt.grid(axis='both', alpha=0.75)
        plt.title('marker positions lost markers',fontsize=12)
        pdf.savefig()

        print("   zzz time for next set of plots: ", mytime.time()-time_last)
        time_last = mytime.time()
        plt.close()
        plt.plot(rl, zl,           'c-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_marker[ii_lost], z_marker[ii_lost], 'ro', ms=2.5,fillstyle='none', rasterized=do_rasterized)
        rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout()
        plt.xlim(1.4,1.8)
        plt.ylim(0.7,1.15)
        plt.grid(axis='both', alpha=0.75)
        plt.title('marker positions lost markers',fontsize=12)
        pdf.savefig()
        
        plt.close()
        plt.plot(rl, zl,           'c-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_marker[ii_lost], z_marker[ii_lost], 'ro', ms=2.5,fillstyle='none', rasterized=do_rasterized)
        rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(h_pad=1.5)
        plt.xlim(2.40,2.46)
        plt.ylim(-0.1, 0.1)
        plt.grid(axis='both', alpha=0.75)
        plt.title('marker positions lost markers',fontsize=12)
        pdf.savefig()
   
        plt.close()
        plt.plot(rl, zl,           'c-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_marker[ii_lost], z_marker[ii_lost], 'ro', ms=2.5, fillstyle='none',rasterized=do_rasterized)
        rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(h_pad=1.5)
        plt.xlim(2.28,2.40)
        plt.ylim(0.35, 0.45)
        plt.grid(axis='both', alpha=0.75)
        plt.title('marker positions lost markers',fontsize=12)
        pdf.savefig()


        
        #  -------------------------------------------------------------
        plt.close()
        plt.figure(figsize=(xsize, ysize))
        plt.plot(rl, zl,           'c-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_ini[ii_lost], z_ini[ii_lost], 'ro', ms=1.0, rasterized=do_rasterized)
        rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(h_pad=1.5)
        plt.grid(axis='both', alpha=0.75)
        plt.title('ini positions lost markers',fontsize=12)
        pdf.savefig()

        print("   ... at position E at time: ", clock.ctime())
        
        plt.close()
        plt.plot(rl, zl,           'c-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_ini[ii_lost], z_ini[ii_lost], 'ro', ms=2.5,fillstyle='none', rasterized=do_rasterized)
        rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout()
        plt.xlim(1.4,1.8)
        plt.ylim(0.7,1.15)
        plt.grid(axis='both', alpha=0.75)
        plt.title('ini positions lost markers',fontsize=12)
        pdf.savefig()
        
        plt.close()
        plt.plot(rl, zl,           'c-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_ini[ii_lost], z_ini[ii_lost], 'ro', ms=2.5,fillstyle='none', rasterized=do_rasterized)
        rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(h_pad=1.5)
        plt.grid(axis='both', alpha=0.75)
        plt.xlim(2.40,2.46)
        plt.ylim(-0.1, 0.1)
        plt.title('ini positions lost markers',fontsize=12)
        pdf.savefig()
   
        plt.close()
        plt.plot(rl, zl,           'c-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_ini[ii_lost], z_ini[ii_lost], 'ro', ms=2.5, fillstyle='none',rasterized=do_rasterized)
        rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(h_pad=1.5)
        plt.xlim(2.28,2.40)
        plt.ylim(0.35, 0.45)
        plt.grid(axis='both', alpha=0.75)
        plt.title('ini positions lost markers',fontsize=12)
        pdf.savefig()

        
        # --------------------------------------------------------------
        # now end positions lost markers
        plt.close()
        plt.figure(figsize=(xsize, ysize))
        plt.plot(rl, zl,           'c-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_end[ii_lost], z_end[ii_lost], 'ro', ms=1.0, rasterized=do_rasterized)
        rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(h_pad=1.5)
        plt.grid(axis='both', alpha=0.75)
        plt.title('end positions lost markers',fontsize=12)
        pdf.savefig()
        
        plt.close()
        plt.plot(rl, zl,           'c-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_ini[ii_lost], z_ini[ii_lost], 'ro', ms=2.5,fillstyle='none', rasterized=do_rasterized)
        rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout()
        plt.xlim(1.4,1.8)
        plt.ylim(0.7,1.15)
        plt.grid(axis='both', alpha=0.75)
        plt.title('ini positions lost markers',fontsize=12)
        pdf.savefig()
        
        plt.close()
        plt.plot(rl, zl,           'c-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_end[ii_lost], z_end[ii_lost], 'ro', ms=2.5,fillstyle='none', rasterized=do_rasterized)
        rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(h_pad=1.5)
        plt.xlim(2.40,2.46)
        plt.ylim(-0.1, 0.1)
        plt.grid(axis='both', alpha=0.75)
        plt.title('end positions lost markers',fontsize=12)
        pdf.savefig()
   
        plt.close()
        plt.plot(rl, zl,           'c-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_end[ii_lost], z_end[ii_lost], 'ro', ms=2.5, fillstyle='none',rasterized=do_rasterized)
        rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(h_pad=1.5)
        plt.xlim(2.28,2.40)
        plt.ylim(0.35, 0.45)
        plt.grid(axis='both', alpha=0.75)
        plt.title('end positions lost markers',fontsize=12)
        pdf.savefig()

        print("   ... at position F at time: ", clock.ctime())
        
        # -----------------------------------------------
        #  now restrict to rho_ini > 0.97


        rend_lost    = r_end[ii_lost]
        zend_lost    = z_end[ii_lost]
        rho_ini_lost = rho_ini[ii_lost]
        time_lost    = time[ii_lost]

        ii_edge_born = (rho_ini_lost>=0.97) & (rho_ini_lost<=1.0)


        ii_90 = (rho_ini_lost<=0.90)
        ii_91 = (rho_ini_lost>0.90) & (rho_ini_lost<=0.91)
        ii_92 = (rho_ini_lost>0.91) & (rho_ini_lost<=0.92)
        ii_93 = (rho_ini_lost>0.92) & (rho_ini_lost<=0.93)
        ii_94 = (rho_ini_lost>0.93) & (rho_ini_lost<=0.94)
        ii_95 = (rho_ini_lost>0.94) & (rho_ini_lost<=0.95)
        ii_96 = (rho_ini_lost>0.95) & (rho_ini_lost<=0.96)
        ii_97 = (rho_ini_lost>0.96) & (rho_ini_lost<=0.97)
        ii_98 = (rho_ini_lost>0.97) & (rho_ini_lost<=0.98)
        ii_99 = (rho_ini_lost>0.98) & (rho_ini_lost<=0.99)
        ii_100 = (rho_ini_lost>0.99) & (rho_ini_lost<=1.00)
        ii_101 = (rho_ini_lost>1.00) & (rho_ini_lost<=1.01)
        ii_102 = (rho_ini_lost>1.01) & (rho_ini_lost<=1.02)
        ii_103 = (rho_ini_lost>1.02) & (rho_ini_lost<=1.03)
        ii_104 = (rho_ini_lost>1.03) & (rho_ini_lost<=1.04)
        ii_big = (rho_ini_lost>1.04)

        jj_90 = (rho_marker_lost<=0.90)
        jj_91 = (rho_marker_lost>0.90) & (rho_marker_lost<=0.91)
        jj_92 = (rho_marker_lost>0.91) & (rho_marker_lost<=0.92)
        jj_93 = (rho_marker_lost>0.92) & (rho_marker_lost<=0.93)
        jj_94 = (rho_marker_lost>0.93) & (rho_marker_lost<=0.94)
        jj_95 = (rho_marker_lost>0.94) & (rho_marker_lost<=0.95)
        jj_96 = (rho_marker_lost>0.95) & (rho_marker_lost<=0.96)
        jj_97 = (rho_marker_lost>0.96) & (rho_marker_lost<=0.97)
        jj_98 = (rho_marker_lost>0.97) & (rho_marker_lost<=0.98)
        jj_99 = (rho_marker_lost>0.98) & (rho_marker_lost<=0.99)
        jj_100 = (rho_marker_lost>0.99) & (rho_marker_lost<=1.00)
        jj_101 = (rho_marker_lost>1.00) & (rho_marker_lost<=1.01)
        jj_102 = (rho_marker_lost>1.01) & (rho_marker_lost<=1.02)
        jj_103 = (rho_marker_lost>1.02) & (rho_marker_lost<=1.03)
        jj_104 = (rho_marker_lost>1.03) & (rho_marker_lost<=1.04)
        jj_big = (rho_marker_lost>1.04)
        

        nn_90 = rend_lost[ii_90].size
        nn_91 = rend_lost[ii_91].size
        nn_92 = rend_lost[ii_92].size
        nn_93 = rend_lost[ii_93].size
        nn_94 = rend_lost[ii_94].size
        nn_95 = rend_lost[ii_95].size
        nn_96 = rend_lost[ii_96].size
        nn_97 = rend_lost[ii_97].size
        nn_98 = rend_lost[ii_98].size
        nn_99 = rend_lost[ii_99].size
        nn_100 = rend_lost[ii_100].size
        nn_101 = rend_lost[ii_101].size
        nn_102= rend_lost[ii_102].size
        nn_103= rend_lost[ii_103].size
        nn_104= rend_lost[ii_104].size
        nn_big= rend_lost[ii_big].size

        qq_90 = rend_lost[jj_90].size
        qq_91 = rend_lost[jj_91].size
        qq_92 = rend_lost[jj_92].size
        qq_93 = rend_lost[jj_93].size
        qq_94 = rend_lost[jj_94].size
        qq_95 = rend_lost[jj_95].size
        qq_96 = rend_lost[jj_96].size
        qq_97 = rend_lost[jj_97].size
        qq_98 = rend_lost[jj_98].size
        qq_99 = rend_lost[jj_99].size
        qq_100 = rend_lost[jj_100].size
        qq_101 = rend_lost[jj_101].size
        qq_102= rend_lost[jj_102].size
        qq_103= rend_lost[jj_103].size
        qq_104= rend_lost[jj_104].size
        qq_big= rend_lost[jj_big].size

        print("--------------------")
        print(" rho_marker and _ini of lost markers")
        print(" 0.00 - 0.90  nlost = ", qq_90, nn_90)
        print(" 0.90 - 0.91  nlost = ", qq_91, nn_91)
        print(" 0.91 - 0.92  nlost = ", qq_92, nn_92)
        print(" 0.92 - 0.93  nlost = ", qq_93, nn_93)
        print(" 0.93 - 0.94  nlost = ", qq_94, nn_94)
        print(" 0.94 - 0.95  nlost = ", qq_95, nn_95)
        print(" 0.95 - 0.96  nlost = ", qq_96, nn_96)
        print(" 0.96 - 0.97  nlost = ", qq_97, nn_97)
        print(" 0.97 - 0.98  nlost = ", qq_98, nn_98)
        print(" 0.98 - 0.99  nlost = ", qq_99, nn_99)
        print(" 0.99 - 1.00  nlost = ", qq_100, nn_100)
        print(" 1.00 - 1.01  nlost = ", qq_101, nn_101)
        print(" 1.01 - 1.02  nlost = ", qq_102, nn_102)
        print(" 1.02 - 1.03  nlost = ", qq_103, nn_103)
        print(" 1.03 - 1.04  nlost = ", qq_104, nn_104)
        print(" > 1.04       nlost = ", qq_big, nn_big)
        print("")

                                                                                               








        
        #pdb.set_trace()
        rend_edge = rend_lost[ii_edge_born]
        zend_edge = zend_lost[ii_edge_born]

        plt.close()
        plt.figure(figsize=(xsize, ysize))
        plt.plot(rl, zl,           'c-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(rend_edge, zend_edge, 'ro', ms=1.5, rasterized=do_rasterized)
        rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(h_pad=1.5)
        plt.title('end positions lost markers (rho_ini>0.97)',fontsize=12)
        pdf.savefig()
        
        plt.close()
        plt.plot(rl, zl,           'c-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(rend_edge, zend_edge, 'ro', ms=3.5,fillstyle='none', rasterized=do_rasterized)
        rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout()
        plt.xlim(1.4,1.8)
        plt.ylim(0.7,1.15)
        plt.title('end positions lost markers (rho_ini>0.97)',fontsize=12)
        pdf.savefig()
        
        plt.close()
        plt.plot(rl, zl,           'c-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(rend_edge, zend_edge, 'ro', ms=3.5,fillstyle='none', rasterized=do_rasterized)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(h_pad=1.5)
        plt.xlim(2.40,2.46)
        plt.ylim(-0.1, 0.1)
        plt.title('end positions lost markers (rho_ini>0.97)',fontsize=12)
        rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2)
        pdf.savefig()

        print("   zzz time for next set of plots: ", mytime.time()-time_last)
        time_last = mytime.time()
        plt.close()
        plt.plot(rl, zl,           'c-', linewidth=1, rasterized=do_rasterized)
        plt.plot(r_wall, z_wall,   'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(rend_edge, zend_edge, 'ro', ms=3.5, fillstyle='none',rasterized=do_rasterized)
        rho_contours_1(geq_rarray, geq_zarray, rhogeq_transpose_2d, rho_list_1, rho_list_2, color_1, color_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(h_pad=1.5)
        plt.xlim(2.28,2.40)
        plt.ylim(0.35, 0.45)
        plt.title('end positions lost markers (rho_in>0.97)',fontsize=12)
        pdf.savefig()

        plt.close()
        # -------------------------------

        rhomarker_lost = rho_marker[ii_lost]
        
        rmarker_lost = r_marker[ii_lost]
        zmarker_lost = z_marker[ii_lost]   
        rini_lost = r_ini[ii_lost]
        zini_lost = z_ini[ii_lost]
        rend_lost = r_end[ii_lost]
        zend_lost = z_end[ii_lost]

        my_zmin =  1.1 * np.min(zl)
        my_zmax =  1.1 * np.max(zl)
        my_rmin =  0.9 * np.min(rl)
        my_rmax =  1.1 * np.max(rl)

        qq_ii = (rhomarker_lost > 0.98)

        rmarker_local = rmarker_lost[qq_ii]
        zmarker_local = zmarker_lost[qq_ii]
        rini_local    =    rini_lost[qq_ii]
        zini_local    =    zini_lost[qq_ii]   
        

 
        mmm_lost = rmarker_local.size
        #print("Number of lost markers: ", mmm_lost)
        plt.plot(rl, zl, 'k-', linewidth=1., rasterized = do_rasterized)

        nn_toplot = np.min([mmm_lost, 300])
        for iii in range(nn_toplot):
        
           rr_loc = np.linspace(rmarker_local[iii], rini_local[iii],2)
           zz_loc = np.linspace(zmarker_local[iii], zini_local[iii],2)

           plt.plot(rr_loc, zz_loc, 'g-', linewidth=1, rasterized=do_rasterized)

           plt.plot(rmarker_local[iii], zmarker_local[iii], 'go', ms=4, fillstyle='none', rasterized=do_rasterized)
           plt.plot(   rini_local[iii],    zini_local[iii], 'ko', ms=4, fillstyle='full', rasterized=do_rasterized)           
           #print("   ... iii = ", iii)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(pad=2)
        plt.xlim((2.28, 2.4))
        plt.ylim((0.35, 0.45))
        plt.title('marker/ini positions lost markers (>0.98)')
        #plt.show()
        pdf.savefig()

        plt.close()
        ##pdb.set_trace()
        print("I have completed plots start and end positions of lost markers")
        # --------------------------------------------------
        #   power lost by sector

        ntf = 18
        sector_number = np.linspace(1., ntf, ntf)
        sector_power_lost = np.zeros(ntf)
        for ijk in range(ntf):
            phi_start = ijk*(360.)/ntf
            phi_end = (ijk+1)*(360.)/ntf
            iii_in = (phi_lost > phi_start) & (phi_lost < phi_end)
            sector_power_lost[ijk] = np.sum(weight_energy_lost[iii_in])

        sector_power_lost = sector_power_lost / np.mean(sector_power_lost)
        print("sector_power_lost:", sector_power_lost)
        plt.figure() 
        plt.plot(sector_number, sector_power_lost, 'b-', linewidth=1, rasterized=do_rasterized)
        
        plt.xlabel('TF sector', fontsize=fontsize_2)
        plt.ylim(0., 2.5)
        plt.title('sector-lost-power /mean', fontsize=fontsize_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()
        
        # -------------------------------------------------
        #   theta_ascot vs theta_new for all markers

        theta_end_mod = theta_end % 360.
        
        plt.figure()
        plt.plot(theta_new, theta_end_mod, 'mo', ms=1, rasterized=do_rasterized)
        plt.grid(True)
        plt.xlabel('theta_new', fontsize=fontsize_2)
        plt.ylabel('theta_ascot', fontsize=fontsize_2)
        plt.title('theta-ascot vs theta-new', fontsize=fontsize_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        print("   ... at position G at time: ", clock.ctime())
        
        plt.figure()
        plt.plot(theta_lost_new, theta_lost, 'mo', ms=1, rasterized=do_rasterized)
        plt.grid(True)
        plt.xlabel('theta_new_lost', fontsize=fontsize_2)
        plt.ylabel('theta_ascot_lost', fontsize=fontsize_2)
        plt.title('theta-ascot-lost vs theta-new-lost', fontsize=fontsize_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        my_nbins = 100
        plt.close()
        plt.figure(figsize=(7.,5.))
        nvalues, bins, patches = plt.hist(theta_lost_new, bins=my_nbins, rwidth=1.,histtype='step',color='k', density=True, weights=weight_marker_lost)
        plt.grid(axis='both', alpha=0.75)
        plt.title('weighted particle loss vs theta_new')
        plt.xlabel('theta')
        plt.ylabel('')   
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        #  repeat with normalized 
        my_xarray = np.zeros(my_nbins)
        for ijk in range(my_nbins):
            my_xarray[ijk] = 0.5 * (bins[ijk] + bins[ijk]+1)

        max_binned = np.max(nvalues)
        my_yarray = nvalues/max_binned
        
        plt.figure(figsize=(7.,5.))
        plt.step(my_xarray, my_yarray,'k', where='mid')
        plt.grid(axis='both', alpha=0.75)
        plt.ylim((0.,1.05))
        plt.xlim(-180.,180.)
        plt.title('weighted, norm particle loss vs theta_new')
        plt.xlabel('theta')
        plt.ylabel('')   
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        #  energy weighted

        energy_weights = np.multiply(energy_lost, weight_marker_lost)

        plt.close()
        plt.figure(figsize=(7.,5.))
        nvalues, bins, patches = plt.hist(theta_lost_new, bins=my_nbins, rwidth=1.,histtype='step',color='k', density=True, weights=energy_weights)

        bin_centers = np.zeros(bins.size-1)
        for iikk in range(bin_centers.size):
            bin_centers[iikk] = 0.5 * (bins[iikk] +bins[iikk+1])

        ii_10_130   = (bin_centers >= -10) & (bin_centers <=130)
        #pdb.set_trace()
        total_total = np.sum(nvalues)
        total_10_130 = np.sum(nvalues[ii_10_130])
        fraction_10_130 = total_10_130 / total_total
        mean_10_130 = np.mean(nvalues[ii_10_130])
        max_10_130  = np.max(nvalues[ii_10_130])
        peak_10_130 = max_10_130 / mean_10_130

        print(" fraction of energy loss, -10 to 130 degrees:  ", fraction_10_130)
        print(" weighted energy-loss theta peaking (-10 to 130 degrees): ",peak_10_130)
    

        my_title = 'weighted eloss vs theta, peaking (-10 135) = {:6.3f}, fraction={:6.3f} )'.format(peak_10_130,fraction_10_130)
        plt.grid(axis='both', alpha=0.75)
        plt.title(my_title, fontsize=12)
        plt.xlabel('theta')
        plt.ylabel('')   
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # ----------------------------------------------------------------------------

        plt.close()
        plt.figure(figsize=(7.,5.))
        nvalues, bins, patches = plt.hist(theta_lost_new, bins=my_nbins, rwidth=1.,histtype='step',color='k', density=True, weights=energy_weights)

        bin_centers = np.zeros(bins.size-1)
        for iikk in range(bin_centers.size):
            bin_centers[iikk] = 0.5 * (bins[iikk] +bins[iikk+1])

        ii_10_25   = (bin_centers >= -10) & (bin_centers <=25)
        #pdb.set_trace()
        total_total = np.sum(nvalues)
        total_10_25 = np.sum(nvalues[ii_10_25])
        fraction_10_25 = total_10_130 / total_total
        mean_10_25 = np.mean(nvalues[ii_10_25])
        max_10_25  = np.max(nvalues[ii_10_25])
        peak_10_25 = max_10_25 / mean_10_25

        print(" fraction of energy loss, -10 to +25 degrees:  ", fraction_10_25)
        print(" weighted energy-loss theta peaking (-10 to 25 degrees): ",peak_10_25)
    

        my_title = 'weighted eloss vs theta(-10 +25), peaking = {:6.3f}, fraction={:6.3f} )'.format(peak_10_25,fraction_10_25)
        plt.grid(axis='both', alpha=0.75)
        plt.title(my_title, fontsize=12)
        plt.xlabel('theta')
        plt.ylabel('')   
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        
        # ----------------------------------------------------------------------------
        #  repeat with normalized
        
        my_xarray = np.zeros(my_nbins)
        for ijk in range(my_nbins):
            my_xarray[ijk] = 0.5 * (bins[ijk] + bins[ijk]+1)

        max_binned = np.max(nvalues)
        my_yarray = nvalues/max_binned
        
        plt.figure(figsize=(9.,7.))
        plt.step(my_xarray, my_yarray,'k', where='mid')
        plt.grid(axis='both', alpha=0.75)
        plt.ylim((0.,1.05))
        plt.xlim((-180.,180.))
        plt.title('weighted, norm energy loss vs theta_new')
        plt.xlabel('theta')
        plt.ylabel('')
        plt.tight_layout(pad=padsize)


        try:
            ##pdb.set_trace()
            bbb = myread.read_any_file('poloidal_spiral.txt')
            my_xx = bbb[:,0]
            my_yy = bbb[:,1]
            my_yy = my_yy / np.max(my_yy)
            plt.step(my_xx, my_yy, 'r', where='mid')

            jj_peak       = (my_xx > 40) & (my_xx < 120)
            total_loss    = np.sum(my_yy)
            peak_loss     = np.sum(my_yy[jj_peak])
            fraction_peak = peak_loss / total_loss
            peak_average  = np.average(my_yy[jj_peak])
            peak_factor   = np.max(my_yy[jj_peak])/peak_average
            
            print('spiral poloidal eloss factors')
            print('   total_loss:   ', total_loss)
            print('   peak_loss:    ', peak_loss)
            print('   fraction_peak: ', fraction_peak)
            print('   peak_factor:   ', peak_factor)
            
        except:
            xx_dummy = 0.

        pdf.savefig()
        plt.close()

        # -----------------------------------------------
        #  repeat log scale

        plt.figure(figsize=(7.,5.))
        plt.step(my_xarray, my_yarray,'k', where='mid')
        plt.grid(axis='both', alpha=0.75)
        plt.yscale('log')
        plt.ylim((0.001,1.2))
        plt.xlim((-180.,180.))
        plt.title('weighted, norm energy loss vs theta_new')
        plt.xlabel('theta')
        plt.ylabel('')   
        plt.tight_layout(pad=padsize)


        try:
            ##pdb.set_trace()
            bbb = myread.read_any_file('poloidal_spiral.txt')
            my_xx = bbb[:,0]
            my_yy = bbb[:,1]
            my_yy = my_yy / np.max(my_yy)
            plt.step(my_xx, my_yy, 'r', where='mid')
        except:
            xx_dummy = 0.

        pdf.savefig()
        plt.close()

        print("   zzz time for next set of plots: ", mytime.time()-time_last)
        time_last = mytime.time()
        
        # -------------------------------------------------

        print("   ... at position H at time: ", clock.ctime())
        
        plt.figure()
        plt.plot(pitch_lost, theta_lost_new, 'mo', ms=0.6, rasterized=do_rasterized)
        plt.grid(True)
        plt.xlabel('pitch', fontsize=fontsize_2)
        plt.ylabel('theta', fontsize=fontsize_2)
        plt.title('theta vs pitch for lost markers', fontsize=fontsize_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.plot(ekev_lost, theta_lost_new, 'mo', ms=0.6, rasterized=do_rasterized)
        plt.grid(True)
        plt.xlabel('pitch', fontsize=fontsize_2)
        plt.ylabel('ekev', fontsize=fontsize_2)
        plt.title('theta vs ekev for lost markers', fontsize=fontsize_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()
        # -------------------------------------------------
        #   theta vs phi of lost markers

        plt.figure()
        plt.plot(phi_lost, theta_lost, 'mo', ms=2, rasterized=do_rasterized)
        plt.grid(True)
        plt.xlabel('phi', fontsize=fontsize_2)
        plt.ylabel('theta', fontsize=fontsize_2)
        plt.title('theta vs phi for markers that are lost', fontsize=fontsize_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.plot(phi_lost_18, theta_lost, 'mo', ms=2, rasterized=do_rasterized)
        plt.grid(True)
        plt.xlabel('phi', fontsize=fontsize_2)
        plt.ylabel('theta', fontsize=fontsize_2)
        plt.title('theta vs phi mod(20) for markers that are lost', fontsize=fontsize_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.plot(phi_lost_12, theta_lost, 'mo', ms=2, rasterized=do_rasterized)
        plt.grid(True)
        plt.xlabel('phi', fontsize=fontsize_2)
        plt.ylabel('theta', fontsize=fontsize_2)
        plt.title('theta vs phi mod(30) for markers that are lost', fontsize=fontsize_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()   

        # -------------------------------------------------
        #  compute binned, weighted particle loss yyyyy

        if (simtime_lost.size > 0):

            nbin_sim          = 150
            nlost_total       = simtime_lost.size
            total_weight_lost = np.sum(weight_marker_lost)
            total_energy_lost = np.sum(np.multiply(energy_lost, weight_marker_lost))
            total_energy_ini  = np.sum(np.multiply(ekev, weight))
            
            simtime_max = 1.3  * np.max(simtime_lost)
            simtime_min = 0.75 * np.min(simtime_lost)

            tlog_max = np.log(simtime_max)
            tlog_min = np.log(simtime_min)

            tarray_log = np.linspace(tlog_min, tlog_max, nbin_sim)
            tarray_sim = np.exp(tarray_log)

            my_time_array   = np.zeros(nbin_sim-1)
            weighted_lost   = np.zeros(nbin_sim-1)
            unweighted_lost = np.zeros(nbin_sim-1)

            weighted_energy_lost          = np.zeros(nbin_sim-1)
            weighted_energy_lost_norm_ini = np.zeros(nbin_sim-1)

            ww_loss_weighted = np.multiply(energy_lost, weight_marker_lost)

            #  debugging:  plot using a histogram

            plt.close()
            plt.hist(simtime_lost, bins=100, histtype='step', log=True)
            plt.ylim(bottom=1)
            plt.title("hist: unweighted marker loss")
            pdf.savefig()

            #  changed from 2.e-5 to 3.e-5  8:40 AM 5/13/2021
            
            qq0 = (simtime_lost > 3.e-5)

            plt.close()
            plt.hist(simtime_lost[qq0], bins=100, histtype='step', log=True)
            plt.ylim(bottom=1)
            plt.title("hist: unweighted marker loss (tloss > 3.e-5)")
            pdf.savefig()

            plt.close()
            qq1 = (1000*simtime_lost <= 0.1)
            plt.hist(simtime_lost[qq1], bins=100, histtype='step', log=True)
            plt.ylim(bottom=1)
            plt.xlabel("time [msec]")
            plt.title("hist: unweighted marker loss (to 0.1 ms)")
            pdf.savefig()

            plt.close()
            qq2 = (simtime_lost >=3.e-5)   # changed from 2.e-5 5/13/2021
            plt.hist(simtime_lost[qq2], bins=50, histtype='step')
            plt.ylim(bottom=0)
            plt.title("hist: unweighted marker loss (tloss>3.e-5)")
            pdf.savefig()
            plt.close()
        
            for ik in range(nbin_sim-1):
                
                my_time_array[ik]        = (tarray_sim[ik] + tarray_sim[ik+1])/2.
                indices                  = (simtime_lost >=tarray_sim[ik]) & (simtime_lost <= tarray_sim[ik+1])
                weighted_lost[ik]        = np.sum(weight_marker_lost[indices]) / total_weight_lost
                unweighted_lost[ik]      = (simtime_lost[indices].size)/ nlost_total
                if(energy_lost[indices].size >0):
                   ##pdb.set_trace()
                   weighted_energy_lost[ik] = np.sum(np.multiply(energy_lost[indices], weight_marker_lost[indices])) / total_energy_lost
                   weighted_energy_lost_norm_ini[ik] = np.sum(np.multiply(energy_lost[indices], weight_marker_lost[indices])) / total_energy_ini
            check_energy = np.sum(weighted_energy_lost)
            check_markers = np.sum(weighted_lost)
            
            running_fraction_elost = np.zeros(nbin_sim)
            
            for ik in range(nbin_sim):
                mmm = (simtime_lost <= tarray_sim[ik])
                running_fraction_elost[ik] = np.sum(ww_loss_weighted[mmm])/total_energy_ini

            print("  check_energy: ",  check_energy)
            print("  check_markers: ", check_markers)

            mywrite.write_a_file("energy_lost_vs_time.txt", my_time_array, weighted_lost)
            mywrite.write_a_file("energy_lost_vs_time_norm_ini.txt", my_time_array,  weighted_energy_lost_norm_ini)
            mywrite.write_a_file("markers_lost_vs_time.txt", my_time_array, weighted_energy_lost)
                
            ##pdb.set_trace()
            # --------------------------
            #    weighted particle loss


            plt.close()
            plt.figure()
            plt.step(1000*my_time_array,weighted_lost, 'k', where='mid', rasterized=do_rasterized)
            plt.yscale('log')
            
            plt.ylim(bottom=1.e-4)
            plt.xlim((0, 0.1))
            plt.grid(True)
            plt.xlabel('time [msec]', fontsize=fontsize_2)
            plt.title('weighted markers that are lost', fontsize=fontsize_2)
            plot_filename = stub + '_particle_loss_time_log.pdf'
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()
            
            plt.close()
            plt.figure()
            plt.step(my_time_array,weighted_lost, 'k', where='mid', rasterized=do_rasterized)
            plt.yscale('log')
            plt.xscale('log')
            plt.ylim(bottom=1.e-5)
            plt.xlim((1.e-9, 0.5))
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('weighted markers that are lost', fontsize=fontsize_2)
            plot_filename = stub + '_particle_loss_time_log.pdf'
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()
            ##pdb.set_trace()

            iinp = (my_time_array >= 3.e-5)   # changed from 2.e-5 5/13/2021

            plt.close()
            plt.figure()
            plt.step(my_time_array[iinp],weighted_lost[iinp], 'k', where='mid', rasterized=do_rasterized)
            plt.xlim((0., 0.005))
            plt.ylim(bottom=0.)
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('nonprompt weighted markers that are lost', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()

            plt.close()
            plt.figure()
            plt.step(my_time_array[iinp],np.cumsum(weighted_lost[iinp]), 'k', where='mid', rasterized=do_rasterized)
            plt.xlim((0., 0.005))
            plt.ylim(bottom=0.)
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('cumulative nonprompt weighted markers that are lost', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()

            
            plt.figure()
            plt.step(my_time_array[iinp],weighted_lost[iinp], 'k', where='mid', rasterized=do_rasterized)
            plt.xlim((0., 0.02))
            plt.ylim(bottom=0.)
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('weighted nonprompt markers that are lost', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()

            plt.figure()
            plt.step(my_time_array[iinp],np.cumsum(weighted_lost[iinp]), 'k', where='mid', rasterized=do_rasterized)
            plt.xlim((0., 0.02))
            plt.ylim(bottom=0.)
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('cumulative nonprompt weighted markers that are lost', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()

            print("   ... at position I at time: ", clock.ctime())
            
            plt.figure()
            plt.step(my_time_array[iinp],weighted_lost[iinp], 'k', where='mid', rasterized=do_rasterized)
            plt.xlim(left=0.)
            plt.ylim(bottom=0.)
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('weighted nonprompt markers that are lost', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()

            plt.figure()
            plt.step(my_time_array[iinp],np.cumsum(weighted_lost[iinp]), 'k', where='mid', rasterized=do_rasterized)
            plt.xlim(left=0.)
            plt.ylim(bottom=0.)
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('cumulative nonprompt weighted markers that are lost', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()


            
            
            plt.close()
            plt.figure()
            plt.step(my_time_array,weighted_lost, 'k', where='mid', rasterized=do_rasterized)
            plt.xscale('log')
            plt.xlim((1.e-9, 0.5))
            plt.ylim(bottom=0.)
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('weighted markers that are lost', fontsize=fontsize_2)
            plot_filename = stub + '_particle_loss_time_log.pdf'
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #  plot weighted energy loss norm to total initial weighted energy
            #  9/27/2020

            plt.close()
            plt.figure()
            plt.step(my_time_array,weighted_energy_lost_norm_ini, 'k', where='mid', rasterized=do_rasterized)
            plt.yscale('log')
            plt.xscale('log')
            plt.ylim(bottom=1.e-10)
            plt.xlim((1.e-9, 0.5))
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('weighted energy loss (norm to Wtot_ini)', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()

            plt.close()
            plt.figure()
            plt.step(tarray_sim,running_fraction_elost, 'k', where='mid', rasterized=do_rasterized)
            plt.yscale('log')
            plt.xscale('log')
            plt.ylim(bottom=1.e-10)
            plt.xlim((1.e-9, 0.5))
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('cumulative weighted energy loss (norm to Wtot_ini)', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            
            plt.close()
            plt.figure()
            plt.step(tarray_sim,running_fraction_elost, 'k', where='mid', rasterized=do_rasterized)
            #plt.yscale('log')
            plt.xscale('log')
            plt.ylim(bottom=1.e-6)
            plt.xlim((1.e-9, 0.5))
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('cumulative weighted energy loss (norm to Wtot_ini)', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()

            plt.close()
            plt.figure()
            plt.step(tarray_sim,running_fraction_elost, 'k', where='mid', rasterized=do_rasterized)
            #plt.yscale('log')
            plt.xscale('log')
            plt.ylim(bottom=1.e-6)
            plt.xlim((1.e-7, 0.3))
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('cumulative weighted energy loss (norm to Wtot_ini)', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()

            plt.close()
            plt.figure()
            plt.step(tarray_sim,running_fraction_elost, 'k', where='mid', rasterized=do_rasterized)
            #plt.yscale('log')
            #plt.xscale('log')
            plt.ylim(bottom=0.)
            plt.xlim(left=0.)
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('cumulative weighted energy loss (norm to Wtot_ini)', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            
            # --------------------------
            #    weighted energy loss
            
            plt.close()
            plt.figure()
            plt.step(my_time_array,weighted_energy_lost, 'k', where='mid', rasterized=do_rasterized)
            plt.yscale('log')
            plt.xscale('log')
            plt.ylim(bottom=1.e-10)
            plt.xlim((1.e-9, 0.5))
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('weighted energy loss', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()



            
            try:
                ##pdb.set_trace()
                bbb = myread.read_any_file('eloss_time.txt')
                spiral_time  = bbb[:,0]
                spiral_eloss = bbb[:,1]

                time_spiral  =  spiral_time[spiral_eloss.argmax()]
                eloss_spiral = spiral_eloss[spiral_eloss.argmax()]

                eloss_ascot      = np.interp(time_spiral, my_time_array, weighted_energy_lost)
                eloss_ascot_norm = 0.9 * weighted_energy_lost * (eloss_spiral/eloss_ascot)

                # #pdb.set_trace()

                plt.close()
                plt.figure()
                plt.step(my_time_array,eloss_ascot_norm, 'k', where='mid', rasterized=do_rasterized)
                plt.step(spiral_time, spiral_eloss,      'r', where='mid', rasterized=do_rasterized)
                plt.yscale('log')
                plt.xscale('log')
                plt.ylim((100., 2.e5))
                
                plt.xlim((1.e-9, 0.5))
                plt.grid(True)
                plt.xlabel('time [sec]', fontsize=fontsize_2)
                plt.title('weighted energy loss', fontsize=fontsize_2)
                plt.tight_layout(pad=padsize)
                pdf.savefig()

                plt.close()
                plt.figure()
                plt.step(my_time_array,eloss_ascot_norm, 'k', where='mid', rasterized=do_rasterized)
                plt.step(spiral_time, spiral_eloss,      'r', where='mid', rasterized=do_rasterized)
                plt.xscale('log')
                plt.ylim(bottom=0.)
                plt.xlim((1.e-9, 0.5))
                plt.grid(True)
                plt.xlabel('time [sec]', fontsize=fontsize_2)
                plt.title('weighted energy loss', fontsize=fontsize_2)
                plt.tight_layout(pad=padsize)
                pdf.savefig()
            except:
                my_dummy = 0.
            
            plt.close()

            plt.close()
            plt.figure()
            plt.step(my_time_array,weighted_energy_lost, 'k', where='mid', rasterized=do_rasterized)
            plt.xscale('log')
            plt.xlim((1.e-9, 0.5))
            plt.ylim(bottom=0.)
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('weighted energy loss', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()

            plt.close()
            plt.figure()
            plt.step(my_time_array,weighted_energy_lost, 'k', where='mid', rasterized=do_rasterized)
            plt.xlim(left=0.)
            plt.ylim(bottom=0.)
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('weighted energy loss', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()

            print("   ... at position J at time: ", clock.ctime())
            
            #  now for unweighted

            plt.close()
            plt.figure()
            plt.step(1000*my_time_array,unweighted_lost, 'k', where='mid', rasterized=do_rasterized)
            plt.yscale('log')
            plt.ylim(bottom=5.e-4)
            plt.xlim((0, 0.1))
            plt.grid(True)
            plt.xlabel('time [msec]', fontsize=fontsize_2)
            plt.title('UNweighted markers that are lost', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()
            
            plt.close()
            plt.figure()
            plt.step(my_time_array,unweighted_lost, 'k', where='mid', rasterized=do_rasterized)
            plt.yscale('log')
            plt.xscale('log')
            plt.ylim(bottom=5.e-5)
            plt.xlim((1.e-9, 0.5))
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('UNweighted markers that are lost', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()

            print("   zzz time for next set of plots: ", mytime.time()-time_last)
            time_last = mytime.time()
            
            plt.close()
            plt.figure()
            plt.step(my_time_array,unweighted_lost, 'k', where='mid', rasterized=do_rasterized)
            plt.xscale('log')
            plt.xlim((1.e-9, 0.5))
            plt.ylim(bottom=0.)
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('UNweighted markers that are lost', fontsize=fontsize_2)
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()
        
        # -------------------------------------------------
        #  compute running fraction of lost

        if (simtime_lost.size > 0):

            nbin_sim    = 200

            nlost_total = simtime_lost.size
        
            simtime_max = np.max(simtime_lost)
            simtime_min = np.min(simtime_lost)

            tlog_max = np.log(simtime_max)
            tlog_min = np.log(simtime_min)

            tarray_log = np.linspace(tlog_min, tlog_max, nbin_sim)
            tarray_sim = np.exp(tarray_log)

            running_fraction_lost = np.zeros(nbin_sim)

            for ik in range(nbin_sim):
                mmm = (simtime_lost <= tarray_sim[ik])
                running_fraction_lost[ik] = (simtime_lost[mmm].size)/nlost_total
            ##pdb.set_trace()

            plt.close()
            plt.figure()
            plt.plot(tarray_sim, running_fraction_lost, 'r-', linewidth=1.5, rasterized=do_rasterized)
            plt.yscale('log')
            plt.xscale('log')
            plt.ylim(1.e-3,1.)
            plt.xlim(1.e-7, 0.1)
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('normalized fraction of markers that are lost', fontsize=fontsize_2)
            plot_filename = stub + '_particle_loss_time_log.pdf'
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()

            plt.figure()
            plt.plot(tarray_sim, running_fraction_lost, 'r-', linewidth=1.5, rasterized=do_rasterized)
            plt.xscale('log')
            plt.ylim(0.,1.)
            plt.xlim(1.e-7, 0.1)
            plt.grid(True)
            plt.xlabel('time [sec]', fontsize=fontsize_2)
            plt.title('normalized fraction of markers that are lost', fontsize=fontsize_2)
            plot_filename = stub + '_particle_loss_time_linear.pdf'
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()


        #  -------------------------------------------------
        #   identify bad points

        rr_diff  = r_ini - r_marker
        ii_bad   = (np.abs(rr_diff) > 0.05)
        rini_bad = r_ini[ii_bad]
        nn_bad   = rini_bad.size

        if (nn_bad > 0):

            r_marker_bad         = r_marker[ii_bad]
            z_marker_bad         = z_marker[ii_bad]
            vphi_marker_bad      = vphi_marker[ii_bad]
            vtot_marker_bad      = vtot_marker[ii_bad]
            pitch_phi_marker_bad = pitch_phi_marker[ii_bad]

            r_ini_bad         = r_ini[ii_bad]
            z_ini_bad         = z_ini[ii_bad]
            vphi_ini_bad      = vphi_ini[ii_bad]
            vtot_ini_bad      = vtot_ini[ii_bad]
            vpar_ini_bad      = vpar_ini[ii_bad]
            pitch_phi_ini_bad = pitch_phi_ini[ii_bad]
            pitch_ini_bad     = pitch_ini[ii_bad]

            print("")
            print("  Markers whose r_ini differs from r_marker by more than 5 cm")
            print("")
            print(" --------------- markers ----------------------------------     ---------------  initial ----------------------------------------")
            print(" ii       R       z           vphi       vtot      pitch_phi     R         z         vphi        vpar        vtot       pitch_phi pitch ")
            print("")
            for ww in range(nn_bad):
                print(" %d  %8.3f  %8.3f   %11.3e %11.3e  %8.4f %8.3f  %8.3f   %11.3e %11.3e %11.3e  %8.4f %8.4f" %                   \
                      (ww, r_marker_bad[ww], z_marker_bad[ww], vphi_marker_bad[ww], vtot_marker_bad[ww], pitch_phi_marker_bad[ww],    \
                           r_ini_bad[ww], z_ini_bad[ww], vphi_ini_bad[ww], vpar_ini_bad[ww], vtot_ini_bad[ww], pitch_phi_ini_bad[ww], \
                           pitch_ini_bad[ww]))
            print("")
        else:
             print("")
             print("  No markers have r_ini differing from r_marker by more than 5 cm")

                       # ----------------------------------------------------------
        #
        #  compute fraction of particles lost as function of rho

        mm = 150
        xarray              = np.linspace(1./(2.*mm), 1.- (1./(2.*mm)), mm)
        particle_lost_ratio = np.zeros(mm)

        for kk in range(mm):

           left_side = (1.*kk)/mm
           right_side = (kk+1.)/mm

           ii_all  = (rho_marker >= left_side) & (rho_marker < right_side)
           ii_lost = (rho_marker_lost >= left_side) & (rho_marker_lost < right_side)

           nn_all  = rho_marker[ii_all].size
           nn_lost = rho_marker_lost[ii_lost].size

           if(nn_all > 0):
              particle_lost_ratio[kk] = (1.*nn_lost)/nn_all
           else:
              particle_lost_ratio[kk] = 0.

        # -----------------------------------------------------------
        #

        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(time[ii_cputime], ekev[ii_cputime],'ro', ms=1, rasterized=do_rasterized)
        plt.ylim(bottom=0.)
        plt.xlim(left=0.)
        plt.grid(True)
        plt.xlabel('simulation time [sec]', fontsize=fontsize_2)
        plt.ylabel('Ekev', fontsize=fontsize_2)
        plt.title('Ekev vs simtime for cputime-end markers', fontsize=fontsize_2)
        plt.tight_layout()
        plot_filename = stub + '_EKEV_vs_simtime_cputimend.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()


        print("   ... at position K at time: ", clock.ctime())

        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(rho_ini[ii_cputime], time[ii_cputime],'ro', ms=1, rasterized=do_rasterized)
        plt.ylim(bottom=0.)
        plt.xlim(left=0.)
        plt.grid(True)
        plt.ylabel('simulation time [sec]')
        plt.xlabel('Initial rho')
        plt.title('simulation-time vs rho_ini for cputime-end markers')
        plt.tight_layout()
        plot_filename = stub + '_cputime_vs_rhoini_cputimend.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(rho_end[ii_cputime], time[ii_cputime],'ro', ms=1, rasterized=do_rasterized)
        plt.ylim(bottom=0.)
        plt.xlim(left=0.)
        plt.grid(True)
        plt.ylabel('simulation time [sec]')
        plt.xlabel('End-rho')
        plt.title('simulation-time vs rho_end for cputime-end markers')
        plt.tight_layout()
        plot_filename = stub + '_cputime_vs_rhoend_cputimend.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        # ----------------------------------------------------------

        rhomarker_survived = rho_marker[ii_survived]
        time_survived      = time[ii_survived]
        cpu_survived       = cputime[ii_survived]

        jj_inner  = (rhomarker_survived < 0.33)
        jj_middle = (rhomarker_survived > 0.33) & (rhomarker_survived < 0.66)
        jj_outer  = (rhomarker_survived > 0.66)
        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(time_survived[jj_inner],  cpu_survived[jj_inner],  'ro', ms=1, rasterized=do_rasterized_2)
        plt.plot(time_survived[jj_middle], cpu_survived[jj_middle], 'go', ms=1, rasterized=do_rasterized_2)
        plt.plot(time_survived[jj_outer],  cpu_survived[jj_outer],  'bo', ms=1, rasterized=do_rasterized_2)
        
        plt.ylim(bottom=0.)
        plt.xlim(left=0.)
        plt.grid(True)

        plt.xlabel('simulation time [sec]')
        plt.ylabel('CPU time [sec]')
        plt.title('CPU time vs sim time (survived) r/g/b=thirds')
        plt.tight_layout()
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        # -----------------------------------------------------------------
        print("   ... about to make histogram: _ekev_survived.pdf")   # yyyyy
        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.hist(ekev[ii_survived], bins=20, histtype='step', rwidth=1.,color='k',log=True)
        plt.grid(axis='y', alpha=0.75)
        plt.title('Ekev of survived markers')
        plt.xlabel('Ekev')
        plt.ylabel('')
        plot_filename = stub + '_ekev_survived.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        energy_ratio = np.divide(ekev[ii_survived],ekev_ini[ii_survived])
        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.hist(energy_ratio, bins=20, rwidth=1.0,histtype='step',color='k', log=True)
        plt.grid(axis='y', alpha=0.75)
        plt.title('Ekev[end/ini] of survived markers')
        plt.xlabel('eKev[end]/eKev[ini]')
        plt.ylabel('')
        plot_filename = stub + '_ekev_endini_survived.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()


        print("   ... about to make histogram: _simtime_survived.pdf")
        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.hist(time[ii_survived], bins=15, rwidth=1.0,histtype='step',color='b',log=True)
        plt.grid(axis='y', alpha=0.75)
        plt.title('simulation time for survived markers')
        plt.xlabel('sec')
        plt.ylabel('')
        plot_filename = stub + '_simtime_survived.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()
        print("   ... have completed histogram: _simtime_survived.pdf")
        # ---------------------------------------------------------------
        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(rho_end[ii_survived],ekev[ii_survived], 'ro', ms=2, rasterized=do_rasterized_2)
        plt.grid(axis='y', alpha=0.75)
        plt.title('Ekev[end] vs rho for survived markers')
        plt.xlabel('rho')
        plt.ylabel('')
        plot_filename = stub + '_ekev_rho.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()


        # ---------------------------------------------------------------
        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(pitch_ini, weight_ratio, 'ro', ms=2, rasterized=do_rasterized)
        plt.title('weight_ratio versus pitch_ini')
        plt.xlabel('pitch_ini')
        plt.ylabel('weight)ratio')
        plot_filename = stub + '_weight_ratio_pitch.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()

         # ---------------------------------------------------------------
        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.hist(cputime[ii_survived], bins=15, rwidth=1., histtype='step',color='k',log=True)
        plt.grid(axis='y', alpha=0.75)
        plt.title('CPU time for survived markers')
        plt.xlabel('sec')
        plt.ylabel('')
        plot_filename = stub + '_CPUtime_survived.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        print("   ... have completed histogram:  _CPUtime_survived.pdf")

        print("   zzz time for next set of plots: ", mytime.time() - time_last)
        time_last = mytime.time()
        
        # -----------------------------------------------------------
        plt.close()
        plt.figure()
        plt.step(xarray, particle_lost_ratio, 'k', rasterized=do_rasterized,where='mid')
        plt.yscale('log')
        plt.ylim(1.e-3,1.)
        plt.grid(True)
        plt.xlim(0.,1.)
        plt.xlabel('rho at birth')
        plt.title('Fraction of markers that are lost')
        plot_filename = stub + '_fraction_particle_loss_log.pdf'
        plt.tight_layout(pad=padsize)

        try:
            bbb = myread.read_any_file('local_losses.txt')
            my_xx = bbb[:,0]
            my_yy = bbb[:,1]
            plt.step(my_xx,my_yy, 'r-', rasterized=do_rasterized)
        except:
            my_dummy = 0.
            
        pdf.savefig()
        plt.close()

        plt.close()
        plt.figure()
        plt.step(xarray, particle_lost_ratio, 'k', rasterized=do_rasterized)

        try:
            bbb = myread.read_any_file('local_losses.txt')
            my_xx = bbb[:,0]
            my_yy = bbb[:,1]
            plt.step(my_xx,my_yy, 'r-', rasterized=do_rasterized)
        except:
            my_dummy = 0.
            
        plt.yscale('log')
        plt.ylim(1.e-3,1.)
        plt.grid(True)
        plt.xlim(0.6,1.)
        plt.xlabel('rho at birth')
        plt.title('Fraction of markers that are lost')
        plot_filename = stub + '_fraction_particle_loss_log.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.step(xarray, particle_lost_ratio, 'k', rasterized=do_rasterized)

        try:
            bbb = myread.read_any_file('local_losses.txt')
            my_xx = bbb[:,0]
            my_yy = bbb[:,1]
            plt.step(my_xx,my_yy, 'r-', rasterized=do_rasterized)
        except:
            my_dummy = 0.

        plt.ylim(bottom=0.)
        plt.grid(True)
        plt.xlim(0.,1.)
        plt.ylim(0.,1.)
        plt.xlabel('rho at birth')
        plt.title('Fraction of markers that are lost')
        plot_filename = stub + '_fraction_particle_loss_linear.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.step(xarray, particle_lost_ratio, 'k', rasterized=do_rasterized)

        try:
            bbb = myread.read_any_file('local_losses.txt')
            my_xx = bbb[:,0]
            my_yy = bbb[:,1]
            plt.step(my_xx,my_yy, 'r-', rasterized=do_rasterized)
        except:
            my_dummy = 0.
            
        plt.ylim(bottom=0.)
        plt.grid(True)
        plt.xlim(0.6,1.)
        plt.ylim(0.,1.)
        plt.xlabel('rho at birth')
        plt.title('Fraction of markers that are lost')
        plot_filename = stub + '_fraction_particle_loss_linear.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # --------------------------------------------------------
        #
        #  histogram of simulation time for lost markers

        ii_prompt   = (simtime_lost <= 3.e-5)   # changed from 2.e-5 5/13/2021
        ii_delayed = (simtime_lost >= 3.e-5)

        nn_prompt_lost  = simtime_lost[ii_prompt].size
        nn_delayed_lost = simtime_lost[ii_delayed].size

        if(nn_prompt_lost > 0):
            percent_delayed = (100.*nn_delayed_lost) / nn_prompt_lost
        else:
            percent_delayed =100.

        print('   percent of delayed orbit loss: ', percent_delayed)
        print('   percent of prompt  orbit loss: ', 100.-percent_delayed)

        # ---------------------------------------------------------------------
        #   after extensive experimentation, found that the "bins='auto'" caused
        #   the code to hag
        # print("   ... starting _simtime_lost_markers.pdf")
        # #pdb.set_trace()
        plt.close()
        plt.figure(figsize=(7.,5.))
        nn, bins, patches = plt.hist(simtime_lost, bins=15, rwidth=1.,histtype='step',color='k',log=True, bottom=0.5)
        plt.grid(axis='y', alpha=0.75)
        plt.title('end time for lost markers: (%6.2f percent delayed)' %(percent_delayed))
        plt.xlabel('sec')
        plt.ylabel('')
        plot_filename = stub + '_simtime_lost_markers.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # print("   ... have completed _simtime_lost_markers.pdf")

        ii_zeros = (simtime_lost == 0.)   # otherwise we get -inf  7/20/2020
        simtime_lost[ii_zeros] = 1.e-11
        simtime_lost_log10 = np.log10(simtime_lost)
        # #pdb.set_trace()
        plt.figure(figsize=(7.,5.))
        #plt.hist(simtime_lost_log10, bins=15, rwidth=0.85,alpha=0.75,color='b', log=True)
        #pdb.set_trace()
        nn, bins, patches = plt.hist(simtime_lost_log10, bins=15, rwidth=1.,histtype='step',color='k', bottom=0.5)
        plt.grid(axis='y', alpha=0.75)
        #plt.xlim(-7., 7.)
        #plt.ylim(-1.e4, 1.e4)
        #plt.ylim(0.1, 1000.)
        plt.title('end time for lost markers: (%6.2f percent delayed)' %(percent_delayed))
        plt.xlabel('log_10(sec)')
        plt.ylabel('')
        plot_filename = stub + '_simtime_lost_markers_log.pdf'
        pdf.savefig()
        plt.close()

        print("   ... at position L at time: ", clock.ctime())
        
        plt.figure(figsize=(7.,5.))

        nn, bins, patches = plt.hist(simtime_lost_log10, bins=15, rwidth=1.,histtype='step',color='k', bottom=0.5,log=True)
        plt.grid(axis='y', alpha=0.75)
        #plt.xlim(-7., 7.)
        #plt.ylim(-1.e4, 1.e4)
        #plt.ylim(0.1, 1000.)
        plt.title('end time for lost markers: (%6.2f percent delayed)' %(percent_delayed))
        plt.xlabel('log_10(sec)')
        plt.ylabel('')
        plot_filename = stub + '_simtime_lost_markers_loglog.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        ##pdb.set_trace()

        # --------------------------------------------------------
        #
        #  histogram of (rho marker)

        rl =  gg.equilibria[eq_index].rlcfs
        zl =  gg.equilibria[eq_index].zlcfs

        rmax = np.max(rl)
        rmin = np.min(rl)
        zmax = np.max(zl)
        zmin = np.min(zl)

        nr = 10
        nz = 20

        delta_R = (rmax - rmin)/nr
        delta_Z = (zmax - zmin)/nz
        
        r_marker_t = np.transpose(r_marker)
        z_marker_t = np.transpose(z_marker)

        plt.close()

        plt.figure(figsize=(4,7.33))
        fig,ax=plt.subplots()
        
        plt.plot(rl, zl, 'k-', linewidth=1)
        plt.xlim(1.2,2.5)
        plt.ylim(-1.2,1.2)
        plt.plot(gg.equilibria[eq_index].rlcfs,
                 gg.equilibria[eq_index].zlcfs,
                 color='b',linewidth=2)
        plt.xlabel('Rmajor [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(pad=padsize)

        estimated_max = (2.*r_marker_t.size)/(nr*nz)
        for ir in range(nr):
            for iz in range(nz):

                Rleft   = rmin    + ir* delta_R
                Rright  = Rleft   + delta_R
                zbottom = zmin    + iz * delta_Z
                ztop    = zbottom + delta_Z

                jj_born   =   (r_marker_t  >= Rleft)  & (r_marker_t  <= Rright)  \
                            & (z_marker_t >= zbottom) & (z_marker_t  <= ztop)    #\
                            # & (endcond != 0)

                markers_born   = np.sum(jj_born)

                if(markers_born > 0.75*estimated_max):
                    this_color='peru'
                elif(markers_born > 0.5*estimated_max):
                    this_color='sandybrown'
                elif(markers_born > 0.25*estimated_max):
                    this_color='peachpuff'
                else:
                    this_color='linen'

                
                ##pdb.set_trace()
                # rect = Rectangle( (Rleft,zbottom),delta_R, delta_Z, fill=False, edgecolor='r', linewidth=3.0) # fill=False
                # rect = Rectangle( (Rleft,zbottom),delta_R, delta_Z, linewidth=1.0,color=this_color)
                                                                   # facecolor=this_color
                rect = Rectangle( (Rleft,zbottom),delta_R, delta_Z, facecolor=this_color, edgecolor='k', linewidth=0.5)
                    
                ax.add_patch(rect)

                xpos = Rleft   + delta_R/2.
                ypos = zbottom + delta_Z/2.

                my_string = '{:d}'.format(markers_born.astype(int))
                plt.text(xpos, ypos, my_string, fontsize=8, ha='center', va='center')
                rho_contours = np.linspace(0.1, 0.9, 9)
                contour_colors=['linen', 'sandybrown']  # not used
                cs=plt.contour(geq_rarray, geq_zarray, np.transpose(rhogeq_transpose_2d), rho_contours, linewidths=1.0, colors='b',zorder=1)
                plt.clabel(cs,fontsize=10)
                
        plt.title(' Markers born', fontsize=fontsize_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # -----------------------------------------------
        plt.figure(figsize=(7.,5.))
        plt.plot(rho_marker, weight/np.max(weight), 'ko', ms=0.5, rasterized=do_rasterized)
        my_yy = np.array(aa_profile["alpha_source"])
        my_yy = my_yy / np.max(my_yy)
        my_yy = my_yy[:,0]
        my_xx = np.array(aa_profile["rho_pol"])

        plt.plot(my_xx, my_yy, 'r-', linewidth=1)
        plt.xlim(0., 1.)
        plt.ylim(0., 1.05)
        plt.xlabel('rho')
        plt.title('norm marker weights (bl) and alpha source (r)')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # ++++++++++++++++++++++++++++++++++++++
        
        plt.figure(figsize=(7.,5.))
        ii_70 = np.abs(rho_marker-0.70).argmin()
        plt.plot(rho_marker, weight/np.max(weight[ii_70]), 'ko', ms=0.5, rasterized=do_rasterized)
        my_yy = np.array(aa_profile["alpha_source"])
        my_yy = my_yy / np.max(my_yy)
        my_yy = my_yy[:,0]
        my_xx = np.array(aa_profile["rho_pol"])

        ii_70 = np.abs(my_xx-0.70).argmin()
        my_yy_70 = my_yy/my_yy[ii_70]
        plt.plot(my_xx, my_yy_70, 'r-', linewidth=1)
        plt.xlim(0.7, 1.)
        plt.ylim(0., 1.05)
        plt.xlabel('rho')
        plt.title('norm marker weights (bl) and alpha source (r) norm at 0.70')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()
                  
        # -----------------------------------------------
        
        my_nbins            = 50
        binned_weights      = np.zeros(my_nbins)
        binned_weights_fake = np.zeros(my_nbins)
        binned_cumulative   = np.zeros(my_nbins)
        cumulative_sum      = 0.
        
        my_rhos        = 1./my_nbins  + np.linspace(0., 1.-1./my_nbins, my_nbins)   # want zone boundaries
        edges          = np.linspace(0., 1., my_nbins+1)
        
        # print('')
        # print(' about to computed binned weights')
        for qq in range(my_nbins):

            # print('   min, max rho of bin ', qq, edges[qq], edges[qq+1])
            
            ii = (rho_marker > edges[qq] ) & (rho_marker < edges[qq+1])
            
            binned_weights[qq] = np.sum(weight[ii])
            cumulative_sum  = cumulative_sum + binned_weights[qq]
            binned_cumulative[qq] = cumulative_sum

            jj = (rho_marker_fake > edges[qq]) & (rho_marker_fake < edges[qq+1])

            binned_weights_fake[qq] = np.sum(weight_marker_fake[jj])

        binned_cumulative   = binned_cumulative / cumulative_sum
        yy_sorted           = np.sort(binned_weights_fake)
        zz_sorted           = yy_sorted[-4:]
        yy_max_fake         = np.mean(zz_sorted)
        binned_weights_fake = binned_weights_fake / yy_max_fake

        # ----------------------------------------------------------    
        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(my_rhos, binned_weights_fake, 'ko-', ms=1)
        plt.xlim(0., 1.05)
        plt.ylim(bottom=0.)
        plt.title('Summed binned (norm)  FAKE weights versus rho')
        plt.xlabel('rho')
        plt.ylabel('')
        pdf.savefig()
        plt.close()
        plt.clf()
        print("  zzz time for next set of plots: ", mytime.time() - time_last)
        time_last = mytime.time()
        
        # ----------------------------------------------------------    
        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(my_rhos**2, binned_weights_fake, 'ko-', ms=1)
        plt.xlim(0., 1.05)
        plt.ylim(bottom=0.)
        plt.title('Summed binned (norm)  FAKE weights versus rho^2')
        plt.xlabel('rho')
        plt.ylabel('')
        pdf.savefig()
        plt.close()
        plt.clf()

        print("   ... at position M at time: ", clock.ctime())
        
        # ----------------------------------------------------------    
        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(my_rhos, binned_weights, 'ko-', ms=1)
        plt.xlim(0., 1.05)
        plt.ylim(bottom=0.)
        plt.title('Summed binned weights versus rho')
        plt.xlabel('rho')
        plt.ylabel('')
        pdf.savefig()
        plt.close()
        plt.clf()

        yy_sorted = np.sort(binned_weights)
        zz_sorted = yy_sorted[-4:]
        yy_max_weights    = np.mean(zz_sorted)

        binned_weights_norm = binned_weights / yy_max_weights

        # --------------------------------------------------------
        #  repeat for fake markers
        
        
        # --------------------------------------------------------
        #  plot alpha_source * dvol

        
        yy_1 = np.array(aa_profile["alpha_source"])
        yy_2 = np.array(aa_profile["vol_zone"])
        nnn = yy_1.size

        my_yy = np.zeros(nnn)
        for qqq in range(nnn):
            my_yy[qqq] = 1.e-18* yy_1[qqq]*yy_2[qqq]



        #  see email to Alex and Pablo 2/26:  the rho-poloidal
        #  array is non-uniform, and the points are closer
        #  together near rho=1.  So we must 'bin' the data
        #  in rho-poloidal just as we do for the binned weights

        my_ascot_nbins    = 20
        binned_source     = np.zeros(my_ascot_nbins)
        cumulative_source = np.zeros(my_ascot_nbins)
        cumulative_sum    = 0.
        
        my_rhos_source = 1./my_ascot_nbins  + np.linspace(0., 1.-1./my_ascot_nbins, my_ascot_nbins)   # want zone boundaries
        edges          = np.linspace(0., 1., my_ascot_nbins+1)
        
        for qq in range(my_ascot_nbins):

            # print('   min, max rho of bin ', qq, edges[qq], edges[qq+1])
            
            ii = (rho_pol > edges[qq] ) & (rho_pol < edges[qq+1])
            
            binned_source[qq]    = np.sum(my_yy[ii])
            cumulative_sum        = cumulative_sum + binned_source[qq]
            cumulative_source[qq] = cumulative_sum
            
        cumulative_source      = cumulative_source/cumulative_sum
        
        yy_sorted           = np.sort(binned_source)
        zz_sorted           = yy_sorted[-4:]
        yy_max              = np.mean(zz_sorted)
        binned_source_norm  = binned_source / yy_max
        
        # -------------------------------------------------------------------------
        
        my_xx = aa_profile["rho_pol"]   # do not use rhosqrt ... two zeros at start

        #  rho_pol is on zone-centers.  want zone-boundaries

        rho_zb = np.zeros(nnn)
        for ww in range(1, nnn-1):
            rho_zb[ww] = 0.5*(my_xx[ww] + my_xx[ww+1])
        rho_zb[nnn-1] = 1.
                              
        # #pdb.set_trace()
        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(my_rhos_source, binned_source, 'ko-', ms=1)
        plt.xlim(0., 1.05)
        plt.ylim(bottom=0.)
        plt.title('Alpha_source * dVol vs rho (on zb)')
        plt.xlabel('rho')
        plt.ylabel('10^18 /sec')
        pdf.savefig()
        plt.close()
        plt.clf()
        ##pdb.set_trace()


        
        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(my_rhos, binned_weights_norm, 'ko-', ms=1)
        plt.plot(my_rhos_source, binned_source_norm, 'r-')
        plt.xlim(0., 1.05)
        plt.ylim(bottom=0.)
        plt.title('Normalized binned_weights and S*dVol')
        plt.xlabel('rho')
        plt.grid(True, alpha=0.75)
        pdf.savefig()
        plt.close()
        plt.clf()
        ##pdb.set_trace()

        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(my_rhos, binned_weights_norm/np.max(binned_weights_norm), 'ko-', ms=1)
        plt.plot(my_rhos_source, binned_source_norm/np.max(binned_source_norm), 'r-')
        plt.xlim(0., 1.05)
        plt.ylim(bottom=0.)
        plt.title('normalize at max')
        plt.xlabel('rho')
        plt.grid(True, alpha=0.75)
        pdf.savefig()
        plt.close()
        plt.clf()
        ##pdb.set_trace()


        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(my_rhos,        binned_cumulative, 'ko-', ms=1)
        plt.plot(my_rhos_source, cumulative_source, 'r-')
        plt.xlim(0., 1.05)
        plt.ylim(bottom=0.)
        plt.title('cumulative bin-weights and source (red)')
        plt.xlabel('rho')
        plt.grid(True, alpha=0.75)
        pdf.savefig()
        plt.close()
        plt.clf()

        print("\n +++++++++++++++++++++++++++++++++++ \n")
        print("  ii   rho  binned_cumulative  1-binned_cumulative\n")

        mylen = len(my_rhos)
        
        for kl in range(mylen):
            print( "%d5  %6.3f  %8.4f   %8.4f"%(kl, my_rhos[kl], binned_cumulative[kl], 1.-binned_cumulative[kl]))
                                                                        
        print("\n ++++++++++++++++++++++++++++++++++")
        
        # --------------------------------------------------------

        print("   ... at position N at time: ", clock.ctime())
        plt.figure(figsize=(7.,5.))
        plt.hist(vr_marker,   bins=50, histtype='step',rwidth=1,color='k')
        plt.hist(vphi_marker, bins=50, histtype='step',rwidth=1,color='r')
        plt.hist(vz_marker,   bins=50, histtype='step',rwidth=1,color='g') 
        plt.title('Distribution of marker vR vphi vZ [b r g]')
        plt.xlabel('v [m/s]')
        plt.ylabel('')
        pdf.savefig()
        plt.close()
        plt.clf()

        # ------------------------------------------------
        #  histogram of rmajor
    
        mmm = (np.abs(z_marker) < 0.1)
        plt.figure(figsize=(7.,5.))
        plt.hist(r_marker[mmm], bins=50, histtype='step',rwidth=1,color='k') 
        plt.title('Distribution of marker Rmajor (for abs(z)<0.1)')
        plt.xlabel('Rmajor [m]')
        plt.ylabel('')
        pdf.savefig()
        plt.close()
        plt.clf()

        mmm = (np.abs(r_marker-1.80) < 0.05)
        plt.figure(figsize=(7.,5.))
        plt.hist(z_marker[mmm], bins=50, histtype='step',rwidth=1,color='k') 
        plt.title('Distribution of marker Z (for 1.75<R<1.85)')
        plt.xlabel('Z [m]')
        plt.ylabel('')
        pdf.savefig()
        plt.close()
        plt.clf()

        # ----------------------------------------------------
        plt.close()
        plt.figure(figsize=(7.,5.))
        nn, bins, patches = plt.hist(rho_marker, bins=50, rwidth=1.,histtype='step',color='k',range=(0.,1.))
        plt.grid(axis='y', alpha=0.75)
        plt.title('starting rho of all markers')
        plt.xlabel('rho')
        plt.ylabel('')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        
        
        plt.close()
        plt.figure(figsize=(7.,5.))
        counts, bins = np.histogram(rho_marker,bins=50)
        counts = counts / np.sum(counts)
        plt.hist(bins[:-1], bins=50, rwidth=1.,histtype='step',color='k',weights=counts, range=(0.,1.))
        plt.grid(axis='y', alpha=0.75)
        plt.title('starting rho of all markers (normalized)')
        plt.xlabel('rho')
        plt.ylabel('')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.close()
        plt.figure(figsize=(7.,5.))
        counts, bins = np.histogram(rho_marker,bins=50)
        counts = counts / np.sum(counts)
        plt.hist(bins[:-1], bins=50, rwidth=1.,histtype='step',color='k',weights=counts, range=(0.001,1.),log=True)
        plt.grid(axis='y', alpha=0.75)
        plt.title('starting rho of all markers (normalized)')
        plt.xlabel('rho')
        plt.ylabel('')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # ----------------------------------------
        #  cumulative marker starts

        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.hist(rho_marker, bins=100, rwidth=1.,histtype='step',color='b',cumulative=True)
        plt.grid(axis='y', alpha=0.75)
        plt.title('starting rho of all markers (cumulative, unweighted)')
        plt.xlabel('rho')
        plt.ylabel('')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        print("zzz time for plots-1: ", mytime.time()-time_last)
        time_last = mytime.time()
        
        # --------------------------------------------------------------

        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.hist(rho_marker, bins=100, rwidth=1.,histtype='step',color='b',cumulative=True,density=True,bottom=0.001,log=True)
        plt.grid(axis='y', alpha=0.75)
        plt.title('starting rho of all markers (cumulative,norm)')
        plt.xlabel('rho')
        plt.ylabel('')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()
        
        plt.close()
        plt.figure(figsize=(7.,5.))
        my_yy, bin_edges, patches = plt.hist(rho_marker, bins=100, rwidth=1.,histtype='step',color='b',cumulative=True,density=True,bottom=0.)
        plt.grid(axis='y', alpha=0.75)
        plt.title('starting rho of all markers (cumulative,norm)')
        plt.xlabel('rho')
        plt.ylabel('')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()
        
        # --------------------------------------------------------------
        #  number markers born within rho and fit
        
        my_xx = bin_edges[1:]
        params, params_covariance = optimize.curve_fit(my_powerlaw, my_xx, my_yy, p0=[1,3])

        my_title = 'cumulative norm: y = {:6.3f} rho**({:6.3f})'.format(params[0], params[1])
        
        plt.close()
        plt.grid(alpha=0.75)
        plt.plot(my_xx, my_yy,'ko', ms=1)
        plt.plot(my_xx, my_powerlaw(my_xx, params[0], params[1]))
        plt.title(my_title,fontsize=12)
        plt.xlabel('rho')
        plt.ylabel('')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()
        # ----------------------------------------------------------
        print("zzz time for plots-2: ", mytime.time()-time_last)
        time_last = mytime.time()
        plt.close()
        plt.figure(figsize=(7.,5.))
        nn, bins, patches = plt.hist(rho_marker_lost, bins=50, rwidth=1.,histtype='step',color='k')
        plt.grid(axis='y', alpha=0.75)
        plt.title('starting rho of all lost markers (unweighted)')
        plt.xlabel('rho')
        plt.ylabel('')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.close()
        plt.figure(figsize=(7.,5.))

        counts, bins = np.histogram(rho_marker_lost,bins=50)
        counts = counts / np.sum(counts)
        plt.hist(bins[:-1], bins=50, rwidth=1.,histtype='step',color='k',weights=counts, range=(0.,1.))
        plt.grid(axis='y', alpha=0.75)
        plt.title('starting rho of all lost markers (norm)')
        plt.xlabel('rho')
        plt.ylabel('')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()


        print("   ... at position O at time: ", clock.ctime())
        
        plt.close()
        plt.figure(figsize=(7.,5.))
        nn, bins, patches = plt.hist(rho_marker_lost, bins=50, rwidth=1.,histtype='step',color='k', density=True, weights=weight_marker_lost)
        plt.grid(axis='y', alpha=0.75)
        plt.title('weighted, norm starting rho of lost markers')
        plt.xlabel('rho')
        plt.ylabel('')
        plt.tight_layout(pad=padsize)

        plt.close()
        plt.figure(figsize=(7.,5.))
        my_bins = 50
        nn, bins, patches = plt.hist(rho_marker_lost, my_bins, rwidth=1.,histtype='step',color='k', weights=energy_weights)
        sum_values = np.sum(nn)
        my_title = 'weighted, norm rho of lost energy.  sum= =  {:6.3f} )'.format(sum_values)
        
        plt.grid(axis='y', alpha=0.75)
        plt.title(my_title,fontsize=fontsize_2)
        plt.xlabel('rho')
        plt.ylabel('')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        check_sum = np.sum(nn) / my_bins
        print("   check on sum of histogram energy weights: ", check_sum)

        # ------------------------------------------------------------------------------------
        
        plt.close()
        plt.figure(figsize=(7.,5.))
        my_zz, bins, patches = plt.hist(rho_marker_lost, bins=50, log=True, cumulative=True, rwidth=1.,histtype='step',color='k', density=True, weights=weight_marker_lost)
        plt.grid(axis='y', alpha=0.75)
        plt.title('weighted, norm, cumulative starting rho of lost markers')
        plt.xlabel('rho')
        plt.ylabel('')
        plt.ylim(1.e-3,1.)
        plot_filename = stub + '_rho_marker_lost_markers_weight_norm.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        
        plt.close()
        plt.figure(figsize=(7.,5.))
        my_zz, bin_edges, patches = plt.hist(rho_marker_lost, bins=50, cumulative=True, rwidth=1.,histtype='step',color='k', density=True, weights=weight_marker_lost)
        plt.grid(axis='y', alpha=0.75)
        plt.title('weighted, norm, cumulative starting rho of lost markers')
        plt.xlabel('rho')
        plt.xlim(0.7,1.0)
        plt.ylim(0.,1.0)
        plt.ylabel('')
        plot_filename = stub + '_rho_marker_lost_markers_weight_norm.pdf'
        print("zzz time for plots-3: ", mytime.time()-time_last)
        time_last = mytime.time()
        try:
            ##pdb.set_trace()
            bbb = myread.read_any_file('rhosafe.txt')
            my_rho_array = bbb[:,0]
            my_fraction =  bbb[:,1]
            plt.step(my_rho_array, my_fraction, 'r', where='post')
        
        except:
            xx_dummy=0.
        
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        my_xx = bin_edges[1:]
        rho_05 = my_xx[np.abs(my_zz-0.005).argmin()]
        rho_1 =  my_xx[np.abs(my_zz-0.01).argmin()]
        rho_2 =  my_xx[np.abs(my_zz-0.02).argmin()]
        rho_5 =  my_xx[np.abs(my_zz-0.05).argmin()]    
        rho_10 = my_xx[np.abs(my_zz-0.10).argmin()]
        rho_25 = my_xx[np.abs(my_zz-0.25).argmin()]
        rho_50 = my_xx[np.abs(my_zz-0.50).argmin()]
        rho_90 = my_xx[np.abs(my_zz-0.90).argmin()]
        rho_95 = my_xx[np.abs(my_zz-0.95).argmin()]
        rho_96 = my_xx[np.abs(my_zz-0.96).argmin()]
        rho_98 = my_xx[np.abs(my_zz-0.98).argmin()]
        
        ##pdb.set_trace()
        
        print("")
        print(' rho for 1 percent loss (weighted): {:6.3f}'.format(rho_1))
        print(' rho for 2 percent loss (weighted): {:6.3f}'.format(rho_2))
        print(' rho for 5 percent loss (weighted): {:6.3f}'.format(rho_5))
        print(' rho for 10 percent loss (weighted): {:6.3f}'.format(rho_10))
        print(' rho for 25 percent loss (weighted): {:6.3f}'.format(rho_25))
        print(' rho for 50 percent loss (weighted): {:6.3f}'.format(rho_50))
        print(' rho for 90 percent loss (weighted): {:6.3f}'.format(rho_90))
        print(' rho for 95 percent loss (weighted): {:6.3f}'.format(rho_95))
        print(' rho for 96 percent loss (weighted): {:6.3f}'.format(rho_96))
        print(' rho for 98 percent loss (weighted): {:6.3f}'.format(rho_98))
        
        print("")
        ##pdb.set_trace()
        # ----------------------------------------------------------------------------------------
        
        plt.close()
        plt.figure(figsize=(7.,5.))
        my_yy, bin_edges, patches = plt.hist(rho_marker_lost, bins=100, rwidth=1.,histtype='step',color='b',cumulative=True,density=True,bottom=0.001,log=True)

        my_xx = bin_edges[1:]
        
        rho_1 =  my_xx[np.abs(my_yy-0.01).argmin()]
        rho_2 =  my_xx[np.abs(my_yy-0.02).argmin()]
        rho_5 =  my_xx[np.abs(my_yy-0.05).argmin()]    
        rho_10 = my_xx[np.abs(my_yy-0.10).argmin()]
        rho_25 = my_xx[np.abs(my_yy-0.25).argmin()]
        rho_50 = my_xx[np.abs(my_yy-0.50).argmin()]
        
        print("")
        print(' rho for 1 percent loss (unweighted): {:6.3f}'.format(rho_1))
        print(' rho for 2 percent loss (unweighted): {:6.3f}'.format(rho_2))
        print(' rho for 5 percent loss (unweighted): {:6.3f}'.format(rho_5))
        print(' rho for 10 percent loss (unweighted): {:6.3f}'.format(rho_10))
        print(' rho for 25 percent loss (unweighted): {:6.3f}'.format(rho_25))
        print(' rho for 50 percent loss (unweighted): {:6.3f}'.format(rho_50))
        print("")
                                
        my_title = 'rho for 1 percent cumulative loss: {:6.3f}'.format(rho_1)
        
        plt.grid(axis='y', alpha=0.75)
        plt.title(my_title)
        plt.xlabel('rho')
        plt.ylabel('')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        my_xx = bin_edges[1:]
        ii_nearest = np.abs(my_yy-0.01).argmin()
        rho_one_percent = my_xx[ii_nearest]

        print,' rho for 1 percent loss: {:6.3f}'.format(rho_one_percent)
 

        # --------------------------------------------------------------

        plt.figure(figsize=(7.,5.))
        nn, bins, patches = plt.hist(rho_marker_wall, bins='auto', rwidth=1., histtype='step',color='k')
        plt.grid(axis='y', alpha=0.75)
        plt.title('starting rho of all wall-lost markers')
        plt.xlabel('rho')
        plot_filename = stub + '_rho_marker_wall_markers.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # --------------------------------------------------------
        #
        # histogram of phi for all (wall + rhomax) lost markers

        number_lost = phi_lost.size
        print('Number of wall+rhomax lost markers: ', number_lost)
        plt.figure()
        nn, bins, patches = plt.hist(phi_lost, bins=60, rwidth=1.,histtype='step',color='k')

        phi_binned_mean = np.mean(nn)
        phi_binned_max  = np.max(nn)
        if(phi_binned_mean > 0.):
            phi_peaking = phi_binned_max / phi_binned_mean
        else:
            phi_peaking = 0.
        print(" phi peaking factor for all lost markers: ",phi_peaking)

        my_title = 'Phi of all lost markers: peaking = {:6.3f} )'.format(phi_peaking)
        
        plt.grid(axis='y', alpha=0.75)
        plt.title(my_title, fontsize=12)
        plt.xlabel('phi [degrees]')
        maxfreq = nn.max()
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # ------------------------------------------------------------------------

        plt.figure()
        nn, bins, patches = plt.hist(phi_lost_18, bins=20, rwidth=1.,histtype='step',color='k')

        phi_binned_mean = np.mean(nn)
        phi_binned_max  = np.max(nn)
        if(phi_binned_mean > 0.):
            phi_peaking = phi_binned_max / phi_binned_mean
        else:
            phi_peaking = 0.
        print(" phi (mod 20-18TF) peaking factor for all lost markers: ",phi_peaking)
        
        my_title = 'Phi (mod 20-18TF): all lost markers: peaking  = {:6.3f} )'.format(phi_peaking)

        print("zzz time for plots-4: ", mytime.time()-time_last)
        time_last = mytime.time()
        plt.grid(axis='y', alpha=0.75)
        plt.title(my_title, fontsize=12)
        plt.xlabel('phi [degrees]')
        maxfreq = nn.max()
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # -----------------------------------------------------------------------------

        plt.figure()
        nn, bins, patches = plt.hist(phi_lost_16, bins=22, rwidth=1.,histtype='step',color='k')

        phi_binned_mean = np.mean(nn)
        phi_binned_max  = np.max(nn)
        if(phi_binned_mean > 0.):
            phi_peaking = phi_binned_max / phi_binned_mean
        else:
            phi_peaking = 0.
        print(" phi (mod 22.5 for TF) peaking factor for all lost markers: ",phi_peaking)
        
        my_title = 'Phi (mod 22.5-16TF): all lost markers: peaking  = {:6.3f} )'.format(phi_peaking)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('phi [degrees]')
        plt.title(my_title, fontsize=12)
        plot_filename = stub + '_phi_lost_markers_16.pdf'
        maxfreq = nn.max()
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # ------------------------------------------------------------------------------
        
        plt.figure()
        nn, bins, patches = plt.hist(phi_lost_14, bins=26, rwidth=1.,histtype='step',color='k')

        phi_binned_mean = np.mean(nn)
        phi_binned_max  = np.max(nn)
        if(phi_binned_mean > 0.):
            phi_peaking = phi_binned_max / phi_binned_mean
        else:
            phi_peaking = 0.
        print(" phi (mod 25.71 for 14 TF) peaking factor for all lost markers: ",phi_peaking)
        
        my_title = 'Phi (mod 25.71 for 14 TF): all lost markers: peaking  = {:6.3f} )'.format(phi_peaking)
        
        plt.grid(axis='y', alpha=0.75)
        plt.title(my_title,fontsize=12)
        plt.xlabel('phi [degrees]')
        maxfreq = nn.max()
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # -------------------------------------------------------------------------------

        print("   ... at position P at time: ", clock.ctime())
        
        plt.figure()
        nn, bins, patches = plt.hist(phi_lost_12, bins=30, rwidth=1.,histtype='step',color='k')

        phi_binned_mean = np.mean(nn)
        phi_binned_max  = np.max(nn)
        if(phi_binned_mean > 0.):
            phi_peaking = phi_binned_max / phi_binned_mean
        else:
            phi_peaking = 0.
        print(" phi (mod 30-12TF) peaking factor for all lost markers: ",phi_peaking)
        
        my_title = 'Phi (mod 30 for 12 TF): all lost markers: peaking  = {:6.3f} )'.format(phi_peaking)
        
        plt.grid(axis='y', alpha=0.75)
        plt.title(my_title,fontsize=12)
        plt.xlabel('phi [degrees]')
        maxfreq = nn.max()
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        print("   zzz time for next set of plots: ", mytime.time() - time_last)
        time_last = mytime.time()
        
        # -----------------------------------------
        #  repeat with weighted

        nn, bins, patches = plt.hist(phi_lost_18, bins=20, rwidth=1.,histtype='step',color='k', weights=weight_lost)

        phi_binned_mean = np.mean(nn)
        phi_binned_max  = np.max(nn)
        if(phi_binned_mean > 0.):
            phi_peaking = phi_binned_max / phi_binned_mean
        else:
            phi_peaking = 0.
        print(" phi (mod 30-12TF) peaking factor for all lost markers (weighted): ",phi_peaking)
        
        my_title = 'Phi (mod 30 for 12 TF): all lost markers (weighted): peaking  = {:6.3f} )'.format(phi_peaking)
        
        plt.grid(axis='y', alpha=0.75)
        plt.title('Phi (mod 20 for 18 TF) of lost markers, weighted',fontsize=12)
        plt.xlabel('phi [degrees]')
        maxfreq = nn.max()
        plt.tight_layout(pad=padsize)
        plt.xlim((-1.,21.))
        plt.ylim(bottom=0)
        pdf.savefig()
        plt.close()
        print("zzz time for plots-5: ", mytime.time()-time_last)
        time_last = mytime.time()
        
        try:
            bbb = myread.read_any_file('losses_vs_phi.txt')
            phi_spiral   = 20. - bbb[:,0]                  # we seem to have a sign error 4/3/2020
            mloss_spiral = bbb[:,1]

            # #pdb.set_trace()

            plt.figure()
            plt.step(phi_spiral, mloss_spiral, 'r', where='mid')

            plt.title('spiral Phi (mod 20 for 18 TF) of lost markers, weighted')
            plt.xlabel('phi [degrees]')
            maxfreq = nn.max()
            plt.tight_layout(pad=padsize)
            plt.xlim((-1.,21.))
            plt.ylim(bottom=0.)
            pdf.savefig()
        except:
            my_dummy=0.
            


        plt.figure()
        nn, bins, patches = plt.hist(phi_lost_16, bins=23, rwidth=1.,histtype='step',color='k', weights=weight_lost)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('phi [degrees]')
        plt.title('Phi (mod 22.5 for 16 TF) of lost markers, weighted')
        maxfreq = nn.max()
        plt.tight_layout(pad=padsize)
        plt.xlim((-1.,23.5))
        plt.ylim(bottom=0.)
        pdf.savefig()

        plt.close()
        plt.figure()
        nn, bins, patches = plt.hist(phi_lost_14, bins=26, rwidth=1.,histtype='step',color='k', weights=weight_lost)
        plt.grid(axis='y', alpha=0.75)
        plt.title('Phi (mod 25.71 for 14 TF) of lost markers, weighted')
        plt.xlabel('phi [degrees]')
        maxfreq = nn.max()
        plt.tight_layout(pad=padsize)
        plt.xlim((-1, 26.71))
        plt.ylim(bottom=0.)
        pdf.savefig()
        plt.close()

        plt.figure()
        nn, bins, patches = plt.hist(phi_lost_12, bins=30, rwidth=1.,histtype='step',color='k', weights=weight_lost)
        plt.grid(axis='y', alpha=0.75)
        plt.title('Phi (mod 30 for 12 TF) of lost markers, weighted')
        plt.xlabel('phi [degrees]')
        maxfreq = nn.max()
        plt.tight_layout(pad=padsize)
        plt.xlim((-1.,31.))
        plt.ylim(bottom=0.)
        pdf.savefig()
        plt.close()

        try:
            
            bbb = myread.read_any_file('losses_vs_phi.txt')
            phi_spiral   = 30. - bbb[:,0]
            mloss_spiral = bbb[:,1]

            plt.figure()
            plt.step(phi_spiral, mloss_spiral, 'r', where='mid')

            plt.title('spiral Phi (mod 30 for 12 TF) of lost markers, weighted')
            plt.xlabel('phi [degrees]')
            maxfreq = nn.max()
            plt.tight_layout(pad=padsize)
            plt.xlim((-1.,31.))
            plt.ylim(bottom=0.)
            pdf.savefig()
            plt.close()
        except:
            my_dummy=0.
        
        # -----------------------------------------------------------------------------
        #
        # histogram of phi for all (wall + rhomax) lost markers ... with fixed phi-bins

        my_bins = np.linspace(0,360,19)

        plt.figure()
        nn, bins, patches = plt.hist(phi_lost, bins='auto', rwidth=1.,histtype='step',color='k')
        plt.grid(axis='y', alpha=0.75)
        plt.title('Phi of all lost markers')
        plt.xlabel('phi [degrees]')
        maxfreq = nn.max()
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

      # histogram of phi for wall-lost markers

        number_lost = phi_lost.size
        print('Number of wall-lost markers: ', number_lost)

        plt.figure()
        nn, bins, patches = plt.hist(phi_wall, my_bins, rwidth=1.,histtype='step',color='k')
        plt.grid(axis='y', alpha=0.75)
        plt.title('Phi of wall-lost markers')
        plt.xlabel('phi [degrees]')
        maxfreq = nn.max()
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # histogram of phi for all wall-lost markers ... with fixed phi-bins

        my_bins = np.linspace(0,360,19)
        plt.figure()
        nn, bins, patches = plt.hist(phi_wall, my_bins, rwidth=1.,histtype='step',color='k')
        plt.grid(axis='y', alpha=0.75)
        plt.title('Phi of wall-lost markers')
        plt.xlabel('phi [degrees]')
        maxfreq = nn.max()
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # -----------------------------------------------------------------------------
        #
        # ... with fixed phi-bins

        my_bins = np.linspace(0,360,19)
        plt.figure()
        nn, bins, patches = plt.hist(phi_lost, bins='auto', rwidth=1.,histtype='step',color='k')
        plt.grid(axis='y', alpha=0.75)
        plt.title('Phi of all lost markers')
        plt.xlabel('phi [degrees]')
        maxfreq = nn.max()
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()
        print("zzz time for plots-6 ", mytime.time()-time_last)
        time_last = mytime.time()
      # histogram of phi for wall-lost markers

        number_lost = phi_lost.size
        print('Number of wall-lost markers: ', number_lost)
        plt.figure()
        nn, bins, patches = plt.hist(phi_wall, my_bins, rwidth=1.,histtype='step',color='k')
        plt.grid(axis='y', alpha=0.75)
        plt.title('Phi of wall-lost markers')
        plt.xlabel('phi [degrees]')
        plot_filename = stub + '_phi_wall_markers.pdf'
        maxfreq = nn.max()
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # histogram of phi for all wall-lost markers ... with fixed phi-bins

        my_bins = np.linspace(0,360,19)

        plt.figure()   
        nn, bins, patches = plt.hist(phi_wall, my_bins, rwidth=1.,histtype='step',color='k')
        plt.grid(axis='y', alpha=0.75)
        plt.title('Phi of wall-lost markers')
        plt.xlabel('phi [degrees]')
        plot_filename = stub + '_phi_wall_markers_xfixed.pdf'
        maxfreq = nn.max()
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # --------------------------------------------------------------
        # compare "marker" and "ini"
        plt.figure()
        xx_fake = np.linspace(1.3,2.5,2)
        yy_fake = xx_fake
        plt.plot(r_marker, r_ini, 'ro', ms=0.25, rasterized=do_rasterized_2)
        plt.plot(xx_fake, yy_fake, 'k', rasterized=do_rasterized, linewidth=0.75, linestyle="dashed")
        plt.title('r_ini vs r_marker (dash: y=x)')
        plt.xlabel('r_marker')
        plt.ylabel('r_ini')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.figure()
        xx_fake = np.linspace(1.3,2.5,2)
        yy_fake = 0. * xx_fake
        plt.plot(r_marker, r_ini-r_marker, 'ro', ms=0.25, rasterized=do_rasterized_2)
        plt.plot(xx_fake, yy_fake, 'k-', rasterized=do_rasterized, linewidth=0.75)
        plt.title('r_ini-r_marker vs r_marker ')
        plt.xlabel('r_marker')
        plt.ylabel('r_ini - r_marker')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.figure()
        xx_fake = np.linspace(-1.2,1.2,2)
        yy_fake = xx_fake
        plt.plot(z_marker, z_ini, 'ro', ms=0.25, rasterized=do_rasterized_2)
        plt.plot(xx_fake, yy_fake, 'k', rasterized=do_rasterized, linewidth=0.75, linestyle="dashed")
        plt.title('z_ini vs z_marker (dash: y=x)')
        plt.xlabel('z_marker')
        plt.ylabel('z_ini')
        plot_filename = stub + '_z_ini_z_marker.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        print("   ... at position Q at time: ", clock.ctime())
        
        plt.figure()
        xx_fake = np.linspace(-1.2,1.2,2)
        yy_fake = 0.*xx_fake
        plt.plot(z_marker, z_ini-z_marker, 'ro', ms=0.25, rasterized=do_rasterized_2)
        plt.plot(xx_fake, yy_fake, 'k-', rasterized=do_rasterized, linewidth=0.75)
        plt.title('z_ini-z_marker vs z_marker ')
        plt.xlabel('z_marker')
        plt.ylabel('z_ini - z_marker')
        plot_filename = stub + '_z_ini_z_marker.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()
        
        # ----------------------------------------
        # 
        #   plots of all starting markers that did
        #   not abort, and then superimpose those that
        #   hit wall or rhomax or 520

        rmarker_wall = r_marker[ii_wall]
        zmarker_wall = z_marker[ii_wall]

        ii_rhomax = (endcond == 32)

        rmarker_rhomax = r_marker[ii_rhomax]
        zmarker_rhomax = z_marker[ii_rhomax]

        rmarker_520 = r_marker[ii_520]
        zmarker_520 = z_marker[ii_520]

        xmin = np.min(aa["r_wall"])
        xmax = np.max(aa["r_wall"])
        ymin = np.min(aa["z_wall"])
        ymax = np.max(aa["z_wall"])

        print("   zzz time for next set of plots: ", mytime.time() - time_last)
        time_last = mytime.time()

        # -------------------------------------------------------
        #

        plt.close()
        xsize = 7.
        ysize = 5.
        plt.figure(figsize=(xsize,ysize))
        plt.tight_layout(pad=2, h_pad=6)
        plt.plot(ekev_ini, pitch_ini, 'go', ms=1, rasterized=do_rasterized_2)   
        plt.title('Energy and pitch angle of initial markers')
        plt.xlabel('[keV]')
        plot_filename = stub + '_ini_ekev_pitch.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        # -------------------------------------------------------
        #  multiplot of R,z pitch

        plt.close()
        xsize = 5.
        ysize = xsize * (ymax-ymin)/(xmax-xmin)
        ysize_ratio = (ymax-ymin)/(xmax-xmin)
        plt.figure(figsize=(xsize,ysize))
        plt.tight_layout(pad=1, h_pad=3, w_pad=0.2)
        plt.suptitle('Initial pitch of lost markers', fontsize=10)

        jj_1 = (pitch_ini_lost >=0.)    ^ (pitch_ini_lost<=0.25)
        jj_2 = (pitch_ini_lost >=0.25)  ^ (pitch_ini_lost<=0.50)
        jj_3 = (pitch_ini_lost >=0.50)  ^ (pitch_ini_lost<=0.75)
        jj_4 = (pitch_ini_lost >=0.75)

        jj_5 = (pitch_ini_lost >=-0.25)  ^ (pitch_ini_lost<=0.)
        jj_6 = (pitch_ini_lost >=-0.50)  ^ (pitch_ini_lost<=-0.25)
        jj_7 = (pitch_ini_lost >=-0.75)  ^ (pitch_ini_lost<=-0.50)
        jj_8 = (pitch_ini_lost <=-0.75)

        plt.subplot(2,4,1)
        plt.plot(r_ini_lost[jj_1], z_ini_lost[jj_1], 'ro', ms=1,rasterized=do_rasterized, fillstyle='none')
        plt.title('0.0 0.25', fontsize=10)
        cur_axes = plt.gca()
        cur_axes.get_xaxis().set_ticklabels([])

        plt.subplot(2,4,2)
        plt.plot(r_ini_lost[jj_2], z_ini_lost[jj_2], 'ro', ms=1,rasterized=do_rasterized, fillstyle='none')
        plt.title('0.25 0.50', fontsize=10)
        cur_axes = plt.gca()
        cur_axes.get_xaxis().set_ticklabels([])
        cur_axes.get_yaxis().set_ticklabels([])

        plt.subplot(2,4,3)
        plt.plot(r_ini_lost[jj_3], z_ini_lost[jj_3], 'ro', ms=1,rasterized=do_rasterized, fillstyle='none')
        plt.title('0.50 0.75', fontsize=10)
        cur_axes = plt.gca()
        cur_axes.get_xaxis().set_ticklabels([])
        cur_axes.get_yaxis().set_ticklabels([])

        plt.subplot(2,4,4)
        plt.plot(r_ini_lost[jj_4], z_ini_lost[jj_4], 'ro', ms=1,rasterized=do_rasterized, fillstyle='none')
        plt.title('0.75 1.00', fontsize=10)
        cur_axes = plt.gca()
        cur_axes.get_xaxis().set_ticklabels([])
        cur_axes.get_yaxis().set_ticklabels([])

        plt.subplot(2,4,5)
        plt.plot(r_ini_lost[jj_5], z_ini_lost[jj_5], 'ro', ms=1,rasterized=do_rasterized, fillstyle='none')
        plt.title('-0.25  0.', fontsize=10)

        plt.subplot(2,4,6)
        plt.plot(r_ini_lost[jj_6], z_ini_lost[jj_6], 'ro', ms=1,rasterized=do_rasterized, fillstyle='none')
        plt.title('-0.5 -0.25', fontsize=10)
        cur_axes = plt.gca()
        cur_axes.get_yaxis().set_ticklabels([])

        plt.subplot(2,4,7)
        plt.plot(r_ini_lost[jj_7], z_ini_lost[jj_7], 'ro', ms=1,rasterized=do_rasterized, fillstyle='none')
        plt.title('-0.75 -0.50', fontsize=10)
        cur_axes = plt.gca()
        cur_axes.get_yaxis().set_ticklabels([])

        plt.subplot(2,4,8)
        plt.plot(r_ini_lost[jj_8], z_ini_lost[jj_8], 'ro', ms=1,rasterized=do_rasterized, fillstyle='none')
        plt.title('-1. -0.75', fontsize=10)
        cur_axes = plt.gca()
        cur_axes.get_yaxis().set_ticklabels([])

        plt.tight_layout(pad=padsize)
        pdf.savefig()
        # -------------------------------------------------------------
        #  initial positions that are lost - positive pitch

        plt.close()
        xsize = 4.
        ysize = xsize * (ymax-ymin)/(xmax-xmin)
        plt.figure(figsize=(xsize,ysize))
        plt.tight_layout(pad=2)

        jj_1 = (pitch_ini_lost >=0.)    ^ (pitch_ini_lost<=0.25)
        jj_2 = (pitch_ini_lost >=0.25)  ^ (pitch_ini_lost<=0.50)
        jj_3 = (pitch_ini_lost >=0.50)  ^ (pitch_ini_lost<=0.75)
        jj_4 = (pitch_ini_lost >=0.75)

        plt.plot(r_ini_lost[jj_1], z_ini_lost[jj_1], 'ko', ms=1, rasterized=do_rasterized)
        plt.plot(r_ini_lost[jj_2], z_ini_lost[jj_2], 'ro', ms=1, rasterized=do_rasterized)
        plt.plot(r_ini_lost[jj_3], z_ini_lost[jj_3], 'go', ms=1, rasterized=do_rasterized)
        plt.plot(r_ini_lost[jj_4], z_ini_lost[jj_4], 'bo', ms=1, rasterized=do_rasterized)

        plt.title('Initial posns lost (k-r-g-b = pitch [0. 0.25 0.5 0.75 1.0])', fontsize=10)
        plt.xlabel('Rmajor [m]')
        plt.ylabel('z [m]')

        plot_filename = stub + '_iniRZ_lost_pos_pitch.pdf'
        pdf.savefig()

        # -------------------------------------------------------------
        #  initial positions that are lost - negative pitch

        plt.close()
        xsize = 4.
        ysize = xsize * (ymax-ymin)/(xmax-xmin)
        plt.figure(figsize=(xsize,ysize))
        plt.tight_layout(pad=2)

        jj_1 = (pitch_ini_lost >=-0.25)  ^ (pitch_ini_lost<=0.)
        jj_2 = (pitch_ini_lost >=-0.50)  ^ (pitch_ini_lost<=-0.25)
        jj_3 = (pitch_ini_lost >=-0.75)  ^ (pitch_ini_lost<=-0.50)
        jj_4 = (pitch_ini_lost <=-0.75)

        plt.plot(r_ini_lost[jj_1], z_ini_lost[jj_1], 'ko', ms=1, rasterized=do_rasterized)
        plt.plot(r_ini_lost[jj_2], z_ini_lost[jj_2], 'ro', ms=1, rasterized=do_rasterized)
        plt.plot(r_ini_lost[jj_3], z_ini_lost[jj_3], 'go', ms=1, rasterized=do_rasterized)
        plt.plot(r_ini_lost[jj_4], z_ini_lost[jj_4], 'bo', ms=1, rasterized=do_rasterized)

        plt.title('Initial posns lost (k-r-g-b = pitch [0. -0.25 -0.5 -0.75 -1.0])', fontsize=10)
        plt.xlabel('Rmajor [m]')
        plt.ylabel('z [m]')

        plot_filename = stub + '_iniRZ_lost_neg_pitch.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        # -------------------------------------------------------------
        #  initial positions that are lost - rho and pitch

        plt.close()
        xsize = 7.
        ysize = 5.
        plt.figure(figsize=(xsize,ysize))
        plt.tight_layout(pad=2, h_pad=6)

        plt.plot(rho_ini_lost, pitch_ini_lost, 'ro', ms=1, rasterized=do_rasterized_2)

        plt.title('Lost markers:  initial conditions')
        plt.xlabel('rho')
        plt.ylabel('pitch')
        plot_filename = stub + '_ini_lost_rho_pitch.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        # --------------------------------------------------------
        #  histogram of all ini markers vs pitch

        plt.close()
        plt.figure(figsize=(7.,5.))
        #pdb.set_trace(header="before histogram of all ini markers vs pitch")
        my_yy, bin_edges, patches = plt.hist(pitch_ini[np.abs(pitch_ini<0.9999)], bins='auto', rwidth=1.,histtype='step',color='k')
        yy_max = np.max(my_yy)

        my_xx = -1. + np.linspace(0., 2., 1000)
        my_yy =  yy_max * np.sqrt(1 - my_xx**2)
        plt.plot(my_xx, my_yy, 'r')
        
        plt.grid(axis='y', alpha=0.75)
        plt.title('initial pitch of all markers:')
        plt.ylabel('')
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        # -----------------------------------------------------
        #    marker pitch with aborted ones

        print("   ... at position R at time: ", clock.ctime())
        
        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.hist(pitch_phi_marker, bins=50, rwidth=1.,histtype='step',color='k')
        if (ii_abort.size>0):
            plt.hist(pitch_phi_marker[ii_abort], bins=50, rwidth=1.,histtype='step',color='r')
        plt.grid(axis='y', alpha=0.75)
        plt.title('initial pitch of markers (red=aborted')
        plt.ylabel('')
        plt.xlabel('v_parallel/v')
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        # --------------------------------------------------------
        #  histogram of lost markers vs pitch

        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.hist(pitch_ini_lost, bins='auto', rwidth=1.,histtype='step',color='k')
        plt.grid(axis='y', alpha=0.75)
        plt.title('initial pitch of lost markers:')
        plt.ylabel('')
        plt.xlabel('v_parallel/v')
        plt.tight_layout(pad=padsize)
        pdf.savefig()


        # --------------------------------------------------------
        #  histogram of lost markers vs pitch

        plt.close()
        plt.figure(figsize=(7.,5.))
        # nn, bins, patches = plt.hist(simtime_lost, bins='auto', rwidth=0.85,alpha=0.75,color='b',log=True)
        plt.hist(pitch_lost, bins='auto', rwidth=1.,histtype='step',color='k')
        plt.grid(axis='y', alpha=0.75)
        plt.title('final pitch of lost markers:')
        plt.ylabel('')
        plt.xlabel('v_parallel/v')
        plot_filename = stub + '_hist_final_pitch_lost.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        # ----------------------------------------------------------------
        #  initial and final pitch of lost markers
        #  
        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.hist(pitch_ini_lost, bins='auto', rwidth=1., histtype='step', alpha = 1., color='k', label='initial')
        plt.hist(pitch_lost,     bins='auto', rwidth=1., histtype='step', alpha = 1., color='r', label='final')
        plt.grid(axis='y', alpha=0.75)
        plt.title('initial and final (red) pitch of lost markers:')
        plt.xlabel('v_parallel/v')
        plt.ylabel('')
        #plt.legend('upper right')
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(7.,5.))

        total_lost_weight = np.sum(weight_lost)
        weight_lost_norm  = weight_lost/total_lost_weight
        
        plt.hist(pitch_ini_lost, bins=50, rwidth=1., histtype='step', alpha = 1., color='k', weights=weight_lost_norm, label='initial')
        plt.hist(pitch_lost,     bins=50, rwidth=1., histtype='step', alpha = 1., color='b', weights=weight_lost_norm, label='final')

        plt.grid(axis='y', alpha=0.75)
        plt.title('initial/final (bl/blue) pitch of lost markers (weighted):')
        plt.xlabel('v_parallel/v')
        plt.ylabel('')
        #plt.legend('upper right')
        plt.tight_layout(pad=padsize)

        try:
            
            bbb = myread.read_any_file('pitch_loss.txt')
            my_pitch   = bbb[:,0]
            my_initial = bbb[:,1]
            my_final   = bbb[:,2]

            total_initial = np.sum(my_initial)
            total_final   = np.sum(my_final)

            my_initial = my_initial / total_initial
            my_final   = my_final   / total_final

            plt.step(my_pitch, my_initial, 'r', where='mid')
            plt.step(my_pitch, my_final,   'orange', where='mid')
            
        except:
            my_dummy = 0.

        plt.xlim((-1,1.))
        plt.ylim(bottom=0.)
        pdf.savefig()
        
        # ------------------------------------------------------------------
        #  plot just the markers that hit the wall or > rhomax

        plt.close()
        xsize = 4.
        ysize = xsize * (ymax-ymin)/(xmax-xmin)
        plt.figure(figsize=(xsize,ysize))
        plt.tight_layout(pad=2, h_pad=2)
        plt.grid('True')


        plt.plot(rmarker_wall,   zmarker_wall,   'ro', ms=1, rasterized=do_rasterized)
        plt.plot(rmarker_rhomax, zmarker_rhomax, 'bo', ms=1, rasterized=do_rasterized)
        plt.plot(rmarker_520,    zmarker_520,    'co', ms=1, rasterized=do_rasterized)

        plt.plot(aa["r_wall"], aa["z_wall"], 'k-', linewidth=2, rasterized=do_rasterized)
        plt.title('Starting positions (wall=gr rhomax=bl 520=cy )', fontsize=12)

        plt.xlabel('Rmajor [m]')
        plt.ylabel('z [m]')

        plot_filename = stub + '_marker_starts_wall_rhomax_520_RZ.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        print("   ... at position S at time: ", clock.ctime())

        #  plot all markers and lost markers
        plt.close()
        xsize = 3.0
        ysize = xsize * (ymax-ymin)/(xmax-xmin)
        plt.figure(figsize=(xsize, ysize))
        plt.plot(rmarker_all, zmarker_all, 'go', ms=0.25, fillstyle='full', rasterized=do_rasterized)
        plt.plot(rmarker_wall, zmarker_wall, 'ro', ms=0.25, rasterized=do_rasterized)
        plt.plot(rmarker_rhomax, zmarker_rhomax, 'bo', ms=0.25, rasterized=do_rasterized)
        plt.plot(rmarker_520, zmarker_520, 'co', ms=0.25, rasterized=do_rasterized)

        plt.title('marker starting positions (gr/re/bl = all / wall / rhomax)', fontsize=fontsize_2)
        plt.xlabel('Rmajor [m]', fontsize=fontsize_2)
        plt.ylabel('z [m]', fontsize=fontsize_2)
        plt.plot(aa["r_wall"], aa["z_wall"], 'k-', linewidth=2, rasterized=do_rasterized)
        plot_filename = stub + '_marker_starts_RZ.pdf'
        #plt.tight_layout(pad=padsize)
        pdf.savefig()
        #plt.show()

        print("   zzz time for next set of plots: ", mytime.time() - time_last)
        time_last = mytime.time()

        # -------------------------------------------------
        #   plots of all starting markers that did abort

        ii_abort = ( endcond == 0)

        rmarker_abort = r_marker[ii_abort]
        zmarker_abort = z_marker[ii_abort]

        
        xmin = np.min(aa["r_wall"])
        xmax = np.max(aa["r_wall"])
        ymin = np.min(aa["z_wall"])
        ymax = np.max(aa["z_wall"])

        xsize = 4.5
        ysize = xsize * (ymax-ymin)/(xmax-xmin)
        plt.figure(figsize=(xsize,ysize))
        plt.tight_layout(pad=2)
        plt.grid(True)
        plt.plot(rmarker_abort, zmarker_abort,  'ro', ms=2, fillstyle='full', rasterized=do_rasterized_2)
        plt.title('markers that aborted')
        plt.xlabel('Rmajor [m]')
        plt.ylabel('z [m]')
        plt.plot(aa["r_wall"], aa["z_wall"], 'k-', linewidth=2, rasterized=do_rasterized_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(xsize,ysize))
        plt.tight_layout(pad=2)
        plt.grid(True)
        plt.plot(r_marker, z_marker,              'ko', ms=1, fillstyle='full', rasterized=do_rasterized_2)
        plt.plot(rmarker_abort, zmarker_abort,  'ro', ms=2, fillstyle='full', rasterized=do_rasterized_2)
        plt.title('markers (b) and aborted (red)')
        plt.xlabel('Rmajor [m]')
        plt.ylabel('z [m]')
        plt.plot(aa["r_wall"], aa["z_wall"], 'b-', linewidth=2, rasterized=do_rasterized_2)
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # ------------------------------------------------------
        #   lots of plots to identify bad markers
        #

        plt.figure(figsize=(xsize, ysize))
        plt.plot(r_marker, z_marker, 'ro', ms=2, rasterized=do_rasterized_2)
        plt.tight_layout(pad=2)
        if (nn_bad > 0):
            plt.plot(r_marker[ii_bad], z_marker[ii_bad], 'bo', ms=5, rasterized=do_rasterized_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        my_title = 'Marker locations (# bad = ' + str(nn_bad) + ' (bad=blue)'
        plt.title(my_title,fontsize=fontsize_2)
        plot_filename = stub + '_marker_RZ_with_bad.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(xsize, ysize))
        plt.plot(r_ini, z_ini, 'ro', ms=2, rasterized=do_rasterized)
        plt.tight_layout(pad=2)
        if (nn_bad > 0):
            plt.plot(r_ini[ii_bad], z_ini[ii_bad], 'bo', ms=5, rasterized=do_rasterized_2)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        my_title = 'Initial locations (# bad = ' + str(nn_bad) + ' (bad=blue)'
        plt.title(my_title, fontsize=fontsize_2)
        plot_filename = stub + '_initial_RZ_with_bad.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()
        print("zzz time for plots-7: ", mytime.time()-time_last)
        time_last = mytime.time()
        plt.figure(figsize=(7.,5.))
        plt.plot(vphi_marker, pitch_phi_marker, 'ro', ms=2, rasterized=do_rasterized_2)
        plt.tight_layout(pad=2)
        if (nn_bad > 0):
            plt.plot(vphi_marker[ii_bad],pitch_phi_marker[ii_bad], 'bo', ms=5, rasterized=do_rasterized_2)
        plt.xlabel('vphi_marker')
        my_title = 'Marker Pitch_phi (# bad = ' + str(nn_bad) + ' (bad=blue)'
        plt.title(my_title)
        plot_filename = stub + '_marker_pitch_phi_with_bad.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(7.,5.))
        plt.plot(vphi_marker, pitch_phi_marker, 'ro', ms=2, rasterized=do_rasterized)
        plt.tight_layout(pad=2)
        if (nn_bad > 0):
            plt.plot(vphi_marker[ii_bad],pitch_phi_marker[ii_bad], 'bo', ms=5, rasterized=do_rasterized)
        plt.xlabel('vphi_marker')
        plt.xlim((1.25e7, 1.35e7))
        plt.ylim((0.98, 1.005))
        my_title = 'Marker Pitch_phi (# bad = ' + str(nn_bad) + ' (bad=blue)'
        plt.title(my_title)
        plot_filename = stub + '_marker_pitch_phi_with_bad_ylim1.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(7.,5.))
        plt.plot(vphi_marker, pitch_phi_marker, 'ro', ms=2, rasterized=do_rasterized)
        if (nn_bad > 0):
            plt.plot(vphi_marker[ii_bad],pitch_phi_marker[ii_bad], 'bo', ms=5, rasterized=do_rasterized)
        plt.xlabel('vphi_marker')
        plt.xlim((-1.35e7, -1.25e7))
        plt.ylim((-1.005, -0.98))
        my_title = 'Marker Pitch_phi (# bad = ' + str(nn_bad) + ' (bad=blue)'
        plt.title(my_title)
        plot_filename = stub + '_marker_pitch_phi_with_bad_ylim2.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # =========================================================
        #  repeat for pitch_phi_ini

        plt.figure(figsize=(7.,5.))
        plt.plot(vphi_ini, pitch_phi_ini, 'ro', ms=2, rasterized=do_rasterized_2)
        plt.tight_layout(pad=2)
        if (nn_bad > 0):
            plt.plot(vphi_ini[ii_bad],pitch_phi_ini[ii_bad], 'bo', ms=5, rasterized=do_rasterized_2)
        plt.xlabel('vphi_ini')
        my_title = 'Initial Pitch_phi (# bad = ' + str(nn_bad) + ' (bad=blue)'
        plt.title(my_title)
        plot_filename = stub + '_initial_pitch_phi_with_bad.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        print("   ... at position T at time: ", clock.ctime())
        
        plt.figure(figsize=(7.,5.))
        plt.plot(vphi_ini, pitch_phi_ini, 'ro', ms=2, rasterized=do_rasterized_2)
        plt.tight_layout(pad=2)
        if (nn_bad > 0):
            plt.plot(vphi_ini[ii_bad],pitch_phi_ini[ii_bad], 'bo', ms=5, rasterized=do_rasterized_2)
        plt.xlabel('vphi_ini')
        plt.xlim((1.25e7, 1.35e7))
        plt.ylim((0.98, 1.005))
        my_title = 'Initial Pitch_phi (# bad = ' + str(nn_bad) + ' (bad=blue)'
        plt.title(my_title)
        plot_filename = stub + '_initial_pitch_phi_with_bad_ylim1.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(7.,5.))
        plt.plot(vphi_ini, pitch_phi_ini, 'ro', ms=2, rasterized=do_rasterized_2)
        if (nn_bad > 0):
            plt.plot(vphi_ini[ii_bad],pitch_phi_ini[ii_bad], 'bo', ms=5, rasterized=do_rasterized_2)
        plt.xlabel('vphi_ini [m/s]')
        plt.xlim((-1.35e7, -1.25e7))
        plt.ylim((-1.005, -0.98))
        my_title = 'Initial Pitch_phi (# bad = ' + str(nn_bad) + ' (bad=blue)'
        plt.title(my_title)
        plot_filename = stub + '_initial_pitch_phi_with_bad_ylim2.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # =========================================================

        nn_cputime = pitch_ini[ii_cputime].size

        if(nn_cputime >0):

           plt.figure(figsize=(7.,5.))
           plt.plot(rho_ini[ii_cputime], pitch_ini[ii_cputime], 'ro', ms=3, rasterized=do_rasterized)
           plt.tight_layout(pad=2)

           plt.xlabel('rho_ini')
           plt.ylabel('Pitch angle')
           plt.xlim(0.,1.)
           plt.ylim(-1,1)
           my_title = 'Pitch angle vs rho of cputime-end'
           plt.title(my_title)
           plot_filename = stub + '_pitch_rho_cputime_end.pdf'
           pdf.savefig()
           plt.close()

        print("zzz time for plots-8: ", mytime.time()-time_last)
        time_last = mytime.time()
        # =========================================================
        #  repeat for pitch_phi_ini

        plt.figure(figsize=(7.,5.))
        plt.plot(vphi_ini, pitch_ini, 'ro', ms=1, rasterized=do_rasterized)
        plt.tight_layout(pad=2)
        if (nn_bad > 0):
            plt.plot(vphi_ini[ii_bad],pitch_ini[ii_bad], 'bo', ms=4, rasterized=do_rasterized_2)
        plt.xlabel('vphi_ini')
        my_title = 'Initial true Pitch (# bad = ' + str(nn_bad) + ' (bad=blue)'
        plt.title(my_title)
        plot_filename = stub + '_initial_true_pitch_with_bad.pdf'
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(7.,5.))
        plt.plot(vphi_ini, pitch_ini, 'ro', ms=1, rasterized=do_rasterized)
        plt.tight_layout(pad=2)
        if (nn_bad > 0):
            plt.plot(vphi_ini[ii_bad],pitch_ini[ii_bad], 'bo', ms=5, rasterized=do_rasterized_2)
        plt.xlabel('vphi_ini')
        plt.xlim((1.25e7, 1.35e7))
        plt.ylim((0.98, 1.005))
        my_title = 'Initial Pitch_phi (# bad = ' + str(nn_bad) + ' (bad=blue)'
        plt.title(my_title)
        plot_filename = stub + '_initial_true_pitch_with_bad_ylim1.pdf'
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(7.,5.))
        plt.plot(vphi_ini, pitch_ini, 'ro', ms=1, rasterized=do_rasterized)
        plt.tight_layout(pad=2)
        if (nn_bad > 0):
            plt.plot(vphi_ini[ii_bad],pitch_ini[ii_bad], 'bo', ms=5, rasterized=do_rasterized_2)
        plt.xlabel('vphi_ini [m/s]')
        plt.xlim((-1.35e7, -1.25e7))
        plt.ylim((-1.005, -0.98))
        my_title = 'Initial Pitch_phi (# bad = ' + str(nn_bad) + ' (bad=blue)'
        plt.title(my_title)
        plot_filename = stub + '_initial_true_pitch_with_bad_ylim2.pdf'
        pdf.savefig()
        plt.close()

        # ================================================

        plt.figure(figsize=(7.,5.))
        plt.plot(vphi_ini, vpar_ini, 'ro', ms=1, rasterized=do_rasterized_2)
        plt.tight_layout(pad=2)
        if (nn_bad > 0):
            plt.plot(vphi_ini[ii_bad],vpar_ini[ii_bad], 'bo', ms=5, rasterized=do_rasterized_2)
        plt.xlabel('vphi_ini[m/s]')
        plt.ylabel('vpar_ini[m/s]')
        my_title = 'vpar_ini vs vphi_ini (# bad = ' + str(nn_bad) + ' (bad=blue)'
        plt.title(my_title, fontsize=fontsize_2)
        plot_filename = stub + '_vpar_vphi_ini_bad.pdf'
        pdf.savefig()
        plt.close()

        print("   zzz time for next set of plots: ", mytime.time() - time_last)
        time_last = mytime.time()
        print("zzz time for plots-9: ", mytime.time()-time_last)
        time_last = mytime.time()
        # ------------------------------------------------
        #  plot distribution of lost particles

        nseg = 80

        nlost_array        = np.zeros(nseg)
        flost_array        = np.zeros(nseg)
        flost_weighted     = np.zeros(nseg)
        flost_weighted_sum = np.zeros(nseg)

        flost_energy_array        = np.zeros(nseg)
        flost_energy_weighted     = np.zeros(nseg)
        flost_energy_weighted_sum = np.zeros(nseg)

        rwall_mid    = np.array(nseg)
        zwall_mid    = np.array(nseg)
        slope_wall   = np.array(nseg)
        slope_vector = np.array(nseg)

        nn_wall = r_wall.size

        segment_lengths = np.array(nseg)

        # ii_lost = (endcond == 8)    # or (endcond == 32)   # this line removed 12/17 2:14 PM

        ii_lost   = (endcond == 8) ^ (endcond == 32)  ^ (endcond == 520) ^ (endcond == 544) # logical OR

        nn_markers = int(endcond.size)
        ones_array = np.ones(nn_markers)
        nn_lost    = int(np.sum(ones_array[ii_lost]))
        ##pdb.set_trace()
        ones_array = np.ones(nn_lost)

        rlost = r_end[ii_lost]
        zlost = z_end[ii_lost]
        weight_lost = weight[ii_lost]
        ekev_lost   = ekev[ii_lost]

        remin = r_end[ii_emin]
        zemin = z_end[ii_emin]

        rsimtime = r_end[ii_simtime]
        zsimtime = z_end[ii_simtime]

        rtherm = r_end[ii_therm]
        ztherm = z_end[ii_therm]

        rsurvived = r_end[ii_survived]
        zsurvived = z_end[ii_survived]

        ##pdb.set_trace()
        total_weight_lost        = np.sum(weight_lost)
        total_weight_energy_lost = np.sum(weight_lost*ekev_lost)
        total_weight_energy_born = np.sum(weight*ekev_ini)   # was 3515 until 4/13/2021

        print("Check on power loss (alphas): ", total_weight_energy_lost / total_weight_energy_born)

        ysize = xsize * (np.max(z_wall) - np.min(z_wall) + 0.8) / (np.max(r_wall) - np.min(r_wall) + 0.8)

        plt.close()

        # decide whether to plot particle or energy loss

        particle_or_energy = 2   # plot energy

        if(particle_or_energy ==1):
            plot_filename = stub + '_weighted_particle_loss.pdf'
            plot_title    = 'weighted particle loss at wall'
        else:
            plot_filename = stub + '_weighted_energy_loss.pdf'
            plot_title    = 'weighted energy loss at wall'
        print("zzz time for plots-9a: ", mytime.time()-time_last)
        time_last = mytime.time()
        plt.figure(figsize=(4.,7.)) 
        plt.plot(r_wall, z_wall, 'k-', linewidth=2)
        plt.title(plot_title)

        my_length = 0.2
        for itheta in range(5):
            theta = itheta/4. * 2. *np.pi
            rr1 = 1.80
            zz1 = 0.
            rr2 = rr1 + my_length*np.cos(theta)
            zz2 = zz1 + my_length*np.sin(theta)
            plt.plot([rr1,rr2], [zz1,zz2], 'c-', linewidth=0.5)

        plt.ylim(np.min(z_wall)-0.2, np.max(z_wall) + 0.6)
        plt.xlim(np.min(r_wall)-0.6, np.max(r_wall) + 0.4)

        step = 0.1    # meters

        rsave         = np.zeros(nseg)
        zsave         = np.zeros(nseg)
        distance_save = np.zeros(nseg)
        theta_save    = np.zeros(nseg)
        
        distance_sum  = np.zeros(nseg)
        distance_summer = 0.
        print("zzz time for plots-9b: ", mytime.time()-time_last)
        time_last = mytime.time()
        for i in range(nseg):

            # ii_start = i     * (nn_wall//nseg)
            # ii_end   = (i+1) * (nn_wall//nseg)

            ii_start = int(   i   * nn_wall/nseg)
            ii_end   = int( (i+1) * nn_wall/nseg)

            ii_end = np.min((ii_end, nn_wall-1))

            ii_mid = int( (ii_start + ii_end)/2)

            rr_start = r_wall[ii_start]
            rr_end   = r_wall[ii_end]
            zz_start = z_wall[ii_start]
            zz_end   = z_wall[ii_end]

            #  80 segments, total length = 6 meters.
            
            wall_segment = 12. * np.sqrt( (rr_end[0]-rr_start[0])**2 + (zz_end[0]-zz_start[0])**2 )
            
            # print("  %5d   %5d    %5d  %5d "%(i, ii_start, ii_end, ii_mid))

            #rwall_mid = r_wall[ (ii_start + ii_end) // 2  ]
            #zwall_mid = z_wall[ (ii_start + ii_end) // 2  ]

            rwall_mid = r_wall[ ii_mid ]
            zwall_mid = z_wall[ ii_mid ]

            rsave[i] = rwall_mid
            zsave[i] = zwall_mid

            if( (rsave[i] >= 1.80) and zsave[i] > 0):
                theta_save[i] = np.arctan( zsave[i]/(rsave[i]-1.80))
            elif (rsave[i] < 1.80):
                theta_save[i] = np.pi + np.arctan( zsave[i]/(rsave[i]-1.80))
            else:
                theta_save[i] = 2. * np.pi + np.arctan( zsave[i]/(rsave[i]-1.80))

 
            
            if(i > 0):
                distance_save[i] = np.sqrt( (rsave[i]-rsave[i-1])**2 + (zsave[i]-zsave[i-1])**2 )
                distance_summer += distance_save[i]
                distance_sum[i] = distance_summer
            else:   
                distance_save[i] = np.sqrt( (rsave[-1]-rsave[0])**2 + (zsave[-1]-zsave[0])**2 )
                distance_summer += distance_save[i]
                distance_sum[i]  = distance_summer

            if(r_wall[ii_end] == r_wall[ii_start]):
                slope_wall = 1.e-15
            else:
                slope_wall = ( z_wall[ii_end] - z_wall[ii_start]) / (r_wall[ii_end] - r_wall[ii_start])
                
            slope_vector = -1. /slope_wall

            delta_R = step / np.sqrt(1 + slope_vector**2)
            delta_Z = delta_R * slope_vector

            r1 = float(r_wall[ii_start] + delta_R)
            z1 = float(z_wall[ii_start] + delta_Z)

            r2 = float(r_wall[ii_start] - delta_R)
            z2 = float(z_wall[ii_start] - delta_Z)

            r3 = float(r_wall[ii_end] - delta_R)
            z3 = float(z_wall[ii_end] - delta_Z)

            r4 = float(r_wall[ii_end] + delta_R)
            z4 = float(z_wall[ii_end] + delta_Z)

            xx_array = [r1, r2, r3, r4, r1]
            yy_array = [z1, z2, z3, z4, z1]

            # plt.plot(xx_array, yy_array, 'r-')

            coords = [ (r1, z1), (r2, z2), (r3, z3), (r4, z4)]
            my_polygon = Polygon(coords)

            my_sum = 0
            for jj in range(2):   # should be nn_lost
                p1 = Point(rlost[jj], zlost[jj])
                inside_rectangle = p1.within(my_polygon)
                if (inside_rectangle):
                    my_sum = my_sum + 1
                    flost_weighted[i]        = flost_weighted[i]        + weight_lost[jj]
                    flost_energy_weighted[i] = flost_energy_weighted[i] + weight_lost[jj] * ekev_lost[jj]
            nlost_array[i] = my_sum


            flost_weighted[i]        = flost_weighted[i]        / wall_segment
            flost_energy_weighted[i] = flost_energy_weighted[i] / wall_segment
            
            # flost_weighted[i]        = flost_weighted[i]        / distance_save[i]   # too much variation in adjacent lenghts
            # flost_energy_weighted[i] = flost_energy_weighted[i] / distance_save[i]

            step_size_particles   = 1 * flost_weighted[i]        / total_weight_lost
            step_size_energy      = 1 * flost_energy_weighted[i] / total_weight_energy_lost
            ##pdb.set_trace()
            if(particle_or_energy == 1):
                step_size = step_size_particles  # one or the other
            else:
                step_size = step_size_energy     # one or the other

            #  decide direction of vector to plot

            rr1 = rwall_mid + delta_R
            zz1 = zwall_mid + delta_Z

            rr2 = rwall_mid - delta_R
            zz2 = zwall_mid - delta_Z

            dist_1 = (rr1 - 1.78)**2 + zz1**2
            dist_2 = (rr2 - 1.78)**2 + zz2**2

            if(dist_1 >= dist_2):
                my_sign = 1
            else:
                my_sign = -1

            #print(i, dist_1, dist_2, my_sign)    
            delta_RR = my_sign * step_size  / np.sqrt( 1 + slope_vector**2)
            delta_ZZ = delta_RR * slope_vector 


            aa = [rwall_mid, rwall_mid + size_factor*delta_RR]
            bb = [zwall_mid, zwall_mid + size_factor*delta_ZZ]


            plt.plot(aa,bb,'g-', linewidth=1.0)

            flost_array[i] = nlost_array[i]/nn_lost

            # print(i, ii_start, ii_end, nlost_array[i], flost_array[i])
        print("zzz time for plots-9c: ", mytime.time()-time_last)
        time_last = mytime.time()
        flost_weighted        = flost_weighted/np.sum(flost_weighted)
        flost_energy_weighted = flost_energy_weighted/np.sum(flost_energy_weighted)
        plt.plot(rl, zl, 'b', linewidth=1.0)
        plt.plot(rwall_mid, zwall_mid, 'ko', ms=4)
        plt.tight_layout(pad=padsize)
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        pdf.savefig()
        plt.close()

        print("   ... at position U at time: ", clock.ctime())
        ##pdb.set_trace()

        #print("")
        #print("  rsave  zsave  theta_save")
        #for kkjj in range(rsave.size):
        #   print(" %8.3f   %8.3f  %8.1f" % (rsave[kkjj], zsave[kkjj], theta_save[kkjj]*180./np.pi))
        # ------------------------------------------------
        #
        #   plot particle-loss fraction versus theta

        weighted_summer = 0.
        weighted_energy_summer = 0.
        print("zzz time for plots-10: ", mytime.time()-time_last)
        time_last = mytime.time()
        for ii in range(nseg):
            weighted_summer += flost_weighted[ii]
            flost_weighted_sum[ii] = weighted_summer

        flost_weighted_sum = flost_weighted_sum / np.max(flost_weighted_sum)

        theta_save = theta_save * 180./np.pi
        theta_new  = theta_new  * 180./np.pi
        theta_lost_new = theta_new[ii_lost]
        
        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(theta_save, flost_weighted, 'b-', linewidth=0.75)
        plt.plot(theta_save, flost_weighted, 'bo', ms=3)
        plt.ylim(bottom=0.)
        plt.xlim((0.,360.))
        plt.xlabel('theta (ccw; 0 = outer midplane)')
        plt.title(' weighted particle loss vs theta')
        plot_filename = stub + '_weighted_particle_loss_vs_theta_360.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        

        # --------------------------------------

        plt.close()
        plt.figure(figsize=(7.,5.))
        plt.plot(theta_save, flost_weighted, 'b-', linewidth=0.75)
        plt.plot(theta_save, flost_weighted, 'bo', ms=3)
        plt.ylim(bottom=0.)
        plt.xlim((0.,360.))
        plt.xlabel('theta (ccw; 0 = outer midplane)')
        plt.title(' weighted particle loss vs theta')
        plot_filename = stub + '_weighted_particle_loss_vs_theta_360.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(7.,5.))
        plt.plot(theta_save, flost_energy_weighted, 'b-', linewidth=0.75)
        plt.plot(theta_save, flost_energy_weighted, 'bo', ms=3)
        plt.ylim(bottom=0.)
        plt.xlim((0.,360.))
        plt.xlabel('theta (ccw; 0 = outer midplane)')
        plt.title(' weighted energy loss vs theta')
        plot_filename = stub + '_weighted_energy_loss_vs_theta_360.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        #  put graph on same axis as Gerrit's plots

        theta_kramer = theta_save.copy()
        for ijk in range(theta_kramer.size):
            if(theta_kramer[ijk] > 180.):
                theta_kramer[ijk] = theta_kramer[ijk]-360.

        ii_kramer    = np.argsort(theta_kramer)
        theta_kramer = theta_kramer[ii_kramer]
        flost_kramer = flost_energy_weighted[ii_kramer]

        plt.figure(figsize=(7.,5.))
        plt.plot(theta_kramer, flost_kramer, 'k-', linewidth=1.)
        plt.plot(theta_kramer, flost_kramer, 'ko', ms=3)
        plt.ylim(bottom=0.)
        plt.xlim((-180.,180.))
        plt.xlabel('theta (ccw; 0 = outer midplane)')
        plt.title(' weighted energy loss vs theta')
        plot_filename = stub + '_weighted_energy_loss_vs_theta_kramer.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        print("   zzz time for next set of plots: ", mytime.time() - time_last)
        time_last = mytime.time()
        # ---------------------------------------

        plt.figure(figsize=(7.,5.))
        plt.plot(theta_save, flost_weighted, 'r-', linewidth=0.75)
        plt.plot(theta_save, flost_weighted, 'ro', ms=3)
        plt.plot(theta_save, flost_energy_weighted, 'b-', linewidth=0.75)
        plt.plot(theta_save, flost_energy_weighted, 'bo', ms=3)
        plt.ylim(bottom=0.)
        plt.xlim((0.,360.))
        plt.xlabel('theta (ccw; 0 = outer midplane)')
        plt.title(' weighted particle(r) energy(b) loss vs theta')
        plot_filename = stub + '_weighted_particle_energy_loss_vs_theta_360.pdf'
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(7.,5.))
        plt.plot(theta_save, flost_weighted, 'r-', linewidth=0.75)
        plt.plot(theta_save, flost_weighted, 'ro', ms=3)
        plt.plot(theta_save, flost_energy_weighted, 'b-', linewidth=0.75)
        plt.plot(theta_save, flost_energy_weighted, 'bo', ms=3)
        plt.ylim(bottom=0.)
        plt.xlim((0.,180.))
        plt.xlabel('theta (ccw; 0 = outer midplane)')
        plt.title(' weighted particle(r) energy(b) loss vs theta')
        plot_filename = stub + '_weighted_particle_energy_loss_vs_theta_180.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        print("   ... at position V at time: ", clock.ctime())

        ##pdb.set_trace()
        # -------------------------------------------------------------
        #  repeat with log scale

        yy_max_p = np.max(flost_weighted)
        yy_max_e = np.max(flost_energy_weighted)
        yy_max   = np.max([yy_max_p, yy_max_e])

        if ( yy_max > 0.):
            
            plt.figure(figsize=(7.,5.))
            plt.yscale('log')
            plt.ylim(yy_max/1000, yy_max*1.2)

            plt.plot(theta_save, flost_weighted, 'r-', linewidth=0.75)
            plt.plot(theta_save, flost_weighted, 'ro', ms=3)
            plt.plot(theta_save, flost_energy_weighted, 'b-', linewidth=0.75)
            plt.plot(theta_save, flost_energy_weighted, 'bo', ms=3)
            plt.xlim((0.,360.))
            plt.xlabel('theta (ccw; 0 = outer midplane)')
            plt.title(' weighted particle(r) energy(b) loss vs theta')
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()
            
            plt.figure(figsize=(7.,5.))
            plt.yscale('log')
            plt.ylim(yy_max/1000, yy_max*1.2)
            plt.plot(theta_save, flost_weighted, 'r-', linewidth=0.75)
            plt.plot(theta_save, flost_weighted, 'ro', ms=3)
            plt.plot(theta_save, flost_energy_weighted, 'b-', linewidth=0.75)
            plt.plot(theta_save, flost_energy_weighted, 'bo', ms=3)
            plt.xlim((0.,180.))
            plt.xlabel('theta (ccw; 0 = outer midplane)')
            plt.title(' weighted particle(r) energy(b) loss vs theta')
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()
            
            # ---------------------------------------

            plt.figure(figsize=(7.,5.))
            plt.plot(theta_save, flost_weighted, 'b-', linewidth=0.75)
            plt.plot(theta_save, flost_weighted, 'bo', ms=3)
            plt.ylim(bottom=0.)
            plt.xlim((0.,180.))
            plt.grid(True)
            plt.xlabel('theta (ccw; 0 = outer midplane)')
            plt.title(' weighted particle loss vs theta')
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(7.,5.))
            plt.plot(theta_save, flost_energy_weighted, 'b-', linewidth=0.75)
            plt.plot(theta_save, flost_energy_weighted, 'bo', ms=3)
            plt.ylim(bottom=0.)
            plt.xlim((0.,180.))
            plt.grid(True)
            plt.xlabel('theta (ccw; 0 = outer midplane)')
            plt.title(' weighted particle loss vs theta')
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()

            # --------------------------------------

            plt.figure(figsize=(7.,5.))
            plt.plot(theta_save, flost_weighted_sum, 'b-', linewidth=0.75)
            plt.plot(theta_save, flost_weighted_sum, 'bo', ms=3)
            plt.ylim((0.,1.1))  
            plt.xlim((0.,360.))
            plt.grid(True)
            plt.xlabel('theta (ccw; 0 = outer midplane)')
            plt.title(' weighted particle loss vs theta')
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()

            # --------------------------------------

            plt.figure(figsize=(7.,5.))
            plt.plot(distance_sum, flost_weighted_sum, 'b-', linewidth=0.75)
            plt.plot(distance_sum, flost_weighted_sum, 'bo', ms=3)
            plt.ylim()  
            plt.xlim(left=0.)
            plt.grid(True)
            plt.xlabel('distance along wall (0 = out midplane)')
            plt.title(' summed weighted particle loss vs distance')
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(7.,5.))
            plt.plot(theta_save, flost_array, 'b-', linewidth=0.75)
            plt.plot(theta_save, flost_array, 'bo', ms=3)
            plt.ylim(bottom=0.)
            plt.xlim((0.,360.))
            plt.xlabel('theta (ccw; 0 = outer midplane)')
            plt.title(' unweighted particle loss vs theta')
        
            plt.tight_layout(pad=padsize)
            pdf.savefig()
            plt.close()

        #plt.show()
        ##pdb.set_trace()
        # -------------------------------------
        #  plot probabililty of particle loss


        # get last closed flux surface

        # fn_geqdsk = '$dir_runs/v1c.geq'    # need to make this an input parameter

        #print("")
        #print("   i    r     z     dist  flost")
        #print("")

        #for i in range(nseg):
        #    print(" %d %7.3f %7.3f %7.3f %9.4f" % (i, rsave[i], zsave[i], distance_save[i], flost_weighted[i]))
        #print("")

        #return

        gg = geqdsk(fn_geqdsk)
        gg.readGfile()

        rl =  gg.equilibria[eq_index].rlcfs
        zl =  gg.equilibria[eq_index].zlcfs

        rmax = np.max(rl)
        rmin = np.min(rl)
        zmax = np.max(zl)
        zmin = np.min(zl)

        nr = 10
        nz = 20

        delta_R = (rmax - rmin)/nr
        delta_Z = (zmax - zmin)/nz


        ploss_frac = -1. * np.ones((nr,nz))

        rmarker_all = r_marker[ii_no_abort]
        zmarker_all = z_marker[ii_no_abort]

        ii_wall = (endcond == 8)

        rmarker_wall = r_marker[ii_wall]
        zmarker_wall = z_marker[ii_wall]

        ii_rhomax = (endcond == 32)

        rmarker_rhomax = r_marker[ii_rhomax]
        zmarker_rhomax = z_marker[ii_rhomax]

        ones_array = np.ones(r_marker.size)

        minimum_markers = 10

        xsize = 3.5
        ysize = 7.

        # -------------------------------------------
        #  plot R,Z of initial positions of cputime-end markers

        print("   ... at position W at time: ", clock.ctime())
        
        plt.close
        plt.figure(figsize=(xsize, ysize))
        fig,ax = plt.subplots()
        plt.xlim(1.2,2.5)
        plt.ylim(-1.2,1.2)

        plt.plot(gg.equilibria[eq_index].rlcfs,
                 gg.equilibria[eq_index].zlcfs,
                 color='black',linewidth=1, rasterized=do_rasterized)

        plt.plot(r_wall, z_wall, 'c-', linewidth=2)
        plt.plot(r_ini[ii_cputime], z_ini[ii_cputime], 'ro', ms=2)
        plt.title('z_ini vs r_ini for cpu-end markers')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        print("   zzz time for next set of plots: ", mytime.time() - time_last)
        time_last = mytime.time()
        # -------------------------------------------
        #  plot R,Z of initial positions

        plt.close
        plt.figure()
        fig,ax = plt.subplots()
        plt.xlim(1.2,2.5)
        plt.ylim(-1.2,1.2)

        plt.plot(gg.equilibria[eq_index].rlcfs,
                 gg.equilibria[eq_index].zlcfs,
                 color='black',linewidth=1, rasterized=do_rasterized)

        plt.plot(r_wall, z_wall, 'c-', linewidth=2)
        plt.plot(r_ini, z_ini, 'ro', ms=1, rasterized=do_rasterized)
        plt.title('z_ini vs r_ini')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # plt.figure(figsize=(xsize,ysize))
        plt.figure()
        fig,ax=plt.subplots()

        plt.plot(rl, zl, 'k-', linewidth=1)
        plt.xlim(1.2,2.5)
        plt.ylim(-1.2,1.2)
        #ax.axis('equal')

        # -----------------------------------------------------------------------------
        
        #  assume we have no aborted markers

        # for some reason, r_marker is vertical but the endcond array is horizontal
        # and this leads to the number of points beins squared.  good grief

        r_marker_t = np.transpose(r_marker)
        z_marker_t = np.transpose(z_marker)

        for ir in range(nr):
            for iz in range(nz):

                Rleft   = rmin    + ir* delta_R
                Rright  = Rleft   + delta_R
                zbottom = zmin    + iz * delta_Z
                ztop    = zbottom + delta_Z

                jj_born   =   (r_marker_t  >= Rleft)  & (r_marker_t  <= Rright)  \
                            & (z_marker_t >= zbottom) & (z_marker_t  <= ztop)    \
                            & (endcond != 0)

                jj_wall   =   (r_marker_t  >= Rleft)  & (r_marker_t  <= Rright)  \
                            & (z_marker_t >= zbottom) & (z_marker_t  <= ztop)    \
                            & (endcond == 8)

                jj_rhomax =   (r_marker_t  >= Rleft)  & (r_marker_t  <= Rright)  \
                            & (z_marker_t >= zbottom) & (z_marker_t  <= ztop)    \
                            & (endcond == 32)

                ##pdb.set_trace()

                markers_born   = np.sum(jj_born)
                markers_wall   = np.sum(jj_wall)
                markers_rhomax = np.sum(jj_rhomax)

                if(markers_born >= minimum_markers):
                    ploss_frac[ir,iz] = 100.*(markers_wall + markers_rhomax) / markers_born

                if(ploss_frac[ir,iz] >= 0.):

                    if(ploss_frac[ir,iz] <= 0.1):
                        this_color = 'forestgreen'
                    elif (ploss_frac[ir,iz] <= 1.):
                        this_color = 'palegreen'
                    elif (ploss_frac[ir,iz] <= 10.):
                        this_color = 'pink'
                    elif (ploss_frac[ir,iz] <= 30.):
                        this_color = 'red'
                    else:
                        this_color='violet'

                    rect = Rectangle( (Rleft,zbottom),delta_R, delta_Z, color=this_color, edgecolor=None, linewidth=0.5)
                    
                    ax.add_patch(rect)

                    xpos = Rleft + delta_R/2.
                    ypos = zbottom + delta_Z/2.

                    my_string = '{:4.1f}'.format(ploss_frac[ir,iz])
                    plt.text(xpos, ypos, my_string, fontsize=6, ha='center', va='center')

        plt.plot(r_wall, z_wall, 'b-', linewidth=1)

        # ------------------------------------------
        #  add contours of constant rho

        plt.plot(gg.equilibria[eq_index].rlcfs,
                 gg.equilibria[eq_index].zlcfs,
                 color='black',linewidth=1)

        CS = plt.contour(gg.equilibria[eq_index].rGrid,
                         gg.equilibria[eq_index].zGrid,
                         gg.equilibria[eq_index].psiPolSqrt,
                         linewidths = 1.0, colors='black', levels = [0.7,0.8,0.9])
                         # cmap='jet'
        plt.clabel(CS, inline=1, fontsize=10)
        plt.title(' Particle loss:  dg<0.1   lg<1   p<10   r<30  v>30', fontsize=fontsize_2)
        filename = stub + '_loss_regions.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # ---------------------------------------------------------
        #    R,z of those markers that reach emin or thermalize   

        plt.close()
        xsize = 4.0
        ysize = xsize * (ymax-ymin)/(xmax-xmin)
        plt.figure(figsize=(xsize, ysize))
        plt.plot(remin, zemin, 'go', ms=1, rasterized=do_rasterized, fillstyle='none')
        plt.plot(rl, zl, 'k-', rasterized=do_rasterized)
        plt.plot(r_wall,z_wall, 'b-', linewidth=2)
        plt.title('End [R,Z] of markers:  Emin')
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plot_filename = stub + '_RZ_emin.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        plt.close()
        xsize = 5.5
        ysize = xsize * (ymax-ymin)/(xmax-xmin)
        plt.figure(figsize=(xsize, ysize))
        plt.plot(rtherm, ztherm, 'go', ms=1, rasterized=do_rasterized, fillstyle='none')
        plt.plot(rl, zl, 'k-', rasterized=do_rasterized)
        plt.plot(r_wall,z_wall, 'b-', linewidth=2)
        plt.title('End [R,Z] of markers:  Therm')
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plot_filename = stub + '_RZ_therm.pdf'
        plt.tight_layout(pad=2)
        pdf.savefig()

        print("   ... at position X at time: ", clock.ctime())
        
        plt.close()
        xsize = 5.5
        ysize = xsize * (ymax-ymin)/(xmax-xmin)
        plt.figure(figsize=(xsize, ysize))
        plt.plot(rsimtime, zsimtime, 'go', ms=1, rasterized=do_rasterized, fillstyle='none')
        plt.plot(rl, zl, 'k-', rasterized=do_rasterized)
        plt.plot(r_wall,z_wall, 'b-', linewidth=2)
        plt.title('End [R,Z] of markers:  simtime')
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(pad=2)
        pdf.savefig()

        plt.close()
        xsize = 4.0
        ysize = xsize * (ymax-ymin)/(xmax-xmin)
        plt.figure(figsize=(xsize, ysize))
        plt.plot(rsurvived, zsurvived, 'go', ms=1, rasterized=do_rasterized_2, fillstyle='none')
        plt.plot(rl, zl, 'k-', rasterized=do_rasterized)
        plt.plot(r_wall,z_wall, 'b-', linewidth=2)
        plt.title('End [R,Z] of markers:  Emin/Therm/simtime')
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.tight_layout(pad=2)
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(7., 5.))
        plt.plot(rho_ini[ii_simtime], delta_rho[ii_simtime], 'go', ms=1, rasterized=do_rasterized, fillstyle='none')
        plt.title('rho_end-rho_ini vs rho_ini (simtime)')
        plt.xlabel('rho_ini')
        plot_filename = stub + '_delta_rho_simtime.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(7., 5.))
        plt.plot(rho_ini[ii_emin], delta_rho[ii_emin], 'go', ms=1, rasterized=do_rasterized_2, fillstyle='none')
        plt.title('rho_end-rho_ini vs rho_ini (emin)')
        plt.xlabel('rho_ini')
        plot_filename = stub + '_delta_rho_emin.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(7., 5.))
        plt.plot(rho_ini[ii_therm], delta_rho[ii_therm], 'go', ms=1, rasterized=do_rasterized_2, fillstyle='none')
        plt.title('rho_end-rho_ini vs rho_ini (therm)')
        plt.xlabel('rho_ini')
        plot_filename = stub + '_delta_rho_therm.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        plt.close()
        plt.figure(figsize=(7., 5.))
        plt.plot(rho_ini[ii_survived], delta_rho[ii_survived], 'go', ms=1, rasterized=do_rasterized_2, fillstyle='none')
        plt.title('rho_end-rho_ini vs rho_ini (survived)')
        plt.xlabel('rho_ini')
        plot_filename = stub + '_delta_rho_survived.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()

       
        # ------------------------------------------------------------------
        #  plot end positions of rhomax

        print("   zzz time for next set of plots: ", mytime.time() - time_last)
        time_last = mytime.time()
        
        plt.close()
        xsize = 4.
        ysize = xsize * (ymax-ymin)/(xmax-xmin)
        plt.figure(figsize=(xsize,ysize))
        plt.tight_layout(pad=2, h_pad=2)
        plt.grid('True')


        plt.plot(r_end[ii_rhomax],   z_end[ii_rhomax],   'ro', ms=1, rasterized=do_rasterized)
        ##pdb.set_trace()
        plt.plot(r_wall, z_wall, 'k-', linewidth=1, rasterized=do_rasterized)
        plt.plot(rl, zl, 'g-', linewidth=1, rasterized=do_rasterized)
        
        plt.title('Ending positions of rhomax markers', fontsize=12)

        plt.xlabel('Rmajor [m]')
        plt.ylabel('z [m]')

        plot_filename = stub + '_endpos_rhomax_RZ.pdf'
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        # ------------------------------------------------

        jj_outside = (rlost>1.6418)             # was 1.6 until 11/13/2020
        
        plt.close()
        # ii_limiter = (phi_loss_limiters >= jk*delta_phi) & (phi_loss_limiters <= (jk+1)*delta_phi)
        plt.figure(figsize=(9.,7.))
        plt.grid('True')
        plt.plot(phi_lost[jj_outside],   zlost[jj_outside],   'ro', ms=0.25, rasterized=do_rasterized)
        plt.title('zlost vs philost for Rend>1.6418', fontsize=12)
        plt.xlabel('phi [deg]')
        plt.ylabel('z [m]')
        plt.tight_layout(pad=padsize)
        pdf.savefig()

                
        
        plt.close()
        my_xx = np.mod(phi_lost[jj_outside],20)
        plt.figure(figsize=(9.,7.))
        plt.grid('True')
        plt.plot(my_xx,   zlost[jj_outside],   'ro', ms=0.25, rasterized=do_rasterized)
        plt.title('zlost vs philost mod(20) for Rend>1.6418', fontsize=12)
        plt.xlabel('phi [deg]')
        plt.ylabel('z [m]')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
    
        # ----------------------------------------------------
        #  repeat but only for delayed losses

        plt.close()
        time_lost_outside = time_lost[jj_outside]
        phi_lost_outside  = phi_lost[jj_outside]
        zlost_outside     = zlost[jj_outside]

        ii_delayed = (time_lost_outside >= 0.01)
        xx = phi_lost_outside[ii_delayed]
        yy = zlost_outside[ii_delayed]
        plt.figure(figsize=(9.,7.))
        plt.grid('True')
        plt.plot(xx,yy,   'ro', ms=0.25, rasterized=do_rasterized)
        plt.title('zlost vs philost for Rend>1.6418 (tloss > 0.01)', fontsize=12)
        plt.xlabel('phi [deg]')
        plt.ylabel('z [m]')
        plt.tight_layout(pad=padsize)
        pdf.savefig()

        print("   ... at position Y at time: ", clock.ctime())   
        
        plt.close()
        my_xx = np.mod(xx,20)
        plt.figure(figsize=(9.,7.))
        plt.grid('True')
        plt.plot(my_xx,   yy,   'ro', ms=0.25, rasterized=do_rasterized)
        plt.title('zlost vs philost mod(20) for Rend>1.6418 (tloss > 0.01)', fontsize=12)
        plt.xlabel('phi [deg]')
        plt.ylabel('z [m]')
        plt.tight_layout(pad=padsize)
        pdf.savefig()
    
        # ----------------------------------------------------

        zlost_out   = zlost[jj_outside]
        weights_out = weight_marker_lost[jj_outside]

        # #pdb.set_trace()
        my_nbins = 50  
        plt.close()
        plt.figure(figsize=(9.,7.))
        nvalues, bins, patches = plt.hist(zlost_out, bins=my_nbins, rwidth=1.,histtype='step',color='k')
        plt.grid(axis='both', alpha=0.75)
        plt.title('unweighted zlost_out for R > 1.6')
        plt.xlabel('z [m]')
        plt.ylabel('')   
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()
        # #pdb.set_trace()
        my_nbins = 50  
        plt.close()
        plt.figure(figsize=(9.,7.))
        nvalues, bins, patches = plt.hist(zlost_out, bins=my_nbins, rwidth=1.,histtype='step',color='k',density=True, weights=weights_out)
        plt.grid(axis='both', alpha=0.75)
        plt.title('weighted zlost_out for R > 1.6')
        plt.xlabel('z [m]')
        plt.ylabel('')   
        plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()

        # ----------------------------------------------------------------------
        #  compute toroidal peaking factor for power to limiters
        #  this works only for 

        ntf = 18
        delta_phi = 360./ntf
        
        weight_energy_lost = weight_energy_lost/np.sum(weight_energy_lost)

        z_extent_limiters = 0.5
        rinner            = 1.6418   # was 1.7 until 11/13/2020

        ii_hit_limiter = (rlost >= rinner) & (np.abs(zlost) <= z_extent_limiters)
        ii_hit_wall    = (zlost >= z_extent_limiters)

        
        eloss_limiters = np.sum(weight_energy_lost[ii_hit_limiter])
        eloss_wall     = np.sum(weight_energy_lost[ii_hit_wall])
        eloss_other    = 1. - eloss_limiters - eloss_wall
        
        phi_loss_limiters = phi_lost[ii_hit_limiter]
        phi_loss_wall     = phi_lost[ii_hit_wall]

        eloss_limiter_array = weight_energy_lost[ii_hit_limiter]
        eloss_wall_array    = weight_energy_lost[ii_hit_wall]
        
        print("")
        print("   fraction of lost power to limiters:    ", eloss_limiters)
        print("   fraction of lost power to upper wall:  ", eloss_wall)
        print("   fraction of lost power elsewhere:      ", eloss_other)

        eloss_tf_limiter_sector = np.zeros(ntf)
        eloss_tf_wall_sector    = np.zeros(ntf)

        for jk in range(ntf):
            
            ii_limiter = (phi_loss_limiters >= jk*delta_phi) & (phi_loss_limiters <= (jk+1)*delta_phi)
            ii_wall    = (phi_loss_wall >= jk*delta_phi) & (phi_loss_wall <= (jk+1)*delta_phi)
            
            eloss_tf_limiter_sector[jk] = np.sum(eloss_limiter_array[ii_limiter])
            eloss_tf_wall_sector[jk]    = np.sum(eloss_wall_array[ii_wall])

        peaking_factor_limiter = np.max(eloss_tf_limiter_sector)/ np.mean(eloss_tf_limiter_sector)
        peaking_factor_wall    = np.max(eloss_tf_wall_sector)   / np.mean(eloss_tf_wall_sector)

        print("")
        print("   peaking factor for energy loss to limiter:     ", peaking_factor_limiter)
        print("   peaking factor for energy loss to upper wall:  ", peaking_factor_wall)
        print("")
        
        xarray = np.linspace(1, ntf, ntf)
        
        my_title = 'eloss by sector (r/g = lim/wall) peaking factors = {:6.3f}, {:6.3f} )'.format(peaking_factor_limiter, peaking_factor_wall)
        
        plt.close()
        
        plt.figure(figsize=(7.,5.))
        plt.grid('True')
        #plt.ylim(bottom=0.)
        plt.plot(xarray, eloss_tf_limiter_sector,   'r-',  rasterized=do_rasterized)
        plt.plot(xarray, eloss_tf_wall_sector,      'g-',  rasterized=do_rasterized)
        plt.title(my_title, fontsize=12)
        plt.xlabel('TF sector')
        plt.tight_layout(pad=padsize)
        pdf.savefig()







      # plt.figure(figsize=(xsize,ysize))
        plt.figure()
        fig,ax=plt.subplots()

        plt.plot(rl, zl, 'k-', linewidth=1)
        plt.xlim(1.2,2.5)
        plt.ylim(-1.2,1.2)
        #ax.axis('equal')

        # -----------------------------------------------------------------------------
        #   grid-plot of number of lost markers in [phi, z] space
        #  9/24/2020
        #  assume we have no aborted markers

        # for some reason, r_marker is vertical but the endcond array is horizontal
        # and this leads to the number of points beins squared.  good grief

        my_max = 177.
        
        plt.figure(figsize=(9.,7.))
        fig,ax=plt.subplots()
        plt.xlim(0.,20.)
        plt.ylim(-1., 1.2)

        my_phi = np.mod(phi_lost[jj_outside],20)
        my_zz  = zlost[jj_outside]

        phi_min = 0.
        phi_max = 20.
        zz_min  = -1.
        zz_max  = 1.2

        nn_phi  = 20
        nn_zz   = 44

        delta_phi = (phi_max - phi_min) / nn_phi
        delta_Z   = (zz_max  - zz_min ) / nn_zz

        
        hits      = np.zeros((nn_phi, nn_zz))
        hits      = hits.astype(int)
        
        max_hits  = 0
        # +++++++++++++++++++++++++++++++++++++++++++++

        for ir in range(nn_phi):
            
            phi_left   = phi_min   + ir * delta_phi
            phi_right  = phi_left  +      delta_phi
            
            for iz in range(nn_zz):

                zbottom    = zz_min    + iz * delta_Z
                ztop       = zbottom   +      delta_Z

                jj_inside  =  (my_phi  >= phi_left) & (my_phi  <= phi_right)  \
                            & (my_zz   >= zbottom)  & (my_zz   <= ztop)    

                markers_inside   = int(np.sum(jj_inside))
                hits[ir,iz]      = markers_inside
                if (markers_inside > max_hits):
                    max_hits = markers_inside
                
        # +++++++++++++++++++++++++++++++++++++++++++++
        
        for ir in range(nn_phi):
            
            phi_left   = phi_min   + ir * delta_phi
            phi_right  = phi_left  +      delta_phi
            
            for iz in range(nn_zz):
                
                zbottom    = zz_min    + iz * delta_Z
                ztop       = zbottom   +      delta_Z

                norm_hits = int(100.* hits[ir,iz] / max_hits)
                    
                if(hits[ir,iz] > 0):

                    if(norm_hits >= 85):
                        this_color = 'gray'
                    elif (norm_hits >= 50):
                        this_color = 'lime'
                    elif (norm_hits >=  15.):
                        this_color = 'palegreen'
                    elif (norm_hits >= 5.):
                        this_color = 'pink'
                    else:
                        this_color='cornsilk'

                    rect = Rectangle( (phi_left,zbottom),delta_phi, delta_Z, facecolor=this_color, edgecolor='none', linewidth=0.5)
                    
                    ax.add_patch(rect)

                    xpos = phi_left + delta_phi/2.
                    ypos = zbottom  + delta_Z/2.
                    #pdb.set_trace()
                    my_string = '{:3d}'.format(norm_hits)
                    plt.text(xpos, ypos, my_string, fontsize=7, ha='center', va='center')

        plt.title(' wall hits (R>1.799) :  corn: <5 pink:5-15   pgreen:15-50   lime:50-85   gray>85', fontsize=10)
        plt.xlabel('phi [degrees]')
        plt.ylabel('Z [m]')
        #plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()
        print("max_hits= ", max_hits)
        
        #pdb.set_trace()


        
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #  compute statistics on spatial dependence of losses

        rmax = 2.441154624 # from sparc_simple_wall.txt
        ntf  = 18

        #  toroidal angles spanned by TF sector, RF antenna, and TF sector limiter (radians)
        
        width_rf_sector_limiter = 0.305
        width_rf_antenna        = (2. * np.pi * rmax/ntf) - width_rf_sector_limiter
        phi_sector_size         = 2. * np.pi / ntf
        phi_rf_size             = width_rf_antenna        / rmax
        phi_antenna_size        = phi_rf_size
        phi_limiter_size        = width_rf_sector_limiter / rmax

        # duh, convert to degrees
        
        phi_sector_size  = phi_sector_size  * 180./np.pi
        phi_rf_size      = phi_rf_size      * 180./np.pi
        phi_antenna_size = phi_antenna_size * 180./np.pi
        phi_limiter_size = phi_limiter_size * 180./np.pi

        print("   zzz time for next set of plots: ", mytime.time() - time_last)
        time_last = mytime.time()
        

        print("\n width_rf_sector_limiter: ", width_rf_sector_limiter)
        print(" width_rf_antenna:          ", width_rf_antenna)
        print("\n phi_sector_size:         ", phi_sector_size)
        print(" phi_rf_size:               ", phi_rf_size)
        print(" phi_limiter_size:          ", phi_limiter_size)
        print("")

        rinner_wall     = 1.6418  # was 1.7 until 11/13/2020
        tall_rf_antenna = 0.50   
        tall_limiter    = 0.50
        taper_limiter   = 0.10
        tall_tbl        = 0.10
        taper_tbl       = 0.02

        phi_rf_start      = phi_limiter_size/2.
        phi_limiter_start = -1. * phi_limiter_size / 2.

        epsilon_limiter_phi = phi_limiter_size  / 40.
        epsilon_limiter_z   = tall_limiter      / 40.
        epsilon_antenna_phi = phi_rf_size       / 40.
        epsilon_antenna_z   = tall_rf_antenna   / 40.
        
        phis_limiter  = np.zeros((2,ntf))
        phis_antenna = np.zeros((2,ntf))

        zbottom_limiter = -1. * tall_limiter + epsilon_limiter_z
        ztop_limiter    =  1. * tall_limiter - epsilon_limiter_z

        zbottom_antenna = -1. * tall_rf_antenna + epsilon_antenna_z
        ztop_antenna    =  1. * tall_rf_antenna - epsilon_antenna_z

        zbottom_tbl     = tall_limiter
        ztop_tbl        = tall_limiter + tall_tbl

        zbottom_valley  = zbottom_tbl
        ztop_valley     = zbottom_tbl + taper_tbl

        zbottom_outer_wall = ztop_tbl
        ztop_outer_wall    = rinner_wall
        
        for jtf in range(ntf):
            
            phis_limiter[0,jtf] = phi_limiter_start + jtf * phi_sector_size 
            phis_limiter[1,jtf] = phis_limiter[0,jtf] + phi_limiter_size    
            
            phis_antenna[0,jtf]  =  phi_rf_start + jtf * phi_sector_size 
            phis_antenna[1,jtf]  =  phis_antenna[0,jtf]+ phi_rf_size

            phis_limiter[0,jtf] = phis_limiter[0,jtf] + epsilon_limiter_phi
            phis_limiter[1,jtf] = phis_limiter[1,jtf] - epsilon_limiter_phi

            phis_antenna[0,jtf] = phis_antenna[0,jtf] + epsilon_antenna_phi
            phis_antenna[1,jtf] = phis_antenna[1,jtf] - epsilon_antenna_phi
        
        nnphi_limiter  =  10 
        nnz_limiter    =  10 
        nnphi_rf       =   5
        nnphi_antenna  = nnphi_rf
        nnz_rf         =   6
        nnz_antenna    =  nnz_rf
        nnphi_tbl      =  72 
        nnz_tbl        =   8 
        nnphi_wall_1   =  72 
        nnz_wall_1     =   6 
        nnphi_wall_2   =  72 
        nnz_wall_2     =  20 

        energy_limiters = np.zeros(ntf)
        energy_antennas = np.zeros(ntf)

        energy_limiter_hstripe = np.zeros(nnz_limiter)
        energy_limiter_vstripe = np.zeros(nnphi_limiter)

        energy_antenna_hstripe = np.zeros(nnz_rf)
        energy_antenna_vstripe = np.zeros(nnphi_rf)

        energy_tbl_hstripe       = np.zeros(nnz_tbl)
        energy_tbl_above_limiter = np.zeros(ntf)
        energy_tbl_above_antenna = np.zeros(ntf)
        energy_tbl_below_limiter = np.zeros(ntf)
        energy_tbl_below_antenna = np.zeros(ntf)

        energy_upper_outer       = np.zeros(ntf)
        energy_upper_inner       = np.zeros(ntf)
        energy_lower_outer       = np.zeros(ntf)
        energy_lower_inner       = np.zeros(ntf)

        energy_lost_total = np.sum(weight_energy_lost)
        
        ii_inner_wall            = (rend_lost < rinner_wall)
        energy_inner_wall        = np.sum(weight_energy_lost[ii_inner_wall])

        ii_inner_wall_top        = (rend_lost < rinner_wall) & (zend_lost>0.)
        energy_inner_wall_top    = np.sum(weight_energy_lost[ii_inner_wall])

        ii_inner_wall_bottom     = (rend_lost < rinner_wall) & (zend_lost<=0.)
        energy_inner_wall_bottom = np.sum(weight_energy_lost[ii_inner_wall_bottom])   

        ii_outer_wall_top        = (rend_lost > rinner_wall) & (zend_lost > zbottom_outer_wall)
        energy_outer_wall_top    = np.sum(weight_energy_lost[ii_outer_wall_top])

        ii_outer_wall_bottom     = (rend_lost > rinner_wall) & (zend_lost < -1.*zbottom_outer_wall)
        energy_outer_wall_bottom = np.sum(weight_energy_lost[ii_outer_wall_bottom])

        ii_tbl                   = (rend_lost > rinner_wall) * (zend_lost > zbottom_tbl) & (zend_lost < ztop_tbl)
        energy_tbl               = np.sum(weight_energy_lost[ii_tbl])
        jj_tbl                   = (rend_lost > rinner_wall) * (zend_lost > -1.*ztop_tbl) & (zend_lost < -1.*zbottom_tbl)
        energy_tbl_lower         = np.sum(weight_energy_lost[jj_tbl])

        for jtf in range(ntf):

            ii_upper_outer = (rend_lost > rinner_wall)  & (zend_lost > ztop_tbl) & \
                             (phi_lost > jtf*(360./ntf)) & (phi_lost < (jtf+1)*360./ntf)
            
            ii_lower_outer = (rend_lost > rinner_wall)  & (zend_lost < -1.*ztop_tbl) & \
                             (phi_lost > jtf*(360./ntf)) & (phi_lost < (jtf+1)*360./ntf)
            

            ii_upper_inner = (rend_lost <= rinner_wall) & (zend_lost > 0.) & \
                             (phi_lost > jtf*(360./ntf)) & (phi_lost < (jtf+1)*360./ntf)

            ii_lower_inner = (rend_lost <= rinner_wall) & (zend_lost <= 0.) & \
                             (phi_lost > jtf*(360./ntf)) & (phi_lost < (jtf+1)*360./ntf)
            
            ii_limiter = (zend_lost > zbottom_limiter) & (zend_lost < ztop_limiter) & \
                         (phi_lost > phis_limiter[0,jtf]) & (phi_lost < phis_limiter[1,jtf])

            #if (jtf ==0):    # 10/25/2020
            #    ii_limiter = (zend_lost > zbottom_limiter) & (zend_lost < ztop_limiter) & \
            #             (phi_lost > 360. + phis_limiter[0,jtf]) or (phi_lost < phis_limiter[1,jtf])
                
            ii_antenna = (zend_lost > zbottom_antenna) & (zend_lost < ztop_antenna) & \
                         (phi_lost > phis_antenna[0,jtf]) & (phi_lost < phis_antenna[1,jtf])
            
            energy_limiters[jtf] = np.sum(weight_energy_lost[ii_limiter])
            energy_antennas[jtf] = np.sum(weight_energy_lost[ii_antenna])

            ii_tbl_above_limiter = (zend_lost > -1.*ztop_tbl) & (zend_lost < -1.*zbottom_tbl) \
                                   & (phi_lost > phis_limiter[0,jtf]) & (phi_lost < phis_limiter[1,jtf])
            
            ii_tbl_above_antenna =(zend_lost > -1.*ztop_tbl) & (zend_lost < -1.*zbottom_tbl) \
                                   & (phi_lost > phis_antenna[0,jtf]) & (phi_lost < phis_antenna[1,jtf])

            ii_tbl_below_limiter = (zend_lost > zbottom_tbl) & (zend_lost < ztop_tbl) \
                                   & (phi_lost > phis_limiter[0,jtf]) & (phi_lost < phis_limiter[1,jtf])
            
            ii_tbl_below_antenna =(zend_lost > zbottom_tbl) & (zend_lost < ztop_tbl) \
                                   & (phi_lost > phis_antenna[0,jtf]) & (phi_lost < phis_antenna[1,jtf])

            energy_tbl_above_limiter[jtf] = np.sum(weight_energy_lost[ii_tbl_above_limiter])
            energy_tbl_above_antenna[jtf] = np.sum(weight_energy_lost[ii_tbl_above_antenna])
            energy_tbl_below_limiter[jtf] = np.sum(weight_energy_lost[ii_tbl_below_limiter])
            energy_tbl_below_antenna[jtf] = np.sum(weight_energy_lost[ii_tbl_below_antenna])

            energy_upper_outer[jtf]       = np.sum(weight_energy_lost[ii_upper_outer])
            energy_upper_inner[jtf]       = np.sum(weight_energy_lost[ii_upper_inner])
            energy_lower_outer[jtf]       = np.sum(weight_energy_lost[ii_lower_outer])
            energy_lower_inner[jtf]       = np.sum(weight_energy_lost[ii_lower_inner])
            
        for jj in range(nnphi_limiter):
                
            phimin = phis_limiter[0,jj] + jj*phi_limiter_size/nnphi_limiter
            phimax = phimin + jj*phi_limiter_size/nnphi_limiter

            ii_stripe_limiter_vertical = (zend_lost > zbottom_limiter) & (zend_lost < ztop_limiter ) \
                                           &  (phi_lost > phimin)           & (phi_lost < phimax)
            energy_limiter_vstripe[jj] = np.sum(weight_energy_lost[ii_stripe_limiter_vertical])

        for jj in range(nnz_limiter):

            height_limiter = ztop_limiter - zbottom_limiter
            zzmin = zbottom_limiter + jj * height_limiter/nnz_limiter
            zzmax = zzmin +  height_limiter/nnz_limiter

            ii_stripe_limiter_horizontal = (zend_lost > zzmin) & (zend_lost < zzmax ) \
                                            &  (phi_lost > phis_limiter[0,jj]) & (phi_lost < phis_limiter[1,jj])
            energy_limiter_hstripe[jj] = np.sum(weight_energy_lost[ii_stripe_limiter_horizontal])

        for jj in range(nnphi_rf):
                
            phimin = phis_antenna[0,jj] + jj*phi_antenna_size/nnphi_rf
            phimax = phimin + jj*phi_antenna_size/nnphi_rf

            ii_stripe_antenna_vertical = (zend_lost > zbottom_antenna) & (zend_lost < ztop_antenna ) \
                                             & (phi_lost > phimin)           & (phi_lost < phimax)
            energy_antenna_vstripe[jj] = np.sum(weight_energy_lost[ii_stripe_antenna_vertical])

        for jj in range(nnz_antenna):

            height_antenna = ztop_antenna - zbottom_antenna
            zzmin = zbottom_antenna + jj * height_antenna/nnz_antenna
            zzmax = zzmin +  height_antenna/nnz_antenna

            ii_stripe_antenna_horizontal = (zend_lost > zzmin) & (zend_lost < zzmax ) \
                                             & (phi_lost > phis_antenna[0,jj]) & (phi_lost < phis_antenna[1,jj])
            energy_antenna_hstripe[jj] = np.sum(weight_energy_lost[ii_stripe_antenna_horizontal])

        for jj in range(nnz_tbl):
            height_tbl = ztop_tbl - zbottom_tbl
            zzmin      = zbottom_tbl + jj * height_tbl/nnz_tbl
            zzmax      = zzmin + height_tbl/nnz_tbl
            ii_tbl_hstripe = (rend_lost > rinner_wall) & (zend_lost > zzmin) & (zend_lost < zzmax)
            energy_tbl_hstripe[jj]      = np.sum(weight_energy_lost[ii_tbl_hstripe])

        # ++++++++++++++++++++++++++++++++++++++++++++++++
        #   toroidal peaking factors

            tpf_tbl_limiter       = 0.
            tpf_tbl_antenna       = 0.
            tpf_tbl_limiter_lower = 0.
            tpf_tbl_antenna_lower = 0.
            tpf_upper_outer       = 0.
            tpf_upper_inner       = 0.
            tpf_lower_outer       = 0.
            tpf_lower_inner       = 0.

            if(np.mean(energy_tbl_above_limiter) > 0.):
                tpf_tbl_limiter       = np.max(energy_tbl_above_limiter) / np.mean(energy_tbl_above_limiter)

            if(np.mean(energy_tbl_above_antenna) > 0.):
                tpf_tbl_antenna       = np.max(energy_tbl_above_antenna) / np.mean(energy_tbl_above_antenna)

            if(np.mean(energy_tbl_below_limiter) > 0.):
                tpf_tbl_limiter_lower = np.max(energy_tbl_below_limiter) / np.mean(energy_tbl_below_limiter)

            if(np.mean(energy_tbl_below_antenna) > 0.):
                tpf_tbl_antenna_lower = np.max(energy_tbl_below_antenna) / np.mean(energy_tbl_below_antenna) 

            if(np.mean(energy_upper_outer) > 0.):
                tpf_upper_outer       = np.max(energy_upper_outer)       / np.mean(energy_upper_outer)

            if(np.mean(energy_upper_inner) > 0.):
                tpf_upper_inner       = np.max(energy_upper_inner)       / np.mean(energy_upper_inner)

            if(np.mean(energy_lower_outer) > 0.):
                tpf_lower_outer       = np.max(energy_lower_outer)       / np.mean(energy_lower_outer)

            if(np.mean(energy_lower_inner) > 0.):
                tpf_lower_inner       = np.max(energy_lower_inner)       / np.mean(energy_lower_inner)      
   
        # ++++++++++++++++++++++++++++++++++++++++++++++++

        print("\n")
        print(" tf   limiter   antenna  tbl-lim    tbl-ant     tbl-llim  tbl-lant \n")
        for jj in range(ntf):
            print(" %d  %11.5f %11.5f %11.5f %11.5f %11.5f %11.5f" % (jj+1, energy_limiters[jj], energy_antennas[jj],    \
                                                            energy_tbl_above_limiter[jj], energy_tbl_above_antenna[jj],  \
                                                            energy_tbl_below_limiter[jj], energy_tbl_below_antenna[jj]))

        print("\n") 
        print(" tf   upper/outer upper/inner lower/outer lower/inner \n   ")
        for jj in range(ntf):
            print(" %d  %12.5f %12.5f %12.5f %12.5f " % (jj+1, energy_upper_outer[jj], energy_upper_inner[jj], \
                                                            energy_lower_outer[jj], energy_lower_inner[jj]))

        print("   ... at position Z at time: ", clock.ctime())
            
        plt.close('all')
        plt.figure(figsize=(9.,7.))
        plt.plot(energy_limiter_hstripe, 'ko', ms=4)
        plt.plot(energy_limiter_hstripe, 'k-')
        plt.title('horizontal stripes on limiter')
        plt.xlabel('stripe number')
        plt.ylim(bottom=0.)
        pdf.savefig()

        plt.close('all')
        plt.figure(figsize=(9.,7.))
        plt.plot(energy_limiter_hstripe, 'ko', ms=4)
        plt.plot(energy_limiter_hstripe, 'k-')
        plt.title('horizontal stripes on limiter')
        plt.xlabel('stripe number')
        plt.ylim(bottom=0.)
        pdf.savefig()

        plt.close('all')
        plt.figure(figsize=(9.,7.))
        plt.plot(energy_limiter_vstripe, 'ko', ms=4)
        plt.plot(energy_limiter_vstripe, 'k-')
        plt.title('vertical stripes on limiter')
        plt.xlabel('stripe number')
        plt.ylim(bottom=0.)
        pdf.savefig()

        plt.close('all')
        plt.figure(figsize=(9.,7.))
        plt.plot(energy_antenna_hstripe, 'ko', ms=4)
        plt.plot(energy_antenna_hstripe, 'k-')
        plt.title('horizontal stripes on antenna')
        plt.xlabel('stripe number')
        plt.ylim(bottom=0.)
        pdf.savefig()

        plt.close('all')
        plt.figure(figsize=(9.,7.))
        plt.plot(energy_antenna_vstripe, 'ko', ms=4)
        plt.plot(energy_antenna_vstripe, 'k-')
        plt.title('vertical stripes on antenna')
        plt.xlabel('stripe number')
        plt.ylim(bottom=0.)
        pdf.savefig()

        plt.close('all')
        plt.figure(figsize=(9.,7.))
        plt.plot(energy_tbl_hstripe, 'ko', ms=4)
        plt.plot(energy_tbl_hstripe, 'k-')
        plt.title('horizontal stripes on upper toroidal belt limiter')
        plt.xlabel('stripe number')
        plt.ylim(bottom=0.)
        pdf.savefig()

        print("   zzz time for next set of plots: ", mytime.time() - time_last)
        time_last = mytime.time()
        
        print("\n horizontal stripes on limiter")
        print(" stripe   limiter \n")
        for jj in range(nnz_limiter):
            print(" %d  %11.5f" % (jj+1, energy_limiter_hstripe[jj]))

        print("\n vertical stripes on limiter")
        print(" stripe   limiter \n")
        for jj in range(nnphi_limiter):
            print(" %d  %11.5f" % (jj+1, energy_limiter_vstripe[jj]))

        print("\n horizontal stripes on antenna")
        print(" stripe   antenna \n")
        for jj in range(nnz_antenna):
            print(" %d  %11.5f" % (jj+1, energy_antenna_hstripe[jj]))

        print("\n vertical stripes on antenna")
        print(" stripe   limiter \n")
        for jj in range(nnphi_antenna):
            print(" %d  %11.5f" % (jj+1, energy_antenna_vstripe[jj]))  

        print("\n horizontal stripes on toroidal belt limiter")
        print(" stripe   tbl\n")
        for jj in range(nnz_tbl):
            print(" %d  %11.5f" % (jj+1, energy_tbl_hstripe[jj]))

        energy_limiter_total = np.sum(energy_limiters)
        energy_antenna_total = np.sum(energy_antennas)
        
        my_sum = energy_tbl + energy_tbl_lower + energy_limiter_total + energy_antenna_total \
                 + energy_inner_wall_top + energy_inner_wall_bottom + energy_outer_wall_top     \
                 + energy_outer_wall_bottom
        
        print("\n  total powers (percent of total) \n")

        print(" upper tbl:         %6.2f  "%energy_tbl)
        print(" lower tbl:         %6.2f  "%energy_tbl_lower)
        print(" limiters:          %6.2f  "%energy_limiter_total)
        print(" antennas:          %6.2f  "%energy_antenna_total)
        print(" inner wall_top:    %6.2f  "%energy_inner_wall_top)
        print(" inner wall_bottom: %6.2f  "%energy_inner_wall_bottom) 
        print(" outer_wall_top     %6.2f  "%energy_outer_wall_top)   
        print(" outer_wall_bottom: %6.2f  "%energy_outer_wall_bottom)
        print(" sum of above:      %6.2f  "%my_sum)
        print(" actual total losses: %6.2f"%energy_lost_total)
        
        print("\n")

        print("\n toroidal peaking factors \n")
        print("upper tbl above limiters %6.3f"%(tpf_tbl_limiter))       
        print("upper tbl above antennas %6.3f"%(tpf_tbl_antenna))       
        print("lower tbl below limiters %6.3f"%(tpf_tbl_limiter_lower)) 
        print("lower tbl below antennas %6.3f"%(tpf_tbl_antenna_lower)) 

        print("upper outer wall %6.3f"%(tpf_upper_outer))         
        print("upper inner wall %6.3f"%(tpf_upper_inner))           
        print("lower outer wall %6.3f"%(tpf_lower_outer))      
        print("lower inner wall %6.3f"%(tpf_lower_inner))          

        # ++++++++++++++++++++++++++++++++++++++++++++++
        print("   zzz time position-a: ", mytime.time() - time_last)
        time_last = mytime.time()
        phi_offset, ii_offset = compute_phi_offset(phi_lost, rend_lost, zend_lost, phi_limiter_size, tall_limiter, rinner_wall)
        print("   zzz time position-b: ", mytime.time() - time_last)
        time_last = mytime.time()
        # wall_xarray = rwall_max * phi_offset * np.pi/180.
              
        plt.figure(figsize=(9.,7.))
        fig,ax=plt.subplots()
              
        plt.xlim((0.,1.1*phi_limiter_size))
        plt.ylim((-1.1*tall_limiter, 1.1*tall_limiter))

        phi_min = 0.
        phi_max = phi_limiter_size
        zz_min  = -1.*tall_limiter
        zz_max  =     tall_limiter

        nn_phi  = 10  # 20
        nn_zz   = 22  #44

        delta_phi = (phi_max - phi_min) / nn_phi
        delta_Z   = (zz_max  - zz_min ) / nn_zz

        energy_limiter_2d      = np.zeros((nn_phi, nn_zz))
    
        max_energy  = 0.
              
        # +++++++++++++++++++++++++++++++++++++++++++++

        for ir in range(nn_phi):
            
            phi_left   = phi_min   + ir * delta_phi
            phi_right  = phi_left  +      delta_phi
            
            for iz in range(nn_zz):

                zbottom    = zz_min    + iz * delta_Z
                ztop       = zbottom   +      delta_Z

                jj_inside  =  (phi_offset  >= phi_left) & (phi_offset  <= phi_right)  \
                            & (zend_lost   >= zbottom)  & (zend_lost   <= ztop)    

                energy_inside            = np.sum(weight_energy_lost[jj_inside])
                energy_limiter_2d[ir,iz]  = energy_inside
              
                if (energy_inside > max_energy):
                    max_energy = energy_inside
        print("   zzz time position-1: ", mytime.time() - time_last)
        time_last = mytime.time()     
        # +++++++++++++++++++++++++++++++++++++++++++++
        
        for ir in range(nn_phi):
            
            phi_left   = phi_min   + ir * delta_phi
            phi_right  = phi_left  +      delta_phi
            
            for iz in range(nn_zz):
                
                zbottom    = zz_min    + iz * delta_Z
                ztop       = zbottom   +      delta_Z

                norm_energy_limiter_2d = int(100.* energy_limiter_2d[ir,iz] / max_energy)
                    
                if(energy_limiter_2d[ir,iz] > 0):

                    if(norm_energy_limiter_2d >= 85):
                        this_color = 'gray'
                    elif (norm_energy_limiter_2d >= 50):
                        this_color = 'lime'
                    elif (norm_energy_limiter_2d >=  15.):
                        this_color = 'palegreen'
                    elif (norm_energy_limiter_2d >= 5.):
                        this_color = 'pink'
                    else:
                        this_color='cornsilk'

                    rect = Rectangle( (phi_left,zbottom),delta_phi, delta_Z, color=this_color, edgecolor=None, linewidth=0.5)
                    
                    ax.add_patch(rect)

                    xpos = phi_left + delta_phi/2.
                    ypos = zbottom  + delta_Z/2.

                    my_string = '{:3d}'.format(norm_energy_limiter_2d)
                    plt.text(xpos, ypos, my_string, fontsize=7, ha='center', va='center')
        print("   zzz time position-2: ", mytime.time() - time_last)
        time_last = mytime.time()
        plt.title(' norm energy onto limiter (R>1.6418) :  corn<5   pink:5-15  pgreen:15-50  lime:50-85   gray>85', fontsize=10)
        plt.xlabel('phi [degrees]')
        plt.ylabel('Z [m]')
        #plt.tight_layout(pad=padsize)
        pdf.savefig()
        plt.close()
        print("max_hits= ", max_hits)

        print("   ... at position AA at time: ", clock.ctime())
        

              
        # ++++++++++++++++++++++++++++++++++++++++++++++++

        plt.close()
        plt.figure(figsize=(7.,5.))
        fig,ax=plt.subplots()
        plt.xlim((-5, 370.))
        plt.ylim((-0.75, 1.25))
        
        plt.plot(phi_lost[jj_outside],   zlost[jj_outside],   'ro', ms=0.05, rasterized=do_rasterized)
        plt.title(" zlost vs philost [jj_outside]")
        for jtf in range(ntf):  # ntf

            delta_phi = phis_limiter[1,jtf] - phis_limiter[0,jtf]
            delta_Z   = ztop_limiter - zbottom_limiter
            rect = Rectangle( (phis_limiter[0,jtf], zbottom_limiter),delta_phi, delta_Z, facecolor='pink') # edgecolor='k', linewidth=0.5)   
            ax.add_patch(rect)
            
        #   print(" limiter: %d  %10.4f %10.4f %10.4f %10.4f" % (jtf, phis_limiter[0,jtf], delta_phi, zbottom_limiter, delta_Z))
            
            delta_phi = phis_antenna[1,jtf] - phis_antenna[0,jtf]
            delta_Z   = ztop_antenna - zbottom_antenna
            rect = Rectangle( (phis_antenna[0,jtf], zbottom_antenna),delta_phi, delta_Z, facecolor='palegreen') # edgecolor='k', linewidth=0.5)  
            ax.add_patch(rect)
            
        delta_phi = 360.
        delta_Z   = ztop_tbl - zbottom_tbl
        rect = Rectangle( (0., zbottom_tbl),delta_phi, delta_Z, facecolor='skyblue')  # edgecolor='k', linewidth=0.5)   
        ax.add_patch(rect)           

        delta_phi = 360.
        delta_Z   = ztop_tbl - zbottom_tbl
        rect = Rectangle( (0., -1.*ztop_tbl),delta_phi, delta_Z, facecolor='skyblue')  # edgecolor='k', linewidth=0.5)   
        ax.add_patch(rect)
        

        pdf.savefig()
        #pdb.set_trace()
        # ++++++++++++++++++++++++++++++++++++++++++++++++

        # ishape = 6   # was 5 until 11/24  disabled 12/3/2020
        # pdb.set_trace()

        print("   zzz time for next set of plots, just before call to surface_power_density: ", mytime.time() - time_last)
        time_last = mytime.time()

        #  removed 4/27/2021
        
        # if(suppress_tbl):
        #    aa = surface_power_density_2(ploss_wall_kw, rend_lost, zend_lost, phi_lost, weight_energy_lost, ishape, pdf, time_lost, fn_parameter)
        # else:
        #    aa = surface_power_density(ploss_wall_kw, rend_lost, zend_lost, phi_lost, weight_energy_lost, ishape, pdf, time_lost,fn_parameter,)

        # print("   zzz time for surface_power_density: ", mytime.time() - time_last)
        time_last = mytime.time()
        # ++++++++++++++++++++++++++++++++++++++++++++++++
        #  loss map

        if (np.abs(zl[0]) < 2.e-4):
            rmajor_outer = rl[0]
        else:
            print(" too bad, need more work")
            sys.exit()
        rmaxis =  gg.equilibria[eq_index].rmaxis

        nnn_rho       = 5000
        rho_array     = np.zeros(nnn_rho)
        rmajor_extend = 0.03
        rmajor_array  = np.linspace(rmaxis, rmajor_outer+rmajor_extend, nnn_rho)
        print("   zzz time position-3: ", mytime.time() - time_last)
        time_last = mytime.time()
        for jk in range(nnn_rho):
            rho_array[jk] =  rho_interpolator(rmajor_array[jk], 0.)

        plt.close()
        plt.figure(figsize=(8.,5.))
        plt.plot(rho_array, rmajor_array, 'b-')
        plt.xlabel('rho_poloidal')
        plt.ylabel('m')
        plt.title('Rmajor vs rho for loss map')
        pdf.savefig()
        

        #  create uniform rho and pitch-angle grid
        
        mmm_rho   = 100
        mmm_pitch = 100
        pitch_lossmap  = np.linspace(-1., 1., mmm_pitch)
        rho_lossmap    = np.linspace(0., 1., mmm_rho)
        rmajor_lossmap = np.zeros(mmm_rho)

        for jk in range(mmm_rho):
            idx = (np.abs(rho_array - rho_lossmap[jk])).argmin()
            rmajor_lossmap[jk] = rmajor_array[idx]

        #pdb.set_trace() 
        print("   ... at position AB at time: ", clock.ctime())
        print("   zzz total time: ", mytime.time()-time_first)
            
        return

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++
                   
def myread_hdf5_multiple(fn_hdf5_list, max_markers=0):

    # fn_hdf5_list = ["ascot_27193488.h5", "ascot_27197137.h5","ascot_27197197.h5","ascot_27197428.h5"]
    
    nn_files = len(fn_hdf5_list)
    print("   myread_hdf5_multiple:  number of files that I will read = ", nn_files)
    
    for jj in range(nn_files):
        print("   ... now reading file: ", fn_hdf5_list[jj])
        if(jj == 0):

            full_filename = fn_hdf5_list[0]
        
            new_or_old = old_or_new(full_filename)
    
            if(new_or_old == 0):
                out_combined = myread_hdf5(full_filename, max_markers)
            elif(new_or_old == 1):
                out_combined = myread_hdf5_new(full_filename, max_markers)
            
            # out_combined = myread_hdf5(fn_hdf5_list[0])
            
            mm           = out_combined["marker_id"].size

        else:

            full_filename = fn_hdf5_list[jj]
            new_or_old = old_or_new(full_filename)
    
            if(new_or_old == 0):
                out_local = myread_hdf5(full_filename,max_markers)
            elif(new_or_old == 1):
                out_local = myread_hdf5_new(full_filename, max_markers)
 
            # out_local = myread_hdf5(fn_hdf5_list[jj])

            marker_ids = out_local["marker_id"] + mm
            end_ids    = out_local["id_end"]    + mm
            ini_ids    = out_local["id_ini"]    + mm

            out_local["marker_id"] = marker_ids
            out_local["id_end"]    = end_ids
            out_local["id_ini"]    = ini_ids

            mm_local = out_local["marker_id"].size
            mm = mm + mm_local

            for key in out_combined:

                array_1 = out_combined[key]
                array_2 = out_local[key]

                array_new = np.concatenate((array_1, array_2))

                out_combined[key] = array_new
          
    return out_combined

# -----------------------------------------------------------

def myread_hdf5(file_name, max_markers=0, do_corr=0):

    time_start = clock.time()
    print(" \n   ... myread_hdf5 note:  reading OLD-style hdf5 file: ", file_name)
    full_filename = construct_full_filename(file_name)
    ff = h5py.File(full_filename, 'r')

    # ----------------------------------------------
    #  read wall data
    
    ww        = ff['wall']
    wall_keys = ww.keys()
    
    for key in wall_keys:
        wwid = key   # assume there is just one
        
    wall = ww[wwid]

    try:   # 2D wall
        rr_wall = wall['r']
        zz_wall = wall['z']

        r_wall = np.array(rr_wall)
        z_wall = np.array(zz_wall)
    except:  # 3D wall
        aaa    = myread.read_any_file('V1E_FalseFloor2_densify.txt')
        my_shape = aaa.shape
        nnn_seg = my_shape[0]
        r_wall = np.zeros((nnn_seg,1)) # for consistency with old code
        z_wall = np.zeros((nnn_seg,1))
        r_wall[:,0] = aaa[:,0]
        z_wall[:,0] = aaa[:,1]
        #pdb.set_trace()
        
    # ---------------------------------------------------
    #  read marker data
    
    mm = ff['marker']
    marker_keys = mm.keys()
    for key in marker_keys:
        mmid = key            # assume there is just one
    markers = mm[mmid]
    #import pdb; #pdb.set_trace()

    marker_rr   = np.array(np.transpose(np.array(markers['r'])))
    marker_zz   = np.array(np.transpose(np.array(markers['z'])))
    marker_phi  = np.array(np.transpose(np.array(markers['phi'])))
    marker_id   = np.array(np.transpose(np.array(markers['id'])))
    marker_vphi = np.array(np.transpose(np.array(markers['vphi'])))
    marker_vz   = np.array(np.transpose(np.array(markers['vz'])))
    marker_vr   = np.array(np.transpose(np.array(markers['vr'])))
    marker_mass = np.array(np.transpose(np.array(markers['mass'])))

    marker_rr   = marker_rr[0,:]
    marker_zz   = marker_zz[0,:]
    marker_phi  = marker_phi[0,:]
    marker_id   = marker_id[0,:]
    marker_vphi = marker_vphi[0,:]
    marker_mass = marker_mass[0,:]
    
    #  **IMPT**  marker_vphi multiplied by -1 from 3/30 until 10/17/2020
    
    marker_vphi = marker_vphi  
    marker_vz   = marker_vz[0,:]
    marker_vr   = marker_vr[0,:]

    # import pdb; #pdb.set_trace()

    # --------------------------------------------
    #  read results data
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

    orbtime = np.array(results[run_id]['orbit']['time'])
    orbid   = np.array(results[run_id]['orbit']['id'])
    
    print(' I will read from : ', run_id)

    anum    = np.array(results[run_id]['endstate']['anum'])
    znum    = np.array(results[run_id]['endstate']['znum'])
    charge  = np.array(results[run_id]['endstate']['charge'])
    mass    = np.array(results[run_id]['endstate']['mass'])
    vpar    = np.array(results[run_id]['endstate']['vpar'])
    vphi    = np.array(results[run_id]['endstate']['vphi'])
    vr      = np.array(results[run_id]['endstate']['vr'])
    vz      = np.array(results[run_id]['endstate']['vz']) 
    weight  = np.array(results[run_id]['endstate']['weight'])
    time    = np.array(results[run_id]['endstate']['time'])
    cputime = np.array(results[run_id]['endstate']['cputime'])
    endcond = np.array(results[run_id]['endstate']['endcond'])
    phi_end = np.array(results[run_id]['endstate']['phiprt'])    ## was 'phi' = until 3/19/2021
    r_end   = np.array(results[run_id]['endstate']['rprt'])      ## was 'r' = until 3/19/2021
    z_end   = np.array(results[run_id]['endstate']['zprt'])      ## was 'z' = until 3/19/2021
    phiprt  = np.array(results[run_id]['endstate']['phiprt'])  ## new
    rprt    = np.array(results[run_id]['endstate']['rprt'])    ## new
    zprt    = np.array(results[run_id]['endstate']['zprt'])    ## new

    id_end  = np.array(results[run_id]['endstate']['id'])      ## was id = until 10/17/2020
    theta_end   = np.array(results[run_id]['endstate']['theta'])   ## was theta = until 10/17/2020

    r_ini    = np.array(results[run_id]['inistate']['rprt'])   # was 'r' until 3/19/2021
    z_ini    = np.array(results[run_id]['inistate']['zprt'])   # was 'z' until 3/19/2021
    phi_ini  = np.array(results[run_id]['inistate']['phiprt']) # was 'phi' until 3/19/2021
    rprt_ini   = np.array(results[run_id]['inistate']['rprt'])   ## new
    zprt_ini   = np.array(results[run_id]['inistate']['zprt'])   ## new
    phiprt_ini = np.array(results[run_id]['inistate']['phiprt']) ## new

    
    id_ini   = np.array(results[run_id]['inistate']['id'])
    vphi_ini = np.array(results[run_id]['inistate']['vphi'])
    vr_ini   = np.array(results[run_id]['inistate']['vr'])
    vz_ini   = np.array(results[run_id]['inistate']['vz'])
    vpar_ini = np.array(results[run_id]['inistate']['vpar'])

    vpar      = vpar       # **IMPT** multiplied by -1 from 3/30 until 10/17/2020
    vphi      = vphi       # **IMPT** multiplied by -1 from 3/30 until 10/17/2020
    
    vphi_ini = vphi_ini    # **IMPT** multiplied by -1 from 3/30 until 10/17/2020
    vpar_ini = vpar_ini    # **IMPT** multiplied by -1 from 3/30 until 10/17/2020

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
    proton_mass = 1.67e-27
    q_e         = 1.602e-19
    ##pdb.set_trace()   
    ekev        = 0.001  * proton_mass * mass * vtot**2/ (2.*1.602e-19)
    ekev_ini    = 0.001  * proton_mass * mass * vtot_ini**2/ (2.*1.602e-19)
    marker_ekev = 0.001  * proton_mass * marker_mass * marker_vtot**2/ (2.*1.602e-19)  # 12/31/2020:read thru aa

    out = {}

    nn = anum.size
    if(max_markers > 0):
        nn = max_markers
    
    out["weight_parent"] = weight_parent[0:nn]

    out['orbtime'] = orbtime[0:nn]
    out['orbid']   = orbid[0:nn] 
    out["anum"]    = anum[0:nn]
    out["znum"]    = znum[0:nn]
    out["charge"]  = charge[0:nn]
    out["ekev"]    = ekev[0:nn]
    out["vtot"]    = vtot[0:nn]
    out["mass"]    = mass[0:nn]
    out["vpar"]    = vpar[0:nn]
    out["vphi"]    = vphi[0:nn]
    out["vr"]      = vr[0:nn]
    out["vz"]      = vz[0:nn]
    out["weight"]  = weight[0:nn]
    out["time"]    = time[0:nn]
    out["cputime"] = cputime[0:nn]
    out["endcond"] = endcond[0:nn]
    out["phi_end"] = phi_end[0:nn]
    out["r_end"]   = r_end[0:nn]
    out["z_end"]   = z_end[0:nn]
    out["id_end"]  = id_end[0:nn]
    out["r_ini"]   = r_ini[0:nn]
    out["z_ini"]   = z_ini[0:nn]
    out["phi_ini"] = phi_ini[0:nn]
    out["id_ini"]  = id_ini[0:nn]
    out["vphi_ini"] = vphi_ini[0:nn]
    out["vr_ini"]   = vr_ini[0:nn]
    out["vz_ini"]   = vz_ini[0:nn]
    out["vpar_ini"] = vpar_ini[0:nn]
    out["pitch_ini"] = pitch_ini[0:nn]
    out["pitch_phi_ini"] = pitch_phi_ini[0:nn]
    out["marker_pitch_phi"] = marker_pitch_phi[0:nn]

    out["marker_r"] = marker_rr[0:nn]
    out["marker_z"] = marker_zz[0:nn]
    out["marker_phi"] = marker_phi[0:nn]
    out["marker_id"] = marker_id[0:nn]
    out["marker_vphi"] = marker_vphi[0:nn]
    out["marker_vr"] = marker_vr[0:nn]
    out["marker_vz"] = marker_vz[0:nn]

    out["r_wall"]    = r_wall
    out["z_wall"]    = z_wall
    out["theta_end"] = theta_end[0:nn]

    out["vtot_ini"]    = vtot_ini[0:nn]
    out["marker_vtot"] = marker_vtot[0:nn]
    out["ekev_ini"]    = ekev_ini[0:nn]
    out["marker_ekev"] = marker_ekev[0:nn]
    
    out["rprt"]       = rprt[0:nn]             ## new
    out["zprt"]       = zprt[0:nn]            ## new
    out["phiprt"]     = phiprt[0:nn]          ## new
    out["rprt_ini"]   = rprt_ini[0:nn]         ## new
    out["zprt_ini"]   = zprt_ini[0:nn]         ## new
    out["phiprt_ini"] = phiprt_ini[0:nn]       ## new

    print("  time required to read .h5 file: ", clock.time() - time_start, " seconds")
    return out

# ---------------------------------------------------------------------------

def myread_hdf5_markers(file_name, max_markers=0):

    time_start = clock.time()
    
    full_filename = construct_full_filename(file_name)
    ff = h5py.File(full_filename, 'r')

    mm = ff['marker']
    marker_keys = mm.keys()
    for key in marker_keys:
        mmid = key            # assume there is just one
    markers = mm[mmid]
    #import pdb; #pdb.set_trace()

    marker_rr     = np.transpose(np.array(markers['r']))
    marker_zz     = np.transpose(np.array(markers['z']))
    marker_phi    = np.transpose(np.array(markers['phi']))
    marker_id     = np.transpose(np.array(markers['id']))
    marker_vphi   = np.transpose(np.array(markers['vphi']))
    marker_vz     = np.transpose(np.array(markers['vz']))
    marker_vr     = np.transpose(np.array(markers['vr']))
    marker_anum   = np.transpose(np.array(markers['anum']))
    marker_znum   = np.transpose(np.array(markers['znum']))
    marker_charge = np.transpose(np.array(markers['charge']))
    marker_mass   = np.transpose(np.array(markers['mass']))
    marker_time   = np.transpose(np.array(markers['time']))
    marker_n      = np.transpose(np.array(markers['n']))
    marker_weight = np.transpose(np.array(markers['weight']))

    marker_rr     =     marker_rr[0,:]
    marker_zz     =     marker_zz[0,:]
    marker_phi    =    marker_phi[0,:]
    marker_id     =     marker_id[0,:]
    marker_vphi   =   marker_vphi[0,:]
    marker_vz     =     marker_vz[0,:]
    marker_vr     =     marker_vr[0,:]
    marker_anum   =   marker_anum[0,:]
    marker_znum   =   marker_znum[0,:]
    marker_charge = marker_charge[0,:]
    marker_mass   =   marker_mass[0,:]
    marker_time   =   marker_time[0,:]
    marker_n      =      marker_n[0,:]
    marker_weight = marker_weight[0,:]

    # marker_vphi   = -1. * marker_vphi   removed this 7/27/2020
    
    out = {}

    nn = marker_rr.size
    if(max_markers > 0):
       nn = max_markers
       
    out["marker_r"]      = marker_rr[0:nn]
    out["marker_z"]      = marker_zz[0:nn]
    out["marker_phi"]    = marker_phi[0:nn]
    out["marker_id"]     = marker_id[0:nn]
    out["marker_vphi"]   = marker_vphi[0:nn]
    out["marker_vr"]     = marker_vr[0:nn]
    out["marker_vz"]     = marker_vz[0:nn]
    out["marker_mass"]   = marker_mass[0:nn]
    out["marker_charge"] = marker_charge[0:nn]
    out["marker_anum"]   = marker_anum[0:nn]
    out["marker_znum"]   = marker_znum[0:nn]
    out["marker_n"]      = marker_n[0:nn]
    out["marker_time"]   = marker_time[0:nn]
    out["marker_weight"] = marker_weight[0:nn]

    print("   ... time required to read markers: ", clock.tome() - time-start, " seconds")
    return out
# ---------------------------------------------------------------------------

def myread_hdf5_markers(file_name, max_markers=0):

    time_start = clock.time()
    
    full_filename = construct_full_filename(file_name)
    ff = h5py.File(full_filename, 'r')

    mm = ff['marker']
    marker_keys = mm.keys()
    for key in marker_keys:
        mmid = key            # assume there is just one
    markers = mm[mmid]
    #import pdb; #pdb.set_trace()

    marker_rr     = np.transpose(np.array(markers['r']))
    marker_zz     = np.transpose(np.array(markers['z']))
    marker_phi    = np.transpose(np.array(markers['phi']))
    marker_id     = np.transpose(np.array(markers['id']))
    marker_vphi   = np.transpose(np.array(markers['vphi']))
    marker_vz     = np.transpose(np.array(markers['vz']))
    marker_vr     = np.transpose(np.array(markers['vr']))
    marker_anum   = np.transpose(np.array(markers['anum']))
    marker_znum   = np.transpose(np.array(markers['znum']))
    marker_charge = np.transpose(np.array(markers['charge']))
    marker_mass   = np.transpose(np.array(markers['mass']))
    marker_time   = np.transpose(np.array(markers['time']))
    marker_n      = np.transpose(np.array(markers['n']))
    marker_weight = np.transpose(np.array(markers['weight']))

    marker_rr     =     marker_rr[0,:]
    marker_zz     =     marker_zz[0,:]
    marker_phi    =    marker_phi[0,:]
    marker_id     =     marker_id[0,:]
    marker_vphi   =   marker_vphi[0,:]
    marker_vz     =     marker_vz[0,:]
    marker_vr     =     marker_vr[0,:]
    marker_anum   =   marker_anum[0,:]
    marker_znum   =   marker_znum[0,:]
    marker_charge = marker_charge[0,:]
    marker_mass   =   marker_mass[0,:]
    marker_time   =   marker_time[0,:]
    marker_n      =      marker_n[0,:]
    marker_weight = marker_weight[0,:]

    # marker_vphi   = -1. * marker_vphi   removed this 7/27/2020
    
    out = {}

    nn = marker_rr.size
    if(max_markers > 0):
       nn = max_markers
       
    out["marker_r"]      = marker_rr[0:nn]
    out["marker_z"]      = marker_zz[0:nn]
    out["marker_phi"]    = marker_phi[0:nn]
    out["marker_id"]     = marker_id[0:nn]
    out["marker_vphi"]   = marker_vphi[0:nn]
    out["marker_vr"]     = marker_vr[0:nn]
    out["marker_vz"]     = marker_vz[0:nn]
    out["marker_mass"]   = marker_mass[0:nn]
    out["marker_charge"] = marker_charge[0:nn]
    out["marker_anum"]   = marker_anum[0:nn]
    out["marker_znum"]   = marker_znum[0:nn]
    out["marker_n"]      = marker_n[0:nn]
    out["marker_time"]   = marker_time[0:nn]
    out["marker_weight"] = marker_weight[0:nn]

    print("   ... time required to read markers = ", clock.time() - time_start, " seconds")
    return out

# +++++++++++++++++++++++++++++++++++++++++++++++++++

def myread_hdf5_lossonly(file_name, weight_mult, max_markers=0):
    """
      reads the hdf5 file but returns only data from lost markers
      the weights array is multiplied by weight_mult
    """
    aa = myread_hdf5(file_name, max_markers, docorr=0)

    out = {}

    ii_lost =   (endcond==  8)^(endcond== 16)^(endcond== 32)^(endcond==40)   \
               ^(endcond==512)^(endcond==520)^(endcond==544) 
    
    for key in aa.keys():
       out[key] = aa[key][ii_lost]

    out["weights"] = out["weights"] * weight_mult

    return out

def myread_hdf5_new_lossonly(file_name, weight_mult, max_markers=0):
    """
      reads the hdf5 file but returns only data from lost markers
      the weights array is multiplied by weight_mult
    """
    aa = myread_hdf5_new(file_name, max_markers, docorr=0)

    out = {}

    ii_lost =   (endcond==  8)^(endcond== 16)^(endcond== 32)^(endcond==40)   \
               ^(endcond==512)^(endcond==520)^(endcond==544) 
    
    for key in aa.keys():
       out[key] = aa[key][ii_lost]

    out["weights"] = out["weights"] * weight_mult

    return out

# ++++++++++++++++++++++++++++++++++++++++

def myread_hdf5_new(file_name, max_markers=0, do_corr=0):

    time_start = clock.time()
    print(" \n   ... myread_hdf5_new: note:  reading NEW-style hdf5 file: ", file_name)

    full_filename = construct_full_filename(file_name)
    ff = h5py.File(full_filename, 'r')

    ww        = ff['wall']
    wall_keys = ww.keys()
    
    for key in wall_keys:
        wwid = key   # assume there is just one
        
    wall = ww[wwid]

    try:   # 2D wall
        print("   ... I am reading a 2D wall")
        rr_wall = wall['r']
        zz_wall = wall['z']

        r_wall = np.array(rr_wall)
        z_wall = np.array(zz_wall)
    except:  # 3D wall
        print("   ... I am reading wall:  V1E_FalseFloor2_density.txt")
        aaa    = myread.read_any_file('V1E_FalseFloor2_densify.txt')
        my_shape = aaa.shape
        nnn_seg = my_shape[0]
        r_wall = np.zeros((nnn_seg,1)) # for consistency with old code
        z_wall = np.zeros((nnn_seg,1))
        r_wall[:,0] = aaa[:,0]
        z_wall[:,0] = aaa[:,1]
        #pdb.set_trace()

    # -----------------
    #  markers
    
    mm = ff['marker']
    marker_keys = mm.keys()
    for key in marker_keys:
        mmid = key            # assume there is just one
    markers = mm[mmid]
    #import pdb; #pdb.set_trace()

    marker_rr   = np.transpose(np.array(markers['r']))
    marker_zz   = np.transpose(np.array(markers['z']))
    marker_phi  = np.transpose(np.array(markers['phi']))
    marker_id   = np.transpose(np.array(markers['id']))
    marker_vphi = np.transpose(np.array(markers['vphi']))
    marker_vz   = np.transpose(np.array(markers['vz']))
    marker_vr   = np.transpose(np.array(markers['vr']))
    marker_mass = np.transpose(np.array(markers['mass']))

    marker_rr   = marker_rr[0,:]
    marker_zz   = marker_zz[0,:]
    marker_phi  = marker_phi[0,:]
    marker_id   = marker_id[0,:]
    marker_vphi = marker_vphi[0,:]
    marker_mass = marker_mass[0,:]
                               
    #  **IMPT**  marker_vphi multiplied by -1 from 3/30 until 10/17/2020
    
    marker_vz   = marker_vz[0,:]
    marker_vr   = marker_vr[0,:]

    marker_vtot      = (marker_vphi**2 + marker_vr**2 + marker_vz**2)**(0.5)
    
    marker_pitch_phi = marker_vphi/marker_vtot
    
    # ------------------------------------------------------
    #  endstate and orbit

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

    orbtime = np.array(results[run_id]['orbit']['time'])
    orbid   = np.array(results[run_id]['orbit']['ids'])

    print('   ... have completed read of orbits')
    
    anum    = np.array(results[run_id]['endstate']['anum'])
    znum    = np.array(results[run_id]['endstate']['znum'])
    charge  = np.array(results[run_id]['endstate']['charge'])
    mass    = np.array(results[run_id]['endstate']['mass'])
    
    #vpar   = np.array(results[run_id]['endstate']['vpar'])              # **NEW**
    #vphi   = np.array(results[run_id]['endstate']['vphi'])              # **NEW**
    #vr     = np.array(results[run_id]['endstate']['vr'])                # **NEW**
    #vz     = np.array(results[run_id]['endstate']['vz'])                # **NEW**
    
    weight     = np.array(results[run_id]['endstate']['weight'])
    time       = np.array(results[run_id]['endstate']['time'])            
    cputime    = np.array(results[run_id]['endstate']['cputime'])
    endcond    = np.array(results[run_id]['endstate']['endcond'])
    phi_end    = np.array(results[run_id]['endstate']['phiprt'])     # was 'phi' until 3/19/2021
    r_end      = np.array(results[run_id]['endstate']['rprt'])       # was 'r'   until 3/19/2021
    z_end      = np.array(results[run_id]['endstate']['zprt'])       # was 'z'   until 3/19/2021
    phiprt_end    = np.array(results[run_id]['endstate']['phiprt'])      # **NEW**
    rprt_end      = np.array(results[run_id]['endstate']['rprt'])        # **NEW**
    zprt_end      = np.array(results[run_id]['endstate']['zprt'])        # **NEW**


    
    id_end     = np.array(results[run_id]['endstate']['ids'])        # **NEW** had been ['endstate']['id']
    theta_end  = np.array(results[run_id]['endstate']['theta'])

    ppar    = np.array(results[run_id]['endstate']['ppar'])        # **NEW**      
    pphiprt = np.array(results[run_id]['endstate']['pphiprt'])     # **NEW**
    prprt   = np.array(results[run_id]['endstate']['prprt'])       # **NEW**
    pzprt   = np.array(results[run_id]['endstate']['pzprt'])       # **NEW**
    mileage = np.array(results[run_id]['endstate']['mileage'])     # **NEW**
    
    #  convert momentum to velocity
    
    proton_mass = 1.67e-27
    
    vpar    = ppar      / (mass*proton_mass)     # **NEW**    
    vphi    = pphiprt   / (mass*proton_mass)     # **NEW**
    vr      = prprt     / (mass*proton_mass)     # **NEW**
    vz      = pzprt     / (mass*proton_mass)     # **NEW**

    vpar    = vpar   # **IMPT** from 3/30 until 10/17/2020 multiplied by -1
    vphi    = vphi   # **IMPT** from 3/30 until 10/17/2020 multiplied by -1
    
    vtot    = (vphi**2 + vr**2 + vz**2)**(0.5)

    print('   ... have completed read of endstate')
    # ------------------------------------------------------
    #  inistate
    
    r_ini    = np.array(results[run_id]['inistate']['rprt'])  # was 'r' until 3/19/2021
    z_ini    = np.array(results[run_id]['inistate']['zprt'])  # was 'z' until 3/19/2021
    phi_ini  = np.array(results[run_id]['inistate']['phiprt']) # was 'phi' until 3/19/2021
    rprt_ini    = np.array(results[run_id]['inistate']['rprt'])      # **NEW**
    zprt_ini    = np.array(results[run_id]['inistate']['zprt'])      # **NEW**
    phiprt_ini  = np.array(results[run_id]['inistate']['phiprt'])    # **NEW**
    
    id_ini   = np.array(results[run_id]['inistate']['ids'])               #  **NEW** (was id)

    #vphi_ini = np.arrayresults[run_id]['inistate']['vphi'])             #  **NEW**
    #vr_ini   = np.arrayresults[run_id]['inistate']['vr'])               #  **NEW**
    #vz_ini   = np.arrayresults[run_id]['inistate']['vz'])               #  **NEW**
    #vpar_ini = np.arrayresults[run_id]['inistate']['vpar'])             #  **NEW**

    ppar_ini    = np.array(results[run_id]['inistate']['ppar'])        # **NEW**      
    pphiprt_ini = np.array(results[run_id]['inistate']['pphiprt'])     # **NEW**
    prprt_ini   = np.array(results[run_id]['inistate']['prprt'])       # **NEW**
    pzprt_ini   = np.array(results[run_id]['inistate']['pzprt'])       # **NEW**
    mass_ini    = np.array(results[run_id]['inistate']['mass'])         # **NEW**

    # convert momentum to velocity

    vpar_ini    = ppar_ini    / (mass_ini*proton_mass)     # **NEW**    
    vphi_ini    = pphiprt_ini / (mass_ini*proton_mass)     # **NEW**
    vr_ini      = prprt_ini   / (mass_ini*proton_mass)     # **NEW**
    vz_ini      = pzprt_ini   / (mass_ini*proton_mass)     # **NEW**

    vphi_ini = vphi_ini      # **IMPT** from 3/30 until 10/17/2020 multiplied by -1
    vpar_ini = vpar_ini      # **IMPT** from 3/30 until 10/17/2020 multiplied by -1

    vtot_ini    = (vphi_ini**2 + vr_ini**2 + vz_ini**2)**(0.5)
    
    pitch_ini        = vpar_ini/(vtot_ini+1.e-9)
    pitch_phi_ini    = vphi_ini/(vtot_ini+1.e-9)

    print('   ... have completed read of inistate')
    
    # ------------------------------------------------------
    #  miscellaneous
    #  correction due to improper weighting of pitch angles

    weight_parent = weight/np.sum(weight)

    if(do_corr == 1):
        weight = weight * np.sqrt(1 - pitch_ini**2)
        weight = weight / np.sum(weight)
    elif(do_corr == 2):
        weight = weight * np.sqrt(1 - pitch_ini**2) * r_ini
        weight = weight / np.sum(weight)
    proton_mass = 1.67e-27
    q_e         = 1.602e-19
      
    ekev        = 0.001  * proton_mass * mass * vtot**2/ (2.*1.602e-19)
    ekev_ini    = 0.001  * proton_mass * mass * vtot_ini**2/ (2.*1.602e-19)
    marker_ekev = 0.001  * proton_mass * marker_mass * marker_vtot**2/ (2.*1.602e-19)

    print(" I have finished reading run: ", run_id)
    out = {}

    nn = anum.size
    if(max_markers > 0):
       nn = max_markers
       
    out["weight_parent"] = weight_parent[0:nn]
    out['orbtime'] = orbtime[0:nn]
    out['orbid']   = orbid[0:nn] 
    out["mileage"] = mileage[0:nn] 
    out["anum"]    = anum[0:nn] 
    out["znum"]    = znum[0:nn] 
    out["charge"]  = charge[0:nn] 
    out["ekev"]    = ekev[0:nn] 
    out["vtot"]    = vtot[0:nn] 
    out["mass"]    = mass[0:nn] 
    out["vpar"]    = vpar[0:nn] 
    out["vphi"]    = vphi[0:nn] 
    out["vr"]      = vr[0:nn] 
    out["vz"]      = vz[0:nn] 
    out["weight"]  = weight[0:nn] 
    out["time"]    = time[0:nn] 
    out["cputime"] = cputime[0:nn] 
    out["endcond"] = endcond[0:nn] 
    
    out["phi_end"]    = phi_end[0:nn] 
    out["r_end"]      = r_end[0:nn] 
    out["z_end"]      = z_end[0:nn] 
    
    out["phiprt_end"] = phiprt_end[0:nn]     # **NEW**
    out["rprt_end"]   = rprt_end[0:nn]      # **NEW**
    out["zprt_end"]   = zprt_end [0:nn]      # **NEW**
    
    out["id_end"]     = id_end[0:nn] 
    
    out["r_ini"]      = r_ini[0:nn] 
    out["z_ini"]      = z_ini[0:nn] 
    out["phi_ini"]    = phi_ini[0:nn] 
    
    out["rprt_ini"]   = rprt_ini[0:nn]       # **NEW**
    out["zprt_ini"]   = zprt_ini[0:nn]       # **NEW**
    out["phiprt_ini"] = phiprt_ini [0:nn]    # **NEW**

    
    out["id_ini"]  = id_ini[0:nn] 
    out["vphi_ini"] = vphi_ini[0:nn] 
    out["vr_ini"]   = vr_ini[0:nn] 
    out["vz_ini"]   = vz_ini[0:nn] 
    out["vpar_ini"] = vpar_ini[0:nn] 
    out["pitch_ini"] = pitch_ini[0:nn] 
    out["pitch_phi_ini"] = pitch_phi_ini[0:nn] 
    out["marker_pitch_phi"] = marker_pitch_phi[0:nn] 

    out["marker_r"] = marker_rr[0:nn] 
    out["marker_z"] = marker_zz[0:nn] 
    out["marker_phi"] = marker_phi[0:nn] 
    out["marker_id"] = marker_id[0:nn] 
    out["marker_vphi"] = marker_vphi[0:nn] 
    out["marker_vr"] = marker_vr[0:nn] 
    out["marker_vz"] = marker_vz[0:nn] 

    out["r_wall"]    = r_wall
    out["z_wall"]    = z_wall
    out["theta_end"] = theta_end[0:nn] 

    out["vtot_ini"]    = vtot_ini[0:nn] 
    out["marker_vtot"] = marker_vtot[0:nn] 
    out["ekev_ini"]    = ekev_ini[0:nn] 
    out["marker_ekev"] = marker_ekev[0:nn] 

    print("   ... time required to read .h5 file: ", clock.time() - time_start, " seconds")
    return out

# ---------------------------------------------------------------------------

if __name__ == '__main__':

    do_corr = 0
    
    mylen = len(sys.argv)
    # pdb.set_trace()
    if (mylen == 1):
        
        aa = get.get_process_parameters()
        
        file_name                 = aa["fn_h5_list"]
        geq_name                  = aa["fn_geq"]
        fraction_alphas_simulated = aa["ff_mult"]
        fn_profiles               = aa["fn_profiles"]
        ploss_wall_kw             = aa["kw_loss"]
        igrid_spd                 = aa["igrid_spd"]
        stub                     = aa["fn_out"]
        fn_parameter              = aa["fn_parameter"]
        suppress_tbl              = aa["suppress_tbl"]
        max_markers               = aa["max_markers"]
        
    elif (mylen != 11):
        
        print(" You provided insufficient arguments.")
        print(" arguments: filename.h5  v1d.geq and string(frac_alpha_sim) and v1e_profiles_3.txt and kw_loss and igrid_spd fn_parameter max_markers" )
        exit()
    else:
        
        print("")
        print("Number of arguments:    ", mylen)
        print("arguments passed to me: ", sys.argv)
        print("")
        
        file_name                 = sys.argv[1]
        geq_name                  = sys.argv[2]    #'v1e.geq'
        fraction_alphas_simulated = float(sys.argv[3])
        fn_profiles               = sys.argv[4]
        ploss_wall_kw             = float(sys.argv[5])
        igrid_spd                 = int(sys.argv[6])
        stub                      = sys.argv[7]
        fn_parameter              = sys.argv[8]
        suppress_tbl              = sys.argv[9]
        max_markers               = sys.argv[10]

    eq_index  = 0
    

    print("")
    print("      ... file_name:                 ", file_name)
    print("      ... geq_name:                  ", geq_name)
    print("      ... fraction_alphas_simulated: ", fraction_alphas_simulated)
    print("      ... fn_profiles:               ", fn_profiles)
    print("      ... ploss_wall_kw:             ", ploss_wall_kw)
    print("      ... igrid_spd:                 ", igrid_spd)
    print("      ... stub filename for output:  ", stub)
    print("      ... parameter filename:        ", fn_parameter)
    print("      ... suppress_tbl:              ", suppress_tbl)
    print("      ... max_markers:               ", max_markers)
    print("")
    print("   ... main:  I am about to invoke print_summary")
    
    print_summary(file_name, geq_name, eq_index, fraction_alphas_simulated, do_corr, fn_profiles, ploss_wall_kw, igrid_spd, stub, suppress_tbl, max_markers, fn_parameter)



# ---------------------------------------------------------------------------

def myread_hdf5_markers(file_name):

    time_start = clock.time()
    full_filename = construct_full_filename(file_name)
    ff = h5py.File(full_filename, 'r')

    mm = ff['marker']
    marker_keys = mm.keys()
    for key in marker_keys:
        mmid = key            # assume there is just one
    markers = mm[mmid]
    #import pdb; #pdb.set_trace()

    marker_rr     = np.transpose(np.array(markers['r']))
    marker_zz     = np.transpose(np.array(markers['z']))
    marker_phi    = np.transpose(np.array(markers['phi']))
    marker_id     = np.transpose(np.array(markers['id']))
    marker_vphi   = np.transpose(np.array(markers['vphi']))
    marker_vz     = np.transpose(np.array(markers['vz']))
    marker_vr     = np.transpose(np.array(markers['vr']))
    marker_anum   = np.transpose(np.array(markers['anum']))
    marker_znum   = np.transpose(np.array(markers['znum']))
    marker_charge = np.transpose(np.array(markers['charge']))
    marker_mass   = np.transpose(np.array(markers['mass']))
    marker_time   = np.transpose(np.array(markers['time']))
    marker_n      = np.transpose(np.array(markers['n']))
    marker_weight = np.transpose(np.array(markers['weight']))

    marker_rr     =     marker_rr[0,:]
    marker_zz     =     marker_zz[0,:]
    marker_phi    =    marker_phi[0,:]
    marker_id     =     marker_id[0,:]
    marker_vphi   =   marker_vphi[0,:]
    marker_vz     =     marker_vz[0,:]
    marker_vr     =     marker_vr[0,:]
    marker_anum   =   marker_anum[0,:]
    marker_znum   =   marker_znum[0,:]
    marker_charge = marker_charge[0,:]
    marker_mass   =   marker_mass[0,:]
    marker_time   =   marker_time[0,:]
    marker_n      =      marker_n[0,:]
    marker_weight = marker_weight[0,:]

    # marker_vphi   = -1. * marker_vphi   removed this 7/27/2020
    
    out = {}

    out["marker_r"]      = marker_rr
    out["marker_z"]      = marker_zz
    out["marker_phi"]    = marker_phi
    out["marker_id"]     = marker_id
    out["marker_vphi"]   = marker_vphi
    out["marker_vr"]     = marker_vr
    out["marker_vz"]     = marker_vz
    out["marker_mass"]   = marker_mass
    out["marker_charge"] = marker_charge
    out["marker_anum"]   = marker_anum
    out["marker_znum"]   = marker_znum
    out["marker_n"]      = marker_n
    out["marker_time"]   = marker_time
    out["marker_weight"] = marker_weight
    print("   ... time required to read markers: ", clock.time()- time_start, " seconds")
    return out

# ----------------------------------------------


