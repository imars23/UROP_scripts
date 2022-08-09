import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import THACO.load_bright as lb
import MDSplus
import scipy.optimize as opt
from matplotlib import rc
from scipy.stats import pearsonr
rc('text',usetex=True)
rc('text.latex', preamble=r'\usepackage{xcolor}')

shot = 1140213022

#Load data from MDSplus

#Get line integrated brightness and axes from moments
psin, times, ch_num, lint_br, lint_br_unc = lb.load_xics_lintbr(shot, plot = False)

#Load in RF Data
RFTree = MDSplus.Tree('rf', shot)
RFbranch = RFTree.getNode(r'\rf::RF_POWEr_net')
RFdata = RFbranch.data()
RFtimes = np.linspace(0, 2.0478, 20479) #Time axis is just 20479 points spaced evenly apart

#Load in Argon Pumpin Data
specTree = MDSplus.Tree('spectroscopy', shot)
specBranch = specTree.getNode(r'\SPECTROSCOPY::top.x_ray_pha:incaa16:gas_b_sd_lo')
argon_pulse = specBranch.data()
argon_time = specBranch.dim_of(0).data()

def bright_sptime(psin, times, ch_num, lint_br):
    """
    Plots line-integrated brightness as a function of time and psin over a surface

    Parameters:
    psin (array): array of psin values (psin[time, channel])
    times (array): array of time values for each temporal channel
    ch_num (array): array of channel numbers
    lint_br (array): array of line-integrated brightness values (lint_br[time, channel])

    Returns: None
    """
    #psin axis is average across times for each channel
    psin_ave = [np.sqrt(np.average(psin[:,i-1])) for i in ch_num]


    def bright(x, y):
        """
        Finds the brightness at a particular time and psin

        Parameters:
        x: time of sample from times list
        y: psin of sample from psin_ave list

        Returns:
        Brightness at specified time and space if below 1.2. If above 1.2 then likely junk, is thrown out for the plot
        """
        x = list(times).index(x)
        y = list(psin_ave).index(y)

        if lint_br[x,y] < 1.2:
            return lint_br[x,y]
        else:
            return 0

    #Create a grid of the times and psin_ave
    X, Y = np.meshgrid(times, psin_ave)

    #Allow function to take vector as a parameter and iterate through values
    fn_vectorized = np.vectorize(bright)
    Z = fn_vectorized(X, Y)

    fig = plt.figure(figsize=(20,10))
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='terrain', edgecolor=None)
    ax.set(xlabel="Time (sec)", ylabel="psin", zlabel="Brightness", zlim = [0, 1.2], title= 'Z Line Brightness')

    plt.show()

def bright_RF_model(sp_chn, lint_times, lint_br, lint_br_unc, RFtimes=[], RFdata=[], masks = [], shot = 0, exp_model= True, show_plot = False, argon = False):
    """
    Makes a plot of the brightness and RF power over time for a given spatial channel, fitting data over different
    RF steps

    Parameters:
    sp_chn (int): Spatial channel to be analyzed
    times (array): List of times to be used as time axis for brightness
    lint_br (array): 2D array of line-integrated brightness over time, space
    RFtimes (array): List of times to be used as time axis for RF power
    RFdata (array): List of RF power over time
    exp_model (bool): Fit the brightness profile using an exponential model (default: False)
    show_plot (bool): Show a plot of the data (default: False)
    masks (array): List of masks over which to fit models
    shot (int): Shot number

    Returns:
    Linear map: list of slopes over RF steps
    Exponential map: list of time scales of decay/growth over RF steps
    """
   
    #Model types for fit
    def exponential(x, A, r):
        cx, cy = 0, 0
        return cy + A*np.e**(r*(x+cx))
    
    def linear(x, m, b):
        return m*x + b

    #Restrict data and uncertainty to given spatial channel
    lint_data = lint_br[:,sp_chn]
    lint_unc = lint_br_unc[:,sp_chn]    

    #Take ln scale of brightness data if using the exponential model
    if exp_model:
        lint_data = np.log(lint_data)

    if show_plot:
        fig, ax = plt.subplots()

        #Plot errorbar of brightness over time
        ax.errorbar(lint_times, lint_data, lint_unc, linewidth=1, color='red', fmt = 'o', capsize=3, label='Brightness')
        ax.tick_params(axis='y', labelcolor ='red')
        ax.set_xlabel('Time (sec)', fontsize = 14)
        if exp_model:
            ax.set_ylabel('ln(Z-line Brightness)', color = 'red', fontsize = 14)
        else:
             ax.set_ylabel('Z-line Brightness', color = 'red', fontsize = 14)

        #Plot the RF Power over time
        ax2 = ax.twinx()
        ax2.plot(RFtimes, RFdata, color = 'black', label = 'RF Power')
        ax2.set_ylabel('RF Power (MW)', fontsize = 14)

        if argon:
            ax3 = ax.twinx()
            ax3.plot(argon_time, argon_pulse, '-', color = 'magenta', label = 'Argon Pump In')
            ax3.set_xlim([0, lint_times[-1]])
            #ax3.set_ylabel('Argon Pump In', color = 'magenta', fontsize = 14)
            ax3.tick_params(axis = 'y', labelcolor = 'magenta')

        fig.legend()
    #Generate model fits for each region of RF step. Exponential model is calculated using ln(brightness)
    slopes = []
    for i, mask in enumerate(masks):
        start = mask[0]
        stop = mask[-1] + 1
        try:
            popt, pcov = opt.curve_fit(f=linear, xdata=lint_times[start:stop], ydata=lint_data[start:stop], sigma=lint_unc[start:stop], absolute_sigma=True)
            slopes.append(popt[0])

            if show_plot:
                ax.plot(lint_times[start:stop], linear(lint_times[start:stop], popt[0], popt[1]), linewidth = 2, label = 'm = {0}'.format(str(popt[0].round(2))))
        except:
            pass
    slopes = np.array(slopes)

    if show_plot:
        plt.title('Pumpout Model Fit')
        fig.legend()
        # plt.savefig('Shot {0} Channel {1}.png'.format(shot, str(sp_chn)))
        plt.show()
        plt.close()
    
    #Return slopes if linear, signed time scale if exponential
    if exp_model:
        return slopes**(-1)
    else:
        return slopes

def fit_models_plot(masks, ch_num, times, lint_br, lint_br_unc, RFtimes, RFdata, shot):
    """
    Fits exponential models to the temporal brightness profile over regions in masks and saves figures. Prints signed time scale of the first mask to compare across channels,
    prints the average and standard deviation signed time scale for each region

    Parameters:
    masks (array): List of masks over which to fit models
    ch_num (array): List of channel numbers for which to fit models
    times (array): List of times to be used as time axis for brightness
    lint_br (array): 2D array of line-integrated brightness over time, space
    RFtimes (array): List of times to be used as time axis for RF power
    RFdata (array): List of RF power over time
    shot (int): Shot number
    """

    slopes_list = []
    for n in ch_num:
        slopes_list.append(bright_RF_model(n-1, times, lint_br, lint_br_unc, RFtimes, RFdata, masks, shot, show_plot = True))

    slopes_list = np.array(slopes_list)
    averages = []
    stds = []
    for i in range(len(slopes_list[0,:])):
        slope_ave = np.average(slopes_list[:, i])
        slope_std = np.std(slopes_list[:,i])
        averages.append(slope_ave)
        stds.append(slope_std)

    print('First Region Timescales: ', slopes_list[:,0])
    print('Average Timescales: ', averages)
    print('Standard Deviation Timescales: ', stds)


# Get spatial distribution of time scales for pumpout steps in 019
def timesc_chn_plot (masks, ch_num, times, lint_br, lint_br_unc, RFtimes, RFdata, shot):
    """
    Prints a plot of the timescale of pumpout or return phenomena over channel for different regions

    Parameters:
    masks (array): List of masks over which to fit models
    ch_num (array): List of channel numbers for which to fit models
    times (array): List of times to be used as time axis for brightness
    lint_br (array): 2D array of line-integrated brightness over time, space
    RFtimes (array): List of times to be used as time axis for RF power
    RFdata (array): List of RF power over time
    shot (int): Shot number
    """
    for mask in masks:

        #Calculate timescales over all channels
        time_scales = []
        for n in ch_num:
            scale = bright_RF_model(n-1, times, lint_br, lint_br_unc, RFtimes, RFdata, [mask], exp_model = True)[0]
            time_scales.append(scale)

        fig, ax = plt.subplots()

        #Calculate pearson correlation coefficient between timescale and brightness
        # corr, _ = pearsonr(np.abs(time_scales), lint_br[mask[0],2:])

        #Title figure with pumpout or return based on phenomenon
        title = 'Timescale and Brightness over Spatial Channel ({0}s - {1}s)'.format(str(times[mask[0]].round(2)), str(times[mask[1]].round(2)))
        if np.average(time_scales) < 0:
            title1 = 'Pumpout '
        else:
            title1 = 'Return '
        fig.suptitle(title1 + title)

        time_scales = np.abs(np.array(time_scales))

        print('Average Time Scale: ', np.average(time_scales))
        print('Standard Deviation of Time Scale: ', np.std(time_scales))

        #Plot Timescales over channel
        ax.plot(ch_num, time_scales, 'k.', label = 'Time Scale')
        ax.tick_params(axis='y', labelcolor ='black')
        ax.set_xlabel('Channel', fontsize = 14)
        ax.set_ylabel('Timescale (s)', fontsize = 14, color = 'black')# plt.show()

        #Plot brightness over channel for each temporal channel in the mask region
        ax2 = ax.twinx()
        for i in range(mask[0],mask[1]+1):
            # print(times[i].round(2))
            ax2.plot(ch_num, lint_br[i, 2:], label = 'Brightness at time {0}s'.format(str(times[i].round(2))))
        ax2.tick_params(axis = 'y', labelcolor = 'black')
        ax2.set_ylabel('Brightness', fontsize = 14, color = 'black')

        fig.legend()
        plt.savefig('{0} {1} Time and Bright vs Chn {2} {3}.png'.format(str(shot)[-3:], title1, str(times[mask[0]].round(2)), str(times[mask[1]].round(2))))
        plt.show()

def bright_change (chn, mask, lint_br):
    """
    Compute the brightness change over a given time for at a given spatial channel

    Parameters:
    chn (int): Spatial channel at which to examine change
    mask (array): Temporal mask over which to examine brightness change
    lint_br (2D array): lint_br (array): 2D array of line-integrated brightness over time, space
    """

    start = mask[0]
    stop = mask[1]

    br_pre = lint_br[start, chn]
    br_post = lint_br[stop, chn]

    change = (br_pre - br_post)/br_pre
    return change

def time_bright_plot(masks, ch_num, times, lint_br, lint_br_unc):
    """
    Create a plot of timescale/brightness change over channel for given segments of time

    Parameters:
    masks (array): List of masks over which to fit models
    ch_num (array): List of channel numbers for which to fit models
    times (array): List of times to be used as time axis for brightness
    lint_br (array): 2D array of line-integrated brightness over time, space
    """
    for mask in masks:

        #Calculate timescales and brightness change over all channels
        time_bright = []
        for n in ch_num:
            scale = bright_RF_model(n-1, times, lint_br, lint_br_unc, RFtimes, RFdata, [mask], exp_model = True)[0]
            bright = bright_change(n-1, mask, lint_br)
            time_bright.append(scale/bright)



        fig, ax = plt.subplots()

        #Calculate pearson correlation coefficient between timescale/brightness and brightness
        corr, _ = pearsonr(np.abs(time_bright), lint_br[mask[1],2:])

        #Title figure with pumpout or return based on phenomenon
        title = 'Timescale/Brightness Change over Spatial Channel ({0}s - {1}s), r = {2}'.format(str(times[mask[0]].round(2)), str(times[mask[1]].round(2)), str(corr.round(2)))
        if np.average(time_bright) < 0:
            title1 = 'Pumpout '
        else:
            title1 = 'Return '
        fig.suptitle(title1 + title)

        time_bright = np.array(time_bright)

        #Plot Timescales over channel
        ax.plot(ch_num, np.abs(time_bright), 'r.', label = 'Time Scale/Brightness Change')
        ax.tick_params(axis='y', labelcolor ='black')
        ax.set_xlabel('Channel', fontsize = 14)
        ax.set_ylabel('Timescale/Brightness Change', fontsize = 14, color = 'black')

        #Plot brightness over channel for each temporal channel in the mask region
        ax2 = ax.twinx()
        for i in range(mask[0],mask[1]+1):
            print(times[i].round(2))
            ax2.plot(ch_num, lint_br[i, 2:], label = 'Brightness at time {0}s'.format(str(times[i].round(2))))
        ax2.tick_params(axis = 'y', labelcolor = 'black')
        ax2.set_ylabel('Brightness', fontsize = 14, color = 'black')

        plt.savefig('{0} {1} Time.Bright vs Chn {2} {3}.png'.format(str(shot)[-3:], title1, str(times[mask[0]].round(2)), str(times[mask[1]].round(2))))
        plt.show()


# time_bright_plot([[12,16]], ch_num[2:], times, lint_br, lint_br_unc)
#bright_sptime(psin, times, ch_num, lint_br)
# print(bright_RF_model(15, times, lint_br, lint_br_unc, RFtimes, RFdata, show_plot = True, argon = True, masks = [[5,12]]))
timesc_chn_plot([[5, 12]], ch_num[2:], times, lint_br, lint_br_unc, RFtimes, RFdata, shot)

# masks_019 = [[5, 9], [10, 13], [14, 16], [17, 20], [21, 24], [25, 26], [27, 29], [30, 33], [34,36], [37,40], [41, -2]] #Regions of constant RF for 019
# masks_016 = [[12, 16], [17, -2], [12, 19], [20, -2]] #[12, 16] or [12,17] is the pumpout
#ch_num = ch_num[2:]
