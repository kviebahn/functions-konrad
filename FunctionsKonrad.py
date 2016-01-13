'''
Written by Konrad on
2015-05-27 Wed 05:56 AM
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.cm as cm
import csv
from scipy.optimize import curve_fit
#print help(curve_fit)

from scipy.interpolate import interp1d
from scipy.interpolate import Rbf

from matplotlib import rc


import phys_const as pc
reload(pc)
import K39_const as K39
reload(K39)








#fit functions
def gaussian(x, sigma, mu, ampl = 1, offset = 0):
    #if len(params)<4:
    #    print 'Error: gaussian takes exactly 4 arguments'
    #    return 1
    #else:
        #sigma, mu, ampl, offset = params[0], params[1], params[2], params[3]
    return ampl/(2*np.pi*sigma**2)*np.exp(-0.5*((x-mu)/sigma)**2)+offset


def parabolaGaussian(x, s1, m1,a1, s2, m2, a2):
    return parabola(x, s1, m1, a1)+gaussian(x, s2,m2, a2)


def lorentzian(x, gamma, x0, ampl, offset = 0):
    return ampl/((x-x0)**2 + gamma**2)+offset

def dampedRabi(x, freq, damping, ampl):
#    if len(params)<3:
#        print 'Error: dampedRabi takes exactly 3 arguments'
#        return 1
#    else:
#        freq, damping, ampl = params[0], params[1], params[2]
    return ampl/2.*(-np.cos(2*np.pi*freq*x)*np.exp(-x/damping)+1.)

def undampedRabi(x, freq, ampl):
    return ampl*(-np.cos(2*np.pi*freq*x)+1.)

#def dampedRabi2(x, freq, damping, ampl):
#    return ampl*(np.sin(np.pi*freq*x))**2*(np.exp(-x/damping)+1.)/2.

def exponential(x, decay, ampl, offset):
#    if len(params)<3:
#        print 'Error: exponential takes exacly 3 arguments'
#        return 1
#    else:
#        decay, ampl, offset = params[0], params[1], params[2]
    return ampl*np.exp(-x/decay) + offset

def fixedExponential(x,decay, ampl = 950.):
    return ampl*np.exp(-x/decay)

def beating(x,theta, omega_perp, omega_para, ampl, offset):
#    if len(params) < 7:
#        print 'Error: wrong number of arguments'
#    else:
#        theta, omega_perp, phi_perp, omega_para, phi_para, ampl, offset = params[0], params[1],params[2],params[3],params[4],params[5],params[6]
    return ampl*(np.cos(theta)*np.cos(omega_para*x) + np.sin(theta)*np.cos(omega_perp*x))+offset

def feshbachResonance(V, V_res = 3.8039):
    '''returns scattering length for K39 from input voltage'''
    return  -29.*(1. + (52.*V_res/402.5)/(V - V_res))

def invFeshbachResonance(a,V_res = 3.8039):
    return 0.00198807157057*(12702 + 503*a)*V_res/(29+a)

def parabola(x,ampl, x_offset, y_offset):
    return ampl*(x-x_offset)**2+y_offset

def linear(x,slope,offset):
    return slope*x + offset




def atomLossPiPulse(tau,Rabifreq):
    '''takes tau/s Rabi freq/Hz and returns proportion of atoms lost in one pi pulse'''
    return 1.-np.exp(-1./(tau*2.*Rabifreq))

def atomLossPiPulseDeriv(a1,a2):
    return 1./(2.*a1**2.*a2)*np.exp(-1./(2.*a1*a2))

def quadraticSum(x1,x2):
    return np.sqrt(x1**2+x2**2)







#round to one significant digit
def round_to_1(x):
    return np.around(x, -np.int(np.floor(np.log10(x))))



def kBragg(theta = 96.75, wavelen = K39.lD2):
    return 2.*np.sin(theta/2.*np.pi/180.)*2*np.pi/wavelen 

def trapFreq(trap_x, trap_y, trap_z):
    return (trap_x*trap_y*trap_z)**(1./3.)

def oscLength(trap_freq, mass = K39.m):
    '''takes trap_freq in Hz (NOT in Hz * 2pi)'''
    return np.sqrt(pc.hbar/(2*np.pi*trap_freq*mass))

def TFradius(N_atom, scatt_len, trap_freq, mass = K39.m):
    return oscLength(trap_freq, mass)*(15.*N_atom*scatt_len/oscLength(trap_freq, mass))**(1./5.)

def TFpeakDensity(N_atom, scatt_len, trap_freq, mass = K39.m):
    '''returns peak density n0'''
    return 15./(8.*np.pi)*N_atom/(TFradius(N_atom, scatt_len, trap_freq, mass = K39.m))**3.

def g(a_scatt, m = K39.m):
    return 4.*np.pi*pc.hbar**2.*a_scatt/m

def muMF(n,a_scatt, m = K39.m):
    '''Takes average density n. '''
    return 4.*np.pi*pc.hbar**2.*n*a_scatt/m

def LHY(n0, a_scatt):
    '''Takes peak density n0'''
    return 32./(3.*np.sqrt(np.pi))*n0**(3./2.)*np.pi*75./512.*a_scatt**(3./2.)*g(a_scatt)


def MFwidth(n0,a_scatt):
    return np.sqrt(8./147.)*n0*g(a_scatt)/pc.h

def finiteSizeWidth(k_Bragg, radius, m = K39.m):
    return np.sqrt(21./8.)*pc.hbar*k_Bragg/(2*np.pi*m*radius)

def gaussianWidth(k_Bragg, trap_freq, m = K39.m):
    return pc.hbar*k_Bragg/(2.**(3./2.)*np.pi*m*oscLength(trap_freq))

def combineWidths(w1,w2,alpha = 2.):
    return (1./(w1)**alpha+1./(w2)**alpha)**(-1./alpha)

def addWidths(w1,w2):
    return np.sqrt(w1**2+w2**2)

def fourierWidth(t_Bragg):
    '''takes t_Bragg in s'''
    return 0.36/t_Bragg 


def importFromTxt(file_name, single_import=True):
      
    #initialize arrays
    x_array = np.array([])
    y_array = np.array([]) 
    my_file = open(file_name)
    
    reader = csv.reader(my_file, dialect='excel', delimiter='\t')
    for row in reader:
        #print row
        if single_import:
            y_array = np.append(y_array, np.double(row[0]))
        else:           
            x_array = np.append(x_array, np.double(row[0]))
            y_array = np.append(y_array, np.double(row[1]))
    my_file.close()
    
    #print x_array
    #print y_array
    return x_array, y_array


def importMultipleFromTxt(file_name):
      
    #initialize array
    dummy_array = np.array([])
    my_file = open(file_name)
    
    reader = csv.reader(my_file, dialect='excel', delimiter='\t')
    count = 0
    my_len = 0
    for row in reader:
        print row
        print len(row)
        if count == 0:
            dummy_array = np.append(dummy_array,np.array(row).astype(np.float))
            count += 1
            my_len = len(row)
        else:
            if (len(row) == my_len)*('' not in row[:my_len]):
                dummy_array = np.vstack((dummy_array,np.array(row[:my_len]).astype(np.float)))
    
    my_file.close()
    return dummy_array


def sortDataGetErrors(value_range, x_data, y_data):
     
    #the plot arrays
    x_plot = np.array([])
    y_mean = np.array([])
    y_std = np.array([])
    for i0 in value_range:
        #print i0
        #print np.where(np.round(i0,3)==np.round(x_data,3))
        coord = np.where(np.round(i0,3)==np.round(x_data,3))
        if (len(coord[0])>0):
            x_plot = np.append(x_plot, i0)
            y_mean = np.append(y_mean, np.mean(y_data[coord[0]]))
            y_std = np.append(y_std, np.std(y_data[coord[0]]))
        else:
            pass
            #print 'hello'
            
    print x_plot, y_mean, y_std
    return x_plot, y_mean, y_std


def fitData(x_data, y_data, fit_type, initial_guess):
      
    a,b = curve_fit(fit_type, x_data, y_data, p0 = initial_guess)
    #print a, b
    return a, np.sqrt(np.diag(b))


def plotData(axes_instance, x_data_array, y_data_array):
    axes_instance.plot(x_data_array,y_data_array, 'bo')
    return axes_instance

def plotCurve(axes_instance, fit_type, value_range, params):
    if len(params) > 8:
        print 'plotCurve: too many arguments'
    if len(params) == 8:
        axes_instance.plot(value_range, fit_type(value_range,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]), 'r-')
    if len(params) == 7:
        axes_instance.plot(value_range, fit_type(value_range,params[0],params[1],params[2],params[3],params[4],params[5],params[6]), 'r-')
    if len(params) == 6:
        axes_instance.plot(value_range, fit_type(value_range,params[0],params[1],params[2],params[3],params[4],params[5]), 'r-')
    if len(params) == 5:
        axes_instance.plot(value_range, fit_type(value_range,params[0],params[1],params[2],params[3],params[4]), 'r-')
    if len(params) == 4:
        axes_instance.plot(value_range, fit_type(value_range,params[0],params[1],params[2],params[3]), 'r-')
    if len(params) == 3:
        axes_instance.plot(value_range, fit_type(value_range,params[0],params[1],params[2]), 'r-')
    if len(params) == 2:
        axes_instance.plot(value_range, fit_type(value_range,params[0],params[1]), 'r-')
    if len(params) == 1:
        axes_instance.plot(value_range, fit_type(value_range,params[0]), 'r-')
    return axes_instance


def getImportParameters(camera_type):
    '''returns importMultiply, importAdd, importLogOffset'''
    if camera_type == 'PIXIS':
        return np.array([2.0**(-12.0), -2.0, -0.5])
    if camera_type == 'pixelfly':
        return np.array([2.0**(-8.0), -8.0, 0.0]) 

def getImagePath(camera_type, date, letter, num_img):
    '''return image path'''
    my_directory = '/Volumes/shared/BEC 2 Data/Images/'
    if camera_type == 'PIXIS':
        image_path =  my_directory + date[0:4] + '/'+ date + '/' + letter + '_' + str(num_img) + '_OD.tif' 
    if camera_type == 'pixelfly':
        my_str = '000' + str(num_img)
        image_path = my_directory + date[0:4] + '-' + date[4:6] + '-' + date[6:8] + '/OD_Processed/OD_' + my_str[len(my_str)-3:len(my_str)] + '.tif'  
    print image_path
    return image_path

def importImage(image_path, camera_type):
    '''returns numpy array of image'''
    
    import_params = getImportParameters(camera_type)
    OD_raw = img.imread(image_path).astype(np.float64) 
    OD_raw[OD_raw>10000] = 10000
    OD_processed = np.real(-np.log(np.exp(-(OD_raw*import_params[0] + import_params[1])) + import_params[2]))
    return OD_processed


def viewImage(img_array):
    imgplot = plt.imshow(img_array, interpolation = 'nearest', vmin = 0, vmax = 3)
    plt.colorbar()
    plt.show()
    plt.close()
    return 0

def getROI(array, ROI):
    array_reshape = array[ROI[0]:ROI[1], ROI[2]:ROI[3]]
    return array_reshape 

def locateCentre(img_array, plot_bool = False):
    
    y_max = np.where(np.round(img_array, 3)==np.amax(np.round(img_array, 3)))[0][0]
    x_max = np.where(np.round(img_array, 3)==np.amax(np.round(img_array, 3)))[1][0]
    
    y_axis_data = np.sum(img_array[:, x_max-10:x_max+10], 1)
    x_axis_data = np.sum(img_array[y_max-10:y_max+10,:], 0)
    y_axis_range = np.arange(0, len(y_axis_data),1)
    x_axis_range = np.arange(0, len(x_axis_data),1)
    initial_guess_y = np.where(y_axis_data==np.amax(y_axis_data))[0][0]
    initial_guess_x = np.where(x_axis_data==np.amax(x_axis_data))[0][0]
    y_params, y_errs = fitData(y_axis_range, y_axis_data, fit_type = gaussian, initial_guess = [5, initial_guess_y, np.amax(y_axis_data)-np.mean(y_axis_data), np.mean(y_axis_data)])
    x_params, x_errs = fitData(x_axis_range, x_axis_data, fit_type = gaussian, initial_guess = [5, initial_guess_x, np.amax(x_axis_data)-np.mean(x_axis_data), np.mean(x_axis_data)])
    if plot_bool:
        fig = plt.figure()
        ax0 = fig.add_subplot(211)
        ax0.plot(y_axis_range, y_axis_data,'bo')
        ax0.set_title('y axis')
        plotCurve(ax0, gaussian, y_axis_range, y_params)
        ax1 = fig.add_subplot(212)
        ax1.plot(x_axis_range, x_axis_data,'go') 
        ax1.set_title('x axis')
        plotCurve(ax1, gaussian, x_axis_range, x_params) 
        plt.show()
    y_params[0] = np.abs(y_params[0])
    x_params[0] = np.abs(x_params[0])
    if y_params[0]>15 or x_params[0]>15:
        return 1
    else:
        return np.round(y_params[1],0), np.round(x_params[1],0) 
    
def saveArray(save_directory, save_name, array):
    np.save(save_directory + 'temp/' + save_name, array)
    return 0

def loadArray(directory_name, load_name):
    dummy_array = np.array([])
    print directory_name + 'temp/' + load_name + '.npy'
    dummy_array = np.load(directory_name + 'temp/' + load_name + '.npy')
    return dummy_array

def makeAxesLabels(axes_instance, title, x_label, y_label):    
    axes_instance.set_title(title,fontsize=16)
    axes_instance.set_xlabel(x_label)
    axes_instance.set_ylabel(y_label)
    return axes_instance

def makeAxesLegend(axes_instance, legend_string):
    axes_instance.legend(['Data', legend_string], labelspacing=1.5, fancybox = True, shadow = True)#'lower left', bbox_to_anchor=(0.6, 0.5))    
    return axes_instance

def makeZeroAxis(ax):
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    return ax


def saveFigure(figure_instance, save_name, save_directory_mac, save_directory_shared):

    dummy_name = save_directory_mac + save_name + '.eps'
    figure_instance.savefig(dummy_name, format = 'eps')
    #figure_instance.savefig(save_directory_shared + 'png/' + save_name +'.png', dpi = 300, format = 'png')
    return figure_instance

def saveFigurePDF(figure_instance, save_name, save_directory_mac, save_directory_shared):

    dummy_name = save_directory_mac + save_name + '.pdf'
    figure_instance.savefig(dummy_name, format = 'pdf')
    #figure_instance.savefig(save_directory_shared + 'png/' + save_name +'.png', dpi = 300, format = 'png')
    return figure_instance

def saveToMasterArray(save_directory,save_name, master_save_name, array):
    '''save_name must contain current master value eg 20150527_spectro_3p35 for fesh = 3.35'''
    master_value = np.float(save_name[-4:-3]+ '.'+ save_name[-2:]) 
    
    #master_array_name = save_directory + 'temp/' + save_name[:-5] + '.npy' 
    master_array_name = save_directory + 'temp/' + master_save_name  + '.npy' 
    
    dummy_line = np.hstack((np.array([master_value]), array))
    
    
    if os.path.isfile(master_array_name):
        dummy_array = loadArray(save_directory, master_save_name)
        #print dummy_array
        dummy_array = np.vstack((dummy_array, dummy_line))
        #print dummy_array
        saveArray(save_directory, master_save_name, dummy_array)
    else:
        print 'hello'
        saveArray(save_directory, master_save_name, dummy_line) 
    
    return 0

def saveMasterArray(save_directory,master_save_name, array):
    '''giving master value as input argument'''
    
    master_array_name = save_directory + 'temp/' + master_save_name  + '.npy' 
    
    dummy_line = array 
    
    if os.path.isfile(master_array_name):
        dummy_array = loadArray(save_directory, master_save_name)
        #print dummy_array
        dummy_array = np.vstack((dummy_array, dummy_line))
        #print dummy_array
        saveArray(save_directory, master_save_name, dummy_array)
    else:
        print 'hello'
        saveArray(save_directory, master_save_name, dummy_line) 
    
    return 0

def selectFromArray(x_array, y_array, z_array, z0, z1):
    '''
    returns a reduced x_array and y_array, based on condition where z0 < z_array < z1
    '''
    mask = (np.round(z_array, 4) > np.round(z0,4))*(np.round(z_array, 4) < np.round(z1,4))
    print np.where(mask == 0.)[0]
    indicies = np.where(mask == 0.)[0]
    new_y_array = np.delete(y_array, indicies)
    new_x_array = np.delete(x_array, indicies)
    return new_x_array, new_y_array

def selectFromArrayBool(array, cond_array, cond):
    '''
    returns a reduced x_array and y_array, based on condition where z_array == cond
    '''
    print np.where(np.round(cond_array,6) != np.round(cond,6))[0]
    indicies = np.where(cond_array != cond)[0]
    new_array = np.delete(array, indicies, axis = 0)
    return new_array

def interpolateData(x,y):
    return interp1d(x,y,kind = 'slinear')

def interpolateGaussian(x,y):
    #d = 1*np.ones((len(x),))
    return Rbf(x,y, smooth = 0.001)

























if __name__ == '__main__':
     
    my_file = '20150606_fesh_3p75'
    my_master_name = '20150606_resonances_incl_atom_num'
    #my_data_directory = '/Volumes/shared/BEC 2 Data/Images/2015/20150522/'
    my_date = '20150527'
    my_letter = 'A'
    my_save_directory_mac = '/Volumes/Macintosh HD/Users/Konrad/UNI/ZH/data/'
    my_save_directory_shared = '/Volumes/shared/BEC 2 Data/LabBook/Bragg_alignment/'
#    my_data = importMultipleFromTxt(my_save_directory_shared+'txt/'+my_file+'.txt') 

    x_name = 'Bragg freq/kHz'
    y_name = 'Transferred fraction'
    
#    saveArray(my_save_directory_mac, my_file, my_data)
     
    my_data = loadArray(my_save_directory_mac, my_file)
#    print my_data
  

    px0,px1 = np.mean(my_data.T[11]), np.std(my_data.T[11])
    print px0,px1
 
    max_diffr_p, max_diffr_m = np.amax(my_data.T[8]), np.amax(my_data.T[9])
    
#    new_x, new_x0 = selectFromArray(my_data.T[0], my_data.T[0], np.abs(my_data.T[8] - my_data.T[9]-(max_diffr_p - max_diffr_m)), 0, 0.07)
#    new_p, new_m = selectFromArray(my_data.T[8], my_data.T[9], np.abs(my_data.T[8] - my_data.T[9]-(max_diffr_p - max_diffr_m)), 0, 0.07)
    new_x, new_p = selectFromArray(my_data.T[0], my_data.T[8], my_data.T[11], px0-px1, px0+px1)
    new_x, new_m = selectFromArray(my_data.T[0], my_data.T[9], my_data.T[11], px0-px1, px0+px1)

    print new_x,new_p, new_m







    my_fit_type = gaussian
    my_initial_guess = np.array([1.5, 20,1,0.])
    start = 0
    end = 40
    step = 0.1

    value_range = np.arange(start,end,step)
   
    my_x_plot_p, my_y_mean_p, my_y_std_p = sortDataGetErrors(value_range, new_x, new_p)
    my_x_plot_m, my_y_mean_m, my_y_std_m = sortDataGetErrors(value_range, new_x, new_m)
    

    my_params_p, my_params_err_p = fitData(my_data.T[0], my_data.T[8], fit_type = my_fit_type, initial_guess = my_initial_guess)
    print my_params_p, my_params_err_p
    my_params_m, my_params_err_m = fitData(my_data.T[0], my_data.T[9], fit_type = my_fit_type, initial_guess = my_initial_guess)
    print my_params_m, my_params_err_m
    
    my_save_params = np.hstack((my_data.T[1][0], px0, px1, my_params_p, my_params_err_p, my_params_m, my_params_err_m, np.mean(my_data.T[10]), np.std(my_data.T[10])))
    print my_save_params

    saveMasterArray(my_save_directory_mac, my_master_name, my_save_params) 
    
    fig = plt.figure()
    
    ax_p = fig.add_subplot(211)
    plt.ylim(ymin = 0, ymax = 0.35)
    ax_m = fig.add_subplot(212)

    plt.ylim(ymin = 0, ymax = 0.35)
    ax_p.errorbar(my_x_plot_p, my_y_mean_p, my_y_std_p, fmt = 'bo')
    ax_m.errorbar(my_x_plot_m, my_y_mean_m, my_y_std_m, fmt = 'ro')
    ax_p = plotCurve(ax_p, my_fit_type, value_range, my_params_p)
    ax_m = plotCurve(ax_m, my_fit_type, value_range, my_params_m)
    


    #plt.ylim(ymin = 0, ymax = 0.35)
    #plt.ylim(ymin = 0,ymax = np.amax(my_y_mean_p))
    #plt.xlim(xmin = start, xmax = end)

    makeAxesLabels(ax_p,my_file + 'selecting only average clouds', '', y_name + '(plus first)')
    makeAxesLabels(ax_m,'', x_name, y_name+ '(minus first)')
    #ax.hist(my_data.T[4], 25)
    my_legend_string_p = 'Fitted gaussian: \n sigma = %2.2f (%1.1f) kHz\n mu = %1.2f (%1.2f)' % (my_params_p[0], round_to_1(my_params_err_p[0]), my_params_p[1], my_params_err_p[1])
    my_legend_string_m = 'Fitted gaussian: \n sigma = %2.2f (%1.1f) kHz\n mu = %1.2f (%1.2f)' % (my_params_m[0], round_to_1(my_params_err_m[0]), my_params_m[1], my_params_err_m[1])





#    my_legend_string = 'Fitted rabi: \n freq = %1.3g (%1.1g) MHz \n damping = %2.f (%1.f) us' % (my_params[0],round_to_1(my_params_err[0]), my_params[1], round_to_1(my_params_err[1]))
#    my_legend_string = 'Fitted exp: \n tau = %3.0f (%1.0f) us' % (my_params[0],round_to_1(my_params_err[0]))

    makeAxesLegend(ax_p, my_legend_string_p)
    makeAxesLegend(ax_m, my_legend_string_m)
   
   
    saveFigure(fig, my_file, my_save_directory_mac, my_save_directory_shared)
   

    plt.show()
    
    plt.close('all')


#    camera_type = 'PIXIS'
#
#    first_img = 47
#    last_img = 104
#
#    preview_num_img = 47
#    
#    dead_shots = np.array([70,71,74,87,88,91,100])
#    choose_ROI = True
#    
#    #ROI in format [y_start, y_stop,x_start, x_stop]
#    ROI = np.array([200,700,200,700])
#    
#    my_x_data = np.array([])
#    my_y_data = np.array([])
#    
#    for i0 in np.arange(first_img, last_img+1,1):
#        #print i0    
#        
#        if i0 in dead_shots:
#            pass
#        else:
#        
#            my_image_path = getImagePath(camera_type, my_date, my_letter, i0)
#            my_array = importImage(my_image_path, camera_type)
#            my_array_reshape = getROI(my_array, ROI)
#            my_y, my_x = locateCentre(my_array_reshape, plot_bool = False)
#            my_x_data = np.append(my_x_data, my_x)
#            my_y_data = np.append(my_y_data, my_y)
#            #print my_y,my_x

