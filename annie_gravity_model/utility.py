import pathlib
import numpy as np
import scipy.io as io
import scipy.signal as sig
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def load_datasets(data_dir, data_prm, plot_prm): 
    """ 
    Extract data sections with dphi=constant for each eta 

    Arguments:
      
      data_dir = location of kinematics data files
      data_prm = dictionary containing fit prmeters
      plot_prm = dictionary containing plotting options

    """
    data_files = get_list_of_data_files(data_dir, v=data_prm['v'], xi=data_prm['xi'])
    alphas, data_files = sort_data_files_by_alpha(data_files)
    datasets = {}

    print('loading datasets')
    for alpha, file in zip(alphas, data_files):
        print(f'  {file}')
        data = io.loadmat(str(file))

        # Extract the data we need for polars
        t = data['t_FT_f'][:,0]
        eta = data['wingkin_f'][:,2]
        phi = data['wingkin_f'][:,3]
        dphi = data['wingkin_f'][:,9]
        fx = data['FT_conv_f'][0,:]
        fy = data['FT_conv_f'][2,:]
        fz = data['FT_conv_f'][1,:]

        # Cut out sections between t_lim[0] and t_lim[1]

        if data_prm['t_lim'] is not None:
            mask_t_lim = np.logical_and(t >= data_prm['t_lim'][0], t <= data_prm['t_lim'][1])
            t = t[mask_t_lim]
            eta = eta[mask_t_lim]
            phi = phi[mask_t_lim]
            dphi = dphi[mask_t_lim]
            fx = fx[mask_t_lim]
            fy = fy[mask_t_lim]
            fz = fz[mask_t_lim]

        # Lowpass filter force data
        dt = t[1] - t[0]
        force_filt = sig.butter(4, data_prm['fcut'], btype='low', output='ba', fs=1/dt)
        fx_filt = sig.filtfilt(*force_filt, fx)
        fy_filt = sig.filtfilt(*force_filt, fy)
        fz_filt = sig.filtfilt(*force_filt, fz)

        # Optional plot showing filtered and unfilterd force data
        if plot_prm.setdefault('filtered_forces', False):
            plot_filtered_forces(t, fx, fy, fz, fx_filt, fy_filt, fz_filt, alpha)

        # Get sections where dphi is equal to maximum
        mask_pos = dphi >= dphi.max() 
        t_pos = t[mask_pos]
        eta_pos = eta[mask_pos]
        phi_pos = phi[mask_pos]
        dphi_pos = dphi[mask_pos]
        fx_filt_pos = fx_filt[mask_pos]
        fy_filt_pos = fy_filt[mask_pos]
        fz_filt_pos = fz_filt[mask_pos]

        # Get sections where dphi is equal to -maximum
        mask_neg = dphi <= dphi.min() 
        t_neg = t[mask_neg]
        eta_neg = eta[mask_neg]
        phi_neg = phi[mask_neg]
        dphi_neg = dphi[mask_neg]
        fx_filt_neg = fx_filt[mask_neg]
        fy_filt_neg = fy_filt[mask_neg]
        fz_filt_neg = fz_filt[mask_neg]

        # Save datasets as function of eta
        eta_pos_val = eta_pos.max()
        eta_neg_val = eta_neg.min()
        
        datasets[eta_pos_val] = { 
                't'    : t_pos, 
                'eta'  : eta_pos,
                'phi'  : phi_pos, 
                'dphi' : dphi_pos,
                'fx'   : fx_filt_pos,
                'fy'   : fy_filt_pos,
                'fz'   : fz_filt_pos, 
                }

        datasets[eta_neg_val] = { 
                't'    : t_neg, 
                'eta'  : eta_neg,
                'phi'  : phi_neg, 
                'dphi' : dphi_neg,
                'fx'   : fx_filt_neg,
                'fy'   : fy_filt_neg,
                'fz'   : fz_filt_neg, 
                }


        # Optional plot showing grab sections for gravitational model
        if plot_prm.setdefault('grab_sections', False):
            datafull = {
                    't'   : t, 
                    'eta' : eta, 
                    'phi' : phi, 
                    'dphi': dphi, 
                    'fx'  : fx_filt, 
                    'fy'  : fy_filt, 
                    'fz'  : fz_filt
                    }
            plot_grab_sections(datafull, datasets[eta_pos_val], datasets[eta_neg_val], alpha)

        if plot_prm.setdefault('force_pos_neg', False):
            plot_force_pos_neg(datasets[eta_pos_val], datasets[eta_neg_val], alpha)

    return  datasets


def get_list_of_data_files(data_dir, v=10, xi=0):
    """ Get list of relvant data files - filter by v and xi. """
    data_dir = pathlib.Path(data_dir)
    data_files = [item for item in data_dir.iterdir() if item.is_file]
    data_files = [item for item in data_files if f'_v_{v}' in item.name]
    data_files = [item for item in data_files if f'_xi_{xi}' in item.name]
    data_files.sort()
    return data_files

def sort_data_files_by_alpha(data_files): 
    """ 
    Sorts the data files by alpha where alpha is extracted from the data file name. It
    is assumed to the an integer coming after _alpha_ and before _xi_ in the file name.
    """
    alphas = []
    for item in data_files:
        n0 = item.name.find('_alpha_') + len('_alpha_')
        n1 = item.name.find('_xi_')
        alphas.append(int(item.name[n0:n1]))
    alphas, data_files = zip(*sorted(zip(alphas, data_files)))  
    return alphas, data_files


def plot_force_pos_neg(datapos, dataneg, alpha): 
    fg, ax = plt.subplots(3,1)
    ax[0].set_title(f'alpha = {alpha}')
    ax[0].set_ylabel('fx')
    ax[0].grid(True)
    ax[1].set_ylabel('fy')
    ax[1].grid(True)
    ax[2].set_ylabel('fz')
    ax[2].grid(True)
    ax[2].set_xlabel('t (sec)')
    for dataset, style in ((datapos, '.r'), (dataneg, '.b')):
        phi = dataset['phi']
        fx = dataset['fx']
        fy = dataset['fy']
        fz = dataset['fz']
        ax[0].plot(phi, fx, style)
        ax[1].plot(phi, fy, style)
        ax[2].plot(phi, fz, style)
    plt.show()


def plot_grab_sections(datafull, datapos, dataneg, alpha): 

    t = datafull['t']
    eta = datafull['eta']
    phi = datafull['phi']
    dphi = datafull['dphi']
    fx = datafull['fx']
    fy = datafull['fy']
    fz = datafull['fz']

    fg, ax = plt.subplots(6, 1, sharex=True)
    ax[0].set_title(f'alpha = {alpha}')
    ax[0].set_ylabel('eta')
    ax[0].grid(True)
    ax[1].set_ylabel('phi')
    ax[1].grid(True)
    ax[2].set_ylabel('dphi')
    ax[2].grid(True)
    ax[3].set_ylabel('fx_filt')
    ax[3].grid(True)
    ax[4].set_ylabel('fy_filt')
    ax[4].grid(True)
    ax[5].set_ylabel('fz_filt')
    ax[5].grid(True)
    ax[5].set_xlabel('t (sec)')

    for dataset, style in ((datafull, 'b'), (datapos, '.r'), (dataneg, '.g')):
        t = dataset['t']
        eta = dataset['eta']
        phi = dataset['phi']
        dphi = dataset['dphi']
        fx = dataset['fx']
        fy = dataset['fy']
        fz = dataset['fz']
        ax[0].plot(t, eta, style)
        ax[1].plot(t, phi, style)
        ax[2].plot(t, dphi, style)
        ax[3].plot(t, fx, style)
        ax[4].plot(t, fy, style)
        ax[5].plot(t, fz, style)
    plt.show()


def plot_filtered_forces(t, fx, fy, fz, fx_filt, fy_filt, fz_filt, alpha): 
    """
    Plot filtered raw and filtered forces.
    """
    fg, ax = plt.subplots(3, 1, sharex=True)
    ax[0].set_title(f'alpha = {alpha}')
    ax[0].plot(t, fx, 'b')
    ax[0].plot(t, fx_filt, 'r')
    ax[0].set_ylabel('fx')
    ax[0].grid(True)

    ax[1].plot(t, fy, 'b')
    ax[1].plot(t, fy_filt, 'r')
    ax[1].set_ylabel('fy')
    ax[1].grid(True)

    ax[2].plot(t, fz, 'b')
    ax[2].plot(t, fz_filt, 'r')
    ax[2].set_ylabel('fz')
    ax[2].grid(True)
    ax[2].set_xlabel('t (sec)')
    plt.show()


def plot_force_surfaces(eta, phi, fx, fy, fz):
    """
    Plots fx, fy, fz force surfaces as a function of eta and phi
    """

    figsize = (11,9)
    figsize = (8,6)
    fg, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)
    ax.set_title('fx surface')
    surf = ax.plot_surface(eta, phi, fx, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    ax.set_xlabel('eta')
    ax.set_ylabel('phi')
    ax.set_zlabel('fx')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    fg.colorbar(surf, shrink=0.5, aspect=5)

    fg, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)
    ax.set_title('fy surface')
    surf = ax.plot_surface(eta, phi, fy, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    ax.set_xlabel('eta')
    ax.set_ylabel('phi')
    ax.set_zlabel('fy')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    fg.colorbar(surf, shrink=0.5, aspect=5)

    fg, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)
    ax.set_title('fz surface')
    surf = ax.plot_surface(eta, phi, fz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    ax.set_xlabel('eta')
    ax.set_ylabel('phi')
    ax.set_zlabel('fz')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    fg.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


















