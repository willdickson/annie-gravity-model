import pathlib
import collections
import numpy as np
import scipy as sp
import scipy.io as io
import scipy.signal as sig
import scipy.interpolate as interp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


class GravityModel:
    """
    Model of gravitational forces acting on Robofly wing.  Use for gravitation
    force subtraction when analysing data from a set of polars. 
    """
    DEFAULT_FIT_PARAM = {
            'v'       : 10,   # velocity (int) in kinematics filename
            'xi'      :  0,   # xi value (int) in kinematics filenames  
            'tlim'    : None, # (tuple) lower/upper limits of time range 
            'fcut'    : 10.0, # time range (tuple) lower and upper bounds
            'num_phi' : 50,   # number of phi data points after resampling
            }

    DEFAULT_PLT_PARAM = {
            'force_surfaces'  : True,
            'filtered_forces' : False, 
            'grab_sections'   : False,
            'force_pos_neg'   : False,
            }

    def __init__(self):
        pass


    def load(self, filename):
        pass


    def fit(self, data_dir, fit_param=None, plt_param=None): 
        """
        Fit gravitational model using data polar from data_dir and parameters
        in fit_param.

        Arguments:

            data_dir = location of kinematics data files

        keyword Arguments:

            fit_param = dictionary containing fit parameters

            {
                'v'       :  # velocity (int) in kinematics filenames
                'xi'      :  # xi value (int) in kinematics filenames
                'tlim'    :  # time range (tuple) lower and upper bounds 
                'fcut'    :  # force data lowpass filter cutoff frequency
                'num_phi' :  # number of phi data points after resampling
            }

            plt_param = dictionary of plotting options.
            {
                'force_surfaces'   : True
                'force_filter'     : False, 
                'data_sections'    : False,
                'pos_neg_sections' : False,
            }

        """
        # Set default options and update with user options
        _fit_param = dict(self.DEFAULT_FIT_PARAM)
        if fit_param is not None:
            _fit_param.update(fit_param)

        _plt_param = dict(self.DEFAULT_PLT_PARAM)
        if plt_param is not None:
            _plt_param.update(plt_param)

        # Read in datasets and extract force sections
        datasets = self.extract_datasets(data_dir, _fit_param, _plt_param)

        # Create fx, fy, and fz force surfaces as functions of eta and phi (meshgrids)
        eta, phi, fx, fy, fz = self.create_force_surfaces(datasets, _fit_param['num_phi'])
        if _plt_param['force_surfaces']:
            plot_force_surfaces(eta, phi, fx, fy, fz)


    def create_force_surfaces(self, datasets, num_phi):
        """ 
        Creates surfaces for fx, fy, fz forces as functions of eta and phi 

        Arguments:

          datasets = dictionary mapping eta values to kinematics and forces. 

        Returns:

          eta = meshgrid of eta values  
          phi = meshgrid of phi values 
          fx  = component of forces in x direction (chordwise direction)
          fy  = component of forces in y direction (spanwise direction)
          fz  = component of forces in z direction (normal to wing surface)

        """
        # Get arrays of eta and phi values 
        eta_vals = np.array(sorted(datasets.keys()))
        phi_max = min([datasets[eta]['phi'].max() for eta in eta_vals])
        phi_min = max([datasets[eta]['phi'].min() for eta in eta_vals])
        phi_vals = np.linspace(phi_min, phi_max, num_phi)

        # Create meshgrid
        eta, phi = np.meshgrid(eta_vals, phi_vals)
        fx = np.zeros(eta.shape)
        fy = np.zeros(eta.shape)
        fz = np.zeros(eta.shape)

        for i, val in enumerate(eta_vals):
            data = datasets[val]
            fx_interp_func = interp.interp1d(data['phi'], data['fx'], kind='linear')
            fy_interp_func = interp.interp1d(data['phi'], data['fy'], kind='linear')
            fz_interp_func = interp.interp1d(data['phi'], data['fz'], kind='linear')
            fx[:,i] = fx_interp_func(phi_vals)
            fy[:,i] = fy_interp_func(phi_vals)
            fz[:,i] = fz_interp_func(phi_vals)
        return eta, phi, fx, fy, fz


    def extract_datasets(self, data_dir, fit_param, plt_param): 
        """ 
        Extract data sections with dphi=constant for each eta 

        Arguments:
          
          data_dir = location of kinematics data files
          fit_param = dictionary containing fit parameters
          plt_param = dictionary containing plotting options

        """
        data_files = self.get_list_of_data_files(data_dir, v=fit_param['v'], xi=fit_param['xi'])
        alphas, data_files = sort_data_files_by_alpha(data_files)
        datasets = {}

        print('loading datasets')
        for alpha, file in zip(alphas, data_files):
            print(f'  {file}')
            data = io.loadmat(str(file))

            # Extract the data we need for polars
            t = data['t_FT_s'][:,0]
            eta = data['wingkin_s'][:,2]
            phi = data['wingkin_s'][:,3]
            dphi = data['wingkin_s'][:,9]
            fx = data['FT_conv_s'][0,:]
            fy = data['FT_conv_s'][2,:]
            fz = data['FT_conv_s'][1,:]

            # Cut out sections between tlim[0] and tlim[1]

            if fit_param['tlim'] is not None:
                mask_tlim = np.logical_and(t >= fit_param['tlim'][0], t <= fit_param['tlim'][1])
                t = t[mask_tlim]
                eta = eta[mask_tlim]
                phi = phi[mask_tlim]
                dphi = dphi[mask_tlim]
                fx = fx[mask_tlim]
                fy = fy[mask_tlim]
                fz = fz[mask_tlim]

            # Lowpass filter force data
            dt = t[1] - t[0]
            force_filt = sig.butter(4, fit_param['fcut'], btype='low', output='ba', fs=1/dt)
            fx_filt = sig.filtfilt(*force_filt, fx)
            fy_filt = sig.filtfilt(*force_filt, fy)
            fz_filt = sig.filtfilt(*force_filt, fz)

            # Optional plot showing filtered and unfilterd force data
            if plt_param['filtered_forces']:
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
            if plt_param['grab_sections']:
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

            if plt_param['force_pos_neg']:
                plot_force_pos_neg(datasets[eta_pos_val], datasets[eta_neg_val], alpha)

        return  datasets




    def get_list_of_data_files(self, data_dir, v=10, xi=0):
        """ Get list of relvant data files - filter by v and xi. """
        data_dir = pathlib.Path(data_dir)
        data_files = [item for item in data_dir.iterdir() if item.is_file]
        data_files = [item for item in data_files if f'_v_{v}' in item.name]
        data_files = [item for item in data_files if f'_xi_{xi}' in item.name]
        data_files.sort()
        return data_files


# Utility functions
# ---------------------------------------------------------------------------------------
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


















