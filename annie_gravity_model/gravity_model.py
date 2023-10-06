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

    DEFAULT_FCUT = 10.0
    DEFAULT_XI = 0
    DEFAULT_V = 10
    DEFAULT_NUM_PHI = 50 

    def __init__(self):
        pass


    def load(self, filename):
        pass


    def fit(self, data_dir, tlim=None, v=DEFAULT_V, xi=DEFAULT_XI, fcut=DEFAULT_FCUT, 
            num_phi=DEFAULT_NUM_PHI):

        datasets = self.extract_datasets(data_dir, tlim=tlim, v=v, xi=xi, fcut=fcut)
        eta, phi, fx, fy, fz = self.create_force_surfaces(datasets, num_phi=num_phi)

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


    def create_force_surfaces(self, datasets, num_phi=DEFAULT_NUM_PHI):
        """ Create surfaces for fx, fy, fz forces as functions of eta and phi """
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


    def extract_datasets(self, data_dir, v=10, xi=0, tlim=None, fcut=DEFAULT_FCUT):
        """ Extract data sections with dphi=constant for each eta """
        data_files = self.get_list_of_data_files(data_dir, v=v, xi=xi)
        alphas, data_files = sort_data_files_by_alpha(data_files)

        datasets = {}

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

            if tlim is not None:
                mask_tlim = np.logical_and(t >= tlim[0], t <= tlim[1])
                t = t[mask_tlim]
                eta = eta[mask_tlim]
                phi = phi[mask_tlim]
                dphi = dphi[mask_tlim]
                fx = fx[mask_tlim]
                fy = fy[mask_tlim]
                fz = fz[mask_tlim]

            # Lowpass filter force data
            dt = t[1] - t[0]
            force_filt = sig.butter(4, fcut, btype='low', output='ba', fs=1/dt)
            fx_filt = sig.filtfilt(*force_filt, fx)
            fy_filt = sig.filtfilt(*force_filt, fy)
            fz_filt = sig.filtfilt(*force_filt, fz)

            # Optional plot showing filtered and unfilterd force data
            if 0:
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
            if 0:
                fg, ax = plt.subplots(6, 1, sharex=True)
                ax[0].set_title(f'alpha = {alpha}')

                ax[0].plot(t, eta,'b')
                ax[0].plot(t_pos, eta_pos, '.r')
                ax[0].plot(t_neg, eta_neg, '.g')
                ax[0].set_ylabel('eta')
                ax[0].grid(True)


                ax[1].plot(t, phi,'b')
                ax[1].plot(t_pos, phi_pos, '.r')
                ax[1].plot(t_neg, phi_neg, '.g')
                ax[1].set_ylabel('phi')
                ax[1].grid(True)

                ax[2].plot(t, dphi)
                ax[2].plot(t_pos, dphi_pos, '.r')
                ax[2].plot(t_neg, dphi_neg, '.g')
                ax[2].set_ylabel('dphi')
                ax[2].grid(True)

                ax[3].plot(t, fx_filt, 'b')
                ax[3].plot(t_pos, fx_filt_pos, '.r')
                ax[3].plot(t_neg, fx_filt_neg, '.g')
                ax[3].set_ylabel('fx_filt')
                ax[3].grid(True)

                ax[4].plot(t, fy_filt, 'b')
                ax[4].plot(t_pos, fy_filt_pos, '.r')
                ax[4].plot(t_neg, fy_filt_neg, '.g')
                ax[4].set_ylabel('fy_filt')
                ax[4].grid(True)

                ax[5].plot(t, fz_filt, 'b')
                ax[5].plot(t_pos, fz_filt_pos, '.r')
                ax[5].plot(t_neg, fz_filt_neg, '.g')
                ax[5].set_ylabel('fz_filt')
                ax[5].grid(True)
                ax[5].set_xlabel('t (sec)')
                plt.show()


            if 0:
                fg, ax = plt.subplots(3,1)
                ax[0].set_title(f'alpha = {alpha}')
                ax[0].plot(phi_pos, fx_filt_pos, '.b')
                ax[0].plot(phi_neg, fx_filt_neg, '.r')
                ax[0].set_ylabel('fx')
                ax[0].grid(True)

                ax[1].plot(phi_pos, fy_filt_pos, '.b')
                ax[1].plot(phi_neg, fy_filt_neg, '.r')
                ax[1].set_ylabel('fy')
                ax[1].grid(True)

                ax[2].plot(phi_pos, fz_filt_pos, '.b')
                ax[2].plot(phi_neg, fz_filt_neg, '.r')
                ax[2].set_ylabel('fz')
                ax[2].grid(True)
                ax[2].set_xlabel('t (sec)')
                plt.show()

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




















