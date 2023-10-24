import pickle
import numpy as np
import scipy.interpolate as interp
from . import utility

class GravityModel:
    """
    Model of gravitational forces acting on Robofly wing.  Use for gravitation
    force subtraction when analysing data from a set of polars. 
    """
    DEFAULT_DATA_PRM = {
            'v'       : 10,   # velocity (int) in kinematics filename
            'xi'      :  0,   # xi value (int) in kinematics filenames  
            't_lim'   : None, # (tuple) lower/upper limits of time range 
            'eta_lim' : None, # (tuple) lower/upper limits of eta range
            'fcut'    : 10.0, # frequency cutoff for force lowpass filter
            'num_phi' : 1000, # number of phi data points after resampling
            }

    DEFAULT_PLOT_PRM = {
            'force_surfaces'  : False,
            'filtered_forces' : False, 
            'grab_sections'   : False,
            'force_pos_neg'   : False,
            }

    def __init__(self):
        self.model = {}

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)

    @property
    def eta(self):
        if self.model:
            val = self.model['force_surface']['eta_pts']
        else:
            val = np.nan
        return val

    @property
    def phi(self):
        if self.model:
            val = self.model['force_surface']['phi_pts']
        else:
            val = np.nan
        return val

    @property
    def eta_grid(self):
        if self.model:
            val = self.model['force_surface']['eta_grid']
        else:
            val = None
        return val

    @property
    def phi_grid(self):
        if self.model:
            val = self.model['force_surface']['phi_grid']
        else:
            val = None
        return val
    
    def fx(self, eta, phi):
        return self.get_force('fx', eta, phi)


    def fy(self, eta, phi):
        return self.get_force('fy', eta, phi)


    def fz(self, eta, phi):
        return self.get_force('fz', eta, phi)
    

    def get_force(self, name, eta, phi):
        if self.model: 
            pts = (phi, eta)
            val = self.model['force_interp'][name](pts)
        else:
            val = 0.0
        return val


    def plot_force_surfaces(self, plot_type='interp'):
        """ Plot gravity model force surfaces """
        eta = self.eta_grid 
        phi = self.phi_grid
        if plot_type == 'interp':
            fx = self.fx(eta, phi) 
            fy = self.fy(eta, phi) 
            fz = self.fz(eta, phi) 
        elif plot_type == 'data':
            fx = self.model['force_surface']['fx']
            fy = self.model['force_surface']['fy']
            fz = self.model['force_surface']['fz']
        else:
            raise ValueError(f'unknown plot_type = {plot_type}')
        utility.plot_force_surfaces(eta, phi, fx, fy, fz)


    def fit(self, data_dir, data_prm=None, plot_prm=None): 
        """
        Fit gravitational model using data polar from data_dir and prmeters
        in data_prm.

        Arguments:

            data_dir = location of kinematics data files

        keyword Arguments:

            data_prm = dictionary containing fit prmeters
            {
                'v'       :  # velocity (int) in kinematics filenames
                'xi'      :  # xi value (int) in kinematics filenames
                't_lim'   :  # time range (tuple) lower and upper bounds 
                'fcut'    :  # force data lowpass filter cutoff frequency
                'num_phi' :  # number of phi data points after resampling
            }

            plot_prm = dictionary of plotting options.
            {
                'force_surfaces'   : True
                'force_filter'     : False, 
                'data_sections'    : False,
                'pos_neg_sections' : False,
            }

        """
        # Set default options and update with user options
        _data_prm = dict(self.DEFAULT_DATA_PRM)
        if data_prm is not None:
            _data_prm.update(data_prm)

        _plot_prm = dict(self.DEFAULT_PLOT_PRM)
        if plot_prm is not None:
            _plot_prm.update(plot_prm)

        # Read in datasets and extract force sections
        datasets = utility.load_datasets(data_dir, _data_prm, _plot_prm)

        # Create fx, fy, and fz force surfaces as functions of eta and phi (meshgrids)
        self.model['force_surface'] = self.create_force_surfaces(datasets, _data_prm)
        
        # Create the force interpolators for fx, fy, fz
        self.model['force_interp'] = self.create_force_interpolators(self.model['force_surface'])
        self.model['data_prm'] = data_prm 

        if _plot_prm['force_surfaces']:
            self.plot_force_surfaces(plot_type='data')


    def create_force_interpolators(self, force_surface):
        """
        Create interpolation functions for forces.
        """
        eta_pts = force_surface['eta_pts']
        phi_pts = force_surface['phi_pts']
        fx = force_surface['fx']
        fy = force_surface['fy']
        fz = force_surface['fz']
        fx_interp = interp.RegularGridInterpolator((phi_pts, eta_pts), fx)
        fy_interp = interp.RegularGridInterpolator((phi_pts, eta_pts), fy)
        fz_interp = interp.RegularGridInterpolator((phi_pts, eta_pts), fz)
        force_interp = {
                'fx': fx_interp,
                'fy': fy_interp,
                'fz': fz_interp,
                }
        return force_interp



    def create_force_surfaces(self, datasets, data_prm):
        """ 
        Creates surfaces for fx, fy, fz forces as functions of eta and phi 

        Arguments:

          datasets   = dictionary mapping eta values to kinematics and forces. 
          data_prm  = dictionary of fitting prmeters 

        Returns:

          eta = meshgrid of eta values  
          phi = meshgrid of phi values 
          fx  = component of forces in x direction (chordwise direction)
          fy  = component of forces in y direction (spanwise direction)
          fz  = component of forces in z direction (normal to wing surface)

        """
        # Get arrays of eta and phi values 
        eta_pts = np.array(sorted(datasets.keys()))
        eta_min, eta_max = data_prm['eta_lim']
        eta_mask = np.logical_and(eta_pts >= eta_min, eta_pts <= eta_max) 
        eta_pts = eta_pts[eta_mask]
        phi_max = min([datasets[eta]['phi'].max() for eta in eta_pts])
        phi_min = max([datasets[eta]['phi'].min() for eta in eta_pts])
        phi_pts = np.linspace(phi_min, phi_max, data_prm['num_phi'])

        # Create meshgrid
        eta, phi = np.meshgrid(eta_pts, phi_pts)
        fx = np.zeros(eta.shape)
        fy = np.zeros(eta.shape)
        fz = np.zeros(eta.shape)

        for i, val in enumerate(eta_pts):
            data = datasets[val]
            fx_interp_func = interp.interp1d(data['phi'], data['fx'], kind='linear')
            fy_interp_func = interp.interp1d(data['phi'], data['fy'], kind='linear')
            fz_interp_func = interp.interp1d(data['phi'], data['fz'], kind='linear')
            fx[:,i] = fx_interp_func(phi_pts)
            fy[:,i] = fy_interp_func(phi_pts)
            fz[:,i] = fz_interp_func(phi_pts)

        surface_data = {
                'eta_pts'  : eta_pts,
                'phi_pts'  : phi_pts,
                'eta_grid' : eta, 
                'phi_grid' : phi,
                'fx'       : fx, 
                'fy'       : fy,
                'fz'       : fz,
                }
        return surface_data


