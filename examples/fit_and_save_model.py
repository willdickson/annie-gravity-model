import pathlib
from annie_gravity_model import GravityModel

data_prm = {
        'v'         : 10,           # velocity (int) in kinematics filename
        'xi'        :  0,           # xi value (int) in kinematics filenames  
        't_lim'     : ( 0.8, 7.5),  # t limits for finding grab sections
        'eta_lim'   : ( -80,  80),  # eta limits for force surfaces
        'fcut'      : 10.0,         # frequency cutoff for force lowpass filter
        'num_phi'   : 1000,         # number of phi data points after resampling
        'gain_corr' : 2.0           # Gain correction scaling factor (analog ref issues)
        }
plot_prm = {
        'filtered_forces' : False, 
        'grab_sections'   : True,
        'force_pos_neg'   : False,
        'force_surfaces'  : False,
        }

data_dir = pathlib.Path('../../results_with_conv')
gravity_model = GravityModel()
gravity_model.fit(data_dir, data_prm=data_prm, plot_prm=plot_prm)
gravity_model.save('gravity_model.pkl')
gravity_model.plot_force_surfaces(plot_type='interp')



