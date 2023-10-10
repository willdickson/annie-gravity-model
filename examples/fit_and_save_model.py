import pathlib
from annie_gravity_model import GravityModel

fit_param = {
        't_lim'   : ( 0.8, 7.5),  # t limits for finding grab sections
        'eta_lim' : ( -80,  80),  # eta limits for force surfaces
        }

data_dir = pathlib.Path('../../results_with_conv')
gravity_model = GravityModel()
gravity_model.fit(data_dir, fit_param=fit_param)
gravity_model.save('gravity_model.pkl')
#gravity_model.plot_force_surfaces(plot_type='interp')


