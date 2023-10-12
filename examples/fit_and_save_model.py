import pathlib
from annie_gravity_model import GravityModel

data_prm = {
        't_lim'   : ( 0.8, 7.5),  # t limits for finding grab sections
        'eta_lim' : ( -80,  80),  # eta limits for force surfaces
        }
plot_prm = {
        'grab_sections' : True,
        }

data_dir = pathlib.Path('../../results_with_conv')
gravity_model = GravityModel()
gravity_model.fit(data_dir, data_prm=data_prm, plot_prm=plot_prm)
gravity_model.save('gravity_model.pkl')
#gravity_model.plot_force_surfaces(plot_type='interp')



