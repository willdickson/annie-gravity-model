import pathlib
from annie_gravity_model import GravityModel

tlim = [0.8, 7.5]
data_dir = pathlib.Path('../../results_with_conv')
gravity_model = GravityModel()
gravity_model.fit(data_dir, tlim=tlim)
