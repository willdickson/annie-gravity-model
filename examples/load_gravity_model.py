import pathlib
from annie_gravity_model import GravityModel

filename = pathlib.Path('gravity_model.pkl')
gravity_model = GravityModel()
gravity_model.load(filename)
gravity_model.plot_force_surfaces()
