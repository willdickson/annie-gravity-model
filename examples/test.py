import pathlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from annie_gravity_model import GravityModel

fit_param = {
        't_lim'   : [0.8, 7.5],
        'eta_lim' : [-80, 80],
        }

data_dir = pathlib.Path('../../results_with_conv')
gravity_model = GravityModel()
gravity_model.fit(data_dir, fit_param=fit_param)

# Recreate force surfaces using interpolation 
eta = gravity_model.eta_grid 
phi = gravity_model.phi_grid
fx = gravity_model.fx(eta, phi) 
fy = gravity_model.fy(eta, phi) 
fz = gravity_model.fz(eta, phi) 

figsize = (11,9)
fg, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)
ax.set_title('fx surface - interpolate')
surf = ax.plot_surface(eta, phi, fx, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)
ax.set_xlabel('eta')
ax.set_ylabel('phi')
ax.set_zlabel('fx')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}')
fg.colorbar(surf, shrink=0.5, aspect=5)

fg, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)
ax.set_title('fy surface - interpolate')
surf = ax.plot_surface(eta, phi, fy, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)
ax.set_xlabel('eta')
ax.set_ylabel('phi')
ax.set_zlabel('fy')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}')
fg.colorbar(surf, shrink=0.5, aspect=5)

fg, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)
ax.set_title('fz surface - interpolate')
surf = ax.plot_surface(eta, phi, fz, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)
ax.set_xlabel('eta')
ax.set_ylabel('phi')
ax.set_zlabel('fz')
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}')
fg.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


