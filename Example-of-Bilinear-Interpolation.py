import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates


subs=100
def fpolartocar(r, t, grid, x, y, order=3):

    X, Y = np.meshgrid(x, y)

    new_r = np.sqrt(X*X+Y*Y)
    new_t = np.arctan2(X, Y)

    ir = interp1d(r, np.arange(len(r)), bounds_error=False)
    it = interp1d(t, np.arange(len(t)))

    new_ir = ir(new_r.ravel())
    new_it = it(new_t.ravel())

    new_ir[new_r.ravel() > r.max()] = len(r)-1
    new_ir[new_r.ravel() < r.min()] = 0

    return map_coordinates(grid, np.array([new_ir, new_it]),
                            order=order).reshape(new_r.shape)

def bilinearinterp2D(data,r,theta,x,y):
    nr=len(r)
    nt=len(theta)
    nx=len(x)
    ny=len(y)
    xydata = np.zeros((nx, ny))
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    data=fpolartocar(r,theta,data,x,y)
    for i in range(nr):
        j = int((x[i] - x[0]) // hx)
        for k in range(nt):
            l = int((y[k] - y[0]) // hy)
            if 0 <= l < ny - 1 and 0 <= j < nx - 1:
                xydata[i, k] = ((y[l + 1] - y[k]) / hy) * (
                            ((x[j + 1] - x[i]) / hx) * data[j, l] + ((x[i] - x[j]) / hx) * data[j + 1, l]) \
                               + ((y[k] - y[l]) / hy) * (
                                           ((x[j + 1] - x[i]) / hx) * data[j, l + 1] + ((x[i] - x[j]) / hx) * data[
                                       j + 1, l + 1])
            else:
                xydata[i, k] = 0
    return xydata
gpol=np.zeros((subs,subs))
gcar=np.zeros((subs,subs))
r=np.linspace(0,10,subs)
theta=np.linspace(-np.pi,np.pi,subs)
x = np.linspace(-10,10,subs)
y = np.linspace(-10,10,subs)
for i in range(0,subs):
    for j in range(0,subs):
        gpol[i,j]=(r[i]**2)
        gcar[i,j]=(x[i]**2+y[j]**2)

fig,ax = plt.subplots(1,2)


ax[0].title.set_text("Image")
ax[0].contourf(x,y,gcar)
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')

ax[1].title.set_text("Bilinear Interpolation of Image")
ax[1].contourf(x, y, bilinearinterp2D(gpol,r,theta,x,y))
ax[1].set_xlabel(r'$x$')
ax[1].set_ylabel(r'$y$')
fig.tight_layout()

plt.show()

