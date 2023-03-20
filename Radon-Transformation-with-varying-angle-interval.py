import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.data import shepp_logan_phantom
from skimage.transform import radon
from PIL import Image
from skimage.transform import rescale
# settings

nx=400
sigma=0
for na in range(400,0,-100):
    # phantom
    u = shepp_logan_phantom()
    u=rescale(u,1)
    theta = np.linspace(0., 180., na)

    # sinogram
    f = radon(u, theta=theta)
    f_noisy = f + sigma* np.random.randn(nx,na)

    #Radon Reconstruction
    u_fbp = iradon(f_noisy,theta=theta)
    u_fbp=rescale(u_fbp,1)
    f=rescale(f,1)
    f_noisy=rescale(f_noisy,1)
    # plot
    fig,ax = plt.subplots(1,2)

    ax[0].title.set_text("Image")
    ax[0].imshow(u,extent=(-1,1,-1,1),vmin=0)
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$')
    ax[0].set_aspect(1)

    ax[1].title.set_text("Image with noise")
    ax[1].imshow(u_fbp,extent=(-1,1,-1,1),vmin=0)
    ax[1].set_xlabel(r'$x$')
    ax[1].set_ylabel(r'$y$')
    ax[1].set_aspect(1)
    fig.tight_layout()
    plt.show()


    #Error Plot

    c=0
    d=np.zeros((nx,nx))
    for i in range(nx):
        for j in range(nx):
            d[i,j]=(u[i,j]-u_fbp[i,j])*nx
            c=d[i,j]/(nx*nx)+c
    print("average error",abs(c))
    d=Image.fromarray(d)

    fig, axd = plt.subplots()

    axd.imshow(d, extent=(-1, 1, -1, 1), vmin=0)
    axd.set_xlabel(r'$x$')
    axd.set_ylabel(r'$y$')
    axd.set_aspect(1)
    plt.show()

