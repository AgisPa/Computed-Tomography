import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import  iradon
from skimage.data import shepp_logan_phantom
from skimage.transform import radon


# settings
nx = 400
na = 400
theta = np.linspace(0., 180., na)

for sigma in range(0,10,1):
    # phantom

    u = shepp_logan_phantom()


    # sinogram
    f = radon(u, theta=theta)
    f_noisy = f + sigma * np.random.randn(nx,na)

    #Radon Reconstruction
    u_fbp = iradon(f_noisy,theta=theta)

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
    from PIL import Image, ImageChops
    c=0
    d=np.zeros((nx,na))
    for i in range(0,nx):
        for j in range(0,na):
            d[i,j]=abs(u[i,j]-u_fbp[i,j])*nx
            c=d[i,j]/(nx*na)+c
    print("average error",c)
    d=Image.fromarray(d)

    fig, axd = plt.subplots()

    axd.imshow(d, extent=(-1, 1, -1, 1), vmin=0)
    axd.set_xlabel(r'$x$')
    axd.set_ylabel(r'$y$')
    axd.set_aspect(1)
    plt.show()

