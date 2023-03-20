import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.data import shepp_logan_phantom

# settings
nx = 400
theta = np.linspace(0., 180., nx)

for sigma in range(0,10):
    #Phantom
    u=shepp_logan_phantom()


    # sinogram
    f = radon(u, theta=theta)
    f_noisy = f + sigma * np.random.randn(nx,nx)

    #fft
    fs = np.fft.fft(u)
    fshift = np.fft.fftshift(fs)
    fs_noisy=np.fft.fft(u)
    fs_noisyshift=np.fft.fftshift(fs_noisy)+ sigma * np.random.randn(nx,nx)

    #Radon Reconstruction
    u_fbp = iradon(f_noisy,theta=theta)


    # plot
    fig,ax = plt.subplots(1,2)

    ax[0].title.set_text("Image")
    ax[0].imshow(u,vmin=0)
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$')

    #Sinograph Plot

    ax[1].title.set_text("Image after Radon transform/inverse")
    ax[1].imshow(iradon(f_noisy,theta=theta),extent=(-1,1,-1,1),vmin=0)
    ax[1].set_xlabel(r'$θ$')
    ax[1].set_ylabel(r'$s$')
    ax[1].set_aspect(1)
    plt.show()


    rows, cols = fs.shape
    crow,ccol = rows//2 , cols//2
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft(f_ishift)
    img_back = np.abs(img_back)

    rows, cols = fs_noisy.shape
    crow,ccol = rows//2 , cols//2
    fs_noisyshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_inoisyshift = np.fft.ifftshift(fs_noisyshift)
    img_back_noisy = np.fft.ifft2(f_inoisyshift)
    img_back_noisy = np.abs(img_back_noisy)

    plt.subplot(121),plt.imshow(u)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img_back_noisy)
    plt.title('Image after FT'), plt.xticks([]), plt.yticks([])

    plt.show()