import numpy as np
from matplotlib import pyplot as plt
from maxent_image import psf_to_linear_map
from maxent_image import estimator

sigma=3.     
h_max=3

r_psf=2*np.ceil(sigma*h_max).astype(np.int)+1


x_min=-r_psf//2+1
x_max=r_psf//2+1
y_min=-r_psf//2+1
y_max=r_psf//2+1
x,y=np.meshgrid(np.arange(x_min,x_max),np.arange(y_min,y_max),indexing="ij")

r2=x**2+y**2

psf=np.where(r2<=(sigma*h_max)**2,np.exp(-r2/(2.*sigma**2)),0.)


psf/=np.sum(psf)



image=plt.imread("lenna.tif")

# pack the image into a vector called z
n_xpix=image.shape[0]
n_ypix=image.shape[1]
z=image.flatten()

# plot the image
def plot_vec(v,title=""):
    
    fig,ax=plt.subplots(1,1,figsize=(8,4))
    im=ax.imshow(v.reshape((n_xpix,n_ypix)),vmin=0.)
    fig.colorbar(im,ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    
plot_vec(z)

A=psf_to_linear_map(psf,image.shape,image.shape)

y=A*z
sigma_y=np.full(y.shape,0.05*y.std())
y=np.random.normal(y,sigma_y)



plot_vec(y)

y=y.reshape((n_xpix,n_ypix))
sigma_y=sigma_y.reshape((n_xpix,n_ypix))
M=estimator(y,sigma_y,psf)

plot_vec(M.x)

M.relax_fit()

plot_vec(M.x)

