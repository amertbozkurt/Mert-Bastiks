import numpy as np
import tensorflow as tf
import scipy as sci
import numba as nb
from numba import jit,njit
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class HistogramNormalize(colors.Normalize):
    def __init__(self, data, vmin=None, vmax=None):
        if vmin is not None:
            data = data[data > vmin]
        if vmax is not None:
            data = data[data < vmax]
            
        sorted_data = np.sort(data.flatten())
        self.sorted_data = sorted_data[np.isfinite(sorted_data)]
        colors.Normalize.__init__(self, vmin, vmax)

    def __call__(self, value, clip=None):
        return np.ma.masked_array(np.searchsorted(self.sorted_data, value) /
                                  len(self.sorted_data))


def fractal(p,c, iteration = 100):
    """ Generates a fractal from the Multibrot
    set.
    
    Parameters
    ----------
    
    p : int
        The degree of the polynomial
    c : complex
        Initial complex number
    iteration: Int
        Number of iterations for the fractals
    
    Returns
    -------
    
    z : complex
        Iterated complex value z
    
    """
    z = 0
    for i in range(iteration):
        z=z**p+c
    
    z[np.where(np.isnan(z)==True)]=0
    
    return z


def find_fractal(p):
    """ This function finds an element of Multibrot
    set by searching random points in the complex
    plane.
    
    Parameters
    ---------
    
    p : Int
        Degree of the Multibrot set.
    
    
    Returns
    ---------
    
    z_v : 128x128 array
        The array of the fractal image
    """
    
    std=0
    
    while std<1e3:
        x0=np.random.uniform(-2,0.5) # a random number for the real part of c
        y0=np.random.uniform(-2,2) # a random number for the imaginary part of c
        d_0=np.random.uniform(0,0.75) # sets the distance in the complex plane
    
        x_v=np.linspace(x0,x0+d_0,num=128)
        y_v=np.linspace(y0,y0+d_0,num=128)
    
        z_v=np.abs(np.array([fractal(p,x_v+1j*y_v[i]) for i in range(len(y_v))]))
        #z_v=np.abs(np.array([mandel_numpy(p,x_v+1j*y_v[i]) for i in range(len(y_v))]))
        
        std=np.std(z_v)
    
    return z_v


def plot_fractal(z_v, c_idx = 0):
    """ Plots the fractal defined by the array z_v with a colormap
    chosen by the user supplied integer c_idx
    
    Parameters
    ---------
    
    z_v : array
        The array of the fractal image
        
    c_idx: int
        Index of the colormaps given below.
    
    Returns
    ---------
    
    The fractal plot
    """
    c_maps=['jet','viridis','plasma','inferno','magma','cividis','rainbow','nipy_spectral','brg', 'RdBu']
    
    normalizer = HistogramNormalize(np.tanh(z_v))
    fig,ax=plt.subplots()
    im=ax.imshow(np.tanh(z_v),cmap=c_maps[c_idx],norm=normalizer)#1/(np.exp(z_v/np.max(z_v))+1)
    ax.axis('off')
    #fig.colorbar(im)
    #plt.show()

    return fig


# +
# for i in range(100):
#     Rand_pic().savefig(r"TPics/fig_{}".format(i),dpi=100,format='png')
# -

p = 3
z = find_fractal(p)
print(z[np.where(z>1)])

color_index = 9
plot_fractal(z, c_idx = color_index)


