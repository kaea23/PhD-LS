import numpy as np
from scipy.fft import fftfreq


class grid:
    
    #nx, ny voxel size
    #h1, h2 unit length per voxel
    #ntot total voxel volume
    
    def __init__(self, nx, ny, L): #constructor
        self.nx = nx
        self.ny = ny
        self.L = L
        
        
        self.h1 = self.L/ self.nx
        self.h2 = self.L/ self.ny
      
        self.ntot = self.nx * self.ny
        
    def init_XI(self):

        ii = np.pi * fftfreq(self.nx, 1./self.nx) / self.nx
        jj = np.pi * fftfreq(self.ny, 1./self.ny) / self.ny
        
        jj,ii = np.meshgrid(jj, ii)
        
        xi = np.zeros((self.nx, self.ny, 2))
        
        xi[:,:,0] =  2.*np.sin(ii) /self.h1
        xi[:,:,1] =  2.*np.sin(jj) /self.h2
        
        return xi
    """
    def init_XI(self):

        ii = np.pi * fftfreq(self.nx,1./self.nx) / self.nx
        jj = np.pi * fftfreq(self.ny,1./self.ny) / self.ny
        
        jj,ii = np.meshgrid(jj, ii)
        
        xi = np.zeros((self.nx, self.ny, 2))
        
        xi[:,:,0] = 2./self.h1 * np.sin(ii)*np.cos(jj)
        xi[:,:,1] = 2./self.h2 * np.cos(ii)*np.sin(jj)
        
        return xi
    
    
   
        
"""