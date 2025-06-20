import numpy as np
from scipy.fft import fftfreq

class grid:
    
    #nx, ny voxel size
    #h1, h2 unit length per voxel
    #ntot total voxel volume
    
    def __init__(self, nx, ny, nz, L, ndims): #constructor
    
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        self.L = L
        
        self.ndims = ndims
      
    def init_XI(self):
        
        self.h1 = self.L/ self.nx
        self.h2 = self.L/ self.ny
        
        if self.nz == 0: 
            
            ii = np.pi * fftfreq(self.nx, 1./self.nx) / self.nx
            jj = np.pi * fftfreq(self.ny, 1./self.ny) / self.ny
        
            jj,ii = np.meshgrid(jj, ii)
        
            xi = np.zeros((self.nx, self.ny, self.ndims))
        
            xi[:,:,0] =  2.*np.sin(ii) /self.h1
            xi[:,:,1] =  2.*np.sin(jj) /self.h2
            
        else:
            
            self.h3 = self.L/ self.nz
            
            ii = np.pi * fftfreq(self.nx, 1./self.nx) / self.nx
            jj = np.pi * fftfreq(self.ny, 1./self.ny) / self.ny
            kk = np.pi * fftfreq(self.nz, 1./self.nz) / self.nz
        
            kk,jj,ii = np.meshgrid(kk, jj, ii)
        
            xi = np.zeros((self.nx, self.ny, self.nz, self.ndims))
        
            xi[:,:,0] =  2.*np.sin(ii) /self.h1
            xi[:,:,1] =  2.*np.sin(jj) /self.h2
            xi[:,:,2] =  2.*np.sin(kk) /self.h3
            
        return xi
