import numpy as np
from scipy.fft import fftfreq

class grid:
    
    # nx, ny, nz voxel size
    # L length of material
    # h1, h2, h3 unit length per voxel
    # ndims number of dimensions (2 or 3)
    
    def __init__(self, nx, ny, nz, L, ndims): #constructor
    
        self.nx = nx
        self.ny = ny
        self.nz = nz
        
        self.L = L
        
        self.ndims = ndims
      
    def init_XI(self, scheme):
        
        self.h1 = self.L/ self.nx
        self.h2 = self.L/ self.ny
        
        if scheme == 'S': 
            
            ii = np.pi * fftfreq(self.nx, 1./self.nx) / self.nx
            jj = np.pi * fftfreq(self.ny, 1./self.ny) / self.ny
            
            if self.ndims == 2: 
        
                jj,ii = np.meshgrid(jj, ii)
        
                xi = np.zeros((self.nx, self.ny, self.ndims))
        
                xi[:,:,0] =  2.*np.sin(ii) /self.h1
                xi[:,:,1] =  2.*np.sin(jj) /self.h2
                
            elif self.ndims == 3:
                
                self.h3 = self.L/ self.nz

                kk = np.pi * fftfreq(self.nz, 1./self.nz) / self.nz
        
                kk,jj,ii = np.meshgrid(kk, jj, ii)
        
                xi = np.zeros((self.nx, self.ny, self.nz, self.ndims))
        
                xi[:,:,:,0] =  2.*np.sin(ii) /self.h1
                xi[:,:,:,1] =  2.*np.sin(jj) /self.h2
                xi[:,:,:,2] =  2.*np.sin(kk) /self.h3
                
        elif scheme == 'H': 
            
            ii = np.pi * fftfreq(self.nx, 1./self.nx) / self.nx
            jj = np.pi * fftfreq(self.ny, 1./self.ny) / self.ny
            
            if self.ndims == 2: 
                
                jj,ii = np.meshgrid(jj, ii)
        
                xi = np.zeros((self.nx, self.ny, self.ndims))
        
                xi[:,:,0] =  2.*np.sin(ii)*np.cos(jj) /self.h1
                xi[:,:,1] =  2.*np.sin(jj)*np.cos(ii) /self.h2
                
            elif self.ndims == 3: 
                
                self.h3 = self.L/ self.nz
            
                kk = np.pi * fftfreq(self.nz, 1./self.nz) / self.nz
        
                kk,jj,ii = np.meshgrid(kk, jj, ii)
        
                xi = np.zeros((self.nx, self.ny, self.nz, self.ndims))
        
                xi[:,:,:,0] =  2.*np.sin(ii)*np.cos(jj)*np.cos(kk) /self.h1
                xi[:,:,:,1] =  2.*np.sin(jj)*np.cos(kk)*np.cos(ii) /self.h2
                xi[:,:,:,2] =  2.*np.sin(kk)*np.cos(ii)*np.cos(jj) /self.h3
                         
        else:
            raise ValueError(f"Unknown scheme '{scheme}'. Supported schemes are 'S' and 'H'.")
            
        return xi
    

