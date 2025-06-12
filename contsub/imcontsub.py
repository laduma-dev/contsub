import numpy as np
from scabha import init_logger
from . import BIN


log = init_logger(BIN.im_plane)

class ContSub():
    """
    a class for performing continuum subtraction on data
    """
    def __init__(self, function, nomask, reshape=False, fitsaxes=True):
        """
        each object can be initiliazed by passing a data cube, a fitting function, and a mask
        cube : a fits cube containing the data
        function : a fitting function should be built on FitFunc class
        mask : a fitting mask where the pixels that should be used for fitting has a `True` value
        """
        self.nomask = nomask
        self.function = function
        self.reshape = reshape
        self.fitsaxes = fitsaxes
        
        
    def fitContinuum(self, xspec, cube, mask):
        """
        fits the data with the desired function and returns the continuum and the line
        """
        if self.fitsaxes: 
            dimy, dimx = cube.shape[-2:]
        else:
            dimx, dimy = cube.shape[:2]
        contx = np.zeros_like(cube)
        line = np.zeros_like(cube)
        nomask = self.nomask
        if nomask:
            mask = None
        fitfunc = self.function
        if not fitfunc.preped:
            fitfunc.prepare(xspec)
            
        
        for ra in range(dimx):
            for dec in range(dimy):
                if self.fitsaxes:
                    slc = slice(None),dec,ra
                else:
                    slc = ra,dec,slice(None)
                mask_ij = mask[slc] if nomask == False else None
                cube_ij = cube[slc]
                # get indices of any nan values
                nanvals_idx = np.where(np.isnan(cube_ij))
                if len(nanvals_idx[0]) == len(cube_ij):
                    continue
                if len(nanvals_idx[0]) > 0:
                    if nomask:
                        mask_ij = np.ones_like(cube_ij)
                        mask_ij[nanvals_idx] = 0
                    else:
                        mask_ij[nanvals_idx] = 0
                
                contx[slc] = fitfunc.fit(xspec, cube_ij, 
                                                weights = mask_ij)
        
        line = cube - contx
        if self.reshape:
            newshape = (2,1,0)
            
            contx = np.transpose(contx, newshape)
            line = np.transpose(line, newshape) 
        
        return contx, line
    