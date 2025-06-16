import numpy as np
from scabha import init_logger
from . import BIN


log = init_logger(BIN.im_plane)

class ContSub():
    """
    a class for performing continuum subtraction on data
    """
    def __init__(self, function, nomask, reshape=False, fitsaxes=True, fit_tol=0):
        """
        each object can be initiliazed by passing a data cube, a fitting function, and a mask
        Args:
            function (Callable) : a fitting function should be built on FitFunc class
            nomask (bool): Ignore mask if set
            fitsaxis (fits): Is the data in the standard FITS convention (y,x,spectral)?
            reshape (bool): Reshape output to FITS convention before exit.
        Returns:
            None
        """
        self.nomask = nomask
        self.function = function
        self.reshape = reshape
        self.fitsaxes = fitsaxes
        self.fit_tol = fit_tol
        
        
    def fitContinuum(self, xspec, cube, mask):
        """
        fits the data with the desired function and returns the continuum and the line
        
        Args:
            xspec (Array): Apectrum coordinates_
            cube (Array): Data cube to subtract continuum from
            mask (Array): Binary data weights. True -> will be used in fir, False will not be used in fit.

        Returns:
            (Array,Array): Continuum fit and residual
        """
        
        if self.fitsaxes: 
            nchan, dimy, dimx = cube.shape
        else:
            dimx, dimy, nchan = cube.shape
            
        contx = np.zeros_like(cube)
        line = np.zeros_like(cube)
        nomask = self.nomask
        if nomask:
            mask = None
            
        fitfunc = self.function
        if not fitfunc.preped:
            fitfunc.prepare(xspec)

        skipped_lines = 0 
        for ra in range(dimx):
            for dec in range(dimy):
                if self.fitsaxes:
                    slc = slice(None),dec,ra
                else:
                    slc = ra,dec,slice(None)
                mask_ij = mask[slc] if nomask == False else None
                cube_ij = cube[slc]
                
                # Find and mask NaNs in the data
                nanvals_idx = np.where(np.isnan(cube_ij))
                nansize = len(nanvals_idx[0])
                if nansize == nchan:
                    contx[slc] = np.full_like(cube_ij, np.nan)
                    continue
                elif nansize > 0:
                    if nomask:
                        mask_ij = np.ones_like(cube_ij)
                    mask_ij[nanvals_idx] = 0
                
                # Flag LOS and continue if too many pixels are flagged
                if self.fit_tol > 0:
                    if isinstance(mask_ij, np.ndarray) and \
                            (nchan - mask_ij.sum()) / nchan > self.fit_tol/100:
                        skipped_lines += 1
                        contx[slc] = np.full_like(cube_ij, np.nan)
                        continue
                
                contx[slc] = fitfunc.fit(xspec, cube_ij, 
                                                weights = mask_ij)
        
        line = cube - contx
        if self.reshape:
            newshape = (2,1,0)
            
            contx = np.transpose(contx, newshape)
            line = np.transpose(line, newshape) 
            
        if skipped_lines > 0:
            log.info(f"This worker set {skipped_lines} spectra to NaN because of --cont-fit-tol.")
            
        return contx, line
    