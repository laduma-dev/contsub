import numpy as np
import datetime
import time
import os
from astropy.io import fits
from astropy.table import QTable
from multiprocessing.pool import ThreadPool as Pool
from astropy.wcs import WCS
import astropy.units as u
from itertools import repeat
import sys
from scabha import init_logger
from . import BIN


log = init_logger(BIN.im_plane)

class alreadyOpen(Exception):
    pass

class alreadyClosed(Exception):
    pass

class noBeamTable(Exception):
    pass

class tableExists(Exception):
    pass

class tableDimMismatch(Exception):
    pass

class CubeDimIsSmall(Exception):
    pass

class WCube():
    def __init__(self, path, overwrite=False):
        self.path = path
        self._open = False
        self._header = None
        self.overwrite = overwrite
        
    def openW(self, header):
        try:
            if self._open:
                raise alreadyOpen()

        except alreadyOpen:
            log.info('table is open, cannot open again')
            
        else:
            pathPts = self.path.split('/')
            tmp = f'{pathPts[0]}/'
            for fol in pathPts[1:-1]:
                tmp = f'{tmp}{fol}/'
                if not os.path.isdir(tmp):
                    os.mkdir(tmp)
            self._cube = fits.StreamingHDU(self.path, header)
            self._header = header
            if self._header['NAXIS'] == 3:
                self._extendCh = (None,)
                self._extendCu = ()
            elif self._header['NAXIS'] == 4:
                self._extendCh = (None, None, )
                self._extendCu = (None,)
            else:
                self._extendCh = ()
                self._extendCu = ()
            self._open = True
            
            self._type_dict = {8 : np.uint8, 16  : np.int16, 32  : np.int32, 64  : np.int64, -32 : np.float32, -64 : np.float64}
        
    def close(self):
        try:
            if not self._open:
                raise alreadyClosed()

        except alreadyClosed:
            log.info('table is closed, cannot close again')
            
        else:
            self._cube.close()
            self._open = False
            
    def spatialCombineFits(self, plist, nxny, oxy, order = None, overwrite = False):
        if len(plist) > 1:
            
            try:
                nxny = int(nxny), int(nxny)
                log.info(f'combine assuming {nxny[0]} parts in y and {nxny[1]} parts in x')
            except:
                log.info(f'combine assuming {nxny[0]} parts in y and {nxny[1]} parts in x')
            
            if order is None:
                log.info('order of the cubes not provided, assuming cubes are fed in order')
                order = np.arange(len(plist), dtype = int)
            elif order == 'natural':
                order = np.transpose(np.arange(nxny[0]*nxny[1]).reshape(nxny[0],-1)).flatten()
                log.info(f'order = {order}')
            
            log.info(f'number of fits files: {len(plist)}')
            objlist = [RCube(plist[o]) for o in order]

            [obj.openR(memmap = True) for obj in objlist]
            
            ny, nx = nxny[0], nxny[1]
            oy, ox = oxy, oxy
            
            refxy = np.zeros((len(objlist), 2))
            dimxy = np.zeros((len(objlist), 2), dtype = int)
            for i,c in enumerate(objlist):
                refxy[i,0] = c._cube[0].header['CRVAL2']
                refxy[i,1] = c._cube[0].header['CRVAL1']
                dimxy[i,0] = int(c._cube[0].header['NAXIS2'])
                dimxy[i,1] = int(c._cube[0].header['NAXIS1'])
                
            log.info('dimxy: {}'.format(dimxy))

            nxy = np.zeros((2), dtype = int)
            for i in range(nx):
                nxy[0] = nxy[0] + dimxy[i,0]
            for i in range(ny):
                nxy[1] = nxy[1] + dimxy[nx*i,1]
            nxy[0] = nxy[0] - ox*(nx-1)
            nxy[1] = nxy[1] - oy*(ny-1)
            
            log.info('spatial dimensions of the new cube: {}'.format(nxy))
            
            hdrObj = FitsHeader(objlist[0].getHeader())
            imheader = hdrObj.getCombineHeader(nxy[0], nxy[1])
            
            if os.path.exists(self.path):
                if overwrite:
                    log.info('Stream cube ({}) exists going to remove and recalculate'.format(self.path))
                    os.remove(self.path)
                    time.sleep(.1)
                else:
                    log.info(f'not allowed to overwrite the cube {self.path}')
                    sys.exit(1)
                
            self.openW(imheader)
            
            self._combCore(objlist, refxy, nx, ny, ox, oy)
            log.info('Combined cubes {} to a single fits file'.format(plist))
        
            self.close()
            
            if len(objlist[0]._cube) > 1:
                log.info('Adding the beam table to the appended FITS file')
                hdrObjTb = FitsHeader(objlist[0].getHeader(1))
                tbheader = hdrObjTb.getTableHeader(objlist[0]._cube[0].shape[-3])
                
                beamtab = objlist[0]._cube[1].data
                    
                qt = RCube._createBmTable(beamtab)
                
                rcube = RCube(self.path)
                
                rcube.openR(update = True, mode = 'append')
                rcube.appendBeamTable(fits.BinTableHDU(qt, tbheader))
                rcube.close()
            
        else:
            log.info('got less than 2 paths, cannot append')
            
            
    def _combCore(self, objs, refxy, dx, dy, ox, oy):
    
        log.info("dx:{}, dy:{}".format(dx, dy))
        log.info("ox:{}, oy:{}".format(ox, oy))

        mem, task = RCube._getHWinfo()

        chdim = objs[0]._cube[0].shape[-3]
        log.info('cube freq dim: {}'.format(chdim))

        for j in range(chdim):
            pool = Pool(task)
            chans = pool.starmap(RCube._getChan, zip(objs, repeat(j)))
            pool.close()
            # [print(f'chn_{ichn}: {chn.shape}') for ichn, chn in enumerate(chans)]
            xh = oy//2
            yh = ox//2
            xr = oy - yh
            yr = ox - xh

            xdir = []
            for i in range(dx):
                ydir = []
                for k in range(dy):
                    ch = chans[k+(i*dy)]
                    dimx, dimy = ch.shape
                    if k==0:
                        if i==0:
                            # ydir.append(ch[:dimx-xr, :dimy-yr])
                            ydir.append(ch[:dimx-xr, :dimy-yr])
                            # log.info('ch[:dimx-xr, :dimy-yr]')
                        elif i==dx-1:
                            # ydir.append(ch[xh:, :dimy-yr])
                            ydir.append(ch[:dimx-xr, yh:])
                            # log.info('ch[xh:, :dimy-yr]')
                        else:
                            # ydir.append(ch[xh:dimx-xr, :dimy-yr])
                            ydir.append(ch[:dimx-xr, yh:dimy-yr])
                            # log.info('ch[xh:dimx-xr, :dimy-yr]')
                    elif k==dy-1:
                        if i==0:
                            # ydir.append(ch[:dimx-xr, yh:])
                            ydir.append(ch[xh:, :dimy-yr])
                            # log.info('ch[:dimx-xr, yh:]')
                        elif i==dx-1:
                            # ydir.append(ch[xh:, yh:])
                            ydir.append(ch[xh:, yh:])
                            # log.info('ch[xh:, yh:]')
                        else:
                            # ydir.append(ch[xh:dimx-xr, yh:])
                            ydir.append(ch[xh:, yh:dimy-yr])
                            # log.info('ch[xh:dimx-xr, yh:]')
                    else:
                        if i==0:
                            # ydir.append(ch[:dimx-xr, yh:dimy-yr])
                            ydir.append(ch[xh:dimx-xr, :dimy-yr])
                            # log.info('ch[:dimx-xr, yh:dimy-yr]')
                        elif i==dx-1:
                            # ydir.append(ch[xh:, yh:dimy-yr])
                            ydir.append(ch[xh:dimx-xr, yh:])
                            # log.info('ch[xh:, yh:dimy-yr]')
                        else:
                            ydir.append(ch[xh:dimx-xr, yh:dimy-yr])
                            # log.info('ch[xh:dimx-xr, yh:dimy-yr]')
                if dy > 1:
                    xdir.append(np.concatenate((ydir), axis = 0))
                else:
                    xdir.append(np.array(ydir))
            if dx > 1:
                combChan = np.concatenate((xdir), axis = 1)
            else:
                combChan = np.array(xdir)
            img = np.array(combChan[self._extendCh], dtype = self._type_dict[self._header['BITPIX']])
            self._cube.write(img, clobber=self.overwrite)
        
    # def _getCombNxy(self, dxdy):
        
    def spectralCombineFits(self, plist, overwrite = False):
        if len(plist) > 1:
            objlist = [RCube(p) for p in plist]

            [obj.openR(memmap = True) for obj in objlist]

            freqList = [obj.retFreq() for obj in objlist]
            
            freqst = np.array([f[0] for f in freqList])
            freqen = np.array([f[-1] for f in freqList])

            order = np.argsort(freqst)
            nchan = len(freqList[order[0]])
            
            chanst = []
            chanst.append(0)
            chanen = []

            for i in range(1, len(order)):
                ch = np.where(freqList[order[i]]>freqen[order[i-1]])[0][0]
                dch = ch - ch//2 
                chanst.append(ch//2)
                chanen.append(len(freqList[order[i-1]])-dch)
                nchan = nchan + len(freqList[order[i]]) - ch
                
            chanen.append(len(freqList[order[-1]]))
            
            for i, chs, che in zip(order, chanst, chanen):
                log.info('Will append cube {} from chan {} to {}'.format(plist[i], chs, che))

            log.info('total number of channels: {}'.format(nchan))
            
            chanst = np.array(chanst)
            chanen = np.array(chanen)
            
            hdrObj = FitsHeader(objlist[order[0]].getHeader())
            imheader = hdrObj.getAppendHeader(nchan)
            
            if os.path.exists(self.path): 
                if overwrite:
                    log.info('Stream cube ({}) exists going to remove and recalculate'.format(self.path))
                    os.remove(self.path)
                    time.sleep(.1)
                else:
                    log.info(f'not allowed to overwrite the cube {self.path}')
                    sys.exit(1)
                
            self.openW(imheader)
            
            for i, chs, che in zip(order, chanst, chanen):
                self._append(objlist[i], chs, che)
                log.info('Appended cube {} to the fits file'.format(plist[i]))
            
            self.close()
            
            if len(objlist[order[0]]._cube) > 1:
                log.info('Adding the beam table to the appended FITS file')
                hdrObjTb = FitsHeader(objlist[order[0]].getHeader(1))
                tbheader = hdrObjTb.getTableHeader(nchan)
                
                beamtab = objlist[order[0]]._cube[1].data[chanst[0]:chanen[0]]

                for i in range(1, len(order)):
                    j = order[i]
                    beamtab = np.append(beamtab, objlist[j]._cube[1].data[chanst[i]:chanen[i]])
                    
                qt = RCube._createBmTable(beamtab)
                
                rcube = RCube(self.path)
                
                rcube.openR(update = True, mode = 'append')
                rcube.appendBeamTable(fits.BinTableHDU(qt, tbheader))
                rcube.close()
            
        else:
            log.info('got less than 2 paths, cannot append')
        
    def _append(self, cube, st, en):        
        for i in range(st, en):
            img = np.array(cube._getChan(i), dtype = self._type_dict[self._header['BITPIX']])
            self._cube.write(img[self._extendCh])
            
    def _appendData(self, data):        
        self._cube.write(data.astype(self._type_dict[self._header['BITPIX']])[self._extendCh])
            
    def _appendCube(self, data):        
        self._cube.write(data.astype(self._type_dict[self._header['BITPIX']])[self._extendCu])
        
    def _appendXY(self, cube, cl, yl, xl):
        byteSz = (cl[1]-cl[0])*(yl[1]-yl[0])*(xl[1]-xl[0])*4
        chk = int(byteSz/256e6 + 1) ##assuming buffer size of 250MB
        chlims = np.linspace(cl[0], cl[1], chk+1, dtype = int)
        for i, f in zip(chlims[:-1], chlims[1:]):
            img = np.array(cube.getChansXY((i, f), yl, xl), dtype = self._type_dict[self._header['BITPIX']])
            self._cube.write(img[self._extendCu])

class RCube():
    def __init__(self, path):
        self.path = path
        self._open = False
        
    def getHeader(self, ind = 0):
        try:
            if not self._open:
                raise alreadyClosed()

        except alreadyClosed:
            log.info('table is closed')
            
        else:
            return self._cube[ind].header
        
    def openR(self, mode='readonly', memmap=None, update = False):
        try:
            if self._open:
                raise alreadyOpen()

        except alreadyOpen:
            log.info('table is open, cannot open again')
            
        else:
            self._cube = fits.open(self.path, mode = mode, memmap = memmap, update = update)
            self._open = True
            if len(self._cube[0].shape) == 4:
                self.s_o = [0, slice(None, None, 1), slice(None, None, 1), slice(None, None, 1)]
            elif len(self._cube[0].shape) == 3:
                self.s_o = [slice(None, None, 1), slice(None, None, 1), slice(None, None, 1)]
            else:
                self.s_o = None
                
        
    def close(self):
        try:
            if not self._open:
                raise alreadyClosed()

        except alreadyClosed:
            log.info('table is closed, cannot close again')
            
        else:
            self._cube.close()
            self._open = False
            
    def getNchan(self):
        try:
            if not self._open:
                raise alreadyClosed()

        except alreadyClosed:
            log.info('table is closed, cannot get Nchan')
            
        else:
            return self._cube[0].shape[-3]
    
    @property
    def shape(self):
        try:
            if not self._open:
                raise alreadyClosed()

        except alreadyClosed:
            log.info('table is closed, cannot get Nchan')
            
        else:
            return self._cube[0].shape
        
    def getXdim(self):
        try:
            if not self._open:
                raise alreadyClosed()

        except alreadyClosed:
            log.info('table is closed, cannot get Xdim')
            
        else:
            return self._cube[0].shape[-1]
        
    def add(self, cube, multiple = 1, folder = None, name = '', overwrite = False):
        base = self.path.replace('.fits', f'_{name}.fits')
        if folder is not None:
            base = folder.rstrip('/') + '/' + base.split('/')[-1]
            
        imheader = self.getHeader().copy()
        
        for ptname in [base]:
            if os.path.exists(ptname):
                if overwrite:
                    log.info('Stream cube ({}) exists going to remove and recalculate'.format(ptname))
                    os.remove(ptname)
                    time.sleep(.1)
                else:
                    log.info(f'not allowed to overwrite the cube {ptname}')
                    sys.exit(1)
                    
        ccube = WCube(base)
        
        ccube.openW(imheader)
        
        for i in range(self.shape[-3]):
            ccube._appendData(self._getChan(i) + multiple*cube._getChan(i))
        
        ccube.close()  
        
        log.info(f'wrote cubes {base}')
        
        if len(self._cube) > 1:
            log.info(f'Adding the beam table to the FITS file: {base}')
            
            beamtab = self._cube[1].data
            qt = RCube._createBmTable(beamtab)
            
            rcube = RCube(base)
            rcube.openR(update = True, mode = 'append')
            rcube.appendBeamTable(fits.BinTableHDU(qt, self.getHeader(1).copy()))
            rcube.close()
            
        return base
    
    def subtract(self, cube, folder = None, name = '', overwrite = False):
        return self.add(cube, multiple = -1, folder = folder, name = name, overwrite = overwrite)
    
    def getYdim(self):
        try:
            if not self._open:
                raise alreadyClosed()

        except alreadyClosed:
            log.info('table is closed, cannot get Ydim')
            
        else:
            return self._cube[0].shape[-2]
        
    def collapse(self, rng_chan = None, folder = None, name = 'collapsed', overwrite = False, chk = 2):
        '''
        collapse the cube along the spectral dimension
        
        Parameters
        ----------
        rng_chan : tuple of ints
            range of channels to be collapsed, default None which means all the channels
        
        folder : string
            directory to save the output file
            
        name : string
            suffix to add the name of existing file
            
        overwrite : boolean
            whether or not to overwrite if the output name exists
            
        chk : int
            chunking number for averaging
        '''
        base = self.path.replace('.fits', f'_{name}.fits')
        if folder is not None:
            base = folder.rstrip('/') + '/' + base.split('/')[-1]
            
        imheader = self.getHeader().copy()
        
        for ptname in [base]:
            if os.path.exists(ptname):
                if overwrite:
                    log.info('Stream cube ({}) exists going to remove and recalculate'.format(ptname))
                    os.remove(ptname)
                    time.sleep(.1)
                else:
                    log.info(f'not allowed to overwrite the cube {ptname}')
                    sys.exit(1)
                    
        ccube = WCube(base)
        
        imheader['NAXIS3'] = 1
        ccube.openW(imheader)
        
        inds = []
        chans = []
        
        if rng_chan is None:
            rng_chan = [0,self.shape[-3]]
        
        for i in range(int(rng_chan[0]), int(rng_chan[1])):
            inds.append(1)
            chans.append(self._getChan(i))

            unq, cnt = np.unique(inds, return_counts=True)
            
            while np.any(cnt == chk):
                t_ind = unq[np.where(cnt == chk)[0][0]]
                idx = np.flip(np.sort(np.where(inds == t_ind)[0]))
                
                inds.append(np.sum([inds.pop(i) for i in idx]))
                chans.append(np.mean([chans.pop(i) for i in idx], axis = 0))

                unq, cnt = np.unique(inds, return_counts=True)
    
        ccube._appendData(np.average(chans, axis = 0, weights = inds))
        
        ccube.close()  
        
        log.info(f'wrote cubes {base}')
        
        return base
        
        
    def getSmallestBeam(self):
        ind = np.argmin(self._cube[1].data['BMAJ'])
        return tuple(self._cube[1].data[ind][:2])
        
    def getPixSize_asec(self):
        if 'deg' in self._cube[0].header['CUNIT2']:
            return np.around(np.abs(self._cube[0].header['CDELT2']) * 3600, decimals=3)
        else:
            log.error('the dimension of CDELT2 is not degrees, returning nan')
            return np.nan
            
    def getChans(self, i, f):
        try:
            if not self._open:
                raise alreadyClosed()
            elif len(self._cube[0].shape) < 3:
                raise CubeDimIsSmall()

        except alreadyClosed:
            log.info('table is closed')
        except CubeDimIsSmall:
            log.info('cube has less than 3 dimensions')
            
        else:
            select = self.s_o.copy()
            select[-3] = slice(i,f)
            return self._cube[0].data[tuple(select)]
        
    def getChansXY(self, cl, yl, xl):
        try:
            if not self._open:
                raise alreadyClosed()
            elif len(self._cube[0].shape) < 3:
                raise CubeDimIsSmall()

        except alreadyClosed:
            log.info('table is closed')
        except CubeDimIsSmall:
            log.info('cube has less than 3 dimensions')
            
        else:
            select = self.s_o.copy()
            select[-3] = slice(cl[0], cl[1], 1)
            select[-2] = slice(yl[0], yl[1], 1)
            select[-1] = slice(xl[0], xl[1], 1)
            return self._cube[0].data[tuple(select)]
            
    def _getChan(self, i):
        select = self.s_o.copy()
        select[-3] = i
        return self._cube[0].data[tuple(select)]
        
    def _getChanXY(self, i, y, x):
        select = self.s_o.copy()
        select[-3] = i
        select[-2] = slice(y[0], y[1], 1)
        select[-1] = slice(x[0], x[1], 1)
        return self._cube[0].data[tuple(select)]
            
    def getBeamTable(self, lims = None):
        try:
            if not self._open:
                raise alreadyClosed()
            if len(self._cube) < 2:
                raise noBeamTable()

        except alreadyClosed:
            log.info('table is closed, cannot read beamtable')
        except noBeamTable:
            log.info('no beam table found')
            
            
        else:
            if lims is not None:
                dn, up = lims
            else:
                dn, up = 0, self._cube[1].data.shape[0]
                
            hdrObj = FitsHeader(self._cube[1].header)
            tableheader = hdrObj.getTableHeader(up-dn)
            beamtab = self._cube[1].data[dn:up]

            qt = RCube._createBmTable(beamtab)

            return fits.BinTableHDU(qt, tableheader)
        
    @staticmethod
    def _createBmTable(beamtab):
        qt = QTable()        
        qt['BMAJ'] = (beamtab['BMAJ']).astype('float32')*u.arcsec
        qt['BMIN'] = (beamtab['BMIN']).astype('float32')*u.arcsec
        qt['BPA'] = (beamtab['BPA']).astype('float32')*u.deg
        qt['CHAN'] = np.arange(len(beamtab['BPA']), dtype = 'int32')
        qt['POL'] = np.zeros(len(beamtab['BPA']), dtype = 'int32')
        return qt
    
    def appendBeamTable(self, beamHDU):
        try:
            if not self._open:
                raise alreadyClosed()
            elif len(self._cube) > 1:
                raise tableExists()
            elif self._cube[0].shape[-3] != len(beamHDU.data):
                raise tableDimMismatch()
                
                
        except alreadyClosed:
            log.info('table is closed, cannot append beamtable')
        except tableExists:
            log.info('A table exists in the data')
        except tableDimMismatch:
            log.info('Number table entries does not match the data dimension')
            
        else:
            self._cube.append(beamHDU)
                
    def _split_3ax(self, xlims=None, ylims=None, chlims=None):
        
        if chlims is not None:
            cdn, cup = chlims
            # log.info('in crc brnc')
        else:
            cdn, cup = 0, self.getNchan()
        
        if xlims is not None:
            xdn, xup = xlims
            # log.info('in crx brnc')
        else:
            xdn, xup = 0, self.getXdim()

        if ylims is not None:
            ydn, yup = ylims
            # log.info('in cry brnc')
        else:
            ydn, yup = 0, self.getYdim()
        
        return (cdn, cup), (ydn, yup), (xdn, xup)
        
    def write_3dChunk(self, ptname, xlims=None, ylims=None, chlims=None, overwrite = False):
        '''
        split out a section of a cube
        
        Parameters
        ----------
        ptname : string
            name/path of the new cube
            
        xlims : pair of ints
            range of x pixels to be split [xlims[0], xlims[1])
            
        ylims : pair of ints
            range of y pixels to be split [ylims[0], ylims[1])
            
        chlims : pair of ints
            range of channels to be split [chlims[0], chlims[1])
            
        overwrite : bool
            whether or not replace the file if it exists
            
        '''
        log.info(f'file {ptname} has lims: {xlims}, {ylims}, {chlims}')
        cl, yl, xl = self._split_3ax(xlims=xlims, ylims=ylims, chlims=chlims)
        hdrObj = FitsHeader(self.getHeader())
        imheader = hdrObj.spatialSplitHeader(xl, yl, cl)
        
        if not '.fits' in ptname:
            ptname = ptname + '.fits'
        
        if os.path.exists(ptname):
            if overwrite:
                log.info('Stream cube ({}) exists going to remove and recalculate'.format(ptname))
                os.remove(ptname)
                time.sleep(.1)
            else:
                log.info(f'not allowed to overwrite the cube {ptname}')
                sys.exit(1)
        
        wcube = WCube(ptname)
        wcube.openW(imheader)
        # log.info(f'try cube {cl}, {yl}, {xl}')
        wcube._appendXY(self, cl, yl, xl)
        wcube.close()    
        
        log.info(f'wrote cube {ptname}')
        
        if len(self._cube) > 1:
            log.info(f'Adding the beam table to the 3DSplit FITS file: {ptname}')
            hdrObjTb = FitsHeader(self.getHeader(1))
            tbheader = hdrObjTb.getTableHeader(cl[1]-cl[0])
                
            beamtab = self._cube[1].data[cl[0]:cl[1]]
                    
            qt = RCube._createBmTable(beamtab)
                
            rcube = RCube(ptname)

            rcube.openR(update = True, mode = 'append')
            rcube.appendBeamTable(fits.BinTableHDU(qt, tbheader))
            rcube.close()
            
        time.sleep(1)
            
    def write_like(self, data, outfits, overwrite = False):
            
        imheader = self.getHeader().copy()
        
        log.info(f'Writing cube image: {outfits}')
        
        ccube = WCube(outfits, overwrite=overwrite)
        ccube.openW(imheader)
        ccube._appendCube(data)
        ccube.close()  
        
        if len(self._cube) > 1:
            log.info(f'Adding the restoring beam table to the FITS file: {outfits}')
            
            beamtab = self._cube[1].data
            qt = RCube._createBmTable(beamtab)
            
            rcube = RCube(outfits)
            rcube.openR(update = True, mode = 'append')
            rcube.appendBeamTable(fits.BinTableHDU(qt, self.getHeader(1).copy()))
            rcube.close()
            
        return outfits
        
        
    def _injectSrcs(self, srcLs, posLs):
        dimc, dimy, dimx = self.shape[-3:]
        chks = np.append(np.arange(0, len(srcLs), 100), len(srcLs))     
        inds = np.vstack((chks[:-1], chks[1:])).T
        for row in inds:
            for pos, src in zip(posLs[row[0]:row[1]], srcLs[row[0]:row[1]]):
                ch, y, x = pos
                dc, dy, dx = src.shape
                log.info(f'injecting a source at {ch}, {y}, {x} with dimensions {dc}, {dy}, {dx}')
                self._cube[0].data[0,ch:ch+dc, y:y+dy, x:x+dx] += src[:np.min([dimc-ch, dc]), :np.min([dimy-y, dy]), :np.min([dimx-x, dx])]
            self._cube.flush()
            
    def _injectFITS(self, pathList):
        mwcs = WCS(self._cube[0].header)
        dimc, dimy, dimx = self.shape[-3:]
        chks = np.append(np.arange(0, len(pathList), 100), len(pathList))     
        inds = np.vstack((chks[:-1], chks[1:])).T
        for row in inds:
            for pth in pathList[row[0]:row[1]]:
                srcFits = fits.open(pth)
                head = srcFits[0].header
                x, y, ch = (mwcs.world_to_pixel_values(np.array([[head['CRVAL1'], head['CRVAL2'], head['CRVAL3'], 0]]))[0][:3]).astype(int)
                dc, dy, dx = srcFits[0].shape
                log.info(f'injecting a source at {ch}, {y}, {x} with dimensions {dc}, {dy}, {dx}')
                self._cube[0].data[0,ch:ch+dc, y:y+dy, x:x+dx] += srcFits[0].data[:np.min([dimc-ch, dc]), :np.min([dimy-y, dy]), :np.min([dimx-x, dx])]
                srcFits.close()
            self._cube.flush()
            
    def _checkDivisible(self, spectral, spatial, spc_overlap, spt_overlap):
        cl, yl, xl = self._split_3ax()
        spcd = cl[1] % spectral == 0
        csp = np.linspace(0, cl[1], spectral+1, dtype = int)
        crng = np.transpose(np.vstack((csp[:-1], csp[1:])))
        if spectral>1:
            crng[1:,0] = crng[1:,0] - spc_overlap
            crng[:-1,1] = crng[:-1,1] + spc_overlap
        spyd = yl[1] % spatial[0] == 0
        ysp = np.linspace(0, yl[1], spatial[0]+1, dtype = int)
        yrng = np.transpose(np.vstack((ysp[:-1], ysp[1:])))
        spxd = xl[1] % spatial[1] == 0
        xsp = np.linspace(0, xl[1], spatial[1]+1, dtype = int)
        xrng = np.transpose(np.vstack((xsp[:-1], xsp[1:])))
        if spatial[0]>1:
            yrng[1:,0] = yrng[1:,0] - spt_overlap
            yrng[:-1,1] = yrng[:-1,1] + spt_overlap
        if spatial[1]>1:
            xrng[1:,0] = xrng[1:,0] - spt_overlap
            xrng[:-1,1] = xrng[:-1,1] + spt_overlap
        log.info(f'cube divisible in spectral:{spcd}, y_dim:{spyd} and x_dim:{spxd}')
        log.info(f'cube dimensions: {self.shape}')
        return np.all([spcd, spyd, spxd]), (crng, yrng, xrng)
        
    def _prepareArgs(self, xrng, yrng, crng, folder, indiv = False):
        base = self.path.replace('.fits', '_{}.fits')
        pts = base.split('/')
        pts.insert(-1, '---')
        tmp = ''
        for i in pts:
            tmp = f'{tmp}{i}/'
        base = tmp.rstrip('/')
        if not folder is None:
            base = folder.rstrip('/') + '/---/' + base.split('/')[-1]
            
        if indiv:
            base = base.replace('/---/', '/part---/')
        else:
            base = base.replace('/---/', '/')
            
        lenc, leny, lenx = len(crng), len(yrng), len(xrng)
        nmlist = []
        crlist = []
        yrlist = []
        xrlist = []
        cnt = 1
        for ic, cr in enumerate(crng):
            for iy, yr in enumerate(yrng):
                for ix, xr in enumerate(xrng):
                    crlist.append(cr)
                    yrlist.append(yr)
                    xrlist.append(xr)
                    if indiv:
                        tmp = base.replace('{}', f'ch-{ic+1:02d}f{lenc:02d}_y-{iy+1:02d}f{leny:02d}_x-{ix+1:02d}f{lenx:02d}')
                        nmlist.append(tmp.replace('part---', f'part{cnt:03d}'))
                    else:
                        nmlist.append(base.replace('{}', f'ch-{ic+1:02d}f{lenc:02d}_y-{iy+1:02d}f{leny:02d}_x-{ix+1:02d}f{lenx:02d}'))
                    cnt += 1

        # log.info(f'flist {nmlist}')                
        return nmlist, xrlist, yrlist, crlist
        
            
    def splitCube(self, spectral = 1, spatial = 1, spc_overlap = 0, spt_overlap = 0,
                  overwrite = False, folder = None, indiv = False):
        try:
            spatial = int(spatial), int(spatial)
            log.info(f'splitting into {spatial[0]} parts in y and {spatial[1]} parts in x')
        except:
            log.info(f'splitting into {spatial[0]} parts in y and {spatial[1]} parts in x')
                
        divis, ranges = self._checkDivisible(spectral, spatial, spc_overlap, spt_overlap)
        if (spectral > 1 or np.any(np.array(spatial) > 1)) and divis:
            crng, yrng, xrng = ranges
            _, cpus = RCube._getHWinfo()
            args = self._prepareArgs(xrng, yrng, crng, folder = folder, indiv = indiv)
            # log.info(f'files have lims: {np.hstack((crng, yrng, xrng))}')
            pool = Pool(cpus)
            pool.starmap(RCube.write_3dChunk, zip(repeat(self), *args, repeat(overwrite)))
            pool.close()
            
        else:
            log.info(f'splitCube with spectral = {spectral} and spatial = {spatial} does not write anything out')

    def cropNaNmask(self, folder = None, overwrite = False):
        base = self.path.replace('.fits', '_NaNmask.fits')
        if folder is not None:
            base = folder.rstrip('/') + '/' + base.split('/')[-1]
        
        for ptname in [base]:
            if os.path.exists(ptname):
                if overwrite:
                    log.info('Stream cube ({}) exists going to remove and recalculate'.format(ptname))
                    os.remove(ptname)
                    time.sleep(.1)
                else:
                    log.info(f'not allowed to overwrite the cube {ptname}')
                    sys.exit(1)
                    
        ccube = WCube(base)
        
        nanMask = np.isnan(self._cube[0].data[0, -1])
        
        xlim = np.all(nanMask, axis = 1)
        ylim = np.all(nanMask, axis = 0)
        xind = np.where(np.logical_not(xlim))[0]
        yind = np.where(np.logical_not(ylim))[0]
        xlims = [xind[0]-1, xind[-1]+1]
        ylims = [yind[0]-1, yind[-1]+1]
        
        hdrObj = FitsHeader(self.getHeader())
        imheader = hdrObj.spatialSplitHeader(xlims, ylims)
        
        ccube.openW(imheader)
        
        for i in range(self.shape[-3]):
            img = np.array(self._cube[0].data[0, i])
            img[nanMask] = np.nan
            ccube._appendData(img[ylims[0]:ylims[1], xlims[0]:xlims[1]])        
        
        ccube.close() 
        
        log.info(f'wrote cube {base}')
        
        if len(self._cube) > 1:
            log.info(f'Adding the beam table to the NaNmasked FITS file: {base}')
            
            beamtab = self._cube[1].data
            qt = RCube._createBmTable(beamtab)
            
            rcube = RCube(base)
            rcube.openR(update = True, mode = 'append')
            rcube.appendBeamTable(fits.BinTableHDU(qt, self.getHeader(1).copy()))
            rcube.close()
            
        log.info(f'Finished writing the NaNmasked FITS file: {base}')
        
        
    def apply_pb(self, band, folder = None, overwrite = False, undo = False):
        base = self.path.replace('.fits', '_{}.fits')
        if folder is not None:
            base = folder.rstrip('/') + '/' + base.split('/')[-1]
            
        if undo:
            corrnm = base.format('pbuncorr')
        else:
            corrnm = base.format('pbcorr')
        pbcnm = base.format('pbeam')
            
        from katbeam import JimBeam
    
        beam = None
        if band == 'L' or band == 'l':
            beam = JimBeam('MKAT-AA-L-JIM-2020')
        else:
            beam = JimBeam('MKAT-AA-UHF-JIM-2020')
            
        imheader = self.getHeader().copy()
        
        for ptname in [corrnm, pbcnm]:
            if os.path.exists(ptname):
                if overwrite:
                    log.info('Stream cube ({}) exists going to remove and recalculate'.format(ptname))
                    os.remove(ptname)
                    time.sleep(.1)
                else:
                    log.info(f'not allowed to overwrite the cube {ptname}')
                    sys.exit(1)
                    
        x,y = self._prepare_pb()
        
        fbeam = lambda chanf: beam.I(x, y, chanf)
        
        ccube = WCube(corrnm)
        pcube = WCube(pbcnm)
        
        ccube.openW(imheader)
        pcube.openW(imheader)
        
        _, cpus = RCube._getHWinfo()
        freq = self.retFreq()
        inds = np.append(np.arange(0, len(freq), cpus, dtype = int), len(freq))
        pool = Pool(cpus)
        
        for i in range(len(inds)-1):
            if undo:
                res = pool.starmap(RCube._unpb_chan, zip(repeat(self), np.arange(inds[i], inds[i+1]), repeat(fbeam), freq[inds[i]:inds[i+1]]))
            else:
                res = pool.starmap(RCube._pb_chan, zip(repeat(self), np.arange(inds[i], inds[i+1]), repeat(fbeam), freq[inds[i]:inds[i+1]]))
            for item in res:
                ccube._appendData(item[0])
                pcube._appendData(item[1])
            log.info('chans {}-{} done'.format(inds[i], inds[i+1]-1))
            
        pool.close()
        
        # log.info(f'try cube {cl}, {yl}, {xl}')
        
        ccube.close()    
        pcube.close()    
        
        log.info(f'wrote cubes {corrnm}, {pbcnm}')
        
        if len(self._cube) > 1:
            log.info(f'Adding the beam table to the pb_applied FITS file: {corrnm}')
            
            beamtab = self._cube[1].data
            qt = RCube._createBmTable(beamtab)
            
            rcube = RCube(corrnm)
            rcube.openR(update = True, mode = 'append')
            rcube.appendBeamTable(fits.BinTableHDU(qt, self.getHeader(1).copy()))
            rcube.close()
        
    
    def _pb_chan(self, chan, fbeam, freq):
        pb = fbeam(freq)
        im = self._cube[0].data[0,chan]/pb    
        return im, pb
    
    def _unpb_chan(self, chan, fbeam, freq):
        pb = fbeam(freq)
        im = self._cube[0].data[0,chan]*pb    
        return im, pb
        
    def _prepare_pb(self):
        cel_wcs = WCS(self._cube[0].header).celestial
        
        freq = self.retFreq()
        ydim, xdim = self._cube[0].shape[-2:]
        yspan = cel_wcs.pixel_to_world(xdim//2,ydim-1).separation(cel_wcs.pixel_to_world(xdim//2,0)).value
        xspan = cel_wcs.pixel_to_world(0,ydim//2).separation(cel_wcs.pixel_to_world(xdim-1, ydim//2)).value
        xlin=np.linspace(-xspan/2.,xspan/2.,xdim)
        ylin=np.linspace(-yspan/2.,yspan/2.,ydim)
        return np.meshgrid(xlin,ylin)
        
    def retFreq(self):
        """
        Extract the part of the cube name that will be used in the name of
        the averaged cube

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            header object from the fits file

        Returns
        -------
        frequency
            a 1D numpy array of channel frequencies in MHz  
        """
        hdrObj = FitsHeader(self._cube[0].header)
        return hdrObj.retFreq()
    
    @staticmethod
    def _getHWinfo():
        return .9*int(os.environ['SLURM_MEM_PER_NODE'])/1024, int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        
    def __getParallelChans(self, mem, cpus):
        
        nchan, ydim, xdim = self._cube[0].shape[-3:]
        
        chan_size = (4*xdim*nchan*8)/(1024**3)
        numCh = mem/chan_size

        log.info('Memory required for 1 slice: {}GB'.format(chan_size))
        log.info('Number of slices possible to process in parallel using all memory: {}'.format(numCh))

        if numCh >= cpus:
            log.info('Number of cpus smaller than max number of parallel slice processes')
            threads = cpus
            parCh = numCh // cpus
            log.info('processing 1 slice per CPU'.format(parCh))

        elif numCh >= 1.0:
            log.info('Number of cpus larger than max number of parallel slice processes')
            threads = numCh
            parCh = 1
            log.info('processing in {} thread(s) (smaller than cpus available: {})'.format(threads, cpus))
        else:
            log.debug('Cannot even load one slice in the entire memory, more memory required')
            sys.exit(1)

        return threads, parCh
    
    def writeClipMask(self, thresh, folder = None, name = '', overwrite = False):
        base = self.path.replace('.fits', f'_{name}.fits')
        if folder is not None:
            base = folder.rstrip('/') + '/' + base.split('/')[-1]
            
        imheader = self.getHeader().copy()
        
        for ptname in [base]:
            if os.path.exists(ptname):
                if overwrite:
                    log.info('Stream cube ({}) exists going to remove and recalculate'.format(ptname))
                    os.remove(ptname)
                    time.sleep(.1)
                else:
                    log.info(f'not allowed to overwrite the cube {ptname}')
                    sys.exit(1)
                    
        ccube = WCube(base)
        
        ccube.openW(imheader)
        
        for i in range(self.shape[-3]):
            ccube._appendData(self._getChan(i) > thresh)
        
        ccube.close()  
        
        log.info(f'wrote cubes {base}')
            
        return base
    
    
class CLEAN():
    '''
    a class for performing Hogbom CLEAN on cubes using a fixed PSF to the requested threshold or by the number of iterations
    '''
    def __init__(self, cube, psf, mask, thresh, n_iter, gain = 0.1, log_residual = True):
        '''
        Parameters
        ----------
        cube : string
            path to the line cube for performing Hogbom CLEAN
        
        psf : string
            path to the PSF cube for performing Hogbom CLEAN
            
        mask : string
            path to the mask cube for performing Hogbom CLEAN
            
        thresh : float
            absolute cleaning threshhold in units of Jy
            
        n_iter : int
            number of CLEAN iterations
            
        gain : float
            CLEAN gain
        
        log_residual : bool
            write out the highest residual pixel during cleaning
            
        '''
        self.lne_path = cube
        self.psf_path = psf
        self.msk_path = mask
        _, self.cpus = RCube._getHWinfo() 
        self.thresh = thresh
        self.niter = n_iter
        self.gain = gain
        self.log = log_residual
        self.nlog = 1000
        
    def openCubes(self):
        '''
        open all the cubes
        '''
        with fits.open(self.lne_path) as fitsCbe:
            self.pix = fitsCbe[0].header['CDELT1'] * 3600
            
        self.lne = CLNCube(self.lne_path, self.pix)
        self.dims = self.lne.shape[-3:]
        self.nch = self.dims[0]
        self.psf = PSFCube(self.psf_path, self.dims)
        self.msk = MSKCube(self.msk_path, self.dims)
    
    def clean(self, folder = None, overwrite = False):
        inds = np.append(np.arange(0, self.nch, self.cpus), self.nch)
        pool = Pool(self.cpus)
        
        
        base = self.lne.path.replace('.fits', '_{}.fits')
        if folder is not None:
            base = folder.rstrip('/') + '/' + base.split('/')[-1]
            
        clnnm = base.format('clean')
        mdlnm = base.format('model')
        resnm = base.format('residual')
        
        imheader = self.lne.getHeader().copy()
        
        for ptname in [clnnm, mdlnm, resnm]:
            if os.path.exists(ptname):
                if overwrite:
                    log.info('Stream cube ({}) exists going to remove and recalculate'.format(ptname))
                    os.remove(ptname)
                    time.sleep(.1)
                else:
                    log.info(f'not allowed to overwrite the cube {ptname}')
                    sys.exit(1)
            
        ocubes = [WCube(cbe) for cbe in [clnnm, mdlnm, resnm]] 
        
        [cbe.openW(imheader) for cbe in ocubes]
        
        for st,en in zip(inds[:-1],inds[1:]):
            resPool = pool.starmap(CLEAN._chCLN, zip(repeat(self), np.arange(st, en)))
            
            for ind, pair in enumerate(resPool):
                [cbe._appendData(pr) for cbe, pr in zip(ocubes, pair)]
    
        pool.close()
        
        [cbe.close() for cbe in ocubes]
        
        if len(self.lne._cube) > 1:
            for cbnm in [clnnm, mdlnm, resnm]:
                log.info(f'Adding the beam table to the pb_applied FITS file: {cbnm}')

                beamtab = self.lne._cube[1].data
                qt = RCube._createBmTable(beamtab)

                rcube = RCube(cbnm)
                rcube.openR(update = True, mode = 'append')
                rcube.appendBeamTable(fits.BinTableHDU(qt, self.lne.getHeader(1).copy()))
                rcube.close()
    
    def setNlog(self, nlog):
        '''
        change the number of CLEAN steps between log lines
        
        Parameters
        ----------
        nlog : int
            number of CLEAN steps between log lines
        '''
        self.nlog = int(nlog)
                
    def _chCLN(self, ch):
        
        dr = self.lne.getCh(ch)
        mk = self.msk.getCh(ch)
        ps = self.psf.getCh(ch)
        if len(self.lne._cube) > 1:
            bm = self.lne.getBM(ch)
        else:
            bm = self.psf.getBM(ch)
            
        # the order is clean, model, residual
        outs = [np.zeros_like(dr), np.zeros_like(dr), dr]
        
        n = 0
        crit = True
        while n <= self.niter and crit:
            
            if n%self.nlog == 0 and self.log:
                indx, indy = np.unravel_index(np.nanargmax(np.absolute(outs[2]*mk)), outs[2].shape)
                log.info("chan {:<4d} & step {:<7d} - residual max is at {:<5d},{:<5d} with value: {:.7e}".format(ch, n, indx, indy, outs[2][indx, indy]))
                # print(f'clSum: {np.sum(outs[0])}, modSum: {np.sum(outs[1])}, resSum = {np.sum(outs[2])}')
                
                
            crit, outs[0], outs[1], outs[2] = self._chSTP(outs[0], outs[1], outs[2], ps, mk, bm)
            n += 1
            
        log.info("{:<6d} iterations DONE for channel {:<4d}".format(n-1, ch))
        
        # print(f'clSum: {np.sum(outs[0] + outs[2])}, modSum: {np.sum(outs[1])}, resSum = {np.sum(outs[2])}')
        
        return outs[0] + outs[2], outs[1], outs[2]
    
    def _chSTP(self, cln, mdl, res, ps, mk, bm):
        immask = res * mk
        indx, indy = np.unravel_index(np.nanargmax(np.absolute(immask)), immask.shape)
        midx, midy = ps.shape[0]//2, ps.shape[1]//2   
        # hght = res[indx, indy]*self.gain
        hght = np.around(res[indx, indy]*self.gain, decimals = 10)
        # hght = np.around(res[indx, indy]*self.gain, decimals = np.random.randint(8,13))
        
        if np.abs(res[indx, indy]) >= self.thresh:
            res -= hght*ps[midx-indx:midx-indx+midx,midy-indy:midy-indy+midy]
            # res = np.around(res - hght*ps[midx-indx:midx-indx+midx,midy-indy:midy-indy+midy], decimals = np.random.randint(8,13))
            mdl[indx, indy] += hght
            cln += hght*bm[midx-indx:midx-indx+midx,midy-indy:midy-indy+midy]
            return True, cln, mdl, res
        else:
            return False, cln, mdl, res
            
    def close(self):
        '''
        close the line, psf, and msk cubes.
        '''
        self.lne.close()
        self.psf.close()
        self.msk.close()
            
class CLNCube(RCube):
    def __init__(self, path, pix):
        super().__init__(path)
        self.openR(memmap = True)
        self.pix = pix
        
    def getCh(self, ch):
        if len(self.shape) == 4:
            dimage = self._cube[0].data[0, ch]
        elif len(self.shape) == 3:
            dimage = self._cube[0].data[ch]
        elif len(self.shape) == 2:
            if ch == 0:
                dimage = self._cube[0].data
            else:
                log.debug('the image has {} dimension(s), but channel {} has been requested'.format(len(self.shape), ch))
                sys.exit(1)
                
        return dimage
    
    def getBM(self, ch):
        px, py = self.shape[-2]*2, self.shape[-1]*2
        beamPars = self._cube[1].data[ch]
        fac = 1/(2*np.sqrt(2*np.log(2)))
        return CLNCube._gaussian(1, px//2, py//2, fac*beamPars[0]/self.pix, fac*beamPars[1]/self.pix, beamPars[2]/180*np.pi)(*np.indices((px,py)))
         
    @staticmethod
    def _gaussian(amp, cen_x, cen_y, wid_x, wid_y, rot):
        wid_x = float(wid_x)
        wid_y = float(wid_y)
        cen_xr = cen_x * np.cos(rot) - cen_y * np.sin(rot)
        cen_yr = cen_y * np.cos(rot) + cen_x * np.sin(rot)
        return lambda x,y: amp*np.exp(
                    -0.5*(np.square((cen_xr-(x * np.cos(rot) - y * np.sin(rot)))/wid_x)+np.square((cen_yr-(y * np.cos(rot) + x * np.sin(rot)))/wid_y)))


class PSFCube(RCube):
    def __init__(self, path, dims):
        super().__init__(path)
        self.openR(memmap = True)
        self.pix = self._cube[0].header['CDELT1'] * 3600
        self.dims = dims
        if np.all(self.shape[-3] == dims[-3]):
            self.match = 0
        elif self.shape[-3] == 1:
            self.match = 1
        else:
            log.error(f'psf dimensions do not comply with CLEAN requirements, {self.shape}')
            sys.exit(1)
            
            
    def getCh(self, ch):
        px, py = self.shape[-2:]
        imx, imy = self.dims[-2:]
        psfChan = np.zeros((imx*2, imy*2))
        midx = px//2
        midy = py//2
        if len(self.shape) == 4:
            if self.match == 1:
                psfTemp = self._cube[0].data[0, 0]
            else:
                psfTemp = self._cube[0].data[0, ch]
        elif len(self.shape) == 3:
            if self.match == 1:
                psfTemp = self._cube[0].data[0]
            else:
                psfTemp = self._cube[0].data[ch]
        elif len(self.shape) == 2:
            psfTemp = self._cube[0].data
        
        if px > imx*2 and py > imy*2:
            xr, yr = midx-imx, midy-imy
            psfChan += psfTemp[xr:-xr, yr:-yr]
        elif px == imx*2 and py == imy*2:
            psfChan += psfTemp
        else:
            xr, yr = imx-midx, imy-midy
            psfChan[xr:-xr, yr:-yr] += psfTemp
            
        return psfChan
    
    def getBM(self, ch):
        px, py = self.dims[-2]*2, self.dims[-1]*2
        beamPars = self._cube[1].data[ch]
        # print(f'data len = {len(self._cube[1].data)}, ch = {ch}')
        fac = 1/(2*np.sqrt(2*np.log(2)))
        return PSFCube._gaussian(1, px//2, py//2, fac*beamPars[0]/self.pix, fac*beamPars[1]/self.pix, beamPars[2]/180*np.pi)(*np.indices((px,py)))
         
    @staticmethod
    def _gaussian(amp, cen_x, cen_y, wid_x, wid_y, rot):
        wid_x = float(wid_x)
        wid_y = float(wid_y)
        cen_xr = cen_x * np.cos(rot) - cen_y * np.sin(rot)
        cen_yr = cen_y * np.cos(rot) + cen_x * np.sin(rot)
        return lambda x,y: amp*np.exp(
                    -0.5*(np.square((cen_xr-(x * np.cos(rot) - y * np.sin(rot)))/wid_x)+np.square((cen_yr-(y * np.cos(rot) + x * np.sin(rot)))/wid_y)))
        
        
        
class MSKCube(RCube):
    def __init__(self, path, dims):
        super().__init__(path)
        self.openR(memmap = True)
        self.dims = dims
        if np.all(self.shape[-3:] == dims):
            self.match = 0
        elif np.all(self.shape[-2:] == dims[-2:]):
            self.match = 1
        else:
            log.error(f'mask dimensions do not comply with CLEAN requirements, {self.shape}')
            sys.exit(1)
        
    def getCh(self, ch):
        mx, my = self.shape[-2:]
        if len(self.shape) == 4 or len(self.shape) == 3:
            if self.match == 1:
                msk = self._getChan(0)
            else:
                msk = self._getChan(ch)
        elif len(self.shape) == 2:
            if self.match == 1:
                msk = self._cube[0].data
            else:
                log.debug('the mask has {} dimension(s), but channel {} has been requested'.format(len(self.shape), ch))
                sys.exit(1)
                
        return msk.astype('f')
         
          
class FitsHeader():
    def __init__(self, header):
        self._header = header.copy()
        
    def retFreq(self):
        """
        Extract the part of the cube name that will be used in the name of
        the averaged cube

        Parameters
        ----------
        header : `~astropy.io.fits.Header`
            header object from the fits file

        Returns
        -------
        frequency
            a 1D numpy array of channel frequencies in MHz  
        """
        
        if not ('TIMESYS' in self._header):
            self._header['TIMESYS'] = 'utc'
        elif self._header['TIMESYS'] != 'utc':
            self._header['TIMESYS'] = 'utc'
        freqDim = self._header['NAXIS3']
        wcs3d=WCS(self._header)
        try:
            wcsfreq = wcs3d.spectral
        except:
            wcsfreq = wcs3d.sub(['spectral'])   
        return np.around(wcsfreq.pixel_to_world(np.arange(0,freqDim)).to(u.MHz).value, decimals = 7)
        
    def getAppendHeader(self, nchan):
        return self.spectralSplitHeader(nchan, orig = 'append_fits')
    
    def getTableHeader(self, nchan):
        self._header['NAXIS2'] = nchan
        self._header['NCHAN'] = nchan
        if 'OBSERVER' in list(self._header.keys()):
            self._header.remove('OBSERVER')
        self._header['DATE'] = str(datetime.datetime.now()).replace(' ','T')
        self._header['ORIGIN'] = 'A. Kazemi-Moridani (table_header)'
        return self._header
    
    def getCombineHeader(self, dimx, dimy):
        self._header['NAXIS1'] = int(dimx)
        self._header['NAXIS2'] = int(dimy)
        xcen = dimx//2+1
        ycen = dimy//2+1
        # self._header['CRPIX1'] = xcen 
        # self._header['CRPIX2'] = ycen
        if 'OBSERVER' in list(self._header.keys()):
            self._header.remove('OBSERVER')
        self._header['DATE'] = str(datetime.datetime.now()).replace(' ','T')
        self._header['ORIGIN'] = 'A. Kazemi-Moridani (combine_spatial)'
        return self._header
    
    
    def getPrimeHeader(self, nchan, ydim, xdim, mask = False, orig = 'prime_header'):
        self._header['NAXIS1'] = int(xdim)
        self._header['NAXIS2'] = int(ydim)
        self._header['NAXIS3'] = int(nchan)
        if mask:
            header['BITPIX'] = 8
            if orig == 'prime_header':
                orig = 'mask_header'
        if 'OBSERVER' in list(self._header.keys()):
            self._header.remove('OBSERVER')
        self._header['DATE'] = str(datetime.datetime.now()).replace(' ','T')
        self._header['ORIGIN'] = f'A. Kazemi-Moridani ({orig})'
        return self._header
    
    def spectralSplitHeader(self, nchan, sfreq = None, orig = 'spectral_split'):
        self._header['NAXIS3'] = nchan
        if sfreq != None:
            self._header['CRVAL3'] = sfreq
        if 'OBSERVER' in list(self._header.keys()):
            self._header.remove('OBSERVER')
        self._header['DATE'] = str(datetime.datetime.now()).replace(' ','T')
        self._header['ORIGIN'] = f'A. Kazemi-Moridani ({orig})'
        return self._header
    
    def spatialSplitHeader(self, xlims, ylims, chans = None):
        xdn, xup = xlims
        ydn, yup = ylims
        freq = self.retFreq()
        if chans != None:
            chdn, chup = chans
            self._header['NAXIS3'] = chup - chdn #make sure this is correct it was 'chup - chdn + 1' before
            self._header['CRVAL3'] = freq[chdn]*1e6
        xor = self._header['CRPIX1'] 
        yor = self._header['CRPIX2'] 
        self._header['CRPIX1'] = xor - xdn
        self._header['CRPIX2'] = yor - ydn
        self._header['NAXIS1'] = xup-xdn
        self._header['NAXIS2'] = yup-ydn
        if 'OBSERVER' in list(self._header.keys()):
            self._header.remove('OBSERVER')
        self._header['DATE'] = str(datetime.datetime.now()).replace(' ','T')
        self._header['ORIGIN'] = 'A. Kazemi-Moridani (spatial_split)'
        return self._header