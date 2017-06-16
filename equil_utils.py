# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 17:02:07 2016

@author: gawe
"""
# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #

# This section is to improve compatability between bython 2.7 and python 3x
from __future__ import absolute_import, division, print_function, \
    unicode_literals

# ----------------------------------------------------------------- #

import os as _os
import numpy as _np
from osa import Client as _cl
from pyversion import version as _ver
import getpass as _gp


from .. import utils as _ut
from ..Struct import Struct
# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #


class VMECrest(Struct):

    # Import the VMEC rest client
    vmec = _cl("http://esb.ipp-hgw.mpg.de:8280/services/vmec_v6?wsdl")
    rooturl = 'http://10.66.24.160/'
    resturl = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/run/'
    url = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/'

    def __init__(self, shortID=None, coords=None, verbose=True):
        self.verbose = verbose
        self.shortID = shortID
        self.coords = coords

        self.roa_grid = None
        self.torPsi_grid = None
        self.Pres_grid = None
        self.iota_grid = None
        self.dVdrho_grid = None
        self.Vol_grid = None
        self.Vol_lcfs = None
        if shortID is not None:
            self.vmecid = shortID
            self.currents = self.getCoilCurrents()
            self.getVMECgridDat()
        # endif
        if coords is not None:
            self.setCoords(coords)
            # self.getCoordData()
        # endif
    # enddef __init__

    # ----------------------------------------- #
    # ----------------------------------------- #

    def getCoilCurrents(self, vmecid=None):
        if vmecid is None:
            vmecid = self.vmecid
        # endif
        self.currents = self.vmec.service.getCoilCurrents(vmecid)
        return self.currents

    def loadCoilCurrents(self, experimentID):
        from . import w7x_currents as _curr
        
        self.currents = _curr.get_w7x_currents(experimentID)
        return self.currents

    def getFluxSurfaces(self, torFi=None, numPoints=100, vmecid=None,
                        s=None):
        if vmecid is None:
            vmecid = self.vmecid
        # endif
        if torFi is None:
            torFi = 0.0
            if self.coords is not None:
                torFi = self.fi
            # endif
        # endif
        if s is None:
            s = self.s
        # endif

        fsFi = self.vmec.service.getFluxSurfaces(vmecid, torFi, s,
                                                 numPoints)
        return fsFi

    def getVMECinput(self, vmecid=None):
        if vmecid is None:
            vmecid = self.vmecid
        # endif
        inp = self.vmec.service.getVmecRunData(vmecid, 'input')
        return inp

    def getVMEClog(self, vmecid=None):
        if vmecid is None:
            vmecid = self.vmecid
        # endif
        log = self.vmec.service.getVmecRunData(vmecid, 'log')
        return log

    def getVMECthreed(self, vmecid=None):
        if vmecid is None:
            vmecid = self.vmecid
        # endif
        threed = self.vmec.service.getVmecRunData(vmecid, 'threed1')
        return threed

    def getVMECwouttxt(self, vmecid=None):
        if vmecid is None:
            vmecid = self.vmecid
        # endif
        wout = self.vmec.service.getVmecRunData(vmecid, 'wout')
        return wout

    def getVMECwoutnc(self, vmecid=None):
        if vmecid is None:
            vmecid = self.vmecid
        # endif
        wout = self.vmec.service.getVmecOutputNetcdf(vmecid)
        return wout

    def getVMECboozer(self, vmecid=None):
        if vmecid is None:
            vmecid = self.vmecid
        # endif
        booz = self.vmec.service.getVmecRunData(vmecid, 'boozer')
        return booz

    def getVMECepseff(self, vmecid=None):
        if vmecid is None:
            vmecid = self.vmecid
        # endif
        epseff = self.vmec.service.getVmecRunData(vmecid, 'epseff')
        return epseff

    def getVMECfieldperiod(self, vmecid=None):
        if vmecid is None:
            vmecid = self.vmecid
        # endif
        FP = self.vmec.service.getFieldPeriod(vmecid)
        return FP
        
    def get_new_id(self):
    
        # get all identifiers so far
        vmec_ids = self.vmec.service.listIdentifiers().VmecIds
        
        # get current username
        user_id = _gp.getuser()
        
        # get all runs associated with the current user
        matching = [s for s in vmec_ids if user_id in s]
        
        # get number of runs which are associated with the current user
        num_runs = _np.size(matching)
        
        # generate new id: username + "_" + (number+1)
        new_id = user_id + "_" + _np.str(num_runs+1)
        
        # debug output
        if (self.verbose):
            print("issuing vmec id " + new_id)
            
        return new_id
        
    def get_new_Mgrid_id(self, comment=''):
    
        # get all identifiers so far
        mgrid_ids = self.vmec.service.listIdentifiers().MgridIds
        
        # get current username
        user_id = _gp.getuser()
        
        # get all runs associated with the current user
        matching = [s for s in mgrid_ids if user_id in s]
        
        # get number of runs which are associated with the current user
        num_runs = _np.size(matching)
        
        # generate new id: username + "_" + (number+1)
        new_id = user_id + "_" + comment + '_' + _np.str(num_runs+1)
        
        # debug output
        if (self.verbose):
            print("issuing Mgrid id " + new_id)
            
        return new_id

    # ----------------------------------------- #
    # ----------------------------------------- #

    def setCoords(self, coords):
        """
        Input cartesian coords as column vectors shape(# points, 3)
        """
        self.nn = _np.size(coords, axis=0)
        self.XX = coords[:, 0]
        self.YY = coords[:, 1]
        self.ZZ = coords[:, 2]

        # Load up the webservices Points3D cartesian object
        self._pcart = self.vmec.types.Points3D(self.nn)
        self._pcart.x1 = self.XX
        self._pcart.x2 = self.YY
        self._pcart.x3 = self.ZZ

        # Load up the webservices Points3D cylindrical object
        self.fi, self.RR = _ut.cart2pol(self.XX, self.YY)

        self._pcyl = self.vmec.types.Points3D(self.nn)
        self._pcyl.x1 = self.RR
        self._pcyl.x2 = self.fi
        self._pcyl.x3 = self.ZZ

        self.getCoordData()
    # enddef

    def getCoordData(self, tol=3e-3):
        # Run
        self.getBcart()
        self.getmodB()
        self.getbhat()

        # Get the VMEC coordinates at the input points
        self.getVMECcoord(tol)

        # Get the quantities that exist on the VMEC grid
        # Then linearly interpolate them to the coordinate positions
        self.getVMECgrid()   # Toroidal flux on the VMEC grid
        self.getgridroa()    # normalized eff. radius on VMEC grid
        self.getFluxTor()    # Interpolate toroidal flux to coords
        self.getPressure()   # Get pressure on grid / at coords
        self.getiota()       # Get rot. transform on grid / at coords
        self.getVol()        # get Volume, dVdrho, Vol_lcfs on grid/coords
        return 1

    # ----------------------------------------- #

    def getBcart(self):
        # Magnetic field utility needs cartesian coordinates

        # Call the webservices magnetic field utility
        b = self.vmec.service.magneticField(self.vmecid, self._pcart)
        self.Bxyz = _np.array(_np.vstack((b.x1, b.x2, b.x3)),
                              dtype=_np.float64)
        self.Bxyz = self.Bxyz.transpose()
        return self.Bxyz

    def getmodB(self):
        # Calculate the magnetic field strength
        self.modB = _np.linalg.norm(self.Bxyz, ord=2, axis=1)
        return self.modB

    def getbhat(self):
        # Calculate the magnetic field unit vector
        modB = self.modB.copy()
        self.bhat = self.Bxyz / (modB.reshape(self.nn, 1) *
                                 _np.ones((1, 3), dtype=_np.float64))
        return self.bhat

    # ----------------------------------------- #

    def getVMECcoord(self, tol=3e-3):
        # Get the VMEC coordinates at each point: s,th,zi
        vmec_coords = \
            self.vmec.service.toVMECCoordinates(self.vmecid,
                                                self._pcyl, tol)
        self.s = _np.array(vmec_coords.x1, dtype=_np.float64)
        self.th = _np.array(vmec_coords.x2, dtype=_np.float64)
        self.zi = _np.array(vmec_coords.x3, dtype=_np.float64)
        self.roa = _np.sqrt(self.s)
        return self.roa

    def CalcFluxCart(self, coords=None):
        if coords is not None:
            self.setCartCoords(coords)
            self.runCalc()
        # endif
        return self.roa

    # ----------------------------------------- #

    def getVMECgridDat(self):
        self.getVMECgrid()
        self.getgridroa()
        self.getgridpres()
        self.getgridiota()
        self.getgridVol()
        self.getgridVol_lcfs()
    # end def

    # ----------------------------------------- #

    def getVMECgrid(self):
        # Get the toroidal flux on the VMEC grid
        torPsi_grid = \
            self.vmec.service.getToroidalFluxProfile(self.vmecid)
        self.torPsi_grid = _np.array(torPsi_grid, dtype=_np.float64)
        return self.torPsi_grid

    def getgridroa(self):
        # This is the natural grid calculated by VMEC (no inversion)
        roa_grid = _np.sqrt(self.torPsi_grid / self.torPsi_grid[-1])
        self.roa_grid = roa_grid
        return roa_grid

    def getFluxTor(self):
        # Linearly interpolate to the new spatial coordinates
        self.torPsi = _np.interp(self.roa, self.roa_grid,
                                 self.torPsi_grid)
        return self.torPsi

    # ---

    def getgridpres(self):
        self.Pres_grid = \
            self.vmec.service.getPressureProfile(self.vmecid)
        self.Pres_grid = _np.array(self.Pres_grid, dtype=_np.float64)

        return self.Pres_grid

    def getPressure(self):
        if self.Pres_grid is None:
            self.Pres_grid = self.getgridpres()
        self.Pres = _np.interp(self.roa, self.roa_grid,
                               self.Pres_grid)
        return self.Pres

    # ---

    def getgridiota(self):
        self.iota_grid = self.vmec.service.getIotaProfile(self.vmecid)
        self.iota_grid = _np.array(self.iota_grid, dtype=_np.float64)
        return self.iota_grid

    def getiota(self):
        if self.iota_grid is None:
            self.iota_grid = self.getgridiota()
        self.iota = _np.interp(self.roa, self.roa_grid,
                               self.iota_grid)
        return self.iota

    # ---

    def getgriddVdrho(self):
        # Get the VMEC volume on the VMEC grid
        #   There is an error in the Volume calculation!!!
        #   This actually returns the derivative

        # As long as there is an error in the getVolumeProfile webservice
        if 1:
            # on s-grid
            self.dVds_grid = self.vmec.service.getVolumeProfile(self.vmecid)
        else:
            self.dVds_grid = self.vmec.service.getvp(self.vmecid)
        # endif
        # Undo normalization from VMEC (2*pi*2*pi)
        self.dVds_grid = _np.array(self.dVds_grid, dtype=_np.float64)
        self.dVds_grid = (4*_np.pi**2)*self.dVds_grid

        # Convert to jacobian for integration
        self.dVdrho_grid = 2*self.roa_grid*self.dVds_grid

        return self.dVdrho_grid

    def getgridVol(self):
        if self.dVdrho_grid is None:
            self.dVdrho_grid = self.getgriddVdrho()  # endif

        # As long as there is an error in the getVolumeProfile webservice
        if 1:
            dxvar = _np.diff(self.roa_grid)
            yvar = 0.5*(self.dVdrho_grid[:-1]+self.dVdrho_grid[1:])
            self.Vol_grid = _np.cumsum(yvar*dxvar)
            self.Vol_grid = _np.hstack((0, self.Vol_grid))
        else:
            self.Vol_grid = self.vmec.service.getVolumeProfile(self.vmecid)
            self.Vol_grid = _np.array(self.Vol_grid, dtype=_np.float64)
        # endif
        return self.Vol_grid

    def getgridVol_lcfs(self):
        # 4pi^2 needs to be taken into account
        self.Vol_lcfs = self.vmec.service.getVolumeLCFS(self.vmecid)
        self.Vol_lcfs = _np.array(self.Vol_lcfs, dtype=_np.float64)
        return self.Vol_lcfs

    def getVol(self):
        # Get the VMEC volume on the VMEC grid
        #   There is an error in the Volume calculation!!!
        #   This actually returns the derivative
        if self.dVdrho_grid is None:
            self.getgridVol()  # endif
        self.dVdrho = _np.interp(self.roa, self.roa_grid,
                                 self.dVdrho_grid)

#        self.Vol_grid = _np.trapz(self.dVdrho_grid, x=self.roa_grid)
#        self.Vol = _np.interp(self.roa, self.roa_grid, self.Vol_grid)

        if self.Vol_lcfs is None:
            self.getgridVol_lcfs()  # endif
        return self.Vol_lcfs, self.dVdrho

    # ------------------------------------------------- #
    
    def createMgrid(self, id, magconf, minR, maxR, minZ, maxZ, resR, resZ, resPhi,
                    fieldPeriods=5, isStellaratorSymmetric=True):
        
        return_id = self.vmec.service.createMgrid(magconf, minR, maxR, resR, minZ, maxZ, resZ, resPhi,
                                      isStellaratorSymmetric, fieldPeriods, id)
        if return_id == id:
            return return_id
        else:
            print("returned ID from Mgrid does not match given: "+return_id)
            return return_id
        
        

#    def plotSurfaces(self):
#
#        return _ax

# end class VMECrest

# ----------------------------------------------------------------- #
# ----------------------------------------------------------------- #

def url_exists(request):
    resp = None
    try: 
        resp = _ver.urllib.urlopen(request)  # @UndefinedVariable
    except: 
#        raise
        raise #Exception()
    finally:
        if resp is not None:
            resp.close()
            urlexists = True
        else:
            urlexists = False
        #endif
    #end try
    return urlexists 

class w7xfield(Struct):

    nwindings = [108, 108, 108, 108, 108, 36, 36]
    url = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/'
    vmec = _cl("http://esb.ipp-hgw.mpg.de:8280/services/vmec_v5?wsdl")

    def __init__(self, currents=None, verbose=True):
        self.verbose = verbose
        if currents is not None:
            self.currents = currents
            self.localdir = ''
            self.ratios = \
                _np.int64(1e3*_np.round(currents/currents[0],
                                        decimals=3))
            self.pickW7Xconfig()

            # vmecname = self.getVMECname()
            [self.RefRuns, self.shortIDs] = self.getVMECrun()
        # endif
    # enddef __init__

        ####################

    def getVMECname(self, ratios=None):
        if ratios is None:
            ratios = self.ratios
        # endif
        vmecname = '%4i_%4i_%4i_%4i_%0+5i_%0+5i' % tuple(ratios[1:])
        return vmecname

#    def getSHORTid(self):

    def pickVMECfile(self, ratios=None):
        if ratios is None:
            ratios = self._ratiosForVMECpicking
        # endif
        vmecname = self.getVMECname(ratios)
        kVMECfile01 = 'wout_w7x.%s.01.0000.txt' % (vmecname,)
        kVMECfile05 = 'wout_w7x.%s.05.0000.txt' % (vmecname,)

        # try to pick from webservice
        baseurl = 'http://svvmec1.ipp-hgw.mpg.de:8080'
        baseurl += '/vmecrest/v1/geiger/w7x/'
        folderurl01 = vmecname+'/01/0000/'
        folderurl05 = vmecname+'/05/0000/'
        try:
            url01 = baseurl+folderurl01+kVMECfile01
            existsUrl01 = url_exists(url01)
#            with _ver.urllib.urlopen(url01) as f:  # @UnusedVariable
#                existsUrl01 = True
        except IOError:
            existsUrl01 = False
        # endtry

        try:
            url05 = baseurl+folderurl05+kVMECfile05
            existsUrl05 = url_exists(url05)
#            with _ver.urllib.urlopen(url05) as f:  # @UnusedVariable
#                existsUrl05 = True
        except IOError:
            existsUrl05 = False
        # endtry

        if existsUrl05:
            return url05
        elif existsUrl01:
            return url01
        else:
            if self.verbose:
                print("No VMEC files found on the webservices." +
                      "Trying locally.")
            # endif
            localpath01 = _os.path.join(self.localdir, kVMECfile01)
            localpath05 = _os.path.join(self.localdir, kVMECfile05)
            if _os.path.exists(localpath05):
                return localpath05
            elif _os.path.exists(localpath01):
                return localpath01
            else:
                if self.verbose:
                    print("No VMEC files found locally. " +
                          "Returning None.")
                # endif
                return None
            # endif
        # endif
    # enddef pickVMECrestid

    def getVMECrun(self, ratios=None):
        if ratios is None:
            ratios = self._ratiosForVMECpicking  # endif
        vmecname = self.getVMECname(ratios)
        idlist = self.vmec.service.listIdentifiers()

        # ID number where magnetic configuration is already calculated
        ids = [idx for idx, fil in enumerate(idlist.ReferenceRuns)
               if (fil.find(vmecname) > 0)]
        RefRuns = [idlist.ReferenceRuns[ii] for ii in ids]
        shortIDs = [idlist.ReferenceShortIds[ii] for ii in ids]
        return RefRuns, shortIDs
    # enddef getVMECrun

        # --------------------------------------------------------- #

    # OLD AND DEPRECATED IN FAVOR OF THE VMEC REST SERVICE METHODS
    def __w7xconfigs__(self):
        # Coil currents per winding

        config = {'A': Struct(), 'B': Struct(),
                  'C': Struct(), 'D': Struct(),
                  'E': Struct(), 'F': Struct(),
                  'G': Struct(), 'H': Struct(),
                  'I': Struct(), 'J': Struct()}

        # Average magnetic field strength on the magnetic-axis
        avgB0 = _np.array([2.50], dtype=_np.float64)

        # On-axis magnetic field at phi=0
        B00 = _np.array([2.613, 2.596, 2.636, 2.499, 2.763, 2.711, 2.619,
                         2.592, 2.666, 2.603], dtype=_np.float64)

        # On-axis magnetic field at phi=25 degrees
        B25 = _np.array([2.428, 2.437, 2.417, 2.499, 2.339, 2.379, 2.422,
                         2.453, 2.397, 2.433], dtype=_np.float64)

        # On-axis magnetic field at phi=36 degrees
        B36 = _np.array([2.396, 2.415, 2.372, 2.499, 2.260, 2.323, 2.394,
                         2.437, 2.341, 2.407], dtype=_np.float64)

        config['A'].configname = 'configA - standard case'
        config['A'].currents = _np.array([13470, 13470, 13470, 13470, 13470,
                                          0, 0], dtype=_np.float64)
        config['A'].avgB0 = avgB0
        config['A'].B00 = B00[0]
        config['A'].B25 = B25[0]
        config['A'].B36 = B36[0]
#        config['A'].shortID = 'w7x_ref_1'

        config['B'].configname = 'configB - Low iota'
        config['B'].currents = _np.array([12200, 12200, 12200, 12200, 12200,
                                          9150, 9150], dtype=_np.float64)
        config['B'].avgB0 = avgB0
        config['B'].B00 = B00[1]
        config['B'].B25 = B25[1]
        config['B'].B36 = B36[1]
#        config['B'].shortID = 'w7x_ref_18'

        config['C'].configname = 'configC - High iota'
        config['C'].currents = _np.array([14880, 14880, 14880, 14880, 14880,
                                          -10260, -10260], dtype=_np.float64)
        config['C'].avgB0 = avgB0
        config['C'].B00 = B00[2]
        config['C'].B25 = B25[2]
        config['C'].B36 = B36[2]
#        config['C'].shortID = 'w7x_ref_15'

        config['D'].configname = 'configD - Low Mirror'
        config['D'].currents = _np.array([12630, 13170, 13170, 14240, 14240,
                                          0, 0], dtype=_np.float64)
        config['D'].avgB0 = avgB0
        config['D'].B00 = B00[3]
        config['D'].B25 = B25[3]
        config['D'].B36 = B36[3]
#        config['D'].shortID = 'w7x_ref_21'

        config['E'].configname = 'configE - High Mirror'
        config['E'].currents = _np.array([14510, 14100, 13430, 12760, 12360,
                                          0, 0], dtype=_np.float64)
        config['E'].avgB0 = avgB0
        config['E'].B00 = B00[4]
        config['E'].B25 = B25[4]
        config['E'].B36 = B36[4]
#        config['E'].shortID = 'w7x_ref_26'

        config['F'].configname = 'configF - Low Shear'
        config['F'].currents = _np.array([15320, 15040, 14230, 11520, 11380,
                                          -9760, 10160], dtype=_np.float64)
        config['F'].avgB0 = avgB0
        config['F'].B00 = B00[5]
        config['F'].B25 = B25[5]
        config['F'].B36 = B36[5]
#        config['F'].shortID = 'w7x_ref_37' #TA configuration,
        # w7x/0991_0929_0752_0742_-0531_+0531/01/00/
#        config['F'].shortID = 'w7x_ref_40' #SE configuration,
        # w7x/0982_0929_0752_0743_-0637_+0663/01/00/

        config['G'].configname = 'configG - Inward shift'
        config['G'].currents = _np.array([13070, 12940, 13210, 14570, 14710,
                                          4090, -8170], dtype=_np.float64)
        config['G'].avgB0 = avgB0
        config['G'].B00 = B00[6]
        config['G'].B25 = B25[6]
        config['G'].B36 = B36[6]
#        config['G'].shortID = 'w7x_ref_43'

        config['H'].configname = 'configH - Outward shift'
        config['H'].currents = _np.array([14030, 14030, 13630, 12950, 12950,
                                          -5670, 5670], dtype=_np.float64)
        config['H'].avgB0 = avgB0
        config['H'].B00 = B00[7]
        config['H'].B25 = B25[7]
        config['H'].B36 = B36[7]
#        config['H'].shortID = 'w7x_ref_46'

        config['I'].configname = 'configI - Limiter Case'
        config['I'].currents = _np.array([14150, 14550, 13490, 12170, 11770,
                                          -3970, 7940], dtype=_np.float64)
        config['I'].avgB0 = avgB0
        config['I'].B00 = B00[8]
        config['I'].B25 = B25[8]
        config['I'].B36 = B36[8]
#        config['I'].shortID = ''

        config['J'].configname = 'configJ - Limiter Case for OP1.1'
        config['J'].currents = _np.array([12780, 12780, 12780, 12780, 12780,
                                          4980, 4980], dtype=_np.float64)
        config['J'].avgB0 = avgB0
        config['J'].B00 = B00[9]
        config['J'].B25 = B25[9]
        config['J'].B36 = B36[9]
#        config['J'].shortID = 'w7x_ref_82'

        return config

        ####################
    def pickW7Xconfig(self):
        # Get the table of configurations
        configs = self.__w7xconfigs__()

        foundit = False
        for key, config in configs.items():  # @UnusedVariable
            currents = config.currents
            Bfactor = self.currents[0]/currents[0]
            B00 = config.B00
            B25 = config.B25
            B36 = config.B36
            avgB0 = config.avgB0

            rats = _np.int64(1e3 * _np.round(currents / currents[0],
                                             decimals=3))
#            res = _np.sum( _np.abs( self.ratios-rats ) )
            res = _np.max(_np.abs(self.ratios - rats))
            if res < 10:
                foundit = True
                break
            # endif
        # endfor

        if (not foundit) or (config.configname.find('configJ') > -1):
            # Probably a variant of configuration J
            self.config = configs['J']
            self.configurationJ_pickindex()
            if self.configname.find('configJ') < 0:
                print('No configuration match: Maliciously breaking code')
                self.config = []
                self.Bfactor = []
            # end if
        else:
            self.config = config
            self.Bfactor = Bfactor
            self.B00 = self.Bfactor*B00
            self.avgB0 = self.Bfactor*avgB0
            self.B25 = self.Bfactor*B25
            self.B36 = self.Bfactor*B36
        # end if
        self.configname = self.config.configname

        if self.verbose:
            print(self.configname)
        # end if
        return self.configname
    # end def

    def configurationJ_pickindex(self):
        """
        Configuration J is the OP1.1 limiter configuration
            index 0 is configuration J with B0 at phi=0 of 2.603T, <B0>=2.500T
            index 1-13 are an iota-scan called variation 8
                       with B0 at phi=0 of 2.52T
        """
        # Table of pre-calculated currents for variation 8 of Config J
        # (2.52T at phi=0)
        coil1 = _np.array([12370, 12374, 12378, 12383, 12387, 12391, 12395,
                           12399, 12403, 12408, 12412, 12416, 12420],
                          dtype=_np.float64)
#        coilA = _np.array([4824, 4826, 4828, 4829, 4831, 4832, 4834, 4836,
#                           4837, 4839, 4841, 4842, 4844], dtype=_np.float64)
        coilB = _np.array([4824, 4424, 4023, 3622, 3221, 2819, 2417, 2015,
                           1612, 1210,  807,  404,    0], dtype=_np.float64)
        B00 = _np.array([2.520, 2.520, 2.520, 2.520, 2.520, 2.520, 2.520,
                         2.520, 2.520, 2.520, 2.520, 2.520, 2.520],
                        dtype=_np.float64)

        ratioB = _np.int64(1e3 * _np.round(coilB / coil1, decimals=3))

        # verify that input currents represent variation 8 of configuration J
        # ~390 == _np.int64( 1e3*_np.round(coilA/coil1,decimals=3) )
        self._ratiosForVMECpicking = self.ratios.copy()
        if self.ratios[5] == 390:
            if self.verbose:
                print('Configuration J confirmed.')
        elif _np.isclose(self.ratios[5], 390, atol=1):
            if self.verbose:
                print('Configuration J confirmed within tolerance.')
            # endif
            # Reset the value to 390 to find the correct files
            self._ratiosForVMECpicking[5] = 390
        else:
            print('Input currents do not match Configuration J!')
        # endif

        if self.ratios[6] in ratioB:
            self.index = 1+_np.where(self.ratios[6] == ratioB)[0][0]
        elif _np.any(_np.isclose(self.ratios[6], ratioB, atol=1)):
            self.index = \
                1+_np.where(_np.isclose(self.ratios[6],
                                        ratioB, atol=1) == True)[0][0]
            self._ratiosForVMECpicking[6] = ratioB[self.index-1]
        else:
            print('Input currents do not match Configuration J!')
        # endif

        # check if currents are around 1000 and convert to that if close
        # This should not be a long-term solution!
        if _np.all(_np.isclose(self.ratios[:5], 1000, atol=1)):
            self._ratiosForVMECpicking[:5] = 1000

        if self.index > 0:
            self.configname = \
                'configJ - Limiter Case for OP1.1 - Variant 8, index %i' \
                % (self.index,)
        # endif

        self.Bfactor = self.currents[0]/coil1[self.index-1]

        # On-axis magnetic field at phi=0.0
        self.B00 = B00[self.index-1]*self.Bfactor
    # enddef configurationJ_pickindex

    def __str__(self):
        return self.configname
# end class w7xconfigs

# ------------------------------------------------------------------------- #
# ------------------------------------------------------------------------- #

if __name__ == '__main__':
    currents = _np.array([12361.89994498,
                          12362.92166191,
                          12362.47447639,
                          12363.31994178,
                          12371.22082426,
                          4817.01994068,
                          4826.08753995], dtype=_np.float64)
    equt = w7xfield(currents)
#    equt.pickW7Xconfig()

    fil = 'w7x_ref_82'
    vmc = VMECrest(fil)
    # vmc.getVMECgridDat()
    # vmc_roa = vmc.getgridroa()
    vmc_roa = vmc.roa_grid.copy()

    vmc_dVdrho = vmc.dVdrho_grid.copy()
    print(vmc.Vol_grid[-1])  # test volume normalization
    print(_np.trapz(vmc.dVdrho_grid, x=vmc_roa))  # test jacobian of integral
    print(vmc.Vol_lcfs)  # print estimate from webservices

    import matplotlib.pyplot as _plt
    _hfig = _plt.figure()
    _ax1 = _plt.subplot(2, 1, 1)
    _ax1.plot(vmc.roa_grid**2.0, vmc.dVdrho_grid)
    # _ax1.set_xlabel(r'$\rho$')
    _ax1.set_ylabel(r'$dV/d\rho[m^{-3}]$')
#    _ax2 = _plt.subplot(3,1,2)
#    _ax2.plot(vmc.roa_grid**2.0, vmc.Vol_grid)
#    # _ax2.set_xlabel(r'$\rho$')
#    _ax2.set_ylabel(r'Volume$[m^{-3}]$')
    _ax3 = _plt.subplot(2, 1, 2)
    _ax3.plot(vmc.roa_grid**2.0, vmc.Vol_grid)
    _ax3.set_xlabel('s')
    _ax3.set_ylabel(r'Volume$[m^{-3}]$')
# endif
