# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 17:02:07 2016

@author: gawe
"""
# ========================================================================== #
# ========================================================================== #

# This section is to improve compatability between bython 2.7 and python 3x
from __future__ import absolute_import, division, print_function, \
    unicode_literals

# ========================================================================== #

import os as _os
import numpy as _np
from osa import Client as _cl

#from W7X import jsonutils as _jsnut
from codecs import getreader as _getrdr
from pyversion import version as _ver
import json as _jsn

from pybaseutils import utils as _ut
from pybaseutils.Struct import Struct
try:
    from pybaseutils.equilutils import VMEC_Struct
except:
    VMEC_Struct = Struct
# end try

__metaclass__ = type

# ========================================================================== #
# ========================================================================== #


class ArchiveError(Exception):
    def __init__(self, **kwargs):
        # Now for your custom code...
        self.status = kwargs.get('status',0)
        self.message = kwargs.get('message',' ')
        if self.message.find(':')>-1:
            ind = self.message.find(':')+1
            self.exception = self.message[:ind]
            self.error = self.message[ind:]
        else:
            self.exception = None
            self.error = self.status
        # end if

        # Call the base class constructor with the parameters it needs
        super(ArchiveError, self).__init__(self.message)
    # end def __init__
# end class ArchiveError

def with_open_json(request):
    reader = _getrdr("utf-8")
    resp = None
    try:
#        if request.find("/views/")>-1:
#            request = parse_alias_stream(request)
#        # end if
        resp = _ver.urllib.urlopen(request)
        jsonsignal = _jsn.load(reader(resp), strict=False)
        if hasattr(jsonsignal,'message') and jsonsignal['message'].find('IllegalArgumentException:')>-1:
            print(jsonsignal['request'])
            print(jsonsignal['status'])
            print(jsonsignal['message'])
            raise ArchiveError
    except:
        jsonsignal = None
        pass
#        raise
    finally:
        if resp is not None:
            resp.close()
        # endif
    # end try
    return jsonsignal

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

# ========================================================================== #
# ========================================================================== #


class VMECrest(VMEC_Struct):
    # Import the VMEC rest client
#    vmec = _cl("http://esb.ipp-hgw.mpg.de:8280/services/vmec_v6?wsdl")
    vmec = _cl("http://esb.ipp-hgw.mpg.de:8280/services/vmec_v8?wsdl")
    rooturl = 'http://10.66.24.160/'
    resturl = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/run/'
    url = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/'
#    url = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/'
#    homeurl = _os.path.join('G://','Workshop','TRAVIS_tree','MagnConfigs','W7X')
    homeurl = _os.path.join('X://','QME-ECE','Mapping-files')
    def __init__(self, shortID=None, **kwargs):
#        shortID = kwargs.get('shortID', None)
        coords = kwargs.get('coords', None)
        verbose = kwargs.get('verbose', True)
        realcurrents = kwargs.get('realcurrents',None)
#        Bfactor = kwargs.get('Bfactor',None)
        self.verbose = verbose
        self.shortID = shortID

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
            if realcurrents is None:
                realcurrents = _np.copy(self.currents)
            # end if
            self.realcurrents = _np.asarray(realcurrents, dtype=_np.float64)
            self.Bfactor = self.realcurrents[0]/self.currents[0]
#            if Bfactor is not None: self.Bfactor=Bfactor  # end if
#            self.getMagneticAxis(torFi=0.0)
            self.getVMECamin()
            self.getVMECfieldperiod()
            self.getB00()
            self.getVMECgridDat()

        # endif
        if coords is not None:
            self.setCoords(coords)
            # self.getCoordData()
        # endif
        self.coords = coords
    # enddef __init__

    # ========================================= #
    # ========================================= #

    def save_wout(self, sfile=None, homeurl=None):
        if homeurl is None:
            homeurl = self.homeurl
        if sfile is None:
            epsfile = self.getVMECepseff()
            self.vmectxtname = 'wout_w7x'+epsfile[epsfile.find('w7x')+3:epsfile.find('.data\n%')]+'.txt'
            sfile = _os.path.join(homeurl, self.vmectxtname)
        # end if
        try:
            wout = open(sfile,'w')
            wout.write(self.getVMECwouttxt())
            wout.close()
        except:            pass
        finally:
            try:            wout.close() # just in case
            except:        pass
    # end def save_wout

    def save_netcdf(self, sfile=None, homeurl=None):
        if homeurl is None:
            homeurl = self.homeurl
        if sfile is None:
            epsfile = self.getVMECepseff()
            self.vmecnetcdf = 'wout_w7x'+epsfile[epsfile.find('w7x')+3:epsfile.find('.data\n%')]+'.nc'
            sfile = _os.path.join(homeurl, self.vmecnetcdf)
        # end if
        try:
            wout = open(sfile,'wb')
            wout.write(self.getVMECwoutnc())
            wout.close()
        except:            pass
        finally:
            try:            wout.close() # just in case
            except:        pass
    # end def save_wout

    def save_epseff(self, sfile=None, reload=True):
        epsfile = self.getVMECepseff()
        self.epstxtname = 'eps_w7x'+epsfile[epsfile.find('w7x')+3:epsfile.find('.data\n%')]+'.txt'

        with open(sfile,'w') as eps_txt:
            eps_txt.write(epsfile)
        # end with
        try:            eps_txt.close() # just in case
        except:         pass
        eps_roa, eps_eff, eps_ftrapped, _, _, _ = _np.loadtxt(
            self.epstxtname, comments='%', delimiter=' ', usecols=(0,1,2))
        self.eps = dict()
        self.eps['roa'] = eps_roa
        self.eps['epseff'] = eps_eff
        self.eps['ftrapped'] = eps_ftrapped
    # end def

    # ========================================= #
    # ========================================= #

    def getCoilCurrents(self, vmecid=None):
        if vmecid is None:
            vmecid = self.vmecid
        # endif
        self.currents = self.vmec.service.getCoilCurrents(vmecid)
        return self.currents

    def loadCoilCurrents(self, experimentID):
        from . import w7x_currents as _curr

        self.realcurrents = _curr.get_w7x_currents(experimentID)
        self.Bfactor = self.realcurrents[0]/self.currents[0]
        return self.realcurrents

    def getFluxSurfaces(self, torFi=None, numPoints=100, vmecid=None, s=None):
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

        fsFi = self.vmec.service.getFluxSurfaces(vmecid, torFi, s, numPoints)
        return fsFi

    def getMagneticAxis(self, torFi=None, vmecid=None):
        if vmecid is None:
            vmecid = self.vmecid
        # endif
        if torFi is None:
            torFi = 0.0
            if hasattr(self,'coords') and self.coords is not None:
                torFi = self.fi
            # endif
        # endif
        acoord = self.vmec.service.getMagneticAxis(vmecid, torFi)
        self.axis_coord = _np.array(_np.vstack((acoord.x1, acoord.x2, acoord.x3)), dtype=_np.float64)
        return self.axis_coord

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
        self.nfp = self.vmec.service.getFieldPeriod(vmecid)
        return self.nfp

    def getVMECamin(self,vmecid=None):
        threed = self.getVMECthreed(vmecid)
        ista = threed.find('Minor Radius          =')
        iend = threed.find('[M] (from Cross Section)')
        _, amin = threed[ista:iend].split('=')
        self.amin = _np.float64(amin)
        return self.amin

    def get_new_id(self):
        import getpass as _gp

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
        import getpass as _gp

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

    # ========================================= #
    # ========================================= #

    def setCoords(self, coords, getData=True):
        """
        Input cartesian coords as column vectors shape(# points, 3)
        """
        coords = _np.atleast_2d(coords)
        if coords.shape[1] != 3:  coords = coords.T  # endif

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

        if getData:
            self.getCoordData()
        # end if
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

    # ========================================= #

    def getB00(self):
        x00, y00, z00 = self.getCARTcoord(_np.asarray([0,0,0],dtype=_np.float64)) # s,th,fi
        self.cart_axis_phi0 = _np.asarray([x00, y00, z00], dtype=_np.float64)
        self.setCoords(coords=_np.atleast_2d([x00,y00,z00]), getData=False)
        self.getBcart(rescale=False)
        self.B00_orig = _np.copy(self.getmodB())

        self.getBcart(rescale=True)
        self.B00 = _np.copy(self.getmodB())

    def getBcart(self, rescale=True):  # TODO:  do you want to scale field outside of this or inside?
        # Magnetic field utility needs cartesian coordinates

        # Call the webservices magnetic field utility
        b = self.vmec.service.magneticField(self.vmecid, self._pcart)
        self.Bxyz = _np.array(_np.vstack((b.x1, b.x2, b.x3)), dtype=_np.float64)
        self.Bxyz = self.Bxyz.transpose()
        if rescale and hasattr(self,'Bfactor'):
            self.Bxyz *= self.Bfactor
        # end if
        return self.Bxyz

    def getmodB(self): # TODO:  do you want to scale field outside of this or inside?
        # Calculate the magnetic field strength
        self.modB = _np.linalg.norm(self.Bxyz, ord=2, axis=1)
        return self.modB

    def getbhat(self):
        # Calculate the magnetic field unit vector
        modB = self.modB.copy()
        self.bhat = self.Bxyz / (modB.reshape(self.nn, 1) * _np.ones((1, 3), dtype=_np.float64))
        return self.bhat

    # ========================================= #


    def getVMECcoord(self, tol=3e-3):
        # Get the VMEC coordinates at each point: s,th,zi
        vmec_coords = self.vmec.service.toVMECCoordinates(self.vmecid, self._pcyl, tol)
        self.s = _np.array(vmec_coords.x1, dtype=_np.float64)
        self.th = _np.array(vmec_coords.x2, dtype=_np.float64)
        self.zi = _np.array(vmec_coords.x3, dtype=_np.float64)
        self.roa = _np.sqrt(self.s)
        return self.roa

    def getCARTcoord(self, vmccoords):
        # Get the cartesian coordinates at each point from VMEC coords: s,th,zi
        vmccoords = _np.asarray(vmccoords, dtype=_np.float64)
        vmccoords = _np.atleast_2d(vmccoords)
        nr, nc = vmccoords.shape
        if nr == 3:
            vmccoords = vmccoords.T
            nr, nc = vmccoords.shape
        # end if

        # Load up the webservices Points3D cylindrical object
        _coords = self.vmec.types.Points3D(nr)
        _coords.x1 = vmccoords[:,0]
        _coords.x2 = vmccoords[:,1]
        _coords.x3 = vmccoords[:,2]

        c = self.vmec.service.toCylinderCoordinates(self.vmecid, _coords)
        coords = _np.array(_np.vstack((c.x1, c.x2, c.x3)), dtype=_np.float64) # cylindrical coordinates
        coords = _np.atleast_2d(coords)
        if coords.shape[0] == 3:
            coords = coords.T
        # end if
        XX, YY = _ut.pol2cart(coords[:,0],coords[:,1])
        return XX, YY, coords[:,2]   # Cartesian coordinates

    def CalcFluxCart(self, coords=None):
        if coords is not None:
            self.setCoords(coords)
            self.getCoordData()
        # endif
        return self.roa

    def get_reff(self):
        reff = self.vmec.service.getReff(self.vmecid, self._pcart)
        self.reff = _np.asarray(reff, dtype=_np.float64)
        return self.reff

    # ========================================= #

    def getVMECgridDat(self):
        self.getVMECgrid()
        self.getgridroa()
        self.getgridpres()
        self.getgridiota()
        self.getgridVol()
        self.getgridVol_lcfs()
    # end def

    # ========================================= #

    def getVMECgrid(self):
        # Get the toroidal flux on the VMEC grid
        torPsi_grid = self.vmec.service.getToroidalFluxProfile(self.vmecid)
        reff_grid = self.vmec.service.getReffProfile(self.vmecid)
        self.torPsi_grid = _np.array(torPsi_grid, dtype=_np.float64)
        self.reff_grid = _np.array(reff_grid, dtype=_np.float64)
        return self.torPsi_grid

    def getgridroa(self):
        # This is the natural grid calculated by VMEC (no inversion)
        roa_grid = _np.sqrt(self.torPsi_grid / self.torPsi_grid[-1])
        self.roa_grid = roa_grid
        return roa_grid

    def getFluxTor(self):
        # Linearly interpolate to the new spatial coordinates
        self.torPsi = _np.interp(self.roa, self.roa_grid, self.torPsi_grid)
        return self.torPsi

    def getgridpres(self):
        self.Pres_grid = self.vmec.service.getPressureProfile(self.vmecid)
        self.Pres_grid = _np.array(self.Pres_grid, dtype=_np.float64)
        return self.Pres_grid

    def getPressure(self):
        if self.Pres_grid is None:
            self.Pres_grid = self.getgridpres()
        # end if
        self.Pres = _np.interp(self.roa, self.roa_grid, self.Pres_grid)
        return self.Pres

    def getKineticEnergy(self):
        self.Wgrid = self.vmec.service.getKineticEnergy(self.vmecid)
        return self.Wgrid

    # ====== #

    def getgridiota(self):
        self.iota_grid = self.vmec.service.getIotaProfile(self.vmecid)
        self.iota_grid = _np.array(self.iota_grid, dtype=_np.float64)
        return self.iota_grid

    def getiota(self):
        if self.iota_grid is None:
            self.iota_grid = self.getgridiota()
        self.iota = _np.interp(self.roa, self.roa_grid, self.iota_grid)
        return self.iota

    # ====== #

    def getgriddVdrho(self):
        # Get the VMEC volume on the VMEC grid
        #   At one point there was an error in the Volume calculation!!!
        #   It actually returned the derivative
        hotfix = 0
        if hotfix: # As long as there is an error in the getVolumeProfile webservice # TODO: check this regularly
            # on s-grid
            self.dVds_grid = self.vmec.service.getVolumeProfile(self.vmecid)
        else:
            # doesn't work: returns volume of LCFS for every flux-surface
            self.dVds_grid = self.vmec.service.getDVolDs(self.vmecid)
#            self.dVds_grid = self.vmec.service.getvp(self.vmecid)
        # endif
        self.dVds_grid = _np.array(self.dVds_grid, dtype=_np.float64)

        if hotfix:
            dxvar = _np.diff(self.torPsi_grid)
            dyvar = -1*_np.diff(self.dVds_grid)
            self.dVds_grid = _np.hstack((0, dyvar/dxvar))
            self.dVds_grid = _np.array(self.dVds_grid, dtype=_np.float64)
        # end if

        # Undo normalization from VMEC (2*pi*2*pi)
        self.dVds_grid = (4*_np.pi**2)*self.dVds_grid

        # Convert to jacobian for integration
        self.dVdrho_grid = 2*self.roa_grid*self.dVds_grid
        return self.dVdrho_grid

    def getgridVol(self):
        if self.dVdrho_grid is None:
            self.dVdrho_grid = self.getgriddVdrho()
        # endif

        hotfix = 0
        if hotfix:  # As long as there is an error in the getVolumeProfile webservice
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
        if self.dVdrho_grid is None: self.getgridVol()  # endif
        self.dVdrho = _np.interp(self.roa, self.roa_grid, self.dVdrho_grid)

        if 1: # As long as there is an error in the getVolumeProfile webservice
            self.Vol_grid = _np.trapz(self.dVdrho_grid, x=self.roa_grid)
            self.Vol = _np.interp(self.roa, self.roa_grid, self.Vol_grid)
        # end if

        if self.Vol_lcfs is None:  self.getgridVol_lcfs()  # endif

        return self.Vol_lcfs, self.dVdrho

    # ===================================================================== #

    def createMgrid(self, idin, magconf, minR, maxR, minZ, maxZ, resR, resZ, resPhi,
                    fieldPeriods=5, isStellaratorSymmetric=True):

        return_id = self.vmec.service.createMgrid(magconf, minR, maxR, resR, minZ, maxZ, resZ, resPhi,
                                      isStellaratorSymmetric, fieldPeriods, idin)
        if return_id == idin:
            return return_id
        else:
            print("returned ID from Mgrid does not match given: "+return_id)
            return return_id
        # end if

    # ===================================================================== #

    def Cross_LCMS(self, r0, rd, amin=None, Rmaj=None):
        """
         Find intersection of the Ray with the last closed magnetic surface
         The ray is R(t)=r0+rd*t  with t>0.
         It is assumed, that the origin of ray lies outside of the last surface

         INPUT:
           r0  -- origin of the ray in cartesian coordinates
           rd  -- unit vector of direction of the ray in cartesian coordinates
         OUTPUT:
          entryPoint is the intersection point of the Ray with the last closed
          magnetic surface;
          distance is the distance between r0 and entryPoint,
          distance >= 0 if the function succeeds, distance < 0 otherwise.
        """
        if amin is None: amin=self.amin   # end if
        if Rmaj is None: Rmaj=self.Rmaj   # end if
        entryPoint = 0
        distance   = -1

        s = (self.CalcFluxCart(r0))**2.0
        if(s<1): return entryPoint, distance # return if the origin lies inside LCMS

        dl = amin*0.04         # step 2cm for W7X
        Nr = rd/_np.sqrt(_np.dot(rd,rd))
        dr = dl*Nr

        N = int(100.0*Rmaj/dl)     # max number of iterations
        r = _np.copy(r0)
        Rmax2 = 4*Rmaj**2
        r2 = _np.dot(r,r)
        while (N>1) or (s>1):   # cycle while not inside
            rprev2 = _np.copy(r2)
            r += dr             # move forward along ray
            r2 = _np.dot(r,r)
            if(r2>rprev2)*(r2>Rmax2): return entryPoint, distance # end if, return if point moves far outwards
            N -= 1
            if (N < 1): return entryPoint, distance   # end if entry point not found
            s = (self.CalcFluxCart(r))**2.0
            if(s<1): break # cycle while not inside
        # end while

        # we are near LCMS, bracket the LCMS between two point rOutside and entryPoint
        rOutside = _np.copy(r)   # 'nearest' outside point
        N  = 32                  # max number of iterations
        s = (self.CalcFluxCart(r))**2.0
        if (s<=1):               # if current position lies inside
            while (N>1)+(s<1):     # cycle while inside LCMS
               r -= dr         # move backward along the ray
               N -= 1
               if (N < 1): return entryPoint, distance  # entry point not found
               s = (self.CalcFluxCart(r))**2.0
               if (s>=1): break        # exit if outside
            # end for                  # cycle while inside LCMS
            rOutside   = _np.copy(r)   # 'nearest' outside point
            entryPoint = r+dr         # move inside and save position
        else:                      # if current position lies slightly outside
            while (N>1)+(s>1):     # cycle while outside LCMS
                r += dr              # move forward along ray
                N -= 1
                if (N < 1): return entryPoint, distance # entry point not found
                s = (self.CalcFluxCart(r))**2.0
                if (s<1): break         # exit if inside
            # end for                 # cycle while outside LCMS
            entryPoint = _np.copy(r)  # inside, save position
        # end if

        N = 100
        # Find intersection of the Ray with the last surface by bisection
        while (N>0):
            r = (entryPoint+rOutside)/2
            s = (self.CalcFluxCart(r))**2.0
            dl = 1.0 - s
            if (dl< 0):
              rOutside = _np.copy(r)
            else:
              entryPoint = _np.copy(r)        # if r lies inside
              if(dl<=1e-3): break # end if    # Ok, done
            # end if
            N -= 1
        # end while

        dr = entryPoint-r0
        distance = _np.sqrt(_np.dot(dr,dr))
        return entryPoint,distance
    # end def Cross_LCMS

    # ===================================================================== #
    # ===================================================================== #

    def _extract_data(self, verbose=None):
        if verbose is None:
            if hasattr(self, 'verbose'):
                verbose = self.verbose
            else:
                verbose = True
            # end if
        # end if

        self.VMEC_Data = Struct() # Instantiate an empty class of type structure

        # stellarator symmetric terms
        self.VMEC_Data.rmnc = self.vmec.service.getFourierCoefficients(self.vmecid, 'RCos')
        self.VMEC_Data.ns = _np.copy(self.VMEC_Data.rmnc.numRadialPoints) # number of radial flux surfaces
        self.VMEC_Data.xn = _np.asarray(self.VMEC_Data.rmnc.toroidalModeNumbers)  # toroidal mode numbers
        self.VMEC_Data.xm = _np.asarray(self.VMEC_Data.rmnc.poloidalModeNumbers)  # poloidal mode numbers

        self.VMEC_Data.rmnc = _np.asarray(self.VMEC_Data.rmnc.coefficients)
        self.VMEC_Data.zmns = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'ZSin').coefficients)
        self.VMEC_Data.bmnc = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'BCos').coefficients)
        self.VMEC_Data.lmns = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'LambdaSin').coefficients)
        self.VMEC_Data.gmnc = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'gCos').coefficients)
        self.VMEC_Data.bsubumnc = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'BsubUCos').coefficients)
        self.VMEC_Data.bsubvmnc = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'BsubVCos').coefficients)
        self.VMEC_Data.bsubsmns = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'BsubSSin').coefficients)
        self.VMEC_Data.bsupumnc = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'BsupUCos').coefficients)
        self.VMEC_Data.bsupvmnc = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'BsupVCos').coefficients)
        self.VMEC_Data.currumnc = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'CurrUCos').coefficients)
        self.VMEC_Data.currvmnc = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'CurrVCos').coefficients)
        self.VMEC_Data.currumns = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'CurrUSin').coefficients)
        self.VMEC_Data.currvmns = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'CurrVSin').coefficients)

#        if self.iasym:
    #        # non-stellarator symmetric terms
    #        self.VMEC_Data.rmns = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'RSin'))
    #        self.VMEC_Data.zmnc = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'ZCos').coefficients)
    #        self.VMEC_Data.bmns = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'BSin').coefficients)
    #        self.VMEC_Data.lmnc = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'LambdaCos').coefficients)
    #        self.VMEC_Data.gmns = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'gSin').coefficients)
    #        self.VMEC_Data.bsubumns = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'BsubUSin').coefficients)
    #        self.VMEC_Data.bsubvmns = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'BsubVSin').coefficients)
    #        self.VMEC_Data.bsubsmnc = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'BsubSCos').coefficients)
    #        self.VMEC_Data.bsupumns = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'BsupUSin').coefficients)
    #        self.VMEC_Data.bsupvmns = _np.asarray(self.vmec.service.getFourierCoefficients(self.vmecid, 'BsupVSin').coefficients)
        # end if non-stellarator symmetric

    # end def

# end class VMECrest


# ========================================================================== #
# ========================================================================== #


class w7xCurrentEncoder(Struct):
    baseurl = "http://svvmec1.ipp-hgw.mpg.de:5000/encode_config?"
    def __init__(self, currents=None):
        if currents is not None:
            self.currents = currents
            self.build_url()
            self.parseConfig()
        # endif
    # end def __init__

    @property
    def currents(self):
        return self._I
    @currents.setter
    def currents(self, value):
        self._I = _np.asarray(value)
    @currents.deleter
    def currents(self):
        del self._I

    def build_url(self):
        self.url = self.baseurl
        self.url += "i1=%.1f&i2=%.1f&i3=%.1f&i4=%.1f&i5=%.1f&ia=%.1f&ib=%.1f&b0_req=2.5"\
            %(self.currents[0],self.currents[1],self.currents[2],self.currents[3],
              self.currents[4],self.currents[5],self.currents[6])

    def parseConfig(self):
        # returns a json string containing a dictionary
        self.response = self.openURL(self.url)
        self.code = self.response['3-letter-code']
        self.name = self.code[:3]
        self.field = self.code[3:]
        self.B00 = self.response['Bax(phi=0)/T']
#        self.Bax_00 = self.response['Bax/T= 0.0000']
        self.currents =  _np.asarray([_np.float64(ic) for ic in _np.atleast_1d(self.response['Bax/T= 2.5000'])[0].split(' ') if len(ic)>0],
                                      dtype=_np.float64) # self.response['Bax/T= 2.5000']
        self.central_iota = self.response['central iota']
        self.info = self.response['info']
        self.mirror_ratio = self.response['mirror ratio']
        self.avgB0 = self.response['techn. av. Bax/T']
        self.vmecid = self.response['vmec-id-string']

    def openURL(self, url):
        return with_open_json(url)

    def __str__(self):
        return _jsn.dumps(self.response)
# end class w7xCurrentEncoder


# ========================================================================== #
# ========================================================================== #


class w7xfield(Struct):
    nwindings = [108, 108, 108, 108, 108, 36, 36]
    url = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/'
    vmec = _cl("http://esb.ipp-hgw.mpg.de:8280/services/vmec_v8?wsdl")

    def __init__(self, currents=None, verbose=True):
        self.verbose = verbose
        if currents is not None:
            currents = _np.asarray(currents, dtype=_np.float64)
            self.currents = currents
            self.localdir = ''
            self.ratios = _np.int64(1e3*_np.round(currents/currents[0], decimals=3))
            self._ratiosForVMECpicking = self.ratios.copy()
            self.pickW7Xconfig()

            # vmecname = self.getVMECname()
            [self.RefRuns, self.shortIDs] = self.getVMECrun(self.ratios)
        # endif
    # enddef __init__

        # =============== #

    def getVMECname(self, ratios=None):
        if ratios is None:
            ratios = self.ratios
        # endif
        vmecname = '%4i_%4i_%4i_%4i_%0+5i_%0+5i' % tuple(ratios[1:])
        return vmecname


    # =============== #

    def pickW7Xconfig(self):
        config = w7xCurrentEncoder(self.currents)
        self.config = config
        self.B00 = _np.float64(config.B00)
        self.avgB0 = _np.float64(config.avgB0)
#        self.Bfactor = self.currents[0]/self.currents[0]
        self.Bfactor = self.currents[0]/config.currents[0] #self.currents[0]
        self.configname = config.name
#        self.Bfactor = 1e-2*_np.abs(_np.float64(config.field))
        self.vmecid = config.vmecid

    # =============== #

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

    def getVMECrun(self, ratios=None, compensate=True):
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

    # =============== #

    def __OP11configs__(self):
        # Coil currents per winding

        config = {'A': Struct(), 'B': Struct(),
                  'C': Struct(), 'D': Struct(),
                  'E': Struct(), 'F': Struct(),
                  'G': Struct(), 'H': Struct(),
                  'I': Struct(), 'J': Struct()}

        # Average magnetic field strength on the magnetic-axis
        avgB0 = _np.array([2.50], dtype=_np.float64)

        # On-axis magnetic field at phi=0
        B00 = _np.array([2.613, 2.596, 2.636, 2.499, 2.763,
                         2.711, 2.619, 2.592, 2.666, 2.603], dtype=_np.float64)

        # On-axis magnetic field at phi=25 degrees
        B25 = _np.array([2.428, 2.437, 2.417, 2.499, 2.339,
                         2.379, 2.422, 2.453, 2.397, 2.433], dtype=_np.float64)

        # On-axis magnetic field at phi=36 degrees
        B36 = _np.array([2.396, 2.415, 2.372, 2.499, 2.260,
                         2.323, 2.394, 2.437, 2.341, 2.407], dtype=_np.float64)

        config['A'].configname = 'configA - standard case - EIM EJM'   # EIM  (EJM with iota compensation)
        config['A'].currents = _np.array([13470, 13470, 13470, 13470, 13470, 0, 0], dtype=_np.float64)
        config['A'].avgB0 = avgB0
        config['A'].B00 = B00[0]
        config['A'].B25 = B25[0]
        config['A'].B36 = B36[0]
#        config['A'].shortID = 'w7x_ref_1'

        config['B'].configname = 'configB - Low iota - DBM'       # DBM
        config['B'].currents = _np.array([12200, 12200, 12200, 12200, 12200, 9150, 9150], dtype=_np.float64)
        config['B'].avgB0 = avgB0
        config['B'].B00 = B00[1]
        config['B'].B25 = B25[1]
        config['B'].B36 = B36[1]
#        config['B'].shortID = 'w7x_ref_18'

        config['C'].configname = 'configC - High iota - FTM'      # FTM
        config['C'].currents = _np.array([14880, 14880, 14880, 14880, 14880, -10260, -10260], dtype=_np.float64)
        config['C'].avgB0 = avgB0
        config['C'].B00 = B00[2]
        config['C'].B25 = B25[2]
        config['C'].B36 = B36[2]
#        config['C'].shortID = 'w7x_ref_15'

        config['D'].configname = 'configD - Low Mirror - AIM'     # AIM
        config['D'].currents = _np.array([12630, 13170, 13170, 14240, 14240, 0, 0], dtype=_np.float64)
        config['D'].avgB0 = avgB0
        config['D'].B00 = B00[3]
        config['D'].B25 = B25[3]
        config['D'].B36 = B36[3]
#        config['D'].shortID = 'w7x_ref_21'

        config['E'].configname = 'configE - High Mirror - KJM KKM'    # KJM
        config['E'].currents = _np.array([14510, 14100, 13430, 12760, 12360, 0, 0], dtype=_np.float64)
        config['E'].avgB0 = avgB0
        config['E'].B00 = B00[4]
        config['E'].B25 = B25[4]
        config['E'].B36 = B36[4]
#        config['E'].shortID = 'w7x_ref_26'

        config['F'].configname = 'configF - Low Shear - ILD JLF'      #  ILD or JLF
        config['F'].currents = _np.array([15320, 15040, 14230, 11520, 11380, -9760, 10160], dtype=_np.float64)
        config['F'].avgB0 = avgB0
        config['F'].B00 = B00[5]
        config['F'].B25 = B25[5]
        config['F'].B36 = B36[5]
#        config['F'].shortID = 'w7x_ref_37' #TA configuration,
        # w7x/0991_0929_0752_0742_-0531_+0531/01/00/
#        config['F'].shortID = 'w7x_ref_40' #SE configuration,
        # w7x/0982_0929_0752_0743_-0637_+0663/01/00/

        config['G'].configname = 'configG - Inward shift - FIS'   # FIS
        config['G'].currents = _np.array([13070, 12940, 13210, 14570, 14710, 4090, -8170], dtype=_np.float64)
        config['G'].avgB0 = avgB0
        config['G'].B00 = B00[6]
        config['G'].B25 = B25[6]
        config['G'].B36 = B36[6]
#        config['G'].shortID = 'w7x_ref_43'

        config['H'].configname = 'configH - Outward shift - DKH'    # DKH
        config['H'].currents = _np.array([14030, 14030, 13630, 12950, 12950, -5670, 5670], dtype=_np.float64)
        config['H'].avgB0 = avgB0
        config['H'].B00 = B00[7]
        config['H'].B25 = B25[7]
        config['H'].B36 = B36[7]
#        config['H'].shortID = 'w7x_ref_46'

        config['I'].configname = 'configI - Limiter Case - OP1.1'   #
        config['I'].currents = _np.array([14150, 14550, 13490, 12170, 11770, -3970, 7940], dtype=_np.float64)
        config['I'].avgB0 = avgB0
        config['I'].B00 = B00[8]
        config['I'].B25 = B25[8]
        config['I'].B36 = B36[8]
#        config['I'].shortID = ''

        config['J'].configname = 'configJ - Limiter Case - OP1.1'
        config['J'].currents = _np.array([12780, 12780, 12780, 12780, 12780, 4980, 4980], dtype=_np.float64)
        config['J'].avgB0 = avgB0
        config['J'].B00 = B00[9]
        config['J'].B25 = B25[9]
        config['J'].B36 = B36[9]
#        config['J'].shortID = 'w7x_ref_82'

        return config


    # =============== #

    def pickOP11config(self):
        # Get the table of configurations
        configs = self.__OP11configs__()

        foundit = False
        for key, config in configs.items():  # @UnusedVariable
            currents = config.currents
            Bfactor = self.currents[0]/currents[0]
            B00 = config.B00
            B25 = config.B25
            B36 = config.B36
            avgB0 = config.avgB0

            rats = _np.int64(1e3 * _np.round(currents / currents[0], decimals=3))
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

    def configurationA_pickindex(self):
        coil1 = _np.asarray([13470.0, 13067.0])
        B00 = [2.613, 2.63]
#        coilA = [0, 53]
        coilB =  [0, -699.0]
        ratioB = _np.int64(1e3 * _np.round(coilB / coil1, decimals=3))

        # verify that input currents represent variation 8 of configuration J
        self._ratiosForVMECpicking = self.ratios.copy()
        if _np.isclose(self.ratios[5], ratioB[0], atol=1):
            # EJM configuration:  EIM with an iota correction for error fields
            self._ratiosForVMECpicking[5:] = 0.0
            self.index = 1
        elif _np.isclose(self.ratios[5], ratioB[1], atol=1):
            # EJM configuration:  EIM with an iota correction for error fields
            self._ratiosForVMECpicking[5:] = 0.0
            self.index = 2

        # check if currents are around 1000 and convert to that if close
        # This should not be a long-term solution!
        if _np.all(_np.isclose(self.ratios[:5], 1000, atol=1)):
            self._ratiosForVMECpicking[:5] = 1000

        if self.index > 0:
            self.configname = 'configA - Standard Case for OP1.2 - EIM / EJM'%(self.index,)
        # endif

        self.Bfactor = self.currents[0]/coil1[self.index-1]

        # On-axis magnetic field at phi=0.0
        self.B00 = B00[self.index-1]*self.Bfactor


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
            self.index = 1+_np.where(_np.isclose(self.ratios[6], ratioB, atol=1) == True)[0][0]
            self._ratiosForVMECpicking[6] = ratioB[self.index-1]
        else:
            print('Input currents do not match Configuration J!')
        # endif

        # check if currents are around 1000 and convert to that if close
        # This should not be a long-term solution!
        if _np.all(_np.isclose(self.ratios[:5], 1000, atol=1)):
            self._ratiosForVMECpicking[:5] = 1000

        if self.index > 0:
            self.configname = 'configJ - Limiter Case for OP1.1 - Variant 8, index %i'%(self.index,)
        # endif

        self.Bfactor = self.currents[0]/coil1[self.index-1]

        # On-axis magnetic field at phi=0.0
        self.B00 = B00[self.index-1]*self.Bfactor
    # enddef configurationJ_pickindex

    def __str__(self):
        return self.configname
# end class w7xconfigs

# ========================================================================== #
# ========================================================================== #


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

    cnfg = w7xCurrentEncoder(currents)
    print(cnfg)

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

# ========================================================================== #
# ========================================================================== #

