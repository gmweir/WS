###############################################################################
###############################################################################
"""
    This is the field line package from Sergey with significant revisions for
    use by G.M. Weir.

    It uses the wsdl services at W7-X to calculate
      __makeConfigFromCurrents - configurations based on input currents
      getW7XMainMagneticField - Bx,By,Bz at input coordinates (X,Y,Z)
      findMagneticAxis - Magnetic Axis coordinates
          getAxisRMean - Mean magnetic axis at those coordinates (R[m])
          getAxisBMean - Mean magnetic field at those coordinates (B[m])
      getAxisBAtPhi - Magnetic field strength on axis at input toroidal angle
      makePoincarePlot - Poincare plot of the field lines for given
                         configuration at a toroidal plane

     Revisions:
          unscale_currents - Unscales experimental currents for webservices

    To do:
        Add function to calculate <gradrho>, <|gradrho|^2>, roa, dV/droa

http://esb-dev2.ipp-hgw.mpg.de:9763/docs/fieldlinetracer.html#traceLineTracing
            roa, iota, diotadroa - trace (MagneticCharacteristics)
"""

from __future__ import absolute_import, with_statement, absolute_import, division, \
                       print_function, unicode_literals
__metaclass__ = type

###############################################################################


import numpy as _np
import matplotlib.pyplot as _plt
import osa
import copy

_plt.rc('font', size=16)
fig_size = _plt.rcParams["figure.figsize"]
fig_size = (1.2*fig_size[0], 1.2*fig_size[1])
figsize0 = _np.array([ 6.7, 4.775])
figsize1 = _np.array([ 5.775, 5.975])
figsize2 = (7.75, 6.0)
fig_size_small = (6, 4.5)

#__url = "http://lxpowerboz:88/services/cpp/FieldLine?wsdl"
__url = "http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl"
__fltsrv = osa.Client(__url)
__npc_num_windings = 108
__pc_num_windings = 36
__ideal_db_coils =  [160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
                     171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
                     182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,
                     193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
                     204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214,
                     215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225,
                     226, 227, 228, 229]
__asbuilt_db_coils = range(522, 592)
__asbuilt_db_coils_EMload = range(1152, 1222)   # op11
#__asbuilt_db_coils_EMload = range(2272, 2342)   # 2018

def __makeConfigFromCurrents(currents, scale=1.0, useGrid=False,
                             gridSymmetry=5, coils=None,
                             useIdealCoils=False, withEMload=False):
    """
        Create magnetic configuration object, given currents.
    """
    if coils is None:
        if useIdealCoils:
            coils = copy.deepcopy(__ideal_db_coils)
            scale *= -1.0
        else:
            if withEMload:
                coils = copy.deepcopy(__asbuilt_db_coils_EMload)
            else:
                coils = copy.deepcopy(__asbuilt_db_coils)

    config = __fltsrv.types.MagneticConfig()
    config.coilsIds = coils
    config.coilsIdsCurrents = []
    config.coilsIdsCurrents.extend(((currents[:5]*__npc_num_windings*scale ).tolist())*10)
    config.coilsIdsCurrents.extend(((currents[5:7]*__pc_num_windings*scale ).tolist())*10)
    if useGrid:
        grid = __fltsrv.types.Grid()
        config.grid = grid
        grid.fieldSymmetry = gridSymmetry
        cyl = __fltsrv.types.CylindricalGrid()
        grid.cylindrical = cyl
        cyl.numR, cyl.numZ, cyl.numPhi = 181, 181, 481
        cyl.RMin, cyl.RMax, cyl.ZMin, cyl.ZMax = 4.05, 6.75, -1.35, 1.35
    return config

# -------------------------------------------------------------------------- #

def unscale_currents(currents):
    return _np.concatenate( (currents[:5]/__npc_num_windings,
                             currents[5:7]/__pc_num_windings), axis=0)

def getW7XMainMagneticField(currents, points, scale = 1.0,
                            useIdealCoils = True):
    """Calculate magnetic field for given currents and points

    Parameters
    ----------
    currents : array
        Currents NPC1, NPC2, NPC3, NPC4, NPC5, PCA, PCB [A]. Currents should
        be per winding. Multiplication with the number of windings is done
        internally.

    points : 2d array
        XYZ coordinates [m] of the points of interest

    scale : float, optional
        Scale factor for currents.
    useIdealCoils : bool, optional
        Use ideal or as built filaments.

    Returns
    -------
    out : 2d array
        XYZ components of the field [T].
    """
    currents = _np.asarray(currents)
    config = __makeConfigFromCurrents(currents, scale = scale, useGrid=False,
                                      useIdealCoils = useIdealCoils)

    p3d = __fltsrv.types.Points3D()
    p3d.x1 = points[:,0].tolist()
    p3d.x2 = points[:,1].tolist()
    p3d.x3 = points[:,2].tolist()

    field = __fltsrv.service.magneticField(p3d, config)
    b = []
    for i in range(points.shape[0]):
        tmp = [field.field.x1[i], field.field.x2[i], field.field.x3[i]]
        b.append(tmp)
    return _np.array(b)

def findMagneticAxis(currents, scale = 1.0, step = 3e-2, useGrid = False, symmetry=5,
                     phisearch=36.0, numpoints = 50, Rstart=5.1959868872590826,
                     zstart = 0.0, accuracy=5e-4, useIdealCoils = True):
    """Calculate and return magnetic axis.

    Parameters
    ----------
    currents : array
        Currents NPC1, NPC2, NPC3, NPC4, NPC5, PCA, PCB [A]. Currents should be per
        winding. Multiplication with the number of windings is done internally.

    points : 2d array
        XYZ coordinates [m] of the points of interest
    scale : float, optional
        Scale factor for currents.
    step : float, optional
        Integration step.
    useGrid : bool, optional
        Use grid for tracing
    symmetry : int, optional
        Use toroidal symmetry.
    phisearch : float, optional
        Phi plane to use for searching
    numpoints : int, optional
        Number of Poincare points to use in searching
    Rstart : float, optional
        Initial guess of the axis position in phi search plane.
    zstart : float, optional
        Initial guess of the axis position in phi search plane.
    accuracy : float, optional
        Require axis accuracy in [m].
    useIdealCoils : bool, optional
        Use ideal or as build filaments.

    Returns
    -------
    out : 2d array
        Axis vertices with required step.
    """
    currents = _np.asarray(currents)
    config = __makeConfigFromCurrents(currents, scale = scale, useGrid=False,
                                      useIdealCoils = useIdealCoils)
    settings = __fltsrv.types.AxisSettings(1)
    settings.prefferedPhi = phisearch*_np.pi/180.0
    settings.reffNumPoints = numpoints
    settings.toroidalPeriod = symmetry
    settings.axisInitX = Rstart*_np.cos(phisearch*_np.pi/180.0)
    settings.axisInitY = Rstart*_np.sin(phisearch*_np.pi/180.0)
    settings.axisInitZ = zstart
    settings.axisAccuracy = accuracy

    res = __fltsrv.service.findAxis(step , config , settings )

    vertices = []
    for i in range(len(res.axis.vertices.x1)):
        tmp = [res.axis.vertices.x1[i],
               res.axis.vertices.x2[i],
               res.axis.vertices.x3[i]]
        vertices.append(tmp)

    return _np.array(vertices)

def getAxisRMean(axis):
    """Given axis points, find mean R.

    Parameters
    ----------
    axis : 2d array

    Returns
    -------
    float
    """
    R = _np.sqrt(axis[:,0]**2 + axis[:,1]**2)
    return R.mean()

def getAxisBMean(currents, axis, scale = 1.0, useIdealCoils = True):
    """Given axis points and coil currents, find mean B-field on axis.

    Parameters
    ----------
    currents : array
        Coil currents
    axis : 2d array
    scale : float, optional
        Currents scale factor.
    useIdealCoils : bool, optional
        Use ideal or as build filaments.

    Returns
    -------
    float
    """
    b = getW7XMainMagneticField(currents, axis, scale = scale,
                                useIdealCoils = useIdealCoils)
    return _np.sqrt((b*b).sum(axis=1)).mean()

def getAxisBAtPhi(currents, axis, phi0, scale = 1.0, useIdealCoils = True):
    """Find field at the axis in plane phi0.

    This uses interpolation from available axis points.

    Parameters
    ----------
    currents : array
        Coil currents
    axis : 2d array
    phi0 : float [deg]
        Plane of interest.
    scale : float, optional
        Currents scale factor.
    useIdealCoils : bool, optional
        Use ideal or as build filaments.

    Returns
    -------
    float
    """
    phi0 = phi0*_np.pi/180.0
    phi0 = _np.fmod(phi0, 2.0*_np.pi)
    if phi0 < 0:
        phi0 += 2.0*_np.pi
    phi = _np.arctan2(axis[:,1], axis[:,0])
    phi[phi<0] += 2.0*_np.pi
    inds = _np.abs(phi - phi0).argsort()
    if phi0 != 0:
        inds = inds[:3]
    else:
        inds = inds[1:4]
    b = getW7XMainMagneticField(currents, axis[inds, :], scale = scale, useIdealCoils = useIdealCoils)
    b = _np.sqrt((b*b).sum(axis=1))
    phi = phi[inds]
    inds = phi.argsort()
    phi = phi[inds]
    b = b[inds]
    p = _np.polyfit(phi, b, 2)
    return _np.polyval(p, phi0)

def makePoincarePlot(step, phi0, numPoints,
                     startPoints=None,
                     startRMin=5.95, startRMax=6.35,
                     startPhi=0.0,
                     startZMin=0, startZMax=0,
                     numStartPoints=51,
                     config=None, configId=None,
                     currents=None, useSymmetry=True,
                     useGrid=True, useIdealCoils=True):
    """
        Calculate a Poincare plot, for given configId or currents.

        Magnetic configuration is either to be supplied directly, or
        by configId or by currents.

        Start points can be provided directly, otherwise linearly spaced
        points between limits are chosen.

        If for a start point tracing results in a lower number of achieved points,
        such results are thrown away.

    Parameters
    ----------
    step : float
        Integration step.
    phi0 : float [deg]
        Plane of interest.
    numPoints : int
        Number of intersection points.
    startPoints : 2d array, optional
        Start points for tracing. If None, equally space points are
        generated.
    startRMin : float, optional
        Minimal R for generating start points.
    startRMax : float, optional
        Maximal R for generating start points.
    startPhi : float, optional
        Phi plane for choosing start points.
    startZMin : float, optional
        Minimal Z for generating start points.
    startZMax : float, optional
        Maximal Z for generating start points.
    numStartPoints : int, optional
        Number of start points to choose.
    config : service MagneticConfig, optional
        Prepared configuration.
    configId : int, optional
        Configuration Id from database.
    currents : array, optional
        Coil currents (per winding).
    useSymmetry: bool, optional
        If true, 5-fold symmetry is used, the result is rotated to plane
        of interest.
    useGrid : bool, optional
        Use default grid for tracing.
    useIdealCoils : bool, optional
        Use ideal or as build filaments.

    Returns
    -------
    startPoints, poincarePoints : 2d arrays
        Start points used for tracing and Poincare points in XYZ [m].
    """
    if startPoints is None:
        startPoints = _np.zeros((numStartPoints, 3), float)
        R = _np.linspace(startRMin, startRMax, numStartPoints)
        z = _np.linspace(startZMin, startZMax, numStartPoints)
        startPoints[:,0] = R*_np.cos(startPhi)
        startPoints[:,1] = R*_np.sin(startPhi)
        startPoints[:,2] = z
    p3d = __fltsrv.types.Points3D()
    p3d.x1 = startPoints[:,0].tolist()
    p3d.x2 = startPoints[:,1].tolist()
    p3d.x3 = startPoints[:,2].tolist()

    if useSymmetry:
        gridSymmetry = 5
    else:
        gridSymmetry = 1
    if config is None and configId is not None:
        config = __makeConfigFromCurrents(_np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, ]), scale = 1.4e6,
                                           useGrid = useGrid, gridSymmetry = gridSymmetry)
        config.coilsIds = None
        config.coilsIdsCurrents = None
        config.configIds = [configId]
    elif config is None and currents is not None:
        currents = _np.asarray(currents)
        config = __makeConfigFromCurrents(currents,
                                          useGrid = useGrid, gridSymmetry = gridSymmetry,
                                          useIdealCoils = useIdealCoils)
    else:
        raise RuntimeError("No valid config")

    task = __fltsrv.types.Task()
    task.step = step
    poi = __fltsrv.types.PoincareInPhiPlane()
    poi.numPoints = numPoints
    if useSymmetry:
        poi.phi0 = (phi0 + _np.linspace(0.0, 2.0*_np.pi, 5, endpoint=False)).tolist()
    else:
        poi.phi0 = phi0
    task.poincare = poi

    machine = __fltsrv.types.Machine(1)
    machine.grid.XMin, machine.grid.XMax = -10, 10
    machine.grid.YMin, machine.grid.YMax = -10, 10
    machine.grid.ZMin, machine.grid.ZMax = -2, 2
    machine.grid.numX = 500
    machine.grid.numY = 500
    machine.grid.numZ = 100
    machine.meshedModelsIds = [164]

    res = __fltsrv.service.trace(p3d, config, task, machine)

    out1 = []
    out2 = []
    if useSymmetry:
        for i in range(startPoints.shape[0]):
            xx = []
            yy = []
            zz = []
            for j in range(5):
                x = res.surfs[i*5+j].points.x1
                y = res.surfs[i*5+j].points.x2
                z = res.surfs[i*5+j].points.x3
                if x is None or len(x) != numPoints:
                    continue
                xx.extend(x)
                yy.extend(y)
                zz.extend(z)
            if len(xx) != 5*numPoints:
                continue
            s = startPoints[i, :]
            out1.append(s)
            xyz = _np.array((xx,yy,zz)).T
            out2.append(xyz)
    else:
        for i in range(startPoints.shape[0]):
            x = res.surfs[i].points.x1
            y = res.surfs[i].points.x2
            z = res.surfs[i].points.x3
            if x is None or len(x) != numPoints:
                continue
            s = startPoints[i, :]
            out1.append(s)
            xyz = _np.array((x,y,z)).T
            out2.append(xyz)

    return _np.array(out1), _np.array(out2), res

def Poincare(phi=0.0, configId=15, numPoints=200, Rstart=5.6, Rend=6.2, Rsteps=80, step=0.01,
             useSymmetry=True, config=None, currents=None, useGrid=True, useIdealCoils=True, _ax=None, iota_out=False):
    """
    You can track the progress of your calculation here:
    http://webservices.ipp-hgw.mpg.de/docs/fieldlinetracer.html#introduction
    """

    if useSymmetry:
        gridSymmetry = 5
    else:
        gridSymmetry = 1
    if config is None and configId is not None:
        config = __makeConfigFromCurrents(_np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, ]), scale = 1.4e6,
                                           useGrid = useGrid, gridSymmetry = gridSymmetry)
        config.coilsIds = None
        config.coilsIdsCurrents = None
        config.configIds = [configId]
    elif config is None and currents is not None:
        currents = _np.asarray(currents)
        config = __makeConfigFromCurrents(currents,
                                          useGrid = useGrid, gridSymmetry = gridSymmetry,
                                          useIdealCoils = useIdealCoils)
    else:
        raise RuntimeError("No valid config")
    # end if

    tracer = osa.Client('http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl')

    pos = tracer.types.Points3D()
    pos.x1 = _np.linspace(Rstart, Rend, Rsteps)
    pos.x2 = _np.zeros(Rsteps)
    pos.x3 = _np.zeros(Rsteps)

    # config = tracer.types.MagneticConfig()
    # config.configIds = [configId]

    poincare = tracer.types.PoincareInPhiPlane()
    poincare.numPoints = numPoints
    poincare.phi0 = _np.asarray(phi).tolist()

    task = tracer.types.Task()
    task.step = step
    task.poincare = poincare

    if iota_out:
        print('including a calculation of magnetic characteristics: iota. \nThis will slow down the calculation: check flag iota_out')
        print('.... note, the tracer is not returning any info yet!')
        # calculate iota as well
        MagChar = tracer.types.MagneticCharacteristics()
        MagChar.iotaAccuracy = 0.01
        # MagChar.iotaSteps = 50000000
#
        axis = tracer.types.AxisSettings()
        # axis.prefferedPhi = 0.628319
        # axis.reffNumPoints = 10.0
        # axis.toroidalPeriod = 5
        # axis.axisInitX = 5.95
        # axis.axisInitY = 0.0
        # axis.axisInitZ = 0.0
        # axis.axisAccuracy = 0.0001
        MagChar.axisSettings = axis

        task.characteristics = MagChar
        task.characteristics.axisSettings = axis
    # end if

    print('Starting the tracer')
    res = tracer.service.trace(pos, config, task, None, None)

    ''' number of PoincareSurface objets: 80'''
    print(len(res.surfs)//len(_np.atleast_1d(phi)))

    return __Poincare_plotter(res, axs=_ax)
# end def

def __Poincare_plotter(res, axs=None):
    nphi = 1
    if res.surfs[0].phi0 != res.surfs[1].phi0:
        phistart = res.surfs[nphi-1].phi0
        while (res.surfs[nphi].phi0 != phistart)*(nphi<10):
            nphi += 1
        # end while
    # end if
    if axs is None:
        axs = [None for jj in range(nphi)]
    # end if
    hfigs = []
    _axs = []
    for jj, _ax in enumerate(_np.atleast_1d(axs)):
        print(''' plotting the points: ''')
        if _ax is None:
            hfig, _ax = _plt.subplots(1,1)
        else:
            hfig = _ax.figure
        # end if
        hfigs.append(hfig)
        _axs.append(_ax)

        # The surfs attribute has a Points3D object in cartesian coordinates [m]
        for ii in range(0, len(res.surfs)//nphi):
            Rplot = _np.sqrt(_np.asarray(res.surfs[ii*nphi+jj].points.x1)**2.0 + _np.asarray(res.surfs[ii*nphi+jj].points.x2)**2.0).tolist()
            _ax.scatter(Rplot, res.surfs[ii*nphi+jj].points.x3, color="black", s=0.05)

        # _ax.axis('equal')
        _ax.set_xlabel('R [m]')
        _ax.set_ylabel('Z [m]')
        _ax.set_title('phi_ref=%3.1f degrees'%(res.surfs[0*nphi+jj].phi0*180.0/_np.pi,))
        hfig.tight_layout()
        # _ax.axis('equal')
        # _plt.show()
    # end for
    return res, hfigs, _axs
# end def __Poincare_plotter

def quickplot_Poincare(currents, phi=0.0, iota_out=False, _ax=None, useIdealCoils=True, figsize=(12.3 , 9.1), Rsettings=None):
    cstring = _np.asarray(1000*_np.asarray(currents[1:])/currents[0], dtype=int)
    cstring = "%4i_%4i_%4i_%4i_%4i_%4i"%(cstring[0], cstring[1], cstring[2], cstring[3], cstring[4], cstring[5])

    phi = _np.atleast_1d(phi)
    if _ax is None:
        hfigs = []
        _ax = []
        for ii in range(len(phi)):
            figsize = _np.atleast_2d(figsize)
            if _np.shape(figsize)[0] == 1:
                _hfig, _ax_ = _plt.subplots(1,1, num='poincare_phi_%3i_deg_%s'%(int(phi[ii]*180/_np.pi),cstring), figsize=tuple(figsize[0]))
            else:
                _hfig, _ax_ = _plt.subplots(1,1, num='poincare_phi_%3i_deg_%s'%(int(phi[ii]*180/_np.pi),cstring), figsize=tuple(figsize[ii]))
            # end if
            hfigs.append(_hfig)
            _ax.append(_ax_)
#            hfig, _ax = _plt.subplots(1,1, num='poincare_phi_%3i_deg_%s'%(int(phi*180/_np.pi),cstring), figsize=figsize)
        # end for
    # end if
    # ========== number of surfaces for finding islands? ========== #
    if Rsettings is None:
        Rstart = 5.6
    #    Rend = 6.35
        Rend = 6.2
        # step_size = 0.001  # 600 surfaces
        step_size = 0.0066  # 90 surfaces
    #    step_size = 0.01  # 60 surfaces
        Rsteps = int((Rend-Rstart)/step_size)
    else:
        Rstart, Rend, Rsteps = tuple(Rsettings)
    # end if

    # ========== Call the code and plot it =========== #

#    res, hfig, _ax = Poincare(phi=phi, currents=currents, numPoints=5000,
    res, hfig, _ax = Poincare(phi=phi, currents=currents, numPoints=1000,
                              Rstart=Rstart, Rend=Rend, Rsteps=Rsteps,
                              useIdealCoils=useIdealCoils, _ax=_ax, iota_out=iota_out)
#                              useIdealCoils=True, _ax=_ax, iota_out=iota_out)
    _plt.show()
    return res, hfig, _ax
# end def


if __name__=="__main__":

    try:
        from .equil_utils import VMECrest
    except:
        from WS.equil_utils import VMECrest
    # end try

    fils = []
    currents = []

#    currents.append([12362, 12364, 12363, 12363, 12371, 4817, 4828])   # XP:20160302.008
#    fils.append(['w7x_ref_59']) # 8

#    currents.append([1000, 1000, 1000, 1000, 1000, -525, -525])
    currents.append([13882, 13882, 13882, 13882, 13882, -7289, -7289])     # XPPROGID:20180927.015
    fils.append(['w7x_ref_358'])  # 15:   358, 364
#    fils.append(['w7x_ref_364'])  # 15:   358, 364

#    currents.append([13607, 13607, 13607, 13607, 13607, -5039, -5039]) # XP20180927.016
#    fils.append(['w7x_ref_326'])  # 16:   327, 326

#    vmc = VMECrest(fils[0])
    vmc = VMECrest(fils[0], realcurrents=currents[0])

    # XPPROGID:20180927.016, w7x_ref_326 by currents / profiles group, w7x_ref_327 by beta
    # currents:
    # main coils: 13607, 13607, 13607, 13607, 13607, -5039, -5039
    # trim coils: -114, -21, 101, 84, -49
    # currents = [13607, 13607, 13607, 13607, 13607, -5039, -5039]

    res, hfig, _ax = quickplot_Poincare(currents=currents[0], phi=[0.0,2.0*_np.pi/10.0], iota_out=False, _ax=None, useIdealCoils=False, figsize=[(5.25 , 9.1), (12.3 , 9.1)])

    hfig, hfig3 = tuple(hfig)
    _ax, _ax3 = tuple(_ax)
    _ax.set_xlim((5.15, 6.35))
    _ax.set_ylim((-1.1, 1.1))

    vmc.fluxsurfaces(_np.asarray([0.5**2.0, 1.0**2.0]), phi=0.0*_np.pi/180.0, Vid=fils[0], _ax=_ax, fmt='r--')
    vmc.fluxsurfaces(_np.asarray([0.5**2.0, 1.0**2.0]), phi=2.0*_np.pi/10.0, Vid=fils[0], _ax=_ax3, fmt='r--')
    hfig.tight_layout()
    hfig3.tight_layout()

#    # ========== diagnostic specific views =========== #
#    # make a nicely sized figure for inspection of the ECE plot and TS plots
#    hfig1, _ax1 = _plt.subplots(1,1, num='ECE_poincare', figsize=(5.25 , 9.1))
#    hfig2, _ax2 = _plt.subplots(1,1, num='TS_poincare', figsize=(12.3 , 9.1))

#    # ========== toroidal angle for the diagnostic view =========== #
#    # ECE
#    x1ece, y1ece, z1ece = -4.7311, -4.5719, 0.2723
#    x2ece, y2ece, z2ece = -4.09251, -3.7044, 0.1503
#    phiece = 6.3*_np.pi/180.0  # ECE
#
#    _Rsece = _np.sqrt(x1ece**2.0+y1ece**2.0)
#    _Rtece = _np.sqrt(x2ece**2.0+y2ece**2.0)
#
#    # Thomson
#    x1ts, y1ts, z1ts = -0.914, -0.271, 1.604
#    u = [-7.729, 1.924, -2.514]  # from u=0, 1 from AEZ31 to AET31
#    x2ts, y2ts, z2ts = x1ts+1.0*u[0], y1ts+1.0*u[1], z1ts+1.0*u[2]
#
#    # phits = 171.455*_np.pi/180.0  # Thomson scattering
#    phits = (27.2+144.255)*_np.pi/180.0  # Thomson scattering
##    phits = 27.2*_np.pi/180.0  # Thomson scattering
#
#    _Rsts = _np.sqrt(x1ts**2.0+y1ts**2.0)*_np.cos(phits-_np.arctan(y1ts/x1ts))
#    _Rtts = _np.sqrt(x2ts**2.0+y2ts**2.0)*_np.cos(phits-_np.arctan(y2ts/x2ts))

#    res1, _hfig, _ax_ = quickplot_Poincare(currents=currents[0], phi=[phiece,phits], iota_out=False, _ax=[_ax1,_ax2], useIdealCoils=False)
#    hfig1, hfig2 = tuple(_hfig)
#    _ax1, _ax2 = tuple(_ax_)
##    _ax1.plot(_np.asarray([_Rtece,_Rsece]), _np.asarray([z2ece, z1ece]), 'r-')
##    _ax2.plot(_np.asarray([_Rtts,_Rsts]), _np.asarray([z2ts, z1ts]), 'r-')
##
#    # resest ECE limits
#    _ax1.set_ylim((-1.1, 1.1))
#    _ax1.set_xlim((5.15, 6.35))
##
#    # plot the flux-surfaces
#    vmc.fluxsurfaces(_np.asarray([0.5**2.0, 1.0**2.0]), phi=phiece, Vid=fils[0], _ax=_ax1, fmt='r--')
#    hfig1.tight_layout()
#
##    #TS plot second
##    vmc.fluxsurfaces(_np.asarray([0.5**2.0, 1.0**2.0]), phi=phits, Vid=fils[0], _ax=_ax2, fmt='r--')
##    hfig2.tight_layout()
#
#    if res1.characteristics is not None:
#         hiota, axs = _plt.subplots(2,1, num='iota_scan')
#         axi1, axi2 = tuple(axs)
#
#         for ii in range(0, len(res1.surfs)):
#             axi1.plot(res1.characteristics[ii].reff, res1.characteristics[ii].iota,'-')
#             axi2.plot(res1.characteristics[ii].reff, res1.characteristics[ii].diota,'-')
#         # end for
#         axi1.set_ylabel(r'$\iota$')
#         axi2.set_ylabel(r'$d\iota$/dr$_{eff}$ [m$^{-1}]')
#         axi2.set_xlabel(r'r$_{eff}$ [m]')
#         hiota.tight_layout()

    # phi0 = 170.5*_np.pi/180.0  # Thomson scattering
    # res2, hfig2, _ax2 = Poincare(phi=phi0, currents=currents)

    # step = 0.01

    # numPoints = 200
    # sflux, xyz = makePoincarePlot(step, phi0, numPoints,
    #                       startPoints = None,
    #                       startRMin = 5.95, startRMax = 6.35,
    #                       startPhi = 0.0,
    #                       startZMin = 0, startZMax = 0,
    #                       numStartPoints = 51,
    #                       config = None, configId = None,
    #                       currents = currents, useSymmetry=True,
    #                       useGrid = True, useIdealCoils = True)
# end if