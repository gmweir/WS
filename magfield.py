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
import osa
import copy

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
__asbuilt_db_coils_EMload = range(1152, 1222)

def __makeConfigFromCurrents(currents, scale = 1.0, useGrid = False, 
                             gridSymmetry = 5, coils = None, 
                             useIdealCoils = True, withEMload = False):
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
                     startPoints = None,
                     startRMin = 5.95, startRMax = 6.35,
                     startPhi = 0.0,
                     startZMin = 0, startZMax = 0,
                     numStartPoints = 51,
                     config = None, configId = None,
                     currents = None, useSymmetry=True,
                     useGrid = True, useIdealCoils = True):
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
        config.configIds = configId
    elif config is None and currents is not None:
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

    return _np.array(out1), _np.array(out2)


