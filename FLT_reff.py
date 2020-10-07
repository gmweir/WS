from osa import Client
import numpy as np

# =============================================== #

# Coils
 coils_old=range(522,592)
 coils_real_hi=range(1782,1852)
 coils_real_hm=range(1852,1922)
 coils_real_st=range(1362,1432)
 coils_real_wo_em=range(1572,1642)

# Some current settings
currents_hi_real=[14188.0*108, 14188.0*108, 14188.0*108, 14188.0*108, 14188.0*108]*10 # NPC
currents_hi_real.extend([-9790.0*36, -9790.0*36]*10) # PC

currents_kkm_real=[12960.0*108, 13214.0*108, 13994.0*108, 12048.0*108, 10920.0*108]*10 # NPC
currents_kkm_real.extend([-749.0*36, -749.0*36]*10) # PC

currents_st_real=[13067.0*108, 13067.0*108, 13067.0*108, 13067.0*108, 13067.0*108]*10 # NPC
currents_st_real.extend([-699.0*36, -699.0*36]*10) # PC

# =============================================== #

def FLT_reff(x,y,z, coils=coils_real_stm, currents=currents_st_real):
    '''FLT_reff(x,y,z,coils=coils_real_stm,currents=currents_st_real)'''

    flt = Client('http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl')


    p = flt.types.Points3D()
    p.x1 = x
    p.x2 = y
    p.x3 = z


    coils = coils_real_st
    currents = currents_st_real

    config = flt.types.MagneticConfig()
    config.inverseField = 0
    config.coilsIds = coils
    config.coilsIdsCurrents = currents

    task = flt.types.Task()
    task.step = 0.2

    task.characteristics = flt.types.MagneticCharacteristics()
    task.characteristics.axisSettings = flt.types.AxisSettings()

    res = flt.service.trace(p, config, task, None, None)

    reff = [res.characteristics[i].reff for i in range(len(res.characteristics))]

    return {'reff':reff}

def FLT_poincare(phi=0, coils=coils_real_stm, currents=currents_st_real):
    '''FLT_poincare(phi=0,coils=coils_real_stm,currents=currents_st_real)'''

    flt = Client('http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl')


    p = flt.types.Points3D()
    p.x1 = x
    p.x2 = y
    p.x3 = z


    coils = coils_real_st
    currents = currents_st_real

    config = flt.types.MagneticConfig()
    config.inverseField = 0
    config.coilsIds = coils
    config.coilsIdsCurrents = currents

    task = flt.types.Task()
    task.step = 0.2

    task.characteristics = flt.types.MagneticCharacteristics()
    task.characteristics.axisSettings = flt.types.AxisSettings()

    res = flt.service.trace(p, config, task, None, None)

    reff = [res.characteristics[i].reff for i in range(len(res.characteristics))]

    return {'reff':reff}

