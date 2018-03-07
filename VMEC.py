# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:22:01 2017

@author: tcw
"""

from osa import Client
import numpy as np
from pyversion.version import urllib
#import urllib2, urllib
import matplotlib.pyplot as plt
import os as _os

VClient = Client("http://esb:8280/services/vmec_v5?wsdl")

def create_local_configs(vmec_local='C:\PortablePrograms\configs\\'):
    ''' creates local copy of boozer files for TRAVIS'''

    url='http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/'

    if not _os.path.exists(vmec_local):
        _os.mkdir(vmec_local)
    label = {}
    stream = {}
    label[0]  = 'DBM000'
    stream[0] = '1000_1000_1000_1000_+0750_+0750/01/00/boozer.txt'
    label[1]  = 'EIM000'
    stream[1] = '1000_1000_1000_1000_+0000_+0000/01/00/boozer.txt'
    label[2]  = 'EIM065'
    stream[2] = '1000_1000_1000_1000_+0000_+0000/01/04m/boozer.txt'
    label[3]  = 'EIM200'
    stream[3] = '1000_1000_1000_1000_+0000_+0000/01/12m/boozer.txt'
    label[4]  = 'KJM000'
    stream[4] = '1020_1080_0930_0843_+0000_+0000/01/00/boozer.txt'
    label[5]  = 'KJM065'
    stream[5] = '0972_0926_0880_0852_+0000_+0000/01/00jh/boozer.txt'
    label[6]  = 'FTM000'
    stream[6] = '1000_1000_1000_1000_-0690_-0690/01/00/boozer.txt'
    label[7]  = 'EEM000'
    stream[7] = '1000_1000_1000_1000_+0390_+0390/05/0000/boozer.txt'

    for i in range(len(label.items())):
        urllib.urlretrieve(url+stream[i],vmec_local+label[i]+'_boozer.bc')


def get_config_url(label,full_path=0):
    ''' returns link to VMEC url

        usage: Vid=get_config_url(label)

        availabel labels:
            DBM000
            EIM000
            EIM065
            EIM200
            KJM000
            KJM065
            FTM000
            EEM000

                    '''
    co={'DBM000': '/w7x/1000_1000_1000_1000_+0750_+0750/01/00',
        'EIM000': '/w7x/1000_1000_1000_1000_+0000_+0000/01/00',
        'EIM065': '/w7x/1000_1000_1000_1000_+0000_+0000/01/04m',
        'EIM200': '/w7x/1000_1000_1000_1000_+0000_+0000/01/12m',
        'KJM000': '/w7x/1020_1080_0930_0843_+0000_+0000/01/00',
        'KJM065': '/w7x/0972_0926_0880_0852_+0000_+0000/01/00jh',
        'FTM000': '/w7x/1000_1000_1000_1000_-0690_-0690/01/00',
        'EEM000': '/w7x/1000_1000_1000_1000_+0390_+0390/05/0000',
       }

    if full_path:
        return 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger' + co[label]
    else:
        return co[label]

def get_config_sid(Bconf):
    co={'DBM000': 'w7x_ref_18',
        'EIM000': 'w7x_ref_1',
        'EIM065': 'w7x_ref_7',
        'EIM200': 'w7x_ref_9',
        'KJM000': 'w7x_ref_163',
        'KJM065': 'w7x_ref_27',
        'FTM000': 'w7x_ref_15',
        'EEM000': 'w7x_ref_82',
       }
    return co[Bconf]


def get_reff(x,y,z,Vid):
    ''' reff = get_reff(x,y,z,VMEC_ID)'''
    p3d = VClient.types.Points3D()
    p3d.x1 = x
    p3d.x2 = y
    p3d.x3 = z
    B = VClient.service.magneticField(Vid,p3d)
    reff=VClient.service.getReff(Vid,p3d)


def get_B(x,y,z,Vid):
    '''B = get_B(x,y,z,VMEC_ID)'''
    p3d = VClient.types.Points3D()
    p3d.x1 = x
    p3d.x2 = y
    p3d.x3 = z
    s = VClient.service.magneticField(Vid,p3d)
    Bx=s.x1
    By=s.x2
    Bz=s.x3
    Babs=np.sqrt(np.square(Bx) + np.square(By) + np.square(Bz))
    return {'Babs':Babs, 'Bx':Bx, 'By':By, 'Bz':Bz}

def get_minor_radius(Vid):
    '''a = get_minor_radius(VMEC_ID)'''
    tmp=VClient.service.getReffProfile(Vid)
    return tmp[len(tmp)-1]


def fluxsurfaces(s,phi,Vid,N=256,disp=0,_ax=plt):
    ''' fluxsurfaces(s,phi,Vid,N=256,disp=0) '''

    if isinstance(s,int):
        n=1

    else:
        n=len(s)
        phi=np.ones(n)*phi
    w=VClient.service.getFluxSurfaces(Vid, phi*np.pi/180, s, N)
    R=np.zeros((n,N))
    z=np.zeros((n,N))

    for i in range(n):

        if isinstance(w[i].x1, list) and isinstance(w[i].x2, list):

            R1=np.sqrt(np.square(w[i].x1) + np.square(w[i].x2))
            z1=w[i].x3
        else:
            R1=w[i].x1
            z1=w[i].x3
        if disp:
            _ax.plot(R1,z1,'k-')
        R[i]=R1
        z[i]=z1
    if disp: _ax.axis('equal')

    return R, z















