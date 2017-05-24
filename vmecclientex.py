# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:46:26 2016

@author: gawe
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 18:34:21 2016

@author: gmw
"""

import os as _os
from osa import Client
import numpy as _np
import matplotlib.pyplot as _plt

vmec = Client("http://esb.ipp-hgw.mpg.de:8280/services/vmec_v5?wsdl")


vmecid='w7x_ref_60'
currents = vmec.service.getCoilCurrents(vmecid)


s = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
numPoints = 100
phi_t = [0.0, _np.pi/5]

## getting VMEC input from w7x_ref_60 run
#inp1 = vmec.service.getVmecRunData('w7x_ref_60', 'input')
#
## calculation ID
#id = 'test20160114_7'
#
## start of the VMEC run
#vmec.service.execVmecString(inp1, id) 
#
## check whether the run is finished
#print(vmec.service.isReady(id))
#
## check whether the run is successful (e.g. all output files exist)
#print(vmec.service.wasSuccessful(id))

# plotting pictures of the run and iota comparison
fs0 = vmec.service.getFluxSurfaces(id, phi_t[0], s, numPoints)
fs36 = vmec.service.getFluxSurfaces(id, phi_t[1], s, numPoints)

iota0 = vmec.service.getIotaProfile(id)
iota1 = vmec.service.getIotaProfile(vmecid)
#http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/0860_0700_0540_0510_+0750_+0750/
pressure = vmec.service.getPressureProfile(id)


_plt.figure(1)
_plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

#left  = 0.125  # the left side of the subplots of the figure
#right = 0.9    # the right side of the subplots of the figure
#bottom = 0.1   # the bottom of the subplots of the figure
#top = 0.9      # the top of the subplots of the figure
#wspace = 3   # the amount of width reserved for blank space between subplots
#hspace = 0.5   # the amount of height reserved for white space between subplots

_plt.subplot(221)

_plt.title('flux surfaces at phi = 0°')
_plt.xlabel('R')
_plt.ylabel('Z')
_plt.axis([4.5, 6.5, -1.0, 1.0])
_plt.plot(fs0[0].x1, fs0[0].x3, 'r.')
for i in range(len(fs0)):
    _plt.plot(fs0[i].x1, fs0[i].x3, 'r')

_plt.subplot(222)
_plt.title('flux surfaces at phi = 36°')
_plt.xlabel('R')
_plt.ylabel('Z')
_plt.axis([4.5, 6.5, -1.0, 1.0])
_plt.plot(fs36[0].x1, fs36[0].x3, 'r.')
for i in range(len(fs36)):
    _plt.plot(fs36[i].x1, fs36[i].x3, 'r')
    
_plt.subplot(223)
_plt.title('iota profiles')
_plt.xlabel('radial coordinate s')
_plt.ylabel('iota')
_plt.plot(iota0)
_plt.plot(iota1, 'g.')
_plt.text(20, 0.85, 'profiles are identical,\n hence one line')
       

_plt.subplot(224)
_plt.title('pressure profile')
_plt.xlabel('radial coordinate s')
_plt.ylabel('pressure')
_plt.plot(pressure)

_plt.show() 

# iota checks
# location and name of the output file
#outdata2="D:\\tya\\E5\\VMEC\\J_conf\\test_VMEC_output.txt"
outdata2 = "G://Workshop/QMEPYL/pybaseutils/test_VMEC_output.txt"

iota0_arr = _np.asarray(iota0)
iota1_arr = _np.asarray(iota1)
a= iota1_arr-iota0_arr
b=max(abs(x) for x in a)
print(max(abs(x) for x in a))

with open(outdata2, "w") as text_file:
    print("Iota JG  and test calc.", file=text_file)
    print(iota1_arr, file=text_file)
    print("\n", file=text_file)
    print(iota0_arr, file=text_file)
    print("\n", file=text_file)
    print(a, file=text_file)
    print("\n", file=text_file)
    print(b, file=text_file)
    

###############################################################################


#wout_netcdf = vmec.service.getVmecOutputNetcdf('test42')
#file = open("wout_test42.nc", "wb")
#file.write(wout_netcdf)
#file.close()

#coeffsRmnCos = client.types.FourierCoefficients(1), then set the values