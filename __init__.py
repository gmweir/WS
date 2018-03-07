# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:28:01 2017

@author: gawe
"""

# ===================================================================== #
# ===================================================================== #

from __future__ import absolute_import, with_statement, absolute_import, \
                       division, print_function, unicode_literals

# ========================================================================== #
# ========================================================================== #

__version__ = "2017.05.24.15"
__all__ = ['equil_utils','magfield']

from ..ssh import CheckWebServicesConnection #, CheckVMECConnection, \
#                  CheckRESTConnection, CheckENCODERConnection

if CheckWebServicesConnection():  # returns a boolean True if the pinger receives a packet back
    try:
        __import__('osa')
        __osaonpath__ = True
    except:
        print('The WebServices module requires the OSA python package be \n'
              + 'installed/on your python path ... it is not')
        __osaonpath__ = False
    # end try

    if __osaonpath__: # and CheckVMECConnection() and CheckRESTConnection() \
                      # and CheckENCODERConnection():
        from . import equil_utils
        from . import magfield
        from . import VMEC
#        from . import w7x_currents  # analysis:ignore
        from .equil_utils import VMECrest  # analysis:ignore
        from .equil_utils import w7xfield  # analysis:ignore
        from .equil_utils import w7xCurrentEncoder  # analysis:ignore

    # endif
else:  # no packet received from Webservices
    print('could not ping the W7-X webservices (probably off-campus)')
# end

# ===================================================================== #
# ===================================================================== #




