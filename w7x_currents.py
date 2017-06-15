# -*- coding: utf-8 -*-

# load currents in the coil system of W7-X from the archive, take mean over
# a given experiment ID timeslice and return the set of 7 numbers

from W7Xrest import read_restdb as _db
from dataaccess import get_program_info as _pif
import numpy as _np

# read all coil signals as one stream
__stream_url = "http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/raw/W7X/CoDaStationDesc.84/DataReductionProcessDesc.30_DATASTREAM/_signal.json"

#%%

def get_w7x_currents(program_id, calc_sigma=False):
    # get start and end time of program
    [start, end] = _pif.get_start_end(program_id)
    
    [valid, times, current_data] = _db.read_restdb(__stream_url+"?from="+_np.str(start)+"&upto="+_np.str(end))
    
    if not valid:
        print("error: could not get valid current data!")
        return None
    
    if calc_sigma:
        # calculate mean and sigma values of current during experiment
        return _np.mean(current_data, axis=1), _np.std(current_data, axis=1)
    else:
        # calculate mean value of current during experiment
        return _np.mean(current_data, axis=1)
