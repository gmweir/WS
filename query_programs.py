# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:35:34 2017

@author: lukb

This module provides a collection of simple functions to fetch data from the
W7-X ArchiveDB and MDSplus. Most of the times the returned data structures are
the raw response of the W7-X ArchiveDB WEB API. Only unnecessary wrappers are
removed. The units of the W7-X ArchiveDB are used (notably time is in nano
seconds).
"""

import MDSplus
import requests
import numpy as np

server_url = 'http://archive-webapi.ipp-hgw.mpg.de'
archive_url = server_url + '/ArchiveDB'
archive_signal_dict = {
    'ECRH_A1': '/ArchiveDB/codac/W7X/CoDaStationDesc.108/'
               'DataModuleDesc.240_DATASTREAM/0/Rf_A1/scaled/',
    'ECRH_B1': '/ArchiveDB/codac/W7X/CoDaStationDesc.108/'
               'DataModuleDesc.240_DATASTREAM/8/Rf_B1/scaled/',
    'ECRH_A5': '/ArchiveDB/codac/W7X/CoDaStationDesc.106/'
               'DataModuleDesc.237_DATASTREAM/0/Rf_A5/scaled/',
    'ECRH_B5': '/ArchiveDB/codac/W7X/CoDaStationDesc.106/'
               'DataModuleDesc.237_DATASTREAM/8/Rf_B5/scaled/',
    'ECRH_C1': '/ArchiveDB/codac/W7X/CoDaStationDesc.104/'
               'DataModuleDesc.236_DATASTREAM/0/Rf_C1/scaled/',
    'ECRH_D1': '/ArchiveDB/codac/W7X/CoDaStationDesc.104/'
               'DataModuleDesc.236_DATASTREAM/8/Rf_D1/scaled/',
    'ECRH_C5': '/ArchiveDB/codac/W7X/CoDaStationDesc.101/'
               'DataModuleDesc.229_DATASTREAM/0/Rf_C5/scaled/',
    'ECRH_D5': '/ArchiveDB/codac/W7X/CoDaStationDesc.101/'
               'DataModuleDesc.229_DATASTREAM/8/Rf_D5/scaled/',
    'ECRH_E1': '/ArchiveDB/codac/W7X/CoDaStationDesc.94/'
               'DataModuleDesc.209_DATASTREAM/0/Rf_E1/unscaled/',
    'ECRH_F1': '/ArchiveDB/codac/W7X/CoDaStationDesc.94/'
               'DataModuleDesc.209_DATASTREAM/8/Rf_F1/scaled/',
    'ECRH_E5': '/ArchiveDB/codac/W7X/CoDaStationDesc.17/'
               'DataModuleDesc.24_DATASTREAM/0/Rf_E5/scaled/',
    'ECRH_F5': '/ArchiveDB/codac/W7X/CoDaStationDesc.17/'
               'DataModuleDesc.179_DATASTREAM/4/Rf_F5/scaled/',
    'ECRH_tot': '/ArchiveDB/codac/W7X/CBG_ECRH/TotalPower_DATASTREAM/'
                'V2/0/Ptot_ECRH/',
    'ECRH_tot_old': '/ArchiveDB/codac/W7X/CBG_ECRH/TotalPower_DATASTREAM/'
                    'V1/0/Ptot_ECRH/',
    'Wdia': '/Test/raw/Minerva1/Minerva.Magnetics15.Wdia/'
            'Wdia_compensated_QXD31CE001x_DATASTREAM/V1/0/'
            'Wdia_compensated_for_diamagnetic_loop%20QXD31CE001x/',
    'ECE_01': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/01/',
    'ECE_02': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/02/',
    'ECE_03': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/03/',
    'ECE_04': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/04/',
    'ECE_05': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/05/',
    'ECE_06': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/06/',
    'ECE_07': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/07/',
    'ECE_08': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/08/',
    'ECE_09': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/09/',
    'ECE_10': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/10/',
    'ECE_11': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/11/',
    'ECE_12': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/12/',
    'ECE_13': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/13/',
    'ECE_14': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/14/',
    'ECE_15': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/15/',
    'ECE_16': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/16/',
    'ECE_17': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/17/',
    'ECE_18': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/18/',
    'ECE_19': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/19/',
    'ECE_20': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/20/',
    'ECE_21': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/21/',
    'ECE_22': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/22/',
    'ECE_23': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/23/',
    'ECE_24': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/24/',
    'ECE_25': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/25/',
    'ECE_26': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/26/',
    'ECE_27': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/27/',
    'ECE_28': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/28/',
    'ECE_29': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/29/',
    'ECE_30': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/30/',
    'ECE_31': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/31/',
    'ECE_32': '/ArchiveDB/views/KKS/QME_ECE/standard_reduced/32/',

    # raw signal, uncalibrated, with phase wraps
    'density': '/ArchiveDB/codac/W7X/CoDaStationDesc.110/'
               'DataModuleDesc.246_DATASTREAM/11/L5_ECA59/',
    # slow signal with considerable time jitter but no phase wrap and low noise
    'density_slow': '/ArchiveDB/codac/W7X/CoDaStationDesc.16339/'
                    'DataModuleDesc.16341_DATASTREAM/0/Line%20integrated%20density/',
    'ne_center': '/Test/raw/W7X/QTB_Central/volume_2_DATASTREAM/V1/1/ne_map/',
    'Te_center': '/Test/raw/W7X/QTB_Central/volume_2_DATASTREAM/V1/0/Te_map/',
    'ne_ts_2': '/Test/raw/W7X/QTB_Profile/volume_2_DATASTREAM/V1/1/ne_map/',
    'ne_ts_3': '/Test/raw/W7X/QTB_Profile/volume_3_DATASTREAM/V1/1/ne_map/',
    'ne_ts_7': '/Test/raw/W7X/QTB_Profile/volume_7_DATASTREAM/V1/1/ne_map/',
    'ne_ts_8': '/Test/raw/W7X/QTB_Profile/volume_8_DATASTREAM/V1/1/ne_map/',
    'ne_ts_9': '/Test/raw/W7X/QTB_Profile/volume_9_DATASTREAM/V1/1/ne_map/',
    'ne_ts_10': '/Test/raw/W7X/QTB_Profile/volume_10_DATASTREAM/V1/1/ne_map/',
    'ne_ts_11': '/Test/raw/W7X/QTB_Profile/volume_11_DATASTREAM/V1/1/ne_map/',
    'ne_ts_12': '/Test/raw/W7X/QTB_Profile/volume_12_DATASTREAM/V1/1/ne_map/',
    'ne_ts_13': '/Test/raw/W7X/QTB_Profile/volume_13_DATASTREAM/V1/1/ne_map/',
    'ne_ts_14': '/Test/raw/W7X/QTB_Profile/volume_14_DATASTREAM/V1/1/ne_map/',
    'ne_ts_15': '/Test/raw/W7X/QTB_Profile/volume_15_DATASTREAM/V1/1/ne_map/',
    'ne_ts_16': '/Test/raw/W7X/QTB_Profile/volume_16_DATASTREAM/V1/1/ne_map/',
    'Te_ts_2': '/Test/raw/W7X/QTB_Profile/volume_2_DATASTREAM/V1/0/Te_map/',
    'Te_ts_3': '/Test/raw/W7X/QTB_Profile/volume_3_DATASTREAM/V1/0/Te_map/',
    'Te_ts_7': '/Test/raw/W7X/QTB_Profile/volume_7_DATASTREAM/V1/0/Te_map/',
    'Te_ts_8': '/Test/raw/W7X/QTB_Profile/volume_8_DATASTREAM/V1/0/Te_map/',
    'Te_ts_9': '/Test/raw/W7X/QTB_Profile/volume_9_DATASTREAM/V1/0/Te_map/',
    'Te_ts_10': '/Test/raw/W7X/QTB_Profile/volume_10_DATASTREAM/V1/0/Te_map/',
    'Te_ts_11': '/Test/raw/W7X/QTB_Profile/volume_11_DATASTREAM/V1/0/Te_map/',
    'Te_ts_12': '/Test/raw/W7X/QTB_Profile/volume_12_DATASTREAM/V1/0/Te_map/',
    'Te_ts_13': '/Test/raw/W7X/QTB_Profile/volume_13_DATASTREAM/V1/0/Te_map/',
    'Te_ts_14': '/Test/raw/W7X/QTB_Profile/volume_14_DATASTREAM/V1/0/Te_map/',
    'Te_ts_15': '/Test/raw/W7X/QTB_Profile/volume_15_DATASTREAM/V1/0/Te_map/',
    'Te_ts_16': '/Test/raw/W7X/QTB_Profile/volume_16_DATASTREAM/V1/0/Te_map/',
}


def query_archive(url, params=None):
    response = requests.get(url, params)
    if not response.ok:
        raise Exception('Request failed (url: {:s}).'.format(response.url))
    data = response.json()
    if type(data) == dict:
        data['url'] = response.url
    return data


def get_archive_byurl(url, shot):
    """Simple access to archive signal by relative url.

    Args:
        url (str): relative URL of signal (e.g. '/ArchiveDB/ ... /')
        shot: MDSplus style shot number

    Returns:
        values (list): signal values
        time array (list): timestamps in ns relative to (first) t1
        units (str): units description provided by Archive
    """
    info = get_program_info(shot_to_program_nr(shot))
    fullurl = server_url + url + '_signal.json'
    response = query_archive(fullurl,
                             params={'from': info['from'],
                                     'upto': info['upto']})
    time = np.array(response.get('dimensions')) - info['trigger']['1'][0]
    return response.get('values'), time, response.get('units')


def get_archive_byname(name, shot):
    """Simple access to archive signal by name in archive_signal_dict

    Args:
        name (str): name of signal in archive_signal_dict
        shot: MDSplus style shot number

    Returns:
        values (list): signal values
        time array (list): timestamps in ns relative to (first) t1
        units (str): units description provided by Archive
    """
    return get_archive_byurl(archive_signal_dict.get(name), shot)


def get_thomson_ne(shot):
    """get thomson scattering ne profiles

    Args:
        shot (int): MDSplus style shot number

    Returns:
        ne (2D array): 2D numpy array with density in 1e19 m^3
            first dimension: time
            second dimenion: channel
        chlist (1D array): list of thomson scattering channels
        t (1D array): time in ns
    """
    return _get_thomson(shot, 'ne_ts_')


def get_thomson_Te(shot):
    """get thomson scattering Te profiles

    Args:
        shot (int): MDSplus style shot number

    Returns:
        Te (2D array): 2D numpy array with Te in keV
            first dimension: time
            second dimenion: channel
        chlist (1D array): list of thomson scattering channels
        t (1D array): time in ns
    """
    return _get_thomson(shot, 'Te_ts_')


def _get_thomson(shot, prefix):
    """subfunction to get thomson density or temperature profiles

    Args:
        shot (int): MDSplus style shot number
        prefix (str): 'ne_ts_' or 'Te_ts'
    """
    chlist = [2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    for n, i in enumerate(chlist):
        tmp, t, _ = get_archive_byname(prefix + str(i), shot)
        if n == 0:
            ne = np.empty([len(tmp), len(chlist)])
        ne[:, n] = tmp
    return ne, chlist, t


def program_nr_to_shot(program_nr):
    program_nr = str(program_nr)
    return int(program_nr[2:].replace('.', ''))


def shot_to_program_nr(shot):
    shot = str(shot)
    return '20' + shot[:6] + '.' + shot[6:]


def get_program_info(program_nr):
    """
    Returns dict with keys ['from', 'upto', 'description', 'scenarios', 'id',
    'sessionInfo', 'name', 'trigger'].
    """
    res = query_archive(server_url + '/programs.json', {'from': program_nr})
    return res['programs'][0]


def get_PCI_trigger(program_nr):
    """
    Returns the trigger for the PCI diagnostic. All values are in nanoseconds.
    """
    tree = MDSplus.Tree('QOC', program_nr_to_shot(program_nr))
    delay = tree.getNode('HARDWARE:RPTRIG:DELAY').data() * 1e9  # in ns
    timing = np.asarray(tree.getNode(
        'HARDWARE:RPTRIG:TIMING').data()) * 1e6  # in ns
    T0_mdsplus = int(tree.getNode('TIMING.T0').data()[0])
    info = get_program_info(program_nr)
    T0 = info['trigger']['0'][0]  # T0 in ns

    if T0 != T0_mdsplus:
        raise Warning('T0 in archive and MDSplus are not equal (XP {:s})! '
                      'MDSplus shot and program number not in sync?'.format(program_nr))

    return {'T0': T0,
            'calibration start': T0 + delay,
            'calibration stop': T0 + delay + timing[1],
            'measurement start': T0 + delay + timing[2],
            'measurement stop': T0 + delay + timing[3]}


def get_ECRH(start, end):
    """
    Returns total ECRH power for given time interval [start, end] in nanoseconds.
    Returned dict contains time trace under 'dimensions' key.
    """
    diag_url = ('/raw/W7X/CoDaStationDesc.18774/'
                'FeedBackProcessDesc.18770_DATASTREAM/0/ECRH%20Total%20Power/')
    request_url = archive_url + diag_url + '_signal.json'
    res = query_archive(request_url, params={'from': start, 'upto': end})
    res['dimensions'] = np.asarray(res['dimensions'])
    res['values'] = np.asarray(res['values'])
    return res

def densityscaler(rawsignal, shotnum):
    """Returns rescaled density signal

    Args:
        rawsignal (array): raw interferometer signal
        shotnum (int): MDSplus-style shot number

    Returns:
        line integrated density (array) in m^-2
    """
    # constants from https://w7x-logbook/components?id=QMJ
    if type(rawsignal) == float:
        return np.nan
    if shotnum < 170914000:
        offset = -9989.0
        scaling = -11179218513509570.0
    elif shotnum < 170927000:
        offset = -26386.0
        scaling = -2227794915994841.0
    elif shotnum < 171005000:
        offset = -26386.0
        scaling = -6683384747984523.0
    elif shotnum < 171018000:
        offset = -19577.0
        scaling = +2279257185939055.5
    elif shotnum > 171018000:
        offset = -np.mean(rawsignal[0:200])
        scaling = +2279257185939055.5
    else:
        raise Exception('no density scaling known for shot')
    return scaling * (rawsignal + offset)
