# cargar datos atmosfericos ad-hoc de Giovanni
#---Agustin Laguarda-
# 29/08/2018
import pandas as pd
import numpy as np
import os
from termcolor import colored
"""
Created on Wed Aug 29 12:46:19 2018

@author: agustin
"""
def cargar_Giovanni_adhoc(RUTAdatos, nombre_arch):
    ruta_arch=RUTAdatos+nombre_arch
    if os.path.isfile(ruta_arch)==False:
        print( ruta_arch+': ' + colored('NO ENCONTRADO ', 'red')) 
    else: 
        print( ruta_arch+': '+ colored('cargado','green'))
        MAT=pd.read_csv(ruta_arch, names=['time','datos'], delimiter=',',
            skiprows=8, na_values=1e15, parse_dates=['time'])
    TIME=np.array(MAT['time'])
    DATOS=np.array(MAT['datos'])
    YEA=pd.DatetimeIndex(TIME).year; MES=pd.DatetimeIndex(TIME).month
    DAY=pd.DatetimeIndex(TIME).day ; HRA=pd.DatetimeIndex(TIME).hour
    MIN=pd.DatetimeIndex(TIME).minute
    DOY=pd.DatetimeIndex(TIME).dayofyear ;
 

    return DATOS, YEA, DOY, HRA, MIN, MES, DAY