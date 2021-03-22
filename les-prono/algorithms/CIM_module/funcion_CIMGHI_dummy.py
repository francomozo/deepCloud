# Agustin Laguarda 1/07/2020
# Generar CIM-ESRA en ubicacion arbitraria en el norte de Uruguay, 
# con varias varialbes por defecto. 
#---NO USAR SERIAMENTE---

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pvlib as pv
import sys
# sys.path.append('./FunPy_202005/')
import algorithms.CIM_module.FunPy_202005.variables_solares as vs
import algorithms.CIM_module.FunPy_202005.modelos_cielo_claro as mcc

# INPUTS 
# T -  serie de tiempo  (en UTC-0)
# Rp-  (dataframe datetimeindex) - (concordante con T)
# lat, lon, alt.
# paso (10 minutal), etc...

def generar_CIM_GHI_dumy(T, Rp, LATdeg, LONdeg, paso, alt=50, est='xxx'):
    # import pdb; pdb.set_trace()
    #PRUEBA
    #est = 'les'
    #LATdeg =-31.28 #LE
    #LONdeg =-57.92 #LE
    #alt = 56
    #paso= 15 #minutal
    
    zo = 8434.5 # msnm.
    pcoef =  np.exp(-alt/zo)

    #############################################################################
    #    1 - CANTIDADES UTILES
    #############################################################################
    # ANGULOS SOLARES
    ANGS = pv.solarposition.get_solarposition( T+pd.Timedelta( str(paso/2)+'min'), LATdeg, LONdeg)#altLOC[j]
    ANGS.index=T #etiqueta al inicio de la hora
    ZEdeg = ANGS.zenith
    CZEho = np.cos(ZEdeg*np.pi/180)
    #CSZ[CSZ<0]=0
    
    [_, Fn,_,_,_,_] = vs.calcular_variables_solares(LATdeg, LONdeg, T.dayofyear, T.hour, T.minute+paso/2, T.year, 0)
    Pcoef =  Fn*0+pcoef

    ###########################################################################
    #    2. Genero estimativos de cielo claro
    ###########################################################################
    #importo ciclos TL
    cicosname = Path(os.path.abspath(__file__)).parent / 'cicos_TL/TL_ciclo_Norte'
    CY_TLesl = pd.read_csv(cicosname,#'./cicos_TL/TL_ciclo_Norte',
                            index_col =0) #ciclo regional 
    #-- series de -TL interpoladoa linealmente
    doy_medio = [15, 45, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350] #doy del centro de cada mes
    TL = np.interp(T.dayofyear, doy_medio, CY_TLesl['Norte'], period = 366)
    [ESR,_,_] = mcc.ESRA(CZEho, Fn, TL, Pcoef)

    #############################################################################
    #    3. Indice de claridad 
    #############################################################################
    #FR = SAT[7]
    #Rp = FR/CZEho
    #R0[R0>16] = 16
    Rmax = 80
    R0 = 10
    C = (Rp - R0)/(Rmax-R0)
    C[C>1.05]=1.05
    C[C<-0.05] = -0.05

    #############################################################################
    #    4. Modelo CIMIndice de claridad 
    #############################################################################
    a = 0.90
    b = 0.0884
    FC = a*(1-C) + b
    GHIcim = ESR*FC
    GHIcim[CZEho<0]=0
    
    return GHIcim
