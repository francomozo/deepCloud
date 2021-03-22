# cargar datos PX -Agustin Laguarda-
# 8/08/2018
import numpy as np
import pandas as pd
import os
from termcolor import colored

def cargar_PX_RAD(RUTAdatos_LES, EST, PXxx, iniYEA, iniDOY, finYEA, finDOY):
    strPX=str(PXxx) # strPX debe ser divisor de 1440
    if PXxx<10: strPX= '0'+strPX
    #creo matriz vacia 
    DAT=pd.DataFrame(columns={"NR","Fecha","N","T","CZ","GHI1","GHI2","DHI","DNI1","DNI2","GTI","TI1A","TI2A","TI1B","TI2B","TA","TL","kt","fd"})
#   /home/agustin/LES/DATOS/datos_LES/datos_PX/
    ruta=RUTAdatos_LES+'/'+EST+'/PX'+strPX+'/'
    for kYEAR in range(iniYEA,finYEA+1):  #recorre del primer ano al ultimo solicitado
        strYEAR=str(kYEAR)
        # --- Para cada year, recorro los dias correspondientes
        Dini = 1; Dfin = 365+((kYEAR%4)==0)
        if kYEAR == iniYEA: Dini = iniDOY
        if kYEAR == finYEA: Dfin = finDOY
        for kDOY in range(Dini,Dfin+1):
            strDOY = str(kDOY)
            if kDOY < 10: strDOY = '00'+strDOY            
            if (kDOY<100 and kDOY >= 10): strDOY = '0'+strDOY       
            nombre_arch = EST+'_PX'+strPX+'_RAD_'+strYEAR+strDOY+'.csv'
            ruta_k = ruta+strYEAR+'/VIS/CSV/'+nombre_arch
            if os.path.isfile(ruta_k)==False:
                print( ruta_k+': ' + colored('NO ENCONTRADO ', 'red')) 
            else: 
                print( ruta_k+': cargando...')
                MATaux=pd.read_csv(ruta_k,dtype={"NR": int, "Fecha":str }, parse_dates=['Fecha'],
                             delimiter=',',skiprows=0,na_values='NA')
                DAT=np.append(DAT,MATaux,axis=0)
    TIME=DAT[:,1];
    YEA=pd.DatetimeIndex(TIME).year; MES=pd.DatetimeIndex(TIME).month
    DAY=pd.DatetimeIndex(TIME).day ; HRA=pd.DatetimeIndex(TIME).hour
#    "NR","Fecha","N","T","CZ","GHI1","GHI2","DHI","DNI1","DNI2","GTI","TI1A","TI2A","TI1B","TI2B","TA","TL","kt","fd"
    DOY =DAT[:,2] ; TMP=DAT[:,3] ; CSZ=DAT[:,4]  ; 
    GHI1=DAT[:,5] ;GHI2=DAT[:,6] ; DHI=DAT[:,7]  ; DNI1=DAT[:,8];
    DNI2=DAT[:,9] ; GTI=DAT[:,10]; TI1A=DAT[:,11]; TI2A=DAT[:,12];
    TI1B=DAT[:,13];TI2B=DAT[:,14]; TA=DAT[:,15]  ; TL=DAT[:,16];
    kt=DAT[:,17]  ;  fd=DAT[:,18];

    return YEA, DOY, HRA, CSZ, TMP, MES, DAY, GHI1, GHI2, DHI, DNI1, DNI2, GTI, TI1A, TI2A, TI1B, TI2B, TA, TL, kt, fd, TIME


def cargar_PX_RMCIS(RUTAdatos_LES, EST, PXxx, iniYEA, iniDOY, finYEA, finDOY):
    strPX=str(PXxx) # strPX debe ser divisor de 1440
    if PXxx<10: strPX= '0'+strPX
    #creo matriz vacia 
    DAT=pd.DataFrame([])#columns={"NR","Fecha","N","T","CZ","GHI1","GHI2","DHI","DNI1","DNI2","GTI","TI1A","TI2A","TI1B","TI2B","TA","TL","kt","fd"})
#   /home/agustin/LES/DATOS/datos_LES/datos_PX/
    ruta=RUTAdatos_LES+'/'+EST+'/PX'+strPX+'/'
    for kYEAR in range(iniYEA,finYEA+1):  #recorre del primer ano al ultimo solicitado
        strYEAR=str(kYEAR)
        # --- Para cada year, recorro los dias correspondientes
        Dini = 1; Dfin = 365+((kYEAR%4)==0)
        if kYEAR == iniYEA: Dini = iniDOY
        if kYEAR == finYEA: Dfin = finDOY
        for kDOY in range(Dini,Dfin+1):
            strDOY = str(kDOY)
            if kDOY < 10: strDOY = '00'+strDOY            
            if (kDOY<100 and kDOY >= 10): strDOY = '0'+strDOY       
            nombre_arch = 'PX'+strPX+'_'+EST+'_RAD_'+strYEAR+strDOY+'.csv'
            ruta_k = ruta+strYEAR+'/CSV/'+nombre_arch
            if os.path.isfile(ruta_k)==False:
                print( ruta_k+': ' + colored('NO ENCONTRADO ', 'red')) 
            else: 
                print( ruta_k+': cargando...')
                MATaux=pd.read_csv(ruta_k, na_values='NA')
                DAT=DAT.append(MATaux).reset_index(drop=True)
    #DAT['Fecha'] = pd.DatetimeIndex(DAT['Fecha'])
    return DAT  #devuelve un dataframeYEA, DOY, HRA, CSZ, TMP, MES, DAY, GHI1, GHI2, DHI, DNI1, DNI2, GTI, TI1A, TI2A, TI1B, TI2B, TA, TL, kt, fd, TIME


def cargar_PX_DEP(RUTAdatos_LES, EST, PXxx, iniYEA, iniDOY, finYEA, finDOY, nivel_dep):
    strPX=str(PXxx) # strPX debe ser divisor de 1440
    if PXxx<10: strPX= '0'+strPX
    #creo matriz vacia 
    #Fecha,N,T,CZ,GHI,DHI,DNI,GTI,TA,NR - dep3
    #/LE/PXDEP3/PX15/2017/VIS/PX_DEP_3/CSV/PX15_LE_DEP_3_2017013.csv
    
    #dep2
    #"Fecha","N","T","CZ","GHI","DHI","DNI","GTI","TA","NR"
    #/LE/PXDEP1/PX15/
    DAT=pd.DataFrame([])#columns={"Fecha","N","T","CZ","GHI","DHI","DNI","GTI","TA","NR"})
    ruta=RUTAdatos_LES+'/'+EST+'/PXDEP'+ str(nivel_dep)+'/PX'+strPX+'/'
    for kYEAR in range(iniYEA,finYEA+1):  #recorre del primer ano al ultimo solicitado
        strYEAR=str(kYEAR)
        # --- Para cada year, recorro los dias correspondientes
        Dini = 1; Dfin = 365+((kYEAR%4)==0)
        if kYEAR == iniYEA: Dini = iniDOY
        if kYEAR == finYEA: Dfin = finDOY
        for kDOY in range(Dini,Dfin+1):
            strDOY = str(kDOY)
            if kDOY < 10: strDOY = '00'+strDOY            
            if (kDOY<100 and kDOY >= 10): strDOY = '0'+strDOY       
            #2017/VIS/PX_DEP_1/CSV/PX15_LE_DEP_1_2017023.csv
            nombre_arch = 'PX'+strPX+'_'+EST+'_DEP_'+str(nivel_dep)+'_'+strYEAR+strDOY+'.csv'
            ruta_k = ruta+strYEAR+'/VIS/PX_DEP_'+str(nivel_dep)+'/CSV/'+nombre_arch
            if os.path.isfile(ruta_k)==False:
                print( ruta_k+': ' + colored('NO ENCONTRADO ', 'red')) 
            else: 
                print( ruta_k+': cargando...')
                #MATaux=pd.read_csv(ruta_k,dtype={"NR": int, "Fecha":str }, parse_dates=['Fecha'],
                #             delimiter=',',skiprows=0,na_values='NA')
                #DAT=np.append(DAT,MATaux,axis=0)
                MATaux=pd.read_csv(ruta_k, na_values='NA')
                DAT=DAT.append(MATaux).reset_index(drop=True)
                #"Fecha","N","T","CZ","GHI","DHI","DNI","GTI","TA","NR"
#    TIME=DAT[:,0];
#    YEA=np.array(pd.DatetimeIndex(TIME).year); MES=np.array(pd.DatetimeIndex(TIME).month)
#    DAY=np.array(pd.DatetimeIndex(TIME).day) ; HRA=np.array(pd.DatetimeIndex(TIME).hour) ;
#    MIN=np.array(pd.DatetimeIndex(TIME).minute)
#    DOY =DAT[:,1] ; TMP =DAT[:,2] ; CSZ=DAT[:,3]  ; 
#    GHI =DAT[:,4] ; DHI =DAT[:,5] ; DNI=DAT[:,6]  ; GTI=DAT[:,7];
#    Tamb=DAT[:,8] ; Tlog=DAT[:,9];
    return DAT #YEA, DOY, HRA, MIN, CSZ, TMP, MES, DAY, GHI, DHI, DNI, GTI, Tamb, Tlog, TIME

def cargar_PX_UV(RUTAdatos_LES, EST, PXxx, iniYEA, iniDOY, finYEA, finDOY):
    strPX=str(PXxx) 
    if PXxx<10: strPX= '0'+strPX
    #creo matriz vacia 
    DAT=pd.DataFrame(columns={"NR","Fecha","N","T","CZ","UVA","UVB","UVE","IUV","TA","TL"})
    ruta=RUTAdatos_LES+'/'+EST+'/PXDEP0/PX'+strPX+'/'
    for kYEAR in range(iniYEA,finYEA+1):  #recorre del primer ano al ultimo solicitado
        strYEAR=str(kYEAR)
        # --- Para cada year, recorro los dias correspondientes
        Dini = 1; Dfin = 365+((kYEAR%4)==0)
        if kYEAR == iniYEA: Dini = iniDOY
        if kYEAR == finYEA: Dfin = finDOY
        for kDOY in range(Dini,Dfin+1):
            strDOY = str(kDOY)
            if kDOY < 10: strDOY = '00'+strDOY            
            if (kDOY<100 and kDOY >= 10): strDOY = '0'+strDOY       
            nombre_arch = EST+'_PX'+strPX+'_UV_'+strYEAR+strDOY+'.csv'#LE_PX15_UV_2015275.csv
            ruta_k = ruta+strYEAR+'/UV/CSV/'+nombre_arch
            if os.path.isfile(ruta_k)==False:
                print( ruta_k+': ' + colored('NO ENCONTRADO ', 'red')) 
            else: 
                print( ruta_k+': cargando...')
                MATaux=pd.read_csv(ruta_k,dtype={"NR": int, "Fecha":str }, parse_dates=['Fecha'],
                             delimiter=',',skiprows=0,na_values='NA')
                DAT=np.append(DAT,MATaux,axis=0)
    TIME=DAT[:,1];
    YEA=np.array(pd.DatetimeIndex(TIME).year)
    MES=np.array(pd.DatetimeIndex(TIME).month)
    DAY=np.array(pd.DatetimeIndex(TIME).day)
    HRA=np.array(pd.DatetimeIndex(TIME).hour); MIN=np.array(pd.DatetimeIndex(TIME).minute)
    DOY =DAT[:,2]; TMP=DAT[:,3]; CSZ=DAT[:,4]; 
    UVA =DAT[:,5];  UVB=DAT[:,6]; UVE=DAT[:,7]; IUV=DAT[:,8];
    Tamb=DAT[:,9]; Tlog=DAT[:,10];
 
    return  YEA, DOY, HRA, MIN, CSZ, TMP, MES, DAY, UVA, UVB, UVE, IUV, Tamb, Tlog, TIME


