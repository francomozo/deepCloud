# cargar DAT CSV de la RMCIS-Agustin Laguarda-
# 8/08/2018
import numpy as np
import os
import pandas as pd
from termcolor import colored

def cargar_RMCIS_E60(RUTAdatos_LES, EST, PRD, iniYEA, iniMES, finYEA, finMES):
#---03/04/2018 (Matlab) : para todos los productos
#solo sirve para archivos E60
#/home/agustin/LES/DATOS/sincro-datos_LES/LB/E60/DHI/2012/LB_DHI_E60_201203.csv
    columna = ['ANO', 'DOY', 'MES', 'DIA', 'HRA', PRD, 'CSZ', 'CDT', 'MCC']
    DAT = pd.DataFrame([])
    #DAT=pd.DataFrame(columns=columna)
    #N_var=9; N_datos_max=(finYEA-iniYEA+1)*366*24
    ruta=RUTAdatos_LES+'/'+EST+'/E60/'+ PRD+'/'
    for kYEAR in range(iniYEA,finYEA+1):  #recorre del primer ano al ultimo solicitado
        strYEAR=str(kYEAR)
        #--- Rango mensual a recorrer
        Mini = 1; Mfin = 12
        if kYEAR == iniYEA: Mini = iniMES
        if kYEAR == finYEA: Mfin = finMES
        # --- Para cada year, recorro los meses correspondientes
        for kMES in range(Mini,Mfin+1):
            strMES = str(kMES) 
            if kMES < 10: strMES ='0'+strMES            
#           --- path
            nombre_arch= EST+'_'+PRD+'_E60_'+strYEAR+strMES+'.csv'
            ruta_k = ruta+strYEAR+'/'+nombre_arch
            if os.path.isfile(ruta_k)==False:
                print( ruta_k+': ' + colored('NO ENCONTRADO ', 'red')) 
            else: 
                print( ruta_k+': cargando...')
                MATaux=pd.read_csv(ruta_k, names = columna, delimiter=',',skiprows=0,na_values=('\tnan','nan', 'NA'))
                DAT = DAT.append(MATaux,ignore_index=True)
    return DAT

def cargar_RMCIS_E60_viejo(RUTAdatos_LES, EST, PRD, iniYEA, iniMES, finYEA, finMES):
#---03/04/2018 (Matlab) : para todos los productos
#solo sirve para archivos E60
#/home/agustin/LES/DATOS/sincro-datos_LES/LB/E60/DHI/2012/LB_DHI_E60_201203.csv
    N_var=9; N_datos_max=(finYEA-iniYEA+1)*366*24
    #creo matriz vacia 
    DAT=np.full([N_datos_max,N_var],np.nan)
    ruta=RUTAdatos_LES+'/'+EST+'/E60/'+ PRD+'/'
    cont=0
    for kYEAR in range(iniYEA,finYEA+1):  #recorre del primer ano al ultimo solicitado
        strYEAR=str(kYEAR)
        #--- Rango mensual a recorrer
        Mini = 1; Mfin = 12
        if kYEAR == iniYEA: Mini = iniMES
        if kYEAR == finYEA: Mfin = finMES
        # --- Para cada year, recorro los meses correspondientes
        for kMES in range(Mini,Mfin+1):
            strMES = str(kMES) 
            if kMES < 10: strMES ='0'+strMES            
#           --- path
            nombre_arch= EST+'_'+PRD+'_E60_'+strYEAR+strMES+'.csv'
            ruta_k = ruta+strYEAR+'/'+nombre_arch
            if os.path.isfile(ruta_k)==False:
                print( ruta_k+': ' + colored('NO ENCONTRADO ', 'red')) 
            else: 
                print( ruta_k+': cargando...')
                MATaux = np.loadtxt(ruta_k,delimiter=',')  #es una matriz auxiliar
                f,c=np.shape(MATaux)
                DAT[cont:cont+f,:]=MATaux
                cont=cont+f
        DAT=DAT[0:cont,:]# recorto la matriz           
#    --- Pasaje a kJ/m2
#    Ighi_date = 3.6*Ighi_date;
    ANO=DAT[:,0];  DOY=DAT[:,1]; MES=DAT[:,2]; DAY=DAT[:,3]; HRA=DAT[:,4]
    GHI=DAT[:,5];  CSZ=DAT[:,6]; TMP=DAT[:,7]; MCC=DAT[:,8]
#    return DAT
    return ANO, DOY, MES, DAY, HRA, GHI, CSZ, TMP, MCC

#/home/agustin/LES/DATOS/datos_LES/datos_CSV/LE/ART/GHI1/2016/LE_GHI1_ART_201602.csv
def cargar_RMCIS_ART(RUTAdatos_LES, EST, PRD, iniYEA, iniMES, finYEA, finMES):
##---03/04/2018 (Matlab) : para todos los productos
#solo sirve para archivos ART
    N_var=10; N_datos_max=(finYEA-iniYEA+1)*366*24*60
    #creo matriz vacia 
    DAT=np.full([N_datos_max,N_var],np.nan)
    ruta = RUTAdatos_LES+'/'+EST+'/ART/'+ PRD+'/'
    cont=0
    for kYEAR in range(iniYEA,finYEA+1):  #recorre del primer ano al ultimo solicitado
        strYEAR=str(kYEAR)
        #--- Rango mensual a recorrer
        Mini = 1; Mfin = 12
        if kYEAR == iniYEA: Mini = iniMES
        if kYEAR == finYEA: Mfin = finMES
        # --- Para cada year, recorro los meses correspondientes
        for kMES in range(Mini,Mfin+1):
            strMES = str(kMES) 
            if kMES < 10: strMES ='0'+strMES            
#           --- path
            nombre_arch= EST+'_'+PRD+'_ART_'+strYEAR+strMES+'.csv'
            ruta_k = ruta+strYEAR+'/'+nombre_arch
            #if os.path.isfile(ruta_k)==False:
            #    print( ruta_k+': ' + colored('NO ENCONTRADO ', 'red')) 
            try: 
                print( ruta_k+': cargando...')
                MATaux = np.loadtxt(ruta_k,delimiter=',')  #es una matriz auxiliar
                f,c=np.shape(MATaux)
                #DAT2=np.append(DAT2, MATaux)
                DAT[cont:cont+f,]=MATaux
                cont=cont+f
            except OSError: print( ruta_k+': ' + colored('NO ENCONTRADO ', 'red'))
    DAT=DAT[0:cont,]# recorto la matriz           
#    --- Pasaje a kJ/m2
#    Ighi_date = 3.6*Ighi_date;
    ANO=DAT[:,0];  DOY=DAT[:,1]; MES=DAT[:,2]; DAY=DAT[:,3]; HRA=DAT[:,4]
    MIN=DAT[:,5];  FLAG=DAT[:,6]; X =DAT[:,7]; CSZ=DAT[:,8]; TMP =DAT[:,9]
#    return DAT
    return ANO, DOY, MES, DAY, HRA, MIN, FLAG, X, CSZ, TMP

def cargar_RMCIS_E15(RUTAdatos_LES, EST, PRD, iniYEA, iniMES, finYEA, finMES):
##---03/04/2018 (Matlab) : para todos los productos
#solo sirve para archivos ART
    N_var=10; N_datos_max=(finYEA-iniYEA+1)*366*24*4
    #creo matriz vacia 
    DAT=np.full([N_datos_max,N_var],np.nan)
    ruta = RUTAdatos_LES+'/'+EST+'/E15/'+ PRD+'/'
    cont=0
    for kYEAR in range(iniYEA,finYEA+1):  #recorre del primer ano al ultimo solicitado
        strYEAR=str(kYEAR)
        #--- Rango mensual a recorrer
        Mini = 1; Mfin = 12
        if kYEAR == iniYEA: Mini = iniMES
        if kYEAR == finYEA: Mfin = finMES
        # --- Para cada year, recorro los meses correspondientes
        for kMES in range(Mini,Mfin+1):
            strMES = str(kMES) 
            if kMES < 10: strMES ='0'+strMES            
#           --- path
            nombre_arch= EST+'_'+PRD+'_E15_'+strYEAR+strMES+'.csv'
            ruta_k = ruta+strYEAR+'/'+nombre_arch
            #if os.path.isfile(ruta_k)==False:
            #    print( ruta_k+': ' + colored('NO ENCONTRADO ', 'red')) 
            try: 
                print( ruta_k+': cargando...')
                MATaux = np.loadtxt(ruta_k,delimiter=',')  #es una matriz auxiliar
                f,c=np.shape(MATaux)
                DAT[cont:cont+f,]=MATaux
                cont=cont+f
            except OSError: print( ruta_k+': ' + colored('NO ENCONTRADO ', 'red'))
    DAT=DAT[0:cont,]# recorto la matriz           
#    --- Pasaje a kJ/m2
#    Ighi_date = 3.6*Ighi_date;
    ANO=DAT[:,0];  DOY=DAT[:,1]; MES=DAT[:,2]; DAY=DAT[:,3]; HRA=DAT[:,4]
    MIN=DAT[:,5];  X=DAT[:,6]; CSZ =DAT[:,7]; TMP=DAT[:,8]; FLAG =DAT[:,9]
#    return DAT
    return ANO, DOY, MES, DAY, HRA, MIN, FLAG, X, CSZ, TMP

def cargar_RMCIS_S60(RUTAdatos_LES, EST, PRD, iniYEA, iniMES, finYEA, finMES):
#---03/04/2018 (Matlab) : para todos los productos
#solo sirve para archivos E60
#/home/agustin/LES/DATOS/sincro-datos_LES/LB/E60/DHI/2012/LB_DHI_E60_201203.csv
    columna = ['ANO', 'DOY', 'MES', 'DIA', 'HRA', PRD, 'CSZ', 'CDT', 'MCC']
    DAT = pd.DataFrame([])
    #DAT=pd.DataFrame(columns=columna)
    #N_var=9; N_datos_max=(finYEA-iniYEA+1)*366*24
    ruta=RUTAdatos_LES+'/'+EST+'/S60/'+ PRD+'/'
    for kYEAR in range(iniYEA,finYEA+1):  #recorre del primer ano al ultimo solicitado
        strYEAR=str(kYEAR)
        #--- Rango mensual a recorrer
        Mini = 1; Mfin = 12
        if kYEAR == iniYEA: Mini = iniMES
        if kYEAR == finYEA: Mfin = finMES
        # --- Para cada year, recorro los meses correspondientes
        for kMES in range(Mini,Mfin+1):
            strMES = str(kMES) 
            if kMES < 10: strMES ='0'+strMES            
#           --- path
            nombre_arch= EST+'_'+PRD+'_S60_'+strYEAR+strMES+'.csv'
            ruta_k = ruta+strYEAR+'/'+nombre_arch
            if os.path.isfile(ruta_k)==False:
                print( ruta_k+': ' + colored('NO ENCONTRADO ', 'red')) 
            else: 
                print( ruta_k+': cargando...')
                MATaux=pd.read_csv(ruta_k, names = columna,delimiter=',+\s*',skiprows=0,na_values=('NaN', '\tnan','nan', 'NA','NaN')) 
                DAT = DAT.append(MATaux,ignore_index=True)
    return DAT



