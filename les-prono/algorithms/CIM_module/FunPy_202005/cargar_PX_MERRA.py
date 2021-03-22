# cargar datos PX -Agustin Laguarda-
# 8/08/2018
#27/08/2019 similar pero devuelve un dataframe con todas las variables
#import numpy as np
import pandas as pd
import os
from termcolor import colored
import pylab as plt

# en la vesion 3 el indice de los arhchivos importados es el datetimeindex
def cargar_PX_MERRA_v3(RUTAdatos_MERRA, EST, PXxx, iniYEA, finYEA):
    #fucnion sin terminar!!!!
    PRD = ['ANG_coef', 'AOT550', 'O3', 'WV']    
    T = pd.date_range(pd.Timestamp(year=iniYEA, day = 1, month=1, hour =0, minute=0), pd.Timestamp(year=finYEA, day = 31, month=12, hour =23, minute=60-PXxx), freq=str(int(PXxx))+'min')
    
    DAT = pd.DataFrame(index = T, columns = PRD)#PRD = ['ANG_coef' 'AOT550' 'O3' 'WV']
    strPX=str(int(PXxx)) # strPX debe ser divisor de 1440
    if PXxx<10: strPX= '0'+strPX

    for prd in PRD:
        ruta = RUTAdatos_MERRA+'/'+EST+'/'+ prd + '/PX'+strPX + '/'
        for kYEAR in range(iniYEA,finYEA+1):  #recorre del primer ano al ultimo solicitado
            strYEA = str(kYEAR)
            #/LE/ANG_coef/PX60/2015_LE_PX60_ANG_coef.csv
            nombre_arch = strYEA + '_'+EST+'_'+'PX'+strPX+'_'+prd+'.csv'
            ruta_k = ruta + nombre_arch
            if os.path.isfile(ruta_k)==False:
                print( ruta_k+': ' + colored('NO ENCONTRADO ', 'red')) 
            else: 
                print( ruta_k+': cargando...')
                MATaux= pd.read_csv(ruta_k, index_col=0, parse_dates=True, na_values='NA')
                #print(MATaux.index.dtype)
                #print(MATaux)
                #MATaux = MATaux.astype('float64')
                #plt.plot(MATaux[prd])
                #Taux = pd.DatetimeIndex(MATaux.Time)
                #MATaux.index=Taux
                Taux = MATaux.index
                DAT[prd][Taux]= MATaux[prd]               
                
    return DAT  #devuelve un dataframe Tiempo, PRD 


def cargar_PX_MERRA_v2(RUTAdatos_MERRA, EST, PXxx, iniYEA, finYEA):
    #fucnion sin terminar!!!!
    PRD = ['ANG_coef', 'AOT550', 'O3', 'WV']    
    T = pd.date_range(pd.Timestamp(year=iniYEA, day = 1, month=1, hour =0), pd.Timestamp(year=finYEA, day = 31, month=12, hour =23), freq=str(int(PXxx))+'min')
    
    DAT = pd.DataFrame(index = T, columns = PRD)#PRD = ['ANG_coef' 'AOT550' 'O3' 'WV']
    strPX=str(int(PXxx)) # strPX debe ser divisor de 1440
    if PXxx<10: strPX= '0'+strPX
    #creo matriz vacia 
    #DAT=pd.DataFrame([])#columns={"NR","Fecha","N","T","CZ","GHI1","GHI2","DHI","DNI1","DNI2","GTI","TI1A","TI2A","TI1B","TI2B","TA","TL","kt","fd"})
    for prd in PRD:
        ruta=RUTAdatos_MERRA+'/'+EST+'/'+ prd + '/PX'+strPX + '/'
        for kYEAR in range(iniYEA,finYEA+1):  #recorre del primer ano al ultimo solicitado
            strYEA = str(kYEAR)
            #/LE/ANG_coef/PX60/2015_LE_PX60_ANG_coef.csv
            nombre_arch = strYEA + '_'+EST+'_'+'PX'+strPX+'_'+prd+'.csv'
            ruta_k = ruta + nombre_arch
            if os.path.isfile(ruta_k)==False:
                print( ruta_k+': ' + colored('NO ENCONTRADO ', 'red')) 
            else: 
                print( ruta_k+': cargando...')
                MATaux= pd.read_csv(ruta_k, na_values='NA')
                Taux = pd.DatetimeIndex(MATaux.Time)
                MATaux.index=Taux
                
            DAT[prd][Taux]= MATaux[prd][Taux]               
                
            
    return DAT  #devuelve un dataframe Tiempo, PRD 



def cargar_PX_MERRA_v1(RUTAdatos_MERRA, EST, PXxx, PRD, iniYEA, finYEA):
    strPX=str(PXxx) # strPX debe ser divisor de 1440
    if PXxx<10: strPX= '0'+strPX
    #creo matriz vacia 
    DAT=pd.DataFrame([])#columns={"NR","Fecha","N","T","CZ","GHI1","GHI2","DHI","DNI1","DNI2","GTI","TI1A","TI2A","TI1B","TI2B","TA","TL","kt","fd"})
    ruta=RUTAdatos_MERRA+'/'+EST+'/'+ PRD + '/PX'+strPX + '/'
    for kYEAR in range(iniYEA,finYEA+1):  #recorre del primer ano al ultimo solicitado
        strYEA = str(kYEAR)
        #/LE/ANG_coef/PX60/2015_LE_PX60_ANG_coef.csv
        nombre_arch = strYEA + '_'+EST+'_'+'PX'+strPX+'_'+PRD+'.csv'
        ruta_k = ruta + nombre_arch
        if os.path.isfile(ruta_k)==False:
            print( ruta_k+': ' + colored('NO ENCONTRADO ', 'red')) 
        else: 
            print( ruta_k+': cargando...')
            MATaux=pd.read_csv(ruta_k, na_values='NA')
            DAT=DAT.append(MATaux).reset_index(drop=True)
    return DAT  #devuelve un dataframe Tiempo, PRD 
