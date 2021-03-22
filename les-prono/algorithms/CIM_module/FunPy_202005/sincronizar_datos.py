#SINCRONIZAR VARIABLES
#---Agustin Laguarda
#---31/08/2018
"""
Created on Fri Aug 31 13:34:21 2018

@author: agustin
"""

# cargar datos PX -Agustin Laguarda-
# 8/08/2018
import numpy as np
from termcolor import colored

#---inputs:
# ITval: indice temporal de referencia 
# X    : datos que quiero organizar (float), puede ser matriz
# ITx  : tiempos de los datos a arreglar
#---output: Xsincro. X pero sincronizado con las etiquetas validas
#OBS: si X (=> Xsincro) es multiple array, cada columna es una variable y cada fila corresponde a un tiempo   
def sincronizar_datos_codtemp(ITval,ITx, X):
    Nval=np.size(ITval);  #nro de filas
    #defino la dimension de la matriz Xsincro
    try: DIMval = (Nval,X.shape[1])
    except IndexError: DIMval = (Nval,)
    Xsincro=np.zeros(DIMval)*np.nan # matriz vacia
    error=0.1/60/24 # una decima de minuto
    for i in range(Nval):#recorro los tiempos validos
        #print(Nval-i)
        tmp_v=ITval[i]
        msk= np.abs(ITx-tmp_v)<error
        #plt.plot(msk)
        try:Xsincro[i]=X[msk]
        except ValueError:
            if sum(msk)>1: print(colored('hay varios datos con la misma etiqueta temporal en la entrada '+str(i),'red'))
            if sum(msk)==0:print(colored('no hay datos en la etiqueta temporal en la entrada: '+str(i),'red')) 
    return Xsincro


#inputs:
# YEAval, DOYval, HRAval: tiempos de referencia (enteros)
# X                     : datos que quiero organizar (float), puede ser matriz
# YEAx, DOYx, HRAx      : tiempos de los datos a arreglar
# output: Xsincro. X pero sincronizado con las etiquetas validas   
def sincronizar_datos_horarios(YEAval, DOYval, HRAval, YEAx, DOYx, HRAx, X):
    Nval=np.size(YEAval);  #nro de filas
    #defino la dimension de la matriz Xsincro
    try:DIMval = (Nval,X.shape[1])
    except IndexError: DIMval = (Nval,)
    Xsincro=np.zeros(DIMval)*np.nan # matriz vacia
    for i in range(Nval):#recorro los tiempos validos
        print('sincronizando...:'+np.str(Nval-i))
        yea_v=YEAval[i]; doy_v=DOYval[i]; hra_v=HRAval[i]
        msk= (YEAx==yea_v) & (DOYx==doy_v) & (HRAx==hra_v)
        #plt.plot(msk)
        if sum(msk)>1: print(colored('hay varios datos con la misma etiqueta temporal','red'))
        #if sum(msk)==1: 
        Xsincro[i,]=X[msk,]
    return Xsincro
    
def sincronizar_datos_minutales(YEAval, DOYval, HRAval,MINval, YEAx, DOYx, HRAx,MINx, X):
    print('caca')
    Nval=np.size(YEAval)  #nro de filas
    #defino la dimension de la matriz Xsincro
    try:DIMval = (Nval,X.shape[1])
    except IndexError: DIMval = (Nval,)
    Xsincro=np.zeros(DIMval)*np.nan # matriz vacia
    for i in range(Nval):#recorro los tiempos validos
        print('sincronizando...:'+ np.str(Nval-i))    
        yea_v=YEAval[i]; doy_v=DOYval[i]; hra_v=HRAval[i]; min_v=MINval[i]
        msk= (YEAx==yea_v) & (DOYx==doy_v) & (HRAx==hra_v) & (MINx==min_v) 
        #plt.plot(msk)
        #if sum(msk)>1: print(colored('hay varios datos con la misma etiqueta temporal','red'))
        try: Xsincro[i,]=X[msk,]
        except ValueError: print('Falta el dato: año-'+ np.str(yea_v)+', doy-'+np.str(doy_v)+', hora-'+np.str(hra_v)+', minuto-'+np.str(min_v))
    return Xsincro
      
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    