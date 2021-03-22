# funcion para calcular varios indicadores
#---Agustin Laguarda

import numpy as np
import pandas as pd

def calculo_MBD_RMS_MAD(Xest, Xval):
    #---Agustin Laguarda
    #---6/7/2015
    #---25/09/2018 version Python
    #---variables
    #Xest   =  datos generados
    #Xval   =  datos validos
    N   = len(Xest)
    MBD = np.sum(Xest-Xval)/N
    RMS = np.sqrt(sum((Xest-Xval)**2)/N)
    MAD = np.sum(abs(Xest-Xval))/N
    DSD = np.sqrt(sum((Xest-Xval-MBD)**2)/N)

    Xval_medio=np.mean(Xval)
    rMBD = MBD/Xval_medio
    rDSD = DSD/Xval_medio
    rRMS = RMS/Xval_medio
    rMAD = MAD/Xval_medio
    return MBD, rMBD, RMS,rRMS, MAD,rMAD, DSD, rDSD 

def todas_las_metricas(Xest, Xval):
    [MBD, rMBD, RMS,rRMS, MAD,rMAD, corr, N, Xval_medio] = calculo_MBD_RMS_MAD_corr_N_mean(Xest, Xval)
    [KSI, OVER, rKSI, rOVER] = calcular_KSI_OVER(Xest,Xval)
    
    METRICS = pd.Series(\
        [MBD, rMBD, RMS,rRMS, MAD,rMAD, corr, N, Xval_medio, KSI, OVER, rKSI, rOVER],\
        index =['MBD', 'rMBD', 'RMS','rRMS', 'MAD','rMAD','corr', 'N', 'Imean', 'KSI', 'OVER', 'rKSI', 'rOVER'])
    return METRICS
    
    

def calculo_MBD_RMS_MAD_corr_N_mean(Xest, Xval):
    #---Agustin Laguarda
    #---6/7/2015
    #---25/09/2018 version Python
    #---variables
    #Xest   =  datos generados
    #Xval   =  datos validos
    Xest = np.array(Xest)
    Xval = np.array(Xval)
    
    msk_est = np.isfinite(Xest)
    msk_val = np.isfinite(Xval)
    msk = msk_est & msk_val
    N   = len(Xest[msk])
    MBD = np.sum(Xest[msk]-Xval[msk])/N
    RMS = np.sqrt(sum((Xest[msk]-Xval[msk])**2)/N)
    MAD = np.sum(abs(Xest[msk]-Xval[msk]))/N
    
    Xval_medio=np.mean(Xval[msk])
    rMBD = MBD/Xval_medio
    rRMS = RMS/Xval_medio
    rMAD = MAD/Xval_medio
    
    corr = np.corrcoef(Xval[msk], Xest[msk])[0,1]
    
    return MBD, rMBD, RMS,rRMS, MAD,rMAD, corr, N, Xval_medio


#------------------------------------
def ecdf(x): #CDF empirica
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

def calcular_KSI_OVER(Xest,Xval,CDF=0): #CDF es para ver las series
    #---6/07/2015
    #---25/09/2018 - version Python
    #---30/04/2019 - version con forma mas eficiente para bajar el tiempo
    #--- Inicialización
    sVAL = len(Xval)
    sEST = len(Xest)
    if sVAL!=sEST: print('longitudes diferentes, valores relativos no validos')
    Vc = 1.63/np.sqrt(sVAL)
    
    [xCDFest, CDFest]=ecdf(Xest)
    [xCDFval, CDFval]=ecdf(Xval)
    
    xCDF_tot=np.unique(np.concatenate((Xval, Xest))) #concantenate: pega los vectores
    
    CDFval_tot = np.interp(xCDF_tot, xCDFval, CDFval) # interpolo para que sean vectores iguales
    CDFest_tot = np.interp(xCDF_tot, xCDFest, CDFest) # interpolo para que sean vectores iguales
    
    Xmax=max(xCDF_tot); Xmin=min(xCDF_tot)
        # --- Dn y On:
    Dn = abs(CDFval_tot - CDFest_tot)
    On = (Dn - Vc)*(Dn > Vc)
    #---indicadores:
    KSI  = np.trapz(Dn, xCDF_tot)
    OVER = np.trapz(On, xCDF_tot)
    
    # --- Relativos
    rKSI  = KSI/ (Vc*(Xmax - Xmin))
    rOVER = OVER/(Vc*(Xmax - Xmin))
    if CDF ==1: 
        return KSI, OVER, rKSI, rOVER, xCDF_tot, CDFval_tot, CDFest_tot, Dn, On, Vc
    else:
        return KSI, OVER, rKSI, rOVER

def calcular_KSI_OVER_original(Xest,Xval):
    #---6/07/2015
    #---25/09/2018 - version Python
    #--- Inicialización
    sVAL = len(Xval)
    sEST = len(Xest)
    if sVAL!=sEST: print('longitudes diferentes, valores relativos no validos')
    Vc = 1.63/np.sqrt(sVAL)
    xCDF=np.unique(np.concatenate((Xval, Xest))) #concantenate: pega los vectores
                                                #unique ordena (sort) y quita las repeticiones
    Xmax=max(xCDF); Xmin=min(xCDF)
    #--- genero las probabilidades acumuladas:
    CDFest = np.zeros(len(xCDF))
    CDFval = np.zeros(len(xCDF))
    print(Xmax)
    print(Xmin)
    print(xCDF)
    print(len(xCDF))
    for i in range(len(xCDF)):
        print(str(i)+' of '+str(len(xCDF)))
        CDFest[i] = np.sum(Xest<=xCDF[i])
        CDFval[i] = np.sum(Xval<=xCDF[i])
    #---normalizo:
    CDFval = CDFval[:]/sVAL
    CDFest = CDFest[:]/sEST
    
    # --- Dn y On:
    Dn = abs(CDFval - CDFest)
    On = (Dn - Vc)*(Dn > Vc)
    #---indicadores:
    KSI  = np.trapz(Dn, xCDF)
    OVER = np.trapz(On, xCDF)
    
    # --- Relativos
    rKSI = KSI/(Vc*(Xmax - Xmin))
    rOVER = OVER/(Vc*(Xmax - Xmin))
    return KSI, OVER, rKSI, rOVER, xCDF, CDFval, CDFest, Dn, On, Vc
