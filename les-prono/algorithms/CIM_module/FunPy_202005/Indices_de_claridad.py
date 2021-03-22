##calcula variables solares:
## Version 1.0 30/08/2014	Rodrigo Alonso Suárez (matlab)
## version 2.0 20/05/2016  Agustin Laguarda (matlab)
## version 3.0 10/08/2018  LAgustín Laguarda
#
## inputs:
## 1-LATdeg y LONdeg = latitud en longitud en grados y CON signo
## 3-YEA   =  vector anho
## 4-DOY   =  vector con el DOY
## 5-HRA   =  vector con la hora del dia (nro entero!!)
## 6-MIN   =  minutos (en minutos)
## 7-UTC   =  UTC especifica a que zona horaria corresponde HRA (greenwich por defecto)
#
#import numpy as np
#
#def calcular_variables_solares(LATdeg, LONdeg, YEA, DOY, HRA, MIN, UTC=0):
#    HRA-=UTC # lo paso a hora greenwich
#    LATrad = np.deg2rad(LATdeg)
#    diasYEAR = 365 + (YEA%4==0)
#    # --- Variables diarias
#    gam = 2*np.pi*(DOY-1)/diasYEAR
#    gam=np.float64(gam)
#    DELTArad = 0.006918 - 0.399912*np.cos(gam) + 0.070257*np.sin(gam) -0.006758*np.cos(2*gam) + 0.000907*np.sin(2*gam) - 0.002697*np.cos(3*gam) + 0.001480*np.sin(3*gam)
#    Fn = 1.000110 + 0.034221*np.cos(gam) + 0.001280*np.sin(gam) + 0.000719*np.cos(2*gam) + 0.000077*np.sin(2*gam)
#    EcTmin = 60*3.8196667*(0.000075 + 0.001868*np.cos(gam) - 0.032077*np.sin(gam) -0.014615*np.cos(2*gam) - 0.04089*np.sin(2*gam)) # en minutos
#    #229.2~60*3.8196667
#    # --- Angulo horario
#    Hsol = HRA + MIN/60 + (EcTmin+ 4*LONdeg)/60 #LONdeg negativo
#    Wrad = (Hsol-12)*(np.pi/12)
#    # --- Coseno y Seno del ángulo cenital
#    CSZ = np.sin(LATrad)*np.sin(DELTArad) + np.cos(LATrad)*np.cos(DELTArad)*np.cos(Wrad)
#    Wrad=np.array(Wrad); CSZ=np.array(CSZ)
#    
#    return CSZ, Fn, DELTArad, Wrad, EcTmin 
#
#

