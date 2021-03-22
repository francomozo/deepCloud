#calcula variables solares:
# Version 1.0 30/08/2014	Rodrigo Alonso Suárez (matlab)
# version 2.0 20/05/2016  Agustin Laguarda (matlab)
# version 3.0 10/08/2018  LAgustín Laguarda

# inputs:
# 1-LATdeg y LONdeg = latitud en longitud en grados y CON signo
# 3-YEA   =  vector anho
# 4-DOY   =  vector con el DOY
# 5-HRA   =  vector con la hora del dia (nro entero!!)
# 6-MIN   =  minutos (en minutos)
# 7-UTC   =  UTC especifica a que zona horaria corresponde HRA (greenwich por defecto)

import numpy as np
from . import CODtemp as TMP

CS= 1367.0 # W/m2

def Gamma(DOY, YEA=2001):
    diasYEAR = 365. + (YEA%4==0)
    gam = 2.*np.pi*(DOY-1)/diasYEAR
    gam=np.float64(gam)
    return gam     

def Declinacion_solar(LATdeg, LONdeg, DOY, YEA=2001):
    gam=Gamma(DOY, YEA)    
    declinacion = 0.006918 - 0.399912*np.cos(gam) + 0.070257*np.sin(gam) -\
    0.006758*np.cos(2.*gam) + 0.000907*np.sin(2.*gam) - 0.002697*np.cos(3.*gam) + 0.001480*np.sin(3.*gam)
    return declinacion

def Fn(gam):    
    fn = 1.000110 + 0.034221*np.cos(gam) + 0.001280*np.sin(gam) + 0.000719*np.cos(2.*gam) + 0.000077*np.sin(2.*gam)
    return fn

def EoT(DOY, YEA):
    gam=Gamma(DOY, YEA)
    EcTmin = 60*3.8196667*(0.000075 + 0.001868*np.cos(gam) - 0.032077*np.sin(gam) -0.014615*np.cos(2.*gam) - 0.04089*np.sin(2.*gam)) # en minutos
    return EcTmin

def acimut(CSZ, delta, omega_rad, lat_deg):
    lat_rad = lat_deg *np.pi/180
    # 19/04/2020 - version Python
    az = np.sign(omega_rad)* np.abs( np.arccos( (np.sin(delta) - CSZ*np.sin(lat_rad) )/(np.cos(lat_rad)*np.sin(np.arccos(CSZ))) ))
    return az


def calcular_variables_solares(LATdeg, LONdeg, DOY, HRA, MIN, YEA, UTC):
    HRAgw=HRA-UTC # lo paso a hora greenwich
    # --- Variables diarias
    DELTArad=Declinacion_solar(LATdeg, LONdeg, DOY, YEA)
    EcTmin=EoT(DOY, YEA)
    gam=Gamma(DOY, YEA)
    fn=Fn(gam)
    #229.2~60*3.8196667
    # --- Angulo horario
    Hsol = HRAgw + MIN/60. + (EcTmin+ 4.*LONdeg)/60. 
    Wrad = (Hsol-12)*(np.pi/12.)
    Wrad=np.float64(Wrad)
    LATrad = np.deg2rad(LATdeg)
    CSZ = np.sin(LATrad)*np.sin(DELTArad) + np.cos(LATrad)*np.cos(DELTArad)*np.cos(Wrad)
    AZrad = acimut(CSZ, DELTArad, Wrad, LATdeg)
    return CSZ, fn, DELTArad, Wrad, EcTmin, AZrad

def ang_cenital_min(LATdeg, LONdeg, DOY, YEA):
    # --- Variables diarias
    DELTArad=Declinacion_solar(LATdeg, LONdeg, DOY, YEA)
    LATrad = np.deg2rad(LATdeg)
    CSZ = np.sin(LATrad)*np.sin(DELTArad) + np.cos(LATrad)*np.cos(DELTArad)
    ZENmin_rad = np.arccos(CSZ)     
    return ZENmin_rad

def ang_horario_acimut_atardecer(LATdeg, LONdeg, DOY, YEA):
    LATrad = LATdeg*np.pi/180
    DECrad= Declinacion_solar(LATdeg, LONdeg, DOY, YEA)
    Wss = np.arccos(-np.tan(LATrad)*np.tan(DECrad))
    AZss = np.abs( np.arccos( (np.sin(DECrad))/(np.cos(LATrad))) )
    return Wss, AZss
    
#igual a la v1, pero se basa en omega (y la hora solar)
def calcular_variables_solares_de_omega(LATdeg, LONdeg, DOY, Wrad, YEA, UTC):
    #HRAgw=HRA-UTC # lo paso a hora greenwich
    # --- Variables diarias
    DELTArad=Declinacion_solar(LATdeg, LONdeg, DOY, YEA)
    EcTmin=EoT(DOY, YEA)
    gam=Gamma(DOY, YEA)
    fn=Fn(gam)
    #229.2~60*3.8196667
    # --- Angulo horario
    Hloc = 12*(1+Wrad/np.pi) + UTC - (EcTmin+ 4 *LONdeg)/60
    #Hsol = HRAgw + MIN/60. + (EcTmin+ 4.*LONdeg)/60. 
    #Wrad = (Hsol-12)*(np.pi/12.)
    #Wrad=np.float64(Wrad)
    LATrad = np.deg2rad(LATdeg)
    CSZ = np.sin(LATrad)*np.sin(DELTArad) + np.cos(LATrad)*np.cos(DELTArad)*np.cos(Wrad)
    AZrad = acimut(CSZ, DELTArad, Wrad, LATdeg)
    return CSZ, fn, DELTArad, Hloc, EcTmin, AZrad


def Irradiancia_ET(LATdeg, LONdeg, DOY, HRA, MIN, YEA, UTC):
    [csz, Fn, DELTArad, Wrad, EcTmin ]=calcular_variables_solares(LATdeg, LONdeg, DOY, HRA, MIN, YEA, UTC)
    GHIet=csz*CS*Fn
    GHIet[GHIet<0]=0
    return GHIet

def Irradiacion_ET_horaria(LATdeg, LONdeg, YEA, DOY, HRA, tipo, UTC):
    #tipo es E60  o S60
    MIN=30
    if tipo =='S': MIN=0 # en el centro del intervalo
    [CSZ, Fn,_,_,_]=calcular_variables_solares(LATdeg, LONdeg, DOY, HRA, MIN, YEA, UTC)
    Iet=CS*Fn*CSZ
    Iet[Iet<=0]=0.0000000001
    return Iet

def Irradiacion_ET_diaria_PH(LATdeg, LONdeg, DOY, YEA):
    LATrad=np.deg2rad(LATdeg)
    fn=Fn(Gamma(DOY, YEA))
    DELTArad=Declinacion_solar(LATdeg, LONdeg, DOY, YEA) 
    W_s= np.arccos(-np.tan(DELTArad)*np.tan(LATrad)) #angulo horario amanecer (-)/atardecer(+)
    Ho=24*60*60*CS/np.pi/1e6 # en MJ/m2;
    H=Ho*fn*(np.cos(DELTArad)*np.cos(LATrad)*np.sin(W_s)+W_s*np.sin(DELTArad)*np.sin(LATrad))
    return H
    
def Irradiacion_ET_diaria_Normal(LATdeg, LONdeg, DOY, YEA):
    LATrad=np.deg2rad(LATdeg)
    fn=Fn(Gamma(DOY, YEA))
    DELTArad=Declinacion_solar(LATdeg, LONdeg, DOY, YEA) 
    W_s= np.arccos(-np.tan(DELTArad)*np.tan(LATrad)) #angulo horario amanecer (-)/atardecer(+)
    Ho=24*60*60*CS/np.pi/1e6 # en MJ/m2;
    Hn = Ho*fn*W_s 
    return Hn

def Irradiacion_ET_mensual_media(LATdeg, LONdeg, MES, YEA=2001,tipo=0):
    # solo para escalares!!!
    # H0   =    vector con las 12 radiaciones promedio por mes
    # tipo =    0: calculo     1: usando el dia medio 
    BIS=(YEA%4==0)
    if tipo==1:
        dia_medio= [ 17, 47, 75 + BIS, 105 + BIS, 135+ BIS, 162+ BIS, 
                    198+ BIS, 228+ BIS,  258+ BIS, 288+ BIS, 318+ BIS, 344+ BIS]
        H0=Irradiacion_ET_diaria(LATdeg, LONdeg, dia_medio[MES-1], YEA)
    n=TMP.dias_del_mes(MES, BIS)
    ni=TMP.mth_day_to_doy(MES, 1, BIS);
    if tipo==0:
        H=0
        for i in range(ni,ni+n):
            h=Irradiacion_ET_diaria(LATdeg, LONdeg, i, YEA)
            H=H+h
        H0=H/n
    return H0

def Masa_de_aire(CSZ,pcoef, tipo='Young'):
    # 4/10/2016
    # 25/09/2018 - version Python
    # usa como input el coseno del angulo cenital
    if tipo=='simple': m = 1./CSZ
    if tipo=='Kasten': m = pcoef / (CSZ+ 0.50572*(np.arccos(CSZ)+6.07995)**(-1.6364))
    # OBS se puede corregir los angulos por la difraccione de la atmosfera
    if tipo=='Young': m = (1.002432* CSZ**2 + 0.148386* CSZ + 0.0096467 ) / ( CSZ**3 + 0.149864* CSZ**2 + 0.0102963* CSZ+ 0.000303978) 

    #m_kasten_corr=1
    #pcoef/ (CSZcosv(COSZ)+ 0.50572*((180/pi)*...
    #    gamav(gamas(COSZ))+6.07995)**(-1.6364))
    return m


    
    
    
