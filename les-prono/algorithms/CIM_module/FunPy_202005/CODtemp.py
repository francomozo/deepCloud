# Agustin Laguarda 
# 1/02/2018
#funciones relacionadas con el manejo de las etiquetas temporales
###############################################################################
# Glosario de funciones:
#        CODtemp
#        invCODtemp (sin hacer)
#        dias del mes
#        doy_to_mth_day    
#        mth_day_to_doy      
###############################################################################
import numpy as np

# generar codigo temporal::::::::::::::::::::::::::::::::::::::::::::::::::::::
def CODtemp(YEA, DOY, HORA, MIN):
    iniYEA=2000
    N=np.size(YEA)
    codtemp=np.zeros(N)
    if N==1:
        YEA=np.array([YEA])
        DOY=np.array([DOY])
        HORA=np.array([HORA])
        MIN=np.array([MIN])
    #dias_iniYEA=(365+ iniYEA%4==0) #num de dias del anho inicial
    for j in range(N):
        if YEA[j]<2000:
            print('Cuidado! solo para fechas desde el 1/1/2000')
        for k_YEA in range(iniYEA,YEA[j]):
            codtemp[j]=codtemp[j] + 365 + (k_YEA%4==0)
#           ahora le sumo la contribucion del anho YEA
        codtemp[j]=codtemp[j]+(DOY[j]-1.)+ (HORA[j] + MIN[j]/60.) /24.;
#       el menos uno es porque 1 de enero de 2010 es el dia cero
    return codtemp
# del codigo temporal general variables temporales::::::::::::::::::::::::::::

# dias de cada mes :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def dias_del_mes(num_mes, bis):
    if num_mes<1:
        print('mes invalido ')
       # break
    a=28+bis
    D=[31,a,31,30,31,30,31,31,30,31,30,31]
    return D[num_mes-1]

# del dia y mes dar el dia del ano:::::::::::::::::::::::::::::::::::::::::::::
def doy_to_mth_day(doy, fLEAP=0):
    # Version adaptada de 1.0 30/08/2014	Rodrigo Alonso Suarez
    mth = 1
    doy_acum = dias_del_mes(mth, fLEAP)
    while doy > doy_acum:
        mth += 1
        doy_acum += dias_del_mes(mth, fLEAP) 
    day = dias_del_mes(mth, fLEAP) - (doy_acum-doy)    
    return mth, day

# del dia del mes y el mes da el doy:::::::::::::::::::::::::::::::::::::::::::
def mth_day_to_doy(month, day, bis):
#   % Version adaptada de 3/02/2014	Rodrigo Alonso Suarez
#   ojo!!!, funciona solo para inputs escalares
    n_acum = 0
    for k in range(month-1):
        n_acum = n_acum + dias_del_mes(k+1, bis) 
    doy = n_acum + day
    return doy

















