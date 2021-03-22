# Agustin Laguarda 
# 1/02/2018
#funciones relacionadas con el manejo de las etiquetas temporales
###############################################################################
# Glosario de funciones:
#        CODtemp
#        CODtemp_to_VARtemp
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
# del codigo temporal general variables temporales:::::::::::::::::::::::::::::
#---version 1.0: 27/03/2015
#---version 1.1: 3/05/2016 se corrige un '+1' en la condicion del if y un '='
#                 para un bug para codemp en el iniYEA
#---version python 2/2/2018
def CODtemp_to_VARtemp(CODtemp): 
    iniYEA=2000
    years=np.array(range(iniYEA,2050))
    dias=np.ones(len(years))*365    #son todos 365
    dias=dias+((years%4)==0)        # es un vector con 365 y 366
    dias_iniYEA=dias[0]             # num de dias del anho inicial
    diasREF=np.cumsum(dias)         #es el acumulado de los dias
    N=np.size(CODtemp)
    YEA=np.zeros(N,dtype=int);DOY=np.zeros(N, dtype=int);HOR=np.zeros(N,dtype=int);MIN=np.zeros(N,dtype=int)
    for i in range(N):
        try: codtemp=CODtemp[i]
        except TypeError: codtemp=CODtemp
        if (codtemp)< dias_iniYEA:
            YEA=iniYEA
            DOY=np.int(codtemp+1)
        else:
            aux=np.max(diasREF[(diasREF-codtemp)<=0]) # es el numero de dias hasta que 
            #comienza el anho buscado
            #print('aux')
            index=np.where(diasREF==aux) #hallo la cantidad de dias que tiene el ano 
            index=np.int(index[0])        
            #al que pertenece codtemp
            YEA[i]=np.int(years[index+1])
            DOY[i]=np.int(codtemp-diasREF[index])+1
            HORA_dec=codtemp-int(codtemp)
            HORA_sex=HORA_dec*24
            HOR[i]=np.int(np.round(HORA_sex,0))
            MIN[i]= np.int(np.round((HORA_sex-HOR[i])*60,0))
    return YEA, DOY, HOR, MIN

# dias de cada mes ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def dias_del_mes(num_mes, bis):
    if num_mes<1:
        print('mes invalido ')
       # break
    a=28+bis
    D=[31,a,31,30,31,30,31,31,30,31,30,31]
    return D[num_mes-1]

# del dia y mes dar el dia del ano:::::::::::::::::::::::::::::::::::::::::::::
def doy_to_mth_day(doy, fLEAP):
    # Version adaptada de 1.0 30/08/2014	Rodrigo Alonso Suarez
    mth = 1
    doy_acum = dias_del_mes(mth, fLEAP)
    while doy > doy_acum:
        mth = mth + 1
        doy_acum = doy_acum + dias_del_mes(mth, fLEAP) 
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

def convert_to_UTC0(YEA, DOY, HRA, MIN, TZ):
    #recibe en UTC+TZ (-3 para uruguay) y devuelve en UTC0
    TEMPtz=CODtemp(YEA, DOY, HRA, MIN)
    TEMP0=TEMPtz-TZ/24
    [YEA0, DOY0, HRA0, MIN0]=CODtemp_to_VARtemp(TEMP0) 
    return YEA0, DOY0, HRA0, MIN0

def convert_to_UTCtz(YEA0, DOY0, HRA0, MIN0, TZ):
    #recibe en UTC0 y devuelve en UTC-TZ (-3 para uruguay)
    TEMP0=CODtemp(YEA0, DOY0, HRA0, MIN0)
    TEMPtz=TEMP0+ TZ/24
    [YEAtz, DOYtz, HRAtz, MINtz]=CODtemp_to_VARtemp(TEMPtz) 
    return YEAtz, DOYtz, HRAtz, MINtz

def es_bisiesto(yea):
    p = np.mod(yea, 4)==0
    q = np.mod(yea, 100)==0
    r = np.mod(yea, 400)==0
    leap = p
    try:
        leap[q==True]=False
        leap[r==True]=True
    except TypeError: leap= p and (q==False or r)
    return leap












