#---Clasificacion de horas claras segun criterios (Remund 2003)
#---Agustin Laguarda
#---29/06/2015 (version generar_bandera_hora_clara.m)
#---20/06/2016 version en forma de funcion 'standalone'
#---7/07/2017 version generica. No importa datos, los usa de input
#---10/04/2018   version 3.0. SOlo usa datos horarios (no incluye D01)
#                se enfoca en elegir horas de dias claros (~completos)
#---25/09/2018 - version para Python
import numpy as np
#from variables_temporales import es_bisiesto
from variables_solares import Irradiacion_ET_horaria, Masa_de_aire
from Fraccion_difusa import Fraccion_Difusa_Horaria_Erbs

def Generar_bandera_GHI_cielo_claro_horaria(ANO, DOY, I, CSZ, kt, ktp):
    # 15/08/2019
    # msk1 : kt perez
    # msk2 : coseno
    # msk3 : 40% de las horas validas de un dia pasan 1 y 2
    # msk4 : la varaibilidad de ktp de un dia es menor a umbral
    
    n=len(ANO)
    #  criterio 2: kt corregida mayor a 0.7 (Molineaux 1995) 
    ktp_min=0.74; ktp_max=1
    msk_1 = (ktp >= ktp_min) & (ktp <= ktp_max)
    
    msk_2= (CSZ > np.sin(7*np.pi/180))#altura solar superior a 7 grados
    
    MSK_A = msk_1 & msk_2  # msk total de los filtros tipo A
    
    #  B: Filtros que dependen de los datos dentro del dia
    msk_3 = np.zeros(n)==1
    # recorro los dias
    msk_dia=(CSZ>0) # horas diurnas
    VAR_ktp = np.zeros(n)*np.nan
    #rVAR_ktp = np.zeros(n)*np.nan
    for kYEA in np.arange(ANO[0], ANO[-1]+1): # recorro los anhos
        for kDOY in np.arange(1,366+1):  #recorro los dias del ano
            msk_tmp = (ANO==kYEA) & (kDOY==DOY)
            #criterio 4: por lo menos el 40% de las horas de un dia tienen que pasar
            #       el filro A
            N_todos = np.sum(msk_tmp)  #nro de datos de ese dia
            I_tmp_tot = I[msk_tmp & msk_dia] #datos diurnos de ese dia 
            I_tmp_fil = I[msk_tmp & MSK_A] # horas claras de ese dia
            N_datos=len(I_tmp_tot)
            N_datos_claros=len(I_tmp_fil)
            try: coef=N_datos_claros/N_datos
            except ZeroDivisionError:coef=0
      
            if coef >= 0.4: msk_3[msk_tmp]= (np.ones(N_todos)==1)
        # criterio 4: la variabilidad de Kt horaria en un dia es menor a un
        #            umbral
            ktp_tmp = ktp[(msk_tmp) & (CSZ>0.12)] # son los ktp validos de ese dia
            var_tmp = np.std(ktp_tmp)# /np.mean(ktp_tmp)
            VAR_ktp[msk_tmp]=np.ones(N_todos)*var_tmp/np.mean(ktp_tmp)#en terminos relativos
            #rVAR_ktp[msk_tmp]= VAR_ktp[msk_tmp]/np.mean(ktp_tmp)
    msk_4=( VAR_ktp < 0.1)  #umbral puesto A OJO!!
        
    MSK_final = msk_1 & msk_2 & msk_3 & msk_4
    return MSK_final, msk_1, msk_2, msk_3, msk_4, VAR_ktp#, rVAR_ktp


# inputs nivel horario
#  ANO , DOY, HOR, I (irradiancia promedio, en W/m2), misma longitud
# tipo: S (mode satelite;alrededor de la hora puntual)
#       E (estandar     ;alrededor de h:30)

def Generar_bandera_cielo_claro_horaria(ANO,DOY,HOR, I,lat, lon, alt, CSZ,tipo, UTC):
    # defino variables secundarias.............................................
    n=len(ANO)
    LAT=lat*np.ones(n)                  # vector de latitudes para los horarios
    LON=lon*np.ones(n)                  # %vector de longitudes
    pcoef=np.exp(-alt/8435.2)*np.ones(n)   #cociente entre presiones
    #BIS=es_bisiesto(ANO)#FUNCIONA MAL
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #---indice de claridad
    I_ET=Irradiacion_ET_horaria(LAT, LON, ANO, DOY, HOR, tipo, UTC)
                                
    kt=I/I_ET
    #---indice de cielo claro para kt corregido (Pedros et. al., 1999)
    m = Masa_de_aire(CSZ, pcoef) 
    print(type(m))
    ktp = kt/ (1.031*np.exp(-1.4/(0.9 + 9.4/m))+0.1)
    #---componente directa
    fd = Fraccion_Difusa_Horaria_Erbs(kt) # modelo Erbs, es solo para 
    B=(1-fd)*I  #radiacion directa
    msk_dia=(CSZ>0)
    Bn=np.zeros(n) 
    Bn[msk_dia]=B[msk_dia]/CSZ[msk_dia]
    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #---2. definir los cuatro criterios para cada hora
    #   A: Filtros que dependen del dato horario
    #  criterio 1: componente NORMAL directa mayor a 200
    Bmin=200
    msk_1=Bn>=Bmin
    #  criterio 2: kt corregida mayor a 0.7 (Molineaux 1995) 
    ktp_min=0.7; ktp_max=1.5
    msk_2 = (ktp >= ktp_min) * (ktp <= ktp_max)
    #  criterio 3: el dia tiene que tener K>0.50 en el total
    #%Kmin=0.40;
    #%mskd=(KT>=Kmin);
    #%DOYsi=DOYd(mskd);ANOsi=ANOd(mskd);%son las fechas que pasan el filtro 3
    #
    msk_3= (np.ones(n)==1)
    #%for i = 1: length(DOYsi)
    #%    msk_tmp=(DOY==DOYsi(i))&(ANO==ANOsi(i));
    #%    msk_3=msk_3 | msk_tmp; %voy construyendo la mascara
    #%    clear msk_tmp
    #%end
    
    msk_5= (CSZ >np.sin(7*np.pi/180))#altura solar superior a 7 grados
    MSK_A = msk_1 & msk_2 & msk_3 & msk_5  # msk total de los filtros tipo A
    #  B: Filtros que dependen de los datos dentro del dia
    msk_4=np.zeros(n)==1
    # recorro los dias
    VAR_ktp=np.zeros(n)*np.nan
    for kYEA in np.arange(ANO[1], ANO[-1]+1): # recorro los anhos
        for kDOY in np.arange(1,366+1):  #recorro los dias del ano
            msk_tmp = (ANO==kYEA) & (kDOY==DOY)
            #criterio 4: por lo menos el 40% de las horas de un dia tienen que pasar
            #       el filro A
            N_todos = np.sum(msk_tmp)  #nro de datos de ese dia
#            print(N_todos)
#            print(msk_tmp.dtype)
#            print(MSK_A.dtype)
#            print(msk_dia.dtype)
            I_tmp_tot = I[msk_tmp & msk_dia] #datos diurnos de ese dia 
            I_tmp_fil = I[msk_tmp & MSK_A] # horas claras de ese dia
        
            N_datos=len(I_tmp_tot)
            N_datos_claros=len(I_tmp_fil)
            try: coef=N_datos_claros/N_datos
            except ZeroDivisionError:coef=0
            
            if coef >= 0.4: msk_4[msk_tmp]= (np.ones(N_todos)==1)
        # criterio 6: la variabilidad de Kt horaria en un dia es menor a un
        #            umbral
            ktp_tmp=ktp[msk_tmp * CSZ>0] # son los ktp validos de ese dia
            var_tmp=np.std(ktp_tmp)
            VAR_ktp[msk_tmp]=np.ones(N_todos)*var_tmp
    msk_6=VAR_ktp<0.15  #umbral puesto A OJO!!
        
    MSK_final=msk_1 & msk_2 & msk_3 & msk_4 & msk_5 & msk_6
    
    return MSK_final, msk_1, msk_2, msk_3, msk_4, msk_5, msk_6, ktp, kt, VAR_ktp, I_ET


