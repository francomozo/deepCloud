
from numpy import exp, cos, sin, arcsin, arccos, pi, log, copy, size


###############################################################################################################
#   ESRA 
###############################################################################################################

# modelo de cielo claro ESRA
# basado en Rigollier 2000

#---Agustin Laguarda
# 6/07/2015
# version 26/01/2017 admite vectores como entrada
# version 03/04/2019 para Python

#############################################################################################

#---variables:
# I0=1367;  en W/m2, cte solar
# CSZ = coseno del angulo cenital
# TL  = coef de turbidez de Linke
# Fn  = correccion de la distancia Tierra-Sol (poner = 1 por defecto), 
#   por la excentricidad de la órbita terrestre
# pcoef = coeficiente entre la presión y la standar p/po, o exp(-z/zo), 
    #donde zo = 8434.5 msnm.

def ESRA(CSZ, Fn, TL, Pcoef): #function [GHI_ESRA, DNI, DHI]= ESRA(COSZ, doy, TL, pcoef, bis)
    
    # defino cantidades auxiliares:
    I0=1367  #W/m2
    #CORRECCION DISTANCIA TIERRA-SOL (Fn)
    #N=365+bis;
    #GAM=2*pi*(doy-1)./N ;
    #Fn = 1.000110 + 0.034221*cos(GAM) + 0.001280*sin(GAM) + 0.000719*cos(2*GAM) + 0.000077*sin(2*GAM);
    
    #   MASA DE AIRE OPTICA RELATIVA
    
    # altura solar en radianes
    gama = arcsin(CSZ)
    dgama= 0.061359*(0.1594 + 1.1230*gama+0.065656*gama**2)/(1+28.9344 * gama + 277.3971*gama**2)
    gama_corr = gama+dgama
    # defino el coseno del angulo corregido
    CSZ_corr = sin(gama_corr);
    
    # correcion del angulo (en radianes)

    # defino m:
    #(gama en radianes)
    M=Pcoef/ (CSZ_corr+ 0.50572*((180/pi)*arcsin(CSZ_corr)+6.07995)**-1.6364)

    #---ESPESOR OPTICO Rayleigh (en funcion de la masa de aire)
    # (m<20)
    inv_delta  = 6.62960 + 1.75130*M - 0.12020*M**2 + 0.00650*M**3 -0.00013*M**4
    inv_delta2 = 10.4+0.718*M  #obs: discrimina entre gamma mayor a 1.9 grados centigrados o menor.(M=20)
    msk_delta = (M >= 20) 
    inv_delta[msk_delta]=inv_delta2[msk_delta]
    deltaR = 1/inv_delta

    #   IRRADIANCIA DIRECTA B
    DNI= I0*Fn*exp(-0.8662*TL*M*deltaR)
    DNI[DNI<0]=0 #cereo por si acaso

    #   IRRADIACION DIFUSA
        
    #funcion de transmision difusa en el zenit (gama=90 grados)
    Trd = -1.5843e-2 + 3.0543e-2*TL+ 3.797e-4*TL**2

    #dfuncion angular difusa Fd, para eso defino A0, A1 y A2 (dependen de TL)
    
    A0 = 2.6463e-1 - 6.1581e-2*TL + 3.1408e-3*TL**2
    msk_A0 = (A0*Trd)<2e-3
    A0[msk_A0]= 2e-3/Trd[msk_A0]
    
    A1 = 2.0402 + 1.8945e-2*TL - 1.1161e-2*TL**2
    A2 =-1.3025 + 3.9231e-2*TL + 8.5079e-3*TL**2

    Fd = A0 + A1*CSZ + A2*CSZ**2
    
    # irradiacion difusa:
    DHI =I0*Fn*Trd*Fd
    DHI[DHI<0]=0  # cereo por si acaso
    
    #IRRADIANCIA TOTAL: 
    #if (COSZ>0) cerear
    GHI = DHI + CSZ*DNI
    
    return GHI, DNI, DHI


###############################################################################################################
#   KASTEN 
###############################################################################################################
# Modelo de Kansten-Young, mejorado por Inechien y perez
# "A New Airmass Independent Formulation For THe Linke Turbidity Coefficient" 2002. P.Inechien, R.Perez
# Agustin Laguarda
# 26/01/2015
# version 29/03/2016 (con la componente directa)
# version 26/01/2017 admite entradas vectoriales, permite elegir entre corregir o no el angulo por difraccion
# version 10/04/2019 para python
# variables:
# CSZ      = coseno del angulo cenital
# z        = altura sobre el nivel del mar (en metros)
# TL       = coeficiente de turbidez de Linke
# corr     = 0, 1 (sin/con correccion del angulo por difraccion)

# OBS: 1- ver de agregar variacion distancia tierra sol
# OBS: 2- ver la correccion del angulo por difraccion

def KASTEN(CSZ, alt, Fn, TL, pcoef, corr): #[GHI, BNI, M, Fn]   antes: KASTEN(CSZ, alt, TL, pcoef, DOY, YEA, corr)
    #   definiciones usuales
    I0 = 1367. # W/m2
    #   correccion distancia Tierra-Sol
    # N = 365+mod(YEA,4)
    # GAM = 2*pi*(DOY-1)/N
    # Fn = 1.000110 + 0.034221*cos(GAM) + 0.001280*sin(GAM) + 0.000719*cos(2*GAM) + 0.000077*sin(2*GAM)
    
    fh1 = exp(-alt / 8000.)
    fh2 = exp(-alt / 1250.)
    a1 = 5.09e-5*alt + 0.8680
    a2 = 3.92e-5*alt + 0.0387
       
    b = 0.664+0.163/fh1
    
    #   defino la masa de aire corregida segun Kasten y Young (Paper ESRA)::::::::::::::::::::::::::::::
    #   altura solar en radianes
    gama = arcsin(CSZ)
    dgama= 0.061359*(0.1594+1.1230*gama+0.065656*gama**2)/(1+28.9344*gama+277.3971*gama**2)
    gama_corr = gama + dgama
    #   defino el coseno del angulo corregido
    CSZ_corr = sin(gama_corr)
    
    if corr==1: CSZ=CSZ_corr
    
    M = pcoef/( CSZ + 0.50572*( arcsin(CSZ)*180/pi + 6.07995)**-1.6364)
    GHI=a1*I0*Fn*CSZ*exp(-a2*M*( fh1+fh2*(TL-1)))
    BNI=b*I0*Fn*exp(-0.09*M*(TL-1))
    
    
    #if TL<2: print('implementar correccion para TL<2, Kasten mejorado')
    GHI[GHI<0]=0
    BNI[BNI<0]=0
    
    DHI = GHI - BNI*CSZ
    DHI[DHI<0]=0
    return GHI, BNI, DHI# M, Fn

###############################################################################################################
#   sSOLIS 
###############################################################################################################

#MODELO SOLIS SIMPLIFICADO. Ineichen.
# 18/12/2014
# 10/04/2019 version para pythobn

#VARIABLES:
#aod700  =    profundidad optica de aerosoles a 700 nm ()
#Igh0    =    Irradiancia extraterrestre
#CSZA    =    coseno del angulo cenital
#pcoef   =    presion atmosférica en Pascales /p0 (p0=1013e3)
#w       =    vapor de columna de agua (en cm)

#rango optimo: altura snm <7000
#              0< aod700  < 0.45
#              0.2 cm < w < 10 cm

def SOLIS_simple(CSZ, aod700, pcoef, w): #[Igh, Idh, Ibn]

    I0 = 1367.
    Igh0 = I0   
    h = arcsin(CSZ)   # h es la altura solar en radianes?

    #   intensidad extraterrestre aumentada Io
    Io0 = 1.08*w**0.0051
    Io1 = 0.97*w**0.0320
    Io2 = 0.12*w**0.5600

    Io = Igh0*(Io2*aod700**2 + Io1*aod700 + Io0 + 0.071*log(pcoef))

    # componente directa
    #   calculo tau_b y b
    tb1= 1.82+0.056*log(w)+0.0071*log(w)**2
    tb0= 0.33+0.045*log(w)+0.0096*log(w)**2
    tbp= 0.0089*w +0.13

    b1 = 0.00925* aod700**2+ 0.0148* aod700- 0.0172
    b0 =-0.75650* aod700**2+ 0.5057* aod700+ 0.4557

    tau_b = tb1*aod700 + tb0 + tbp*log(pcoef)
    b = b1*log(w) + b0

    # radiacion global
    #	calculo tau_g y g:
    tg1 = 1.24+ 0.047* log(w)+ 0.0061 * log(w)**2
    tg0 = 0.27+ 0.043* log(w)+ 0.0090 * log(w)**2
    tgp = 0.0079 * w +0.1

    tau_g =  tg1 * aod700 + tg0+ tgp* log(pcoef)
    g =-0.0147 *log(w)-0.3079* aod700**2 + 0.2846 * aod700 +0.3798

    # radiacion difusa
    #    calculo tau_d:
    #	cosas cambiadas
    #---(aod700<0.05)
    td4 = 86.0 * w - 13800.
    td3 =-3.110* w + 79.4
    td2 =-0.230* w + 74.8
    td1 = 0.092* w - 8.86
    td0 = 0.0042*w + 3.12
    tdp =-0.83 *(1+aod700)**-17.2
    #---(aod700 >= 0.05)
    msk1 = [aod700 >=0.05]
    td4[msk1] =-0.210 * w[msk1] + 11.6
    td3[msk1] = 0.270 * w[msk1] - 20.7
    td2[msk1] =-0.134 * w[msk1] + 15.5
    td1[msk1] = 0.0554* w[msk1] - 5.71
    td0[msk1] = 0.0057* w[msk1] + 2.94
    tdp[msk1] =-0.710 * (1 + aod700[msk1])**-15.0

    dp = 1/(18.+152.*aod700)

    tau_d = td4 * aod700**4 + td3*aod700**3+ td2*aod700**2 + td1*aod700 + td0 + tdp*log(pcoef)
    d=-0.337*aod700**2 + 0.63*aod700 + 0.116 + dp*log(pcoef)

    #    calculo las salidas: Ibn, Igh, Idh
    Ibn = Io*exp(-tau_b/(sin(h)**b) )
    Igh = Io*exp(-tau_g/(sin(h)**g) )*sin(h)
    Idh = Io*exp(-tau_d/(sin(h)**d) )

    return Igh, Ibn, Idh 


###############################################################################################################
#   REST2 
###############################################################################################################

# inspirado en forma y nomenclatura en Gueymard 2008
# version 2.0 13/08/2018

#---variables:
# z                  =  angulo cenital en grados (array)
# Pcoef              =  presion hP/1013.25 ()
# u0, un             =  ozono y dioxido nitrogeno en atm-cm
# w                  =  vapor de agua precipitable en cm
# alfa1, alfa2, beta =  coeficientes de turbidez de Armstrong
# rog1, rog2         =  albedo terrestre para bandas 1 y 2


## Calculation of optical mass as per SMARTS.  character string
## indicating the process (between 'Ray' (Rayleigh dispersion), 'O3':
## absorption of ozone, 'NO2' absoption of NO2, 'Mix': absorption of
## gases uniformly mixed (02 and C02), Wva: absorption of water vapor,
## 'Aer' dispersion of aerosols, 'Kas': formula of Kaster for air
## optical mass

def OpticalMass(z, proc): #z angulo cenital en GRADOS, proc-> proceso ('Ray', 'O3', 'NO2', 'Mix', 'WV', 'Aer', 'Kas')
    PAR = {'Ray': [4.5665e-1, 0.07, 96.4836, -1.697],\
           'O3' : [2.6845e2, 0.5, 115.42, -3.2922],\
           'NO2': [6.023e2, 0.5, 117.96, -3.4536],\
           'Mix': [4.5665e-1, 0.07, 96.4836, -1.697],\
           'Wva': [0.10648, 0.11423, 93.781, -1.9203],\
           'Aer': [0.16851, 0.18198, 95.318, -1.9542],\
           'Kas': [0.50572, 0, 96.07995, -1.6364]}
    # Choose the adequate variable
    A= PAR[proc]
    cos_z = cos( z*pi/180)
    m = (cos_z + A[0]*z**A[1]*(A[2] - z)**A[3])**-1
    return m


def REST2(z, Pcoef, beta, alfa1, alfa2, uo, un, w, rog1, rog2):
    n=size(z)
    cos_z = cos (z*pi/180)
    # defino constantes
    E0n1 = 635.4 #w/m2 irradiancia normal total en la banda 1
    E0n2 = 709.7 #w/m2                                banda 2

    omega1= 0.95#0.95#0.95  # Tomados de antonanzas.      coeficientes experimentales recomendados para la difusa ideal
    omega2= 0.90
    
    #---defino las cuatro masas de aire: mR, mo, mw, y ma, que dependen de z (Gueymard 2003 parte 1)
    mR = OpticalMass(z, 'Ray')
    mo = OpticalMass(z, 'O3')
    mw = OpticalMass(z, 'Wva')
    ma = OpticalMass(z, 'Aer')    
    mp = mR* Pcoef
    
    #mR = 1/( cos_z+ 0.48353*z**0.095846/(96.741-z)**1.754 )    
    #mo = 1/( cos_z+ 1.06510*z**0.6379  /(101.80-z)**2.2694 )    
    #mw = 1/( cos_z+ 0.10648*z**0.11423 /(93.781-z)**1.9203 )    
    #ma = 1/( cos_z+ 0.16851*z**0.18198 /(95.318-z)**1.9542 )    
    #defino la masa de aire corregida por la presion de Rayleigh (m prima)
  
    # ---otras cantidades utiles
    # coeficientes de turbidez de Armstrong------------------------------------
    beta1 = copy(beta)*0.7**(alfa1-alfa2)
    beta2 = copy(beta)
    
    #--- ua, que depende de otros parametros:
    ua1 = log(1 + ma * beta1)
    #---longitud de onda equivalente (lambda eq = lae) para cada banda------------------
    d0 = 0.57664- 0.024743*alfa1
    d1 = ( 0.093942- 0.2269* alfa1+ 0.12848*alfa1**2)/(1+ 0.6418 *alfa1)
    d2 = (-0.093819+ 0.36668*alfa1- 0.12775*alfa1**2)/(1- 0.11651*alfa1)
    d3 = alfa1*(0.15232- 0.087214*alfa1+ 0.012664*alfa1**2)/(1- 0.90454*alfa1+ 0.26167*alfa1**2)
    ##
    a1Low = (alfa1 <= 1.3)
    d0[a1Low] = 0.544474
    d1[a1Low] = 0.00877874
    d2[a1Low] = 0.196771
    d3[a1Low] = 0.294559
    ##
    lae1 = (d0+ d1*ua1+ d2*ua1**2)/(1+ d3*ua1**2)
    lae1[lae1 < 0.3] = 0.3
    lae1[lae1 > 0.65]= 0.65
    
    
    ua2 = log(1 + ma* beta2)# esta ecuacione es dudosa, no se si usar beta1 o beta2
    e0 = (1.183 - 0.022989* alfa2+ 0.020829* alfa2**2)/(1 +0.11133*alfa2)
    e1 = (-0.50003- 0.18329*alfa2+ 0.23835 * alfa2**2)/(1 + 1.6756*alfa2)
    e2 = (-0.50001+ 1.1414*alfa2+ 0.0083589* alfa2**2)/(1 + 11.168*alfa2)
    e3 = (-0.70003- 0.73587 *alfa2+ 0.51509* alfa2**2)/(1 + 4.7665*alfa2)
    
    ##  Cambiado segun codigo fuente fortran version REST2 8.3
    a2Low = (alfa2 <= 1.3)
    e0[a2Low] = 1.038076
    e1[a2Low] =-0.105559
    e2[a2Low] = 0.0643067
    e3[a2Low] =-0.109243
    ##
    
    lae2 = (e0+ e1*ua2+ e2*ua2**2)/(1+ e3*ua2)
    lae2[lae2 < 0.75] = 0.75
    lae2[lae2 > 1.5] = 1.5
    
    
    #---profundidad optica de aerosol equivalente (tau sub a = ta) para cada banda
    ta1 = beta1*lae1**(-alfa1)
    ta2 = beta2*lae2**(-alfa2)        
    ###########################################################
    #---defino las trasmitancias.l subindice 1 o 2 refiere a la banda (1: 0.29 a 0.7 micrometros, 2: 0.7 a 4 micras)
    #TR     scattering de Rayleigh--------------------------------------------------------------
    TR1 = (1+ 1.8169* mp -0.033454* mp**2)/(1+ 2.063*mp+ 0.31978*mp**2)
    TR2 = (1- 0.010394*mp)/(1- 0.00011042*mp**2)
    #Tg     absorcion por gases uniformemente mezclados---------------------------------
    Tg1 = (1+ 0.95885*mp+ 0.012871 * mp**2)/(1+ 0.96321*mp+ 0.015455*mp**2)
    Tg2 = (1+ 0.27284*mp- 0.00063699*mp**2)/(1+ 0.30306*mp)
    #To    absorcion por ozono----------------------------------------------------------
    f1=uo*( 10.979- 8.5421 *uo) / (1 + 2.0115*uo+ 40.189*uo**2)    
    f2=uo*(-0.027589- 0.005138*uo)/(1- 2.4857*uo+ 13.942*uo**2)
    f3=uo*( 10.995- 5.5001 *uo) / (1 + 1.6784*uo+ 42.406*uo**2)
    To1=(1+f1*mo+f2*mo**2)/(1+f3*mo)
    To2 = 1
    #Tn   absorcion por dioxido de nitrogeno ----------------------------------------------
    g1 = (0.17499 + 41.654*un -2146.4*un**2)/(1+ 22295.0*un**2)    
    g2 = un*(-1.2134 + 59.324*un)/(1+ 8847.8*un**2)
    g3 = (0.17499 +61.658*un +9196.4*un**2)/(1 +74109.0*un**2)
    Tn1 = (1+ g1*mw+ g2*mw**2)/(1+ g3*mw)
    if n > 1: Tn1[Tn1>1] = 1 #maximo =1
    if n == 1: Tn1=min(1,Tn1)
        
    Tn2 = 1
    #Tw  absorcion por vapor de agua------------------------------------------------------
    h1 = w*(0.065445+ 0.00029901*w)/(1+ 1.2728*w)
    h2 = w*(0.065687+ 0.0013218* w)/(1+ 1.2008*w)
    Tw1=(1+ h1*mw)/(1+ h2*mw)
        
    c1 = w*(19.566 - 1.6506 *w+ 1.0672 * w**2)/(1+ 5.4248* w +1.6005* w**2)
    c2 = w*(0.50158- 0.14732*w+ 0.047584*w**2)/(1+ 1.1811* w +1.0699* w**2)
    c3 = w*(21.286 - 0.39232*w+ 1.2692 * w**2)/(1+ 4.8318* w +1.412 * w**2)    
    c4 = w*(0.70992- 0.23155*w+ 0.096514*w**2)/(1+ 0.44907*w +0.75425*w**2)
    Tw2= (1+ c1*mw+ c2*mw**2)/(1+ c3*mw+ c4*mw**2)
    #Ta  extincion por aerosoles-----------------------------------------------
    Ta1 = exp(-ma*ta1)
    Ta2 = exp(-ma*ta2)
    
    #calculo la radiacion normal directa en cada banda-------------------------
    E0n1 = 635.4
    Ebn1 = TR1*Tg1*To1*Tn1*Tw1*Ta1*E0n1
    
    E0n2 = 709.7
    Ebn2 = TR2*Tg2*To2*Tn2*Tw2*Ta2*E0n2
    #radiacion total normal directa: 
    Ebn = Ebn1 + Ebn2
    ###########################################################################    
    #calculo la irradiancia difusa "ideal", para eso hallo mas factores
    #---forward scattering de Rayleig para ambas bandas
    BR1 = 0.5*(0.89013 - 0.0049558*mR + 0.000045721*mR**2)
    BR2 = 0.5
    #---forward scattering de aerosol
    Ba = 1-exp(-0.6931- 1.8326*cos_z)
    #---factores de correccionpara ambas bandas:
    g0 = ( 3.715 + 0.368 * ma + 0.036294*ma**2)/(1 +0.0009391*ma**2)
    g1 = (-0.164 -0.72567* ma + 0.20701* ma**2)/(1 +0.0019012*ma**2)
    g2 = (-0.052288 +0.31902*ma +0.17871*ma**2)/(1 +0.0069592*ma**2)
    F1 = (g0+g1*ta1)/(1+g2*ta1)
 
    h0 = (3.4352+ 0.65267*ma +0.00034328*ma**2)/(1 +0.034388*ma**(1.5))
    h1 = (1.231 - 1.63853* ma + 0.20667 *ma**2 )/(1 +0.1451 *ma**(1.5))
    h2 = (0.8889- 0.55063*ma + 0.50152 * ma**2 )/(1 +0.14865*ma**(1.5))
    F2 = (h0+ h1*ta2)/(1+ h2*ta2)
    
    Tas1 = exp(-ma*ta1*omega1) #omega1=0.92
    Tas2 = exp(-ma*ta2*omega2) #omega2=0.84
    
    #---defino Tw prima y Tn prima, es decir con m=1.66
    #((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((
    #es como Tw, pero con masa de aire m impuesta m=1.66 
    m = 1.66 + mR*0 #para que tenga iguales dims que mR
    h1 = w*(0.065445+ 0.00029901*w)/(1+ 1.2728*w)
    h2 = w*(0.065687+ 0.0013218 *w)/(1+ 1.2008*w)
    Twp1 = (1+ h1*m)/(1+ h2*m)
    
    c1 = w*(19.566 - 1.6506 *w+ 1.0672 * w**2)/(1+ 5.4248*w + 1.6005* w**2)
    c2 = w*(0.50158- 0.14732*w+ 0.047584*w**2)/(1+ 1.1811*w + 1.0699* w**2)
    c3 = w*(21.286 - 0.39232*w+ 1.2692 * w**2)/(1+ 4.8318*w + 1.412 * w**2)    
    c4 = w*(0.70992- 0.23155*w+ 0.096514*w**2)/(1+ 0.44907*w+ 0.75425*w**2)
    Twp2 = (1+c1*m+c2*m**2)/(1+c3*m+c4*m**2)
    
    g1 = (0.17499 + 41.654*un -2146.4*un**2)/(1+ 22295.0*un**2)    
    g2 = un*(-1.2134 + 59.324*un)/(1+ 8847.8*un**2)
    g3 = (0.17499+ 61.658*un+ 9196.4*un**2)/(1+ 74109.0*un**2)
    Tnp1 = (1+ g1*m+ g2*m**2)/(1+ g3*m) #; Tnp1[Tnp1>1]=1 #maximo 1
    if n > 1: Tnp1[Tnp1>1] = 1 #maximo =1
    if n == 1: Tnp1=min(1,Tnp1)
    
    Tnp2 = 1
    #))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

    #--- difusa en superficie perfecta (cero albedo) para cada banda#####
    Edp1 = To1*Tg1*Tnp1*Twp1*( BR1*(1-TR1)*Ta1**0.25 + Ba*F1*TR1*(1-Tas1**0.25)) *E0n1*cos_z
    Edp2 = To2*Tg2*Tnp2*Twp2*( BR2*(1-TR2)*Ta2**0.25 + Ba*F2*TR2*(1-Tas2**0.25)) *E0n2*cos_z
    ###########################################################################
    #---calculo el backscattering.
    #antes calculo el albedo del cielo (ro sub s = rog) para cada banda
    ros1 = (0.13363 + 0.00077358*alfa1 +beta1*(0.37567 +0.22946 *alfa1)/(1- 0.10832*alfa1))\
        /(1 + beta1*(0.84057+ 0.68683*alfa1)/(1 -0.08158*alfa1))
    ros2 = (0.010191+ 0.00085547*alfa2 +beta2*(0.14618 +0.062758*alfa2)/(1- 0.19402*alfa2))\
        /(1 + beta2*(0.58101+ 0.17426*alfa2)/(1 -0.17586*alfa2))
    #backscattering############################################################
    Edd1 = rog1*ros1*( Ebn1*cos_z + Edp1)/(1-rog1*ros1)
    Edd2 = rog2*ros2*( Ebn2*cos_z + Edp2)/(1-rog2*ros2)
    ###########################################################################
    #---radiacion difusa total
    Ed1= Edp1+ Edd1
    Ed2= Edp2+ Edd2
    Ed = Ed1 + Ed2
    #---radiacion global
    Ed[Ed<0]=0
    Ebn[Ebn<0]=0
    
    Eg = Ebn *cos_z + Ed
    Eg[Eg<0]=0
    ###########################################################################
    # ILUMINANCIA
    Eb1=Ebn1*cos_z
    Eg1=Eb1+Ed1
    beta_e = beta1*lae1**(1.3-alfa1)
    
    if n > 1 : m_11=copy(mR); m_11[m_11>11] = 11 #maximo =11
    if n == 1: m_11=min(11,mR)
    
    #np.min(m_R,11)
    r0 = 0.21437 +  0.021878 *m_11 -0.0037737*m_11**2+ 0.00032857*m_11**3- 2.0789e-5 *m_11**4+ 6.7972e-7 *m_11**5
    r1 = 0.0040867+ 0.031571 *m_11 +0.0037634*m_11**2+ 0.003198  *m_11**3+ 5.6847e-4 *m_11**4- 2.7302e-5 *m_11**5
    r2 =-0.030167 + 0.013214 *m_11 -0.02685  *m_11**2+ 0.0076755 *m_11**3- 9.3458e-4 *m_11**4+ 3.6227e-5 *m_11**5
    r3 = 0.67565 -  1.3181   *m_11 +0.87706  *m_11**2- 0.1964    *m_11**3+ 0.022028  *m_11**4- 0.000846  *m_11**5
    s0 = 0.21317 +  0.010589 *m_11 -0.0033043*m_11**2+ 0.00041787*m_11**3- 2.7531e-5 *m_11**4+ 7.8175e-7 *m_11**5
    s1 =-0.19312 +  0.16898  *m_11 -0.072244 *m_11**2+ 0.013549  *m_11**3- 9.2559e-4 *m_11**4+ 2.1105e-5 *m_11**5
    s2 = 0.034794-  0.05233  *m_11 +0.023064 *m_11**2- 0.0046273 *m_11**3+ 3.151e-4  *m_11**4- 6.9504e-6 *m_11**5
    s3 =-0.81119 +  0.64533  *m_11 -0.2673   *m_11**2+ 0.048401  *m_11**3- 0.0032342 *m_11**4+ 7.2347e-5 *m_11**5
    
    Kb = ( r0 + r1* beta_e + r2 * beta_e**2 ) / ( 1 + r3* beta_e**2)
    Kg = ( s0 + s1* beta_e + s2 * beta_e**2 ) / ( 1 + s3* beta_e )
    
    Lg = Eg1 * Kg    
    Lb = Eb1 * Kb    
    
    ###########################################################################
    # RADIACIÓN PAR
    #m_15 1⁄4 minðm R ; 15Þ
    if n > 1 : m_15=copy(mR); m_15[m_15>15] = 15 #maximo =15
    if n == 1: m_15=min(15,mR)
     
    t0 = ( 0.90227 + 0.29000*m_15 +0.22928 *m_15**2 -0.0046842* m_15**3)/(1 +0.35474*m_15 + 0.19721 *m_15**2)
    t1 = (-0.10591 + 0.15416*m_15 -0.048486*m_15**2 +0.0045932* m_15**3)/(1 -0.29044*m_15 + 0.026267*m_15**2)
    t2 = ( 0.47291 - 0.44639*m_15 +0.1414 * m_15**2 -0.014978 * m_15**3)/(1 -0.37798*m_15 + 0.052154*m_15**2)
    t3 = ( 0.077407+ 0.18897*m_15 -0.072869*m_15**2 +0.0068684* m_15**3)/(1 -0.25237*m_15 + 0.020566*m_15**2)

    v0 = ( 0.82725 + 0.86015  *m_15 +0.007136 *m_15**2 +0.00020289*m_15**3)/(1 +0.90358*m_15 +0.015481*m_15**2)
    v1 = (-0.089088+ 0.089226 *m_15 -0.021442 *m_15**2 +0.0017054 *m_15**3)/(1 -0.28573*m_15 +0.024153*m_15**2)
    v2 = (-0.05342 - 0.0034387*m_15 +0.0050661*m_15**2 -0.00062569*m_15**3)/(1 -0.32663*m_15 +0.029382*m_15**2)
    v3 = (-0.17797 + 0.13134  *m_15 -0.030129 *m_15**2 +0.0023343 *m_15**3)/(1 -0.28211*m_15 +0.023712*m_15**2)
    
    Mb = (t0 + t1* beta_e + t2*beta_e**2)/(1+ t3*beta_e**2)
    Mg = (v0 + v1* beta_e + v2*beta_e**2)/(1+ v3*beta_e**2)    
    
    Pg= Eg1* Mg
    Pb= Eb1* Mb
    
    return Eg, Ebn, Ed, mR, Lg, Lb, Pg, Pb




















