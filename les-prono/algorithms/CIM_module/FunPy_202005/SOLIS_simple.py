# MODELO SOLIS SIMPLIFICADO. Ineichen.
# 18/12/2014
# Version 25/09/2018 para Python

#VARIABLES:
#aod700  =    profundidad optica de aerosoles a 700 nm ()
#Igh0    =    Irradiancia extraterrestre
#CSZA    =    coseno del angulo cenital
#pcoef   =    presion atmosférica en Pascales /p0 (p0=1013e3)
#w       =    vapor de columna de agua (en cm)

#I0=1367;  en W/m2, se usa esta constante
#rango optimo: altura snm <7000
#              0< aod700  < 0.45
#              0.2 cm < w < 10 cm
import numpy as np

def SOLIS_simple(CSZ, aod700, pcoef, w):
    Igh0=1367
    
    #obs: sin de la altura solar = CSZ
    logWV= np.log(w)
    logPc= np.log(pcoef)
    #intensidad extraterrestre aumentada Io
    Io0 = 1.08*(w**0.0051)
    Io1 = 0.97*(w**0.032)
    Io2 = 0.12*(w**0.56)
    Io = Igh0 * (Io2*aod700**2 + Io1*aod700 + Io0 + 0.071*logPc)
    
    # cosas para Ibn..........................................................
    # calculo tau_b y b
    tb1 = 1.82 + 0.056*logWV+0.0071*logWV**2
    tb0 = 0.33 + 0.045*logWV+0.0096*logWV**2
    tbp = 0.0089*w +0.13

    b1 = 0.00925 * aod700**2 + 0.0148 *aod700 - 0.0172
    b0 =-0.7565 * aod700**2 + 0.5057 * aod700 + 0.4557

    tau_b = tb1*aod700 + tb0 + tbp*logPc
    b = b1*logWV+ b0
    # cosas para Igh...........................................................
    # calculo tau_g y g:
    tg1 = 1.24 + 0.047*logWV+ 0.0061 * logWV**2
    tg0 = 0.27 + 0.043*logWV+ 0.0090 * logWV**2
    tgp= 0.0079 *w +0.1

    tau_g =  tg1* aod700 + tg0+ tgp* logPc
    g =-0.0147 *logWV-0.3079* aod700**2+ 0.2846 * aod700 +0.3798

    # cosas para Idh...........................................................
    # calculo tau_d:
    #cosas cambiadas
    #---(aod700<0.05)
    td4 = 86*w -13800
    td3 =-3.11*w + 79.4
    td2 =-0.23*w +74.8
    td1 = 0.092*w - 8.86
    td0 = 0.0042*w + 3.12
    tdp =-0.83 *(1+aod700)**(-17.2)
    #---(aod700 >= 0.05)
    msk1 = (aod700>=0.05)
    td4[msk1] =-0.21 * w[msk1]+11.6
    td3[msk1] = 0.27 * w[msk1]-20.7
    td2[msk1] =-0.134 * w[msk1] + 15.5
    td1[msk1] = 0.0554 * w[msk1]-5.71
    td0[msk1] = 0.0057 * w[msk1] + 2.94
    tdp[msk1] =-0.71 * (1+aod700[msk1])**(-15.0)
    #.........................................
    dp=1/(18+152*aod700)
    
    tau_d = td4 * aod700**4+ td3*aod700**3+ td2*aod700**2 + td1*aod700 + td0 + tdp*logPc
    d=-0.337*aod700**2+0.63*aod700+0.116+dp *logPc
    
    #por ultimo calculo las salidas: Ibn, Igh, Idh
    Ibn=Io*np.exp(-tau_b/(CSZ**b) )
    Igh=Io*np.exp(-tau_g/(CSZ**g) )*CSZ
    Idh=Io*np.exp(-tau_d/(CSZ**d) )

    return Igh, Idh, Ibn

