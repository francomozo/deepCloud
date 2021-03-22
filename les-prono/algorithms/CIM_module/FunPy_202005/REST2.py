#PROYECTO -Agustin Laguarda-
#modelo REST2 de dia claro.
#la idea es implementar el modelo REST2 de dia claro
#el siguiente script esta inspirado en forma y nomenclatura en Gueymard 2007
#version 2.0 13/08/2018

from __future__ import division, print_function
from numpy import log, exp, cos, pi, size, copy

#---variables:
# z                  =  angulo cenital en grados (array)
# Pcoef              =  presion hP/1013.25 ()
# u0, un             =  ozono y dioxido nitrogeno en atm-cm
# w                  =  vapor de agua precipitable en cm
# alfa1, alfa2, beta =  coeficientes de turbidez de Armstrong
# rog1, rog2         =  albedo terrestre para bandas 1 y 2

def REST2(z, Pcoef, beta, alfa1, alfa2, uo, un, w, rog1, rog2):
    n=size(z)
    cos_z = cos (z*pi/180)
    # defino constantes
    E0n1 = 635.4 #w/m2 irradiancia normal total en la banda 1
    E0n2 = 709.7 #w/m2                                banda 2

    omega1=0.92  #coeficientes experimentales recomendados para la difusa ideal
    omega2=0.84
    
    #---defino las cuatro masas de aire: mR, mo, mw, y ma, que dependen de z (Gueymard 2003 parte 1)
    mR = 1/( cos_z+ 0.48353*z**0.095846/(96.741-z)**1.754 )    
    mo = 1/( cos_z+ 1.06510*z**0.6379  /(101.80-z)**2.2694 )    
    mw = 1/( cos_z+ 0.10648*z**0.11423 /(93.781-z)**1.9203 )    
    ma = 1/( cos_z+ 0.16851*z**0.18198 /(95.318-z)**1.9542 )    
    #defino la masa de aire corregida por la presion de Rayleigh (m prima)
    mp = mR* Pcoef
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
    lae1 = (d0+ d1*ua1+ d2*ua1**2)/(1+ d3*ua1**2)
    
    ua2 = log(1 + ma* beta2)# esta ecuacione es dudosa, no se si usar beta1 o beta2
    e0 = (1.183 - 0.022989* alfa2+ 0.020829* alfa2**2)/(1 +0.11133*alfa2)
    e1 = (-0.50003- 0.18329*alfa2+ 0.23835 * alfa2**2)/(1 + 1.6756*alfa2)
    e2 = (-0.50001+ 1.1414*alfa2+ 0.0083589* alfa2**2)/(1 + 11.168*alfa2)
    e3 = (-0.70003- 0.73587 *alfa2+ 0.51509* alfa2**2)/(1 + 4.7665*alfa2)
    lae2 = (e0+ e1*ua2+ e2*ua2**2)/(1+ e3*ua2)
    
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
    #Ed[Ed<0]=0
    #Ebn[Ebn<0]=0
    
    Eg = Ebn *cos_z + Ed
    #Eg[Eg<0]=0
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
    
    
    
    
    
    
    
    
    
    


































