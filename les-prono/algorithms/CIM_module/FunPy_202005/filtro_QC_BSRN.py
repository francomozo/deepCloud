#---BSRN Global Network recommended QC tests, V2.0 - C. N. Long and E. G. Dutton-
#---Agustin Laguarda
#---13/08/2019
#---21/04/2020 . se agregan filtros físicamente posibles y de comparación

import numpy as np

#---Extremely rare limits------------------------------------------------------
def filtro_BSRN_GHI_er(Iet, CSZ, GHI):
	#---Iet = 1367 * Fn
	#---GHI en W/m2
	MSK = (GHI>-2) & (GHI <= (Iet*1.2*(CSZ**1.2) + 50)  )
	return MSK 

def filtro_BSRN_DHI_er(Iet, CSZ, DHI):
	#---Iet = 1367 * Fn
	#---DHI en W/m2
	msk_min = DHI > -2
	msk_max = DHI <= Iet*0.75*(CSZ**1.2) + 30
	MSK = msk_min & msk_max
	return MSK 

def filtro_BSRN_DNI_er(Iet, CSZ, DNI):
	#---Iet = 1367 * Fn
	#---DNI en W/m2
	msk_min = DNI > -2
	msk_max = DNI <= Iet*0.95*(CSZ**0.2) + 10
	MSK = msk_min & msk_max
	return MSK 


#---Físicamente posibles-------------------------------------------------------
def filtro_BSRN_GHI_fp(Iet, CSZ, GHI):
	#---Iet = 1367 * Fn
	#---GHI en W/m2
	MSK = (GHI>-4) & (GHI <= (Iet*1.5*(CSZ**1.2) + 100)  )
	return MSK 

def filtro_BSRN_DHI_fp(Iet, CSZ, DHI):
	#---Iet = 1367 * Fn
	#---DHI en W/m2
	msk_min = DHI > -4
	msk_max = DHI <= Iet*0.95*(CSZ**1.2) + 50
	MSK = msk_min & msk_max
	return MSK 

def filtro_BSRN_DNI_fp(Iet, CSZ, DNI):
	# ---Iet = 1367 * Fn
	# ---DNI en W/m2
    msk_min = DNI > -4
    msk_max = DNI <= Iet
    MSK = msk_min & msk_max
    return MSK 


# comparaciones si están las 3 componentes:------------------------------------
def comparacion_clausura(CSZ, GHI, DHI, DNI):
    Gest = DNI*CSZ + DHI
    ratio = GHI/Gest
    umbral = 75
    ZEN = np.arccos(CSZ)*180/np.pi
    
    mk_l75 = abs(ratio-1)<=0.08
    mk_g75 = abs(ratio-1)<=0.15
    
    MSK = CSZ>10 #todos false
    
    MSK[ZEN<umbral] =mk_l75[ZEN< umbral]
    MSK[(ZEN>=umbral) & (ZEN<93) ]=mk_g75[(ZEN>=umbral)& (ZEN<93)]
    #---Iet = 1367 * Fn
    #-- en W/m2
    MSK[Gest<50]= 1# el test no se hace
    return MSK 

def comparacion_difusa(CSZ, GHI, DHI):
    fd = DHI / GHI
    
    umbral = 75
    ZEN = np.arccos(CSZ)*180/np.pi
    
    mk_l75 = fd<=1.05
    mk_g75 = fd<=1.10
    
    MSK = CSZ>10 #todos false
    
    MSK[ZEN<umbral] =mk_l75[ZEN< umbral]
    MSK[(ZEN>=umbral) & (ZEN<93) ]=mk_g75[(ZEN>=umbral) & (ZEN<93)]
    #---Iet = 1367 * Fn
    #-- en W/m2
    MSK[GHI<50]= 1# el test no se hace
    return MSK 


















