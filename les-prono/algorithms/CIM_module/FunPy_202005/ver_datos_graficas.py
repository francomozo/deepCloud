#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:08:32 2019

@author: agustin
"""
import numpy as np
import pylab as plt
from matplotlib import cm
import datetime
import pandas as pd
#import matplotlib.dates as mdates
import pvlib
import variables_solares as vs


#inspeccion visual de datos.
#UTC es un escalar (-3 para Uru)
def graficar_datos_estacion(I, Tiempo, PX, EST, UTC,titulo, MAX):
    print('...espere unos segundos mientras se realiza la figura')
    strUTC = str(UTC)
    if UTC>0: strUTC = '+'+strUTC
    I = np.array(I)
    #NO ADMITE HUECOS (los PX no tienen)
    dat_dia = int(24*60/PX) 
    
    iniTIME=datetime.datetime(Tiempo.year[0] ,Tiempo.month[0] ,Tiempo.day[0] ) 
    finTIME=datetime.datetime(Tiempo.year[-1],Tiempo.month[-1],Tiempo.day[-1])
    
    Tdiario = pd.date_range(iniTIME, finTIME, freq='D')#, dtype='datetime64[ns]')#+tz_cod+']')
    
    Imat =  np.full((dat_dia,len(Tdiario)),np.nan)
    for i in range(len(Tdiario)):
        msk = Tiempo.date == Tdiario.date[i]
        try: Imat[:,i] = I[msk]
        except: pass
    #Imat[Imat<0]=0 
    #Imat = I.reshape( int(len(I)/dat_dia) , dat_dia)
    
    #defino donde pongo el valor del año en el xlabel
    cambio_de_anho = np.where((Tdiario.year[1:-1] - Tdiario.year[0:-2])==1)
    index = cambio_de_anho[0]
    index = np.append(0,index)+1 # agrego el primer anho al comienzo ('0')
   # GRAFICO #############################################################
    
    plt.rcParams['font.size'] = 16
#    plt.rcParams['legend.fontsize'] = 'large'
#    plt.rcParams['figure.titlesize'] = 'large'
    plt.rcParams['axes.labelsize'] = 'large'
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
#    plt.rcParams['lines.markeredgewidth'] = 1.0
#    plt.rcParams['axes.spines.top'] = False
#    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.left'] = True
#    plt.rcParams['figure.figsize'] = 8, 8
    plt.rcParams['axes.facecolor']   = 'white'
    plt.rcParams['figure.facecolor'] ='white'
#    
    plt.figure(figsize=(40,5))
#    ax = FIG.add_subplot(111)
#    [H,x,y]=np.histogram2d(DAT['UVA'], UVAest[Nmod][mska], bins=100, range=[[0,80],[0,80]])
#    H[H==0]=np.nan#para que quede blanco donde no hay puntos
#    plt.grid(linestyle='-', linewidth=1, color='lightgray')
    heatmap = plt.imshow(Imat, origin='low', aspect = 'auto', cmap=cm.magma, vmin=0, vmax=MAX)#,extent=[0, len(Tiempo+1), 0, 23])#,vmin=0, vmax=70) #magma, plasma, inferno
    cbar = plt.colorbar(heatmap)
    cbar.set_label(titulo)#r'Irradiación [$W h /m^2$]
     # c.set_label(r'Irradiación [$W h /m^2$]', labelpad=-40, y=1.05, rotation=0) 
    plt.title('estación  : '+EST)
    plt.ylabel('Hora estándar '+ strUTC)
    plt.yticks(np.arange(0,24*60/PX,3*60/PX),np.arange(0,24,3))
    plt.xticks(index, Tdiario[index].year)
#    plt.tick_params(labelsize=20)
    plt.ylim([3*60/PX, 23*60/PX])
#    t=' MBD  = '+str(round(I1a[0],2))+r'$W/m^2$ rMBD = '+str(round(I1a[1]*100,1)) +'% \n RMSD = '+ str(round(I1a[2],1))+r'$W/m^2$ rRMSD = '+str(round(I1a[3]*100,1))+' %' 
#    ax=plt.subplot(1,1,1)
#    ax.text(1.3,70,t, fontsize=18, bbox=dict(facecolor='white', alpha=1, edgecolor = 'lightgray'))
    plt.tight_layout()
    #plt.show()
#    if GUARDAR == 'SI': plt.savefig('scatter_UVA_modelo'+Nmod+'_'+CODest[j]+'.png')

#################################################################
#Diagrama solar
#    GMT = 'Etc/GMT+3'  # para uruguay es +3!!
def diagrama_solar(I, Tiempo, PX, EST, LATdeg, LONdeg, alt, GMT):
    print('...espere unos segundos mientras se realiza el diagrama solar')
    I = np.array(I)
    I0 = 1367
    Tiempo = pd.DatetimeIndex(Tiempo , tz = GMT)#; Tiempo = aux.tz_convert(tz_cod_out)
    ANGS= pvlib.solarposition.get_solarposition(Tiempo + pd.Timedelta(minutes=PX/2) , LATdeg, LONdeg, altitude=alt)
    
    fn = vs.Fn(vs.Gamma(Tiempo.dayofyear, Tiempo.year))
    
    #D  = pvlib.solarposition.pyephem_earthsun_distance(Tiempo) #distancia tierra sol en UA
    
    GAM = np.array(ANGS.azimuth); HSO = np.array(ANGS.elevation); 
    #GAM[GAM>180] = GAM[GAM>180]-360 
    ZEN = np.array(ANGS.zenith)
    
    index= np.argsort(I)
    
    GAMs = GAM[index]; HSOs = HSO[index]; ZENs = ZEN[index]; Is = I[index]; fns = fn[index] # ordeno para que no se pisen en el grafico
    Iet= I0* fns
    Kts = Is/(Iet * np.cos(ZENs*np.pi/180))
    msk = HSOs>3 #altura mayor a 3 grados
    
    #Kts[Kts<0]=0; Kts[Kts>1]=1; 
   # GRAFICO #############################################################
    plt.figure(figsize=(16,8))
    plt.scatter(GAMs[msk], HSOs[msk], c = Kts[msk], cmap=cm.afmhot,s=5, alpha = 0.6, vmin=0, vmax=1)#
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 'large'
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.grid(linestyle='-', linewidth=1, color='lightgray')
    plt.title('estación  : '+EST)
    plt.xlabel(r'Ángulo azimutal [$^o$]')
    plt.ylabel(r'Altura solar [$^o$]')
    plt.tight_layout()
    plt.show()

# fin################
