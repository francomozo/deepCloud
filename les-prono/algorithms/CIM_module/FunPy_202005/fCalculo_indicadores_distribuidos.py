import numpy as np
import fCalculo_indicadores as indicadores
import pylab as plt
from matplotlib import cm

def calculo_indicadores_distribuidos(Xest, Xval,MSK, kt, CSZ, kTdiv, CZdiv):

# function [MBD, RMS, Cmat, Ctot, MSKrms] = calculo_distribucion_indicadores(E, MSK, kTI, COZ, kTdiv, CZdiv)

# MBD: Matriz de MBD para X clasificada segun cz y kt
# RMS: Matriz de RMS para X clasificada segun cz y kt
# Cmat: Matriz con la cantidad de elemntos clasificados segun cz y kt
# Ctot: Cantidad total de elementos en Cmat
# MSKrms: Mascara que indica donde Cmat es distinto de cero

    lk = len(kTdiv) 
    lc = len(CZdiv)

    MBD  = np.zeros([lk-1, lc-1])
    RMS  = np.zeros([lk-1, lc-1]) 
    Cmat = np.zeros([lk-1, lc-1]) # Cantidad de muestras utilizadas por cada punto de grilla para calcular stderr

    for i in range(lk-1):
    
        kti_sup = kTdiv[i+1]
        kti_inf = kTdiv[i] # defino intervalo de valores de kti
    
        for j in range(lc-1):
            cos_sup = CZdiv[j+1]
            cos_inf = CZdiv[j] # defino intervalo de valores de coseno
        
            MSKkti = (kt >= kti_inf) & (kt < kti_sup) # Defino rango de indice de claridad
            MSKcos = (CSZ >= cos_inf) & (CSZ < cos_sup) # Defino rango de coseno del angulo zenital
            MSKtot = (MSKkti) & (MSKcos) & MSK
            Cok = sum(MSKtot)

            if (Cok > 0):
                Cmat[i,j] = Cok
                [MBD[i,j],_,RMS[i,j],_,_,_,_,_,_] = indicadores.calculo_MBD_RMS_MAD_corr_N_mean(Xest[MSKtot], Xval[MSKtot])

    MSKrms = np.isfinite(Cmat)&(Cmat!=0) # No considero entradas donde no hay muestras
    Ctot = sum(Cmat[MSKrms])        # Cantidad de elementos clasificados 

    return MBD, RMS, Cmat, Ctot, MSKrms

def graficar_indicadores_distribuidos(Xest, Xval,MSK, kt, CSZ, titulo):#, kTdiv, CZdiv):
    
    kTdiv = np.arange(0,1,0.1)
    CZdiv = np.arange(0.1,1.1,.1)
    
    kTtick=np.arange(0,len(kTdiv),2)
    CZtick=np.arange(0,len(CZdiv),2)
    
    [MBD, RMS, Cmat, Ctot, MSKrms] = calculo_indicadores_distribuidos(Xest, Xval,MSK, kt, CSZ, kTdiv, CZdiv)
    MSK = (Cmat==0)
    Cmat[MSK] = np.nan
    rMBD = MBD/np.mean(Xval)*100; rMBD[MSK] = np.nan
    rRMS = RMS/np.mean(Xval)*100; rRMS[MSK] = np.nan
    MSK = (Cmat==0)    
    
    plt.rcParams['font.size'] = 16
#    plt.rcParams['legend.fontsize'] = 'large'
#    plt.rcParams['figure.titlesize'] = 'large'
    #plt.rcParams['axes.labelsize'] = 'large'
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
    
    # DATOS 
    plt.figure(figsize=(8,6))#,top=0.97, bottom=0.1, left=0.1, right=0.97, hspace=0.2, wspace=0.2)
    heatmap = plt.imshow(Cmat, origin='lower', aspect = 'auto', cmap=cm.plasma, vmin=0, extent = [0, len(kTdiv)-1, 0, len(CZdiv)-1])
    cbar = plt.colorbar(heatmap)
    plt.title('Number of data points '+titulo)
    #cbar.set_label('Número de puntos')#r'Irradiación [$W h /m^2$]
     # c.set_label(r'Irradiación [$W h /m^2$]', labelpad=-40, y=1.05, rotation=0) 
    plt.xlabel('cosine of the solar zenith angle')
    plt.ylabel('clearness index ($k_{t}$)')
    #plt.xlabel(r'coseno del angulo zenital')
    #plt.ylabel('índice de claridad ($k_{t}$)')
    plt.xticks(CZtick,np.round(CZdiv[CZtick],1))
    plt.yticks(kTtick,np.round(kTdiv[kTtick],1))
    plt.tight_layout()
    plt.show()
    
    #MBD y RMS
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    heatmap = plt.imshow(rMBD, origin='lower', aspect = 'auto', cmap=cm.Spectral, vmin=-10, vmax=10, extent = [0, len(kTdiv)-1, 0, len(CZdiv)-1])#, extent = [0, len(kTdiv)-1, 0, len(CZdiv)-1])#,extent=[0, len(Tiempo+1), 0, 23])#,vmin=0, vmax=70) #magma, plasma, inferno
    cbar = plt.colorbar(heatmap)
    plt.title('relative MBD (%)'+titulo)
    #cbar.set_label('rMBD (%)')#r'Irradiación [$W h /m^2$]
     # c.set_label(r'Irradiación [$W h /m^2$]', labelpad=-40, y=1.05, rotation=0) 
    plt.xlabel('cosine of the solar zenith angle')
    plt.ylabel('clearness index ($k_{t}$)')
    plt.xticks(CZtick,np.round(CZdiv[CZtick],1))
    plt.yticks(kTtick,np.round(kTdiv[kTtick],1))
    
##    t=' MBD  = '+str(round(I1a[0],2))+r'$W/m^2$ rMBD = '+str(round(I1a[1]*100,1)) +'% \n RMSD = '+ str(round(I1a[2],1))+r'$W/m^2$ rRMSD = '+str(round(I1a[3]*100,1))+' %' 
##    ax=plt.subplot(1,1,1)
##    ax.text(1.3,70,t, fontsize=18, bbox=dict(facecolor='white', alpha=1, edgecolor = 'lightgray'))
    plt.subplot(1,2,2)
    heatmap = plt.imshow(rRMS, origin='lower', aspect = 'auto', cmap=cm.viridis, vmin=0, vmax=15, extent = [0, len(kTdiv)-1, 0, len(CZdiv)-1])
    cbar = plt.colorbar(heatmap)
    plt.title('relative RMSD (%)'+titulo)#r'Irradiación [$W h /m^2$]
    #cbar.set_label('rRMSD (%)')#r'Irradiación [$W h /m^2$]
     # c.set_label(r'Irradiación [$W h /m^2$]', labelpad=-40, y=1.05, rotation=0) 
    plt.xlabel('cosine of the solar zenith angle')
    plt.ylabel('clearness index ($k_{t}$)')
    plt.xticks(CZtick,np.round(CZdiv[CZtick],1))
    plt.yticks(kTtick,np.round(kTdiv[kTtick],1))
    #plt.tick_params(labelsize=20)
    
    plt.tight_layout()
    plt.show()
    
def graficar_indicadores_distribuidos_espanol(Xest, Xval,MSK, kt, CSZ, titulo, nombre,guardar):#, kTdiv, CZdiv):
    #nombre= ruta para guardar las figuras
    #guardar = 1 guarda, o no
    kTdiv = np.arange(0,1,0.1)
    CZdiv = np.arange(0.1,1.1,.1)
    
    kTtick=np.arange(0,len(kTdiv),2)
    CZtick=np.arange(0,len(CZdiv),2)
    
    [MBD, RMS, Cmat, Ctot, MSKrms] = calculo_indicadores_distribuidos(Xest, Xval,MSK, kt, CSZ, kTdiv, CZdiv)
    MSK = (Cmat==0)
    Cmat[MSK] = np.nan
    rMBD = MBD/np.mean(Xval)*100; rMBD[MSK] = np.nan
    rRMS = RMS/np.mean(Xval)*100; rRMS[MSK] = np.nan
    MSK = (Cmat==0)    
    
    plt.rcParams['font.size'] = 14
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.facecolor']   = 'white'
    plt.rcParams['figure.facecolor'] ='white'
    
    # DATOS 
    plt.figure(figsize=(6.5,7))#,top=0.97, bottom=0.1, left=0.1, right=0.97, hspace=0.2, wspace=0.2)
    heatmap = plt.imshow(Cmat, origin='lower', aspect = 'auto', cmap=cm.plasma, vmin=0, extent = [0, len(kTdiv)-1, 0, len(CZdiv)-1])
    cbar = plt.colorbar(heatmap, orientation = 'horizontal', pad = 0.12)
   # plt.title('# datos '+titulo)
    cbar.set_label('# datos')#r'Irradiación [$W h /m^2$]
     # c.set_label(r'Irradiación [$W h /m^2$]', labelpad=-40, y=1.05, rotation=0) 
    plt.xlabel(r'Coseno del ángulo cenital')
    plt.ylabel(r'Índice de claridad')
    #plt.xlabel(r'coseno del angulo zenital')
    #plt.ylabel('índice de claridad ($k_{t}$)')
    plt.xticks(CZtick,np.round(CZdiv[CZtick],1))
    plt.yticks(kTtick,np.round(kTdiv[kTtick],1))
    plt.tight_layout()
    plt.show()
    if guardar: plt.savefig(nombre+'_puntos.png')
    
    #MBD y RMS
    #plt.figure(figsize=(16,6))
    plt.figure(figsize=(6.5,7))
        #plt.subplot(1,2,1)
    heatmap = plt.imshow(rMBD, origin='lower', aspect = 'auto', cmap=cm.Spectral, vmin=-70, vmax=70, extent = [0, len(kTdiv)-1, 0, len(CZdiv)-1])#, extent = [0, len(kTdiv)-1, 0, len(CZdiv)-1])#,extent=[0, len(Tiempo+1), 0, 23])#,vmin=0, vmax=70) #magma, plasma, inferno
    cbar = plt.colorbar(heatmap, orientation = 'horizontal', pad = 0.12)
    #plt.title('rMBD (%)'+titulo)
    cbar.set_label('rMBD (%)')#r'Irradiación [$W h /m^2$]
     # c.set_label(r'Irradiación [$W h /m^2$]', labelpad=-40, y=1.05, rotation=0) 
    plt.xlabel(r'Coseno del ángulo cenital')
    plt.ylabel(r'Índice de claridad')
    plt.xticks(CZtick,np.round(CZdiv[CZtick],1))
    plt.yticks(kTtick,np.round(kTdiv[kTtick],1))
    plt.tight_layout()
    plt.show()
    if guardar: plt.savefig(nombre+'_MBD.png')
##    t=' MBD  = '+str(round(I1a[0],2))+r'$W/m^2$ rMBD = '+str(round(I1a[1]*100,1)) +'% \n RMSD = '+ str(round(I1a[2],1))+r'$W/m^2$ rRMSD = '+str(round(I1a[3]*100,1))+' %' 
##    ax=plt.subplot(1,1,1)
##    ax.text(1.3,70,t, fontsize=18, bbox=dict(facecolor='white', alpha=1, edgecolor = 'lightgray'))
    #plt.subplot(1,2,2)
    plt.figure(figsize=(6.5,7))
    heatmap = plt.imshow(rRMS, origin='lower', aspect = 'auto', cmap=cm.viridis, vmin=0, vmax=90, extent = [0, len(kTdiv)-1, 0, len(CZdiv)-1])
    cbar = plt.colorbar(heatmap, orientation = 'horizontal', pad = 0.12)
    #plt.title('rRMSD (%)'+titulo)#r'Irradiación [$W h /m^2$]
    cbar.set_label('rRMSD (%)')#r'Irradiación [$W h /m^2$]
     # c.set_label(r'Irradiación [$W h /m^2$]', labelpad=-40, y=1.05, rotation=0) 
    plt.xlabel(r'Coseno del ángulo cenital')
    plt.ylabel(r'Índice de claridad')
    plt.xticks(CZtick,np.round(CZdiv[CZtick],1))
    plt.yticks(kTtick,np.round(kTdiv[kTtick],1))
    #plt.tick_params(labelsize=20)
    plt.tight_layout()
    plt.show()
    if guardar: plt.savefig(nombre+'_RMS.png')
    
    
    