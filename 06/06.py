# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 21:35:07 2018

@author: Admin
"""
import time
from PIL import Image
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from scipy.integrate import ode
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg as lin
from scipy.optimize import fsolve
from scipy.linalg import solve
from scipy.linalg import solve_banded
from scipy.special import jn_zeros #prvi parameter je order, drugi št. ničel
from scipy.special import jv #prvi order drugi argument
#from scipy.special import beta
import scipy.special as spec
import scipy.spatial as spat
import scipy.sparse
from scipy.optimize import root
from scipy.integrate import quad
from scipy.integrate import romb
from scipy.integrate import complex_ode
from scipy.integrate import trapz
from scipy.optimize import linprog
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.linalg import svd
from matplotlib.patches import Ellipse
from matplotlib import gridspec
from numba import jit
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rc("text",usetex=True)
matplotlib.rcParams["text.latex.unicode"] = True
plt.close("all")
PI = np.pi

POLJE = np.array([0,0,1])

@jit(nopython=True)
def deltaE(j,stara,nova,J,h,N): 
    return np.dot(nova[j]-stara[j],-J*(stara[(j+1)%N]+stara[(j-1)%N])+h*POLJE)

@jit(nopython=True)
def deltaE2(i,stara,nova,J,h,N):
    j = (i+1)%N
    razlika1 = np.dot(nova[i]-stara[i],-J*(stara[(i-1)%N])+h*POLJE)
    razlika2 = np.dot(nova[j]-stara[j],-J*(stara[(j+1)%N])+h*POLJE)
    return razlika1+razlika2

@jit(nopython=True)
def poteza(stara,N):
    nova = np.copy(stara)
    j = np.random.randint(0,N) #0 inclusive N exclusive
    fi = np.random.rand()*2*PI
    theta = np.arccos(2*np.random.rand()-1)
    nova[j] = np.array([np.sin(theta)*np.cos(fi),np.sin(theta)*np.sin(fi),np.cos(theta)])
    return (nova,j)

@jit(nopython=True)
def poteza2(stara,N):
    nova = np.copy(stara)
    i = np.random.randint(0,N) #0 inclusive N exclusive
    j = (i+1)%N
    vsota = (stara[i]+stara[j])/np.sqrt(np.dot(stara[i]+stara[j],stara[i]+stara[j]))
    K = np.zeros((3,3))*1.0
    K[0][1]=-vsota[2]
    K[0][2]=vsota[1]
    K[1][2]=-vsota[0]
    K = K - np.transpose(K)
    fi = np.random.rand()*2*PI
    nova[i] = stara[i] + np.sin(fi)*np.dot(K,stara[i]) + (1-np.cos(fi))*np.dot(np.dot(K,K),stara[i])
    nova[j] = stara[j] + np.sin(fi)*np.dot(K,stara[j]) + (1-np.cos(fi))*np.dot(np.dot(K,K),stara[j])
    #nova[i] =stara[i]*np.cos(fi)+ np.cross(vsota,stara[i])*np.sin(fi)+vsota*np.dot(vsota,stara[i])*(1-np.cos(fi))
    #nova[j] =stara[j]*np.cos(fi)+ np.cross(vsota,stara[j])*np.sin(fi)+vsota*np.dot(vsota,stara[j])*(1-np.cos(fi))
    return (nova,i)

@jit(nopython=True)
def E(stara,J,h,N):
    en = h*np.sum(np.array([stara[j][2] for j in range(N)]))
    for j in range(N):
        if j==N-1:
            en-= J*np.dot(stara[j],stara[0])
            break
        en -= J*np.dot(stara[j],stara[j+1])
    return en

@jit(nopython=True)
def genZac(N):
    thete = np.arccos(2*np.random.rand(N)-np.ones(N))
    fiji = np.random.rand(N)*2*PI
    return np.transpose(np.vstack((np.sin(thete)*np.cos(fiji),np.sin(thete)*np.sin(fiji),np.cos(thete))))

@jit(nopython=True)
def genZac2(N):
    n = int(N/2)
    thete = np.arccos(2*np.random.rand(n)-np.ones(n))
    fiji = np.random.rand(n)*2*PI
    rez = np.transpose(np.vstack((np.sin(thete)*np.cos(fiji),np.sin(thete)*np.sin(fiji),np.cos(thete))))
    rez2 = np.concatenate((rez,-rez),axis=0)
    np.random.shuffle(rez2)
    return rez2

@jit(nopython=True)
def metropolisBrez(nsteps,beta,J,h,N,zacetni,anneal=False,rate=0.01,pogostost=100):
    bbeta = beta
    prejsnji = np.copy(zacetni)
    trenutni = np.copy(prejsnji)
    for i in range(nsteps):
        if anneal and i%pogostost==0:
            bbeta += rate
        prejsnji = trenutni
        trenutni, j = poteza(prejsnji,N)
        delta = deltaE(j,prejsnji,trenutni,J,h,N)
        if delta < 0:
            continue
        else:
            U = np.random.rand()
            if U < np.exp(-bbeta*delta):
                continue
            else:
                trenutni=prejsnji
    return trenutni

@jit(nopython=True)
def metropolisZ(nsteps,beta,J,h,N,anneal=False,rate=0.01,pogostost=100):
    bbeta = beta
    povp=np.zeros((N,3))
    vrnjene = np.zeros((int(nsteps/pogostost),N,3))
    prejsnji = genZac(N)
    trenutni = np.copy(prejsnji)
    for i in range(nsteps):
        povp += trenutni
        if anneal and i%pogostost==0:
            vrnjene[int(i/pogostost)]=povp/pogostost
            povp = np.zeros((N,3))
            bbeta += rate
        prejsnji = trenutni
        trenutni, j = poteza(prejsnji,N)
        delta = deltaE(j,prejsnji,trenutni,J,h,N)
        if delta < 0:
            continue
        else:
            U = np.random.rand()
            if U < np.exp(-bbeta*delta):
                continue
            else:
                trenutni=prejsnji
    return vrnjene

@jit(nopython=True)
def metropolisKor(nsteps,beta,J,h,N,cakaj):
    vrnjene = np.zeros(nsteps)
    prejsnji = genZac(N)
    trenutni = np.copy(prejsnji)
    for i in range(cakaj):
        prejsnji = trenutni
        trenutni, j = poteza(prejsnji,N)
        delta = deltaE(j,prejsnji,trenutni,J,h,N)
        if delta < 0:
            continue
        else:
            U = np.random.rand()
            if U < np.exp(-beta*delta):
                continue
            else:
                trenutni=prejsnji
    #prejsnji = np.copy(stara)
    #trenutni = np.copy(prejsnji)
    print(666)
    for i in range(nsteps):
        prejsnji = trenutni
        trenutni, j = poteza(prejsnji,N)
        delta = deltaE(j,prejsnji,trenutni,J,h,N)
        if delta < 0:
            continue
        else:
            U = np.random.rand()
            if U < np.exp(-beta*delta):
                continue
            else:
                trenutni=prejsnji
        vrnjene[i]=E(trenutni,J,h,N)
    return vrnjene

"""
def susc(beta,podatki,N):
    n = len(podatki)
    avgm = 1/N*1/n*np.sum(np.sum(podatki,axis=1),axis=0)
    avgm2 = 1/N*1/n*np.sum(np.sum(np.sum(podatki**2,axis=2),axis=1),axis=0)
    return beta*(avgm2-np.sum(avgm**2))
"""

def susc(beta,podatki,N):
    n = len(podatki)
    avgm = 1/N*1/n*np.sum(np.sum(podatki[:,:,2],axis=1),axis=0)
    avgm2 = 1/N*1/n*np.sum(np.sum(podatki[:,:,2]**2,axis=1),axis=0)
    return beta*(avgm2-avgm**2)

def kapac(beta,podatki,J,h,N):
    n = len(podatki)
    energije = np.array([E(podatki[i],J,h,N) for i in range(n)])
    return beta*beta*(np.sum(energije**2)/n-np.sum(energije/n)**2)


def magnetizacija(beta,podatki,N):
    n = len(podatki)
    avgm = 1/N*1/n*np.sum(np.sum(podatki,axis=1),axis=0)
    avgm2 = 1/N*1/n*np.sum(np.sum(np.sum(podatki**2,axis=2),axis=1),axis=0)
    return (np.sqrt(np.sum(avgm**2)),1/np.sqrt(n)*np.sqrt(avgm2-np.sum(avgm**2)))

def magnetizacijaZ(beta,podatki,N):
    n = len(podatki)
    avgm = 1/N*1/n*np.sum(np.sum(podatki[:,:,2],axis=1),axis=0)
    avgm2 = 1/N*1/n*np.sum(np.sum(podatki[:,:,2]**2,axis=1),axis=0)
    return (avgm,1/np.sqrt(n)*np.sqrt(avgm2-avgm**2))

def energija(beta,podatki,J,h,N):
    n = len(podatki)
    energije = np.array([E(podatki[i],J,h,N) for i in range(n)])
    return (np.sum(energije)/n,1/np.sqrt(n)*(1/n*np.sum(energije**2)-np.sum(1/n*energije)**2))

def vseBeta(beta,J,h,N,stevilo=1000,cakaj=500000,povprecuj=50,prva=[0]):
    if len(prva)==1:
        prva = metropolisBrez(cakaj,beta,J,h,N,genZac2(N))
    podatki = np.ones((stevilo,N,3))
    for i in range(stevilo):
        prva = metropolisBrez(povprecuj,beta,J,h,N,prva)
        podatki[i]=prva
    suscepibilnost = susc(beta,podatki,N)
    kapaciteta = kapac(beta,podatki,J,h,N)
    temp = magnetizacija(beta,podatki,N)
    mag = temp[0]
    napaka1=temp[1]
    temp = magnetizacijaZ(beta,podatki,N)
    mag1 = temp[0]
    napaka11=temp[1]
    temp = energija(beta,podatki,J,h,N)
    en = temp[0]
    napaka2=temp[1]
    return [np.array([suscepibilnost, kapaciteta, mag, napaka1, mag1, napaka11, en, napaka2]),podatki[-1]]


if 0:
    #korelacije v odv od beta
    J=1
    h=1
    N = 100
    bete = [0,2,4,6,8,10]
    prva = [0]
    nsteps=1000000
    for beta in bete:
        korr = np.zeros(100)*1j
        print(beta)
        if len(prva)==1:
            prva = metropolisBrez(nsteps,beta,J,h,N,genZac(N))
        else:
            prva = metropolisBrez(nsteps,beta,J,h,N,genZac(N))
        for i in range(1000):
            prva = metropolisBrez(50,beta,J,h,N,prva)
            kor = np.fft.ifft(np.fft.fft(prva[:,2])*np.conj(np.fft.fft(prva[:,2])))
            kor = kor/max(kor)
            korr += kor
            
        x = [i for i in range(int(len(kor)/2))]
        plt.plot(x,korr[:int(len(korr)/2)]/1000,label=r"$\beta={}$".format(beta))        
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel(r"$j$",fontsize=14)
    #plt.xticks([i for i in range(1,21)])
    plt.ylabel(r"$C$",fontsize=14)
    plt.title(r"Spinske korelacije ($\sigma^z$), $N=100, J=1, h=1$",fontsize=13)
    plt.savefig("Figures/spinkor2.pdf")
if 0:
    #odv od beta za vec N
    J=1
    h=1
    Nji = [10,20,30,40,50]
    bete = [0.1*i for i in range(101)]
    rezultati = []
    for N in Nji:
        prva = [0]
        rez = []
        for beta in bete:
            print(beta)
            temp = vseBeta(beta,J,h,N,prva=prva)
            rez.append(temp[0])
            prva = temp[1]
        rez = np.array(rez)
        rezultati.append(rez)
    for i in range(len(Nji)):
        N = Nji[i]
        plt.plot(bete,rezultati[i][:,0],"--",label=r"$N={}$".format(N))
        #plt.plot(bete,rezultati[i][:,1],"--",label=r"$N={}$".format(N))
        #plt.plot(bete,rez[:,2],"--",color="red",label=r"$M$")
        #plt.errorbar(bete,rezultati[i][:,2],yerr=rezultati[i][:,3],linestyle="--",label=r"$N={}$".format(N))
        #plt.errorbar(bete,rez[:,4],yerr=rez[:,5],color="orange",linestyle="--",label=r"$M_z$")
        #plt.plot(bete,rez[:,4],"--",color="magenta",label=r"$E$")
        #plt.errorbar(bete,rezultati[i][:,6],yerr=rezultati[i][:,7],linestyle="--",label=r"$N={}$".format(N))
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel(r"$\beta$",fontsize=14)
    #plt.xticks([i for i in range(1,21)])
    #plt.ylabel(r"$C$",fontsize=14)
    plt.title(r"Odvisnost $\chi$, $J=1, h=1$",fontsize=13)
    plt.savefig("Figures/betavez1.pdf")
    
if 0:
    #vse stvari v odv od beta
    J=0
    h=1
    N=100
    bete = [0.1*i for i in range(101)]
    rez = []
    prva = [0]
    for beta in bete:
        print(beta)
        temp = vseBeta(beta,J,h,N,prva=prva)
        rez.append(temp[0])
        prva = temp[1]
    print("lol")
    rez = np.array(rez)
    plt.plot(bete,rez[:,0],"--",color="blue",label=r"$\chi$")
    plt.plot(bete,rez[:,1],"--",color="green",label=r"$C_V$")
    #plt.plot(bete,rez[:,2],"--",color="red",label=r"$M$")
    plt.errorbar(bete,rez[:,2],yerr=rez[:,3],color="red",linestyle="--",label=r"$M$")
    plt.errorbar(bete,rez[:,4],yerr=rez[:,5],color="orange",linestyle="--",label=r"$M_z$")
    #plt.plot(bete,rez[:,4],"--",color="magenta",label=r"$E$")
    plt.errorbar(bete,rez[:,6],yerr=rez[:,7],color="magenta",linestyle="--",label=r"$E$")
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel(r"$\beta$",fontsize=14)
    #plt.xticks([i for i in range(1,21)])
    #plt.ylabel(r"$C$",fontsize=14)
    plt.title(r"Odvisnost kolicin od $\beta$, $N=100, J=0, h=1$",fontsize=13)
    plt.savefig("Figures/grafvez3.pdf")

if 0:
    #vse stvari v odv od J/h
    h=1
    N=100
    bete = [0.5*i for i in range(21)]
    rez = []
    beta=5
    prva = [0]
    for J in bete:
        print(J)
        temp = vseBeta(beta,J,h,N,prva=prva)
        rez.append(temp[0])
        prva = temp[1]
    print("lol")
    rez = np.array(rez)
    plt.plot(bete,rez[:,0],"--",color="blue",label=r"$\chi$")
    #plt.plot(bete,rez[:,1],"--",color="green",label=r"$C_V$")
    #plt.plot(bete,rez[:,2],"--",color="red",label=r"$M$")
    plt.errorbar(bete,rez[:,2],yerr=rez[:,3],color="red",linestyle="--",label=r"$M$")
    plt.errorbar(bete,rez[:,4],yerr=rez[:,5],color="orange",linestyle="--",label=r"$M_z$")
    #plt.plot(bete,rez[:,4],"--",color="magenta",label=r"$E$")
    #plt.errorbar(bete,rez[:,6],yerr=rez[:,7],color="magenta",linestyle="--",label=r"$E$")
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel(r"$J$",fontsize=14)
    #plt.xticks([i for i in range(1,21)])
    #plt.ylabel(r"$C$",fontsize=14)
    plt.title(r"Odvisnost kolicin od $J$, $N=100, \beta=5, h=1$",fontsize=13)
    plt.savefig("Figures/grafvez44.pdf")


if 0:
    #prikaz
    N=100
    J=1
    h=1
    beta=0
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    koti = np.array([i/N*2*PI for i in range(N)])
    ax.set_zlim(-1,1)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    x = np.cos(koti)
    y = np.sin(koti)
    z = np.zeros(N)
    rez = metropolisBrez(100000000,beta,J,h,N,genZac2(N),anneal=1,rate=0.01,pogostost=10000)
    ax.quiver(x,y,z,rez[:,0],rez[:,1],rez[:,2],normalize=True,arrow_length_ratio=0.05,color="blue")
    #rez = metropolisBrez(1000000,beta,J,h,N,genZac2(N),anneal=1,rate=1,pogostost=10000)
    #ax.quiver(x,y,z,rez[:,0],rez[:,1],rez[:,2],normalize=True,arrow_length_ratio=0.05,color="red")
    #rez = metropolisBrez(1000000,beta,J,h,N,genZac2(N),anneal=1,rate=1,pogostost=10000)
    #ax.quiver(x,y,z,rez[:,0],rez[:,1],rez[:,2],normalize=True,arrow_length_ratio=0.05,color="magenta")
    plt.title(r"Osnovno stanje z vezjo, $N=100, J=1, h=1$")
    koti=np.array([i/1000*2*PI for i in range(1000)])
    ax.plot(np.cos(koti),np.sin(koti),np.zeros(1000),ls="-",color="green")
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    frame1.axes.zaxis.set_ticklabels([])
    #plt.axis("off")
    ax.quiver(0,0,0,POLJE[0],POLJE[1],POLJE[2],color="orange")
    plt.tight_layout()
    plt.savefig("Figures/prikazvez1.pdf")

if 1:
    #prikaz z animacijo
    N=10
    J=1
    h=1
    beta=0
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    koti = np.array([i/N*2*PI for i in range(N)])
    ax.set_zlim(-1,1)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    x = np.cos(koti)
    y = np.sin(koti)
    z = np.zeros(N)
    rez = metropolisZ(10000000,beta,J,h,N,anneal=1,rate=0.1,pogostost=10000)
    #rez = metropolisBrez(1000000,beta,J,h,N,genZac2(N),anneal=1,rate=1,pogostost=10000)
    #ax.quiver(x,y,z,rez[:,0],rez[:,1],rez[:,2],normalize=True,arrow_length_ratio=0.05,color="red")
    #rez = metropolisBrez(1000000,beta,J,h,N,genZac2(N),anneal=1,rate=1,pogostost=10000)
    #ax.quiver(x,y,z,rez[:,0],rez[:,1],rez[:,2],normalize=True,arrow_length_ratio=0.05,color="magenta")
    koti=np.array([i/1000*2*PI for i in range(1000)])
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    frame1.axes.zaxis.set_ticklabels([])
    #plt.axis("off")
    #plt.savefig("Figures/prikaz2.pdf")
    def animiraj(t):
        print(t)
        ax.clear()
        ax.quiver(x,y,z,rez[t][:,0],rez[t][:,1],rez[t][:,2],normalize=True,arrow_length_ratio=0.05,color="blue")
        ax.plot(np.cos(koti),np.sin(koti),np.zeros(1000),ls="-",color="green")
        ax.quiver(0,0,0,POLJE[0],POLJE[1],POLJE[2],color="orange")
        plt.suptitle(r"Brez vezi: $N=100, J=1, h=1, \beta={}$".format(round(beta+0.1*t)))
        plt.tight_layout()
    ani = animation.FuncAnimation(fig,animiraj,range(1000),interval=50)   
    #plt.show()
    ani.save("relaks2.mp4")  
    print("evo")

if 0:
    # Osnovna energija od J/h
    N=10
    nsteps=10000
    pogostost=1
    J=1
    h=1
    beta=0
    bete = [beta+i*0.1 for i in range(int(nsteps/pogostost))]
    """
    rez = metropolisZ(nsteps,10,J,h,N,True,0.1,pogostost)
    plt.plot(bete,[E(i,J,h,N) for i in rez],label=r"$J={}, h={}$".format(J,h))    
    print("lol")
    """
    J=1
    h=0
    rez = metropolisZ(nsteps,beta,J,h,N,True,0.1,pogostost)
    energije =[E(i,J,h,N) for i in rez]
    #energije = [sum(energije[100*i:100*i+100])/100 for i in range(int(len(energije)/100))]
    plt.plot(bete,energije,label=r"$J={}, h={}$".format(J,h))      
    J=0
    h=1
    rez = metropolisZ(nsteps,beta,J,h,N,True,0.1,pogostost)
    energije =[E(i,J,h,N) for i in rez]
    #energije = [sum(energije[100*i:100*i+100])/100 for i in range(int(len(energije)/100))]
    plt.plot(bete,energije,label=r"$J={}, h={}$".format(J,h))  
    plt.axhline(-N,bete[0],bete[-1],ls="--")
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel(r"$\beta$",fontsize=14)
    #plt.xticks([i for i in range(1,21)])
    plt.ylabel(r"$E$",fontsize=14)
    plt.title(r"Energija pri ohlajanju. $N=10$",fontsize=13)
    plt.savefig("Figures/energija.pdf")

if 0:
    #avtokorelacija
    cakaj=1000000
    nsteps=1000
    J=1
    h=1
    beta=5
    for N in [10,50,100,200,500]:
        print(beta)
        rez = metropolisKor(nsteps,beta,J,h,N,cakaj)
        rez -= np.average(rez)
        #plt.plot(list(range(len(rez))),rez)
        kor = np.abs(np.fft.ifft(np.fft.fft(rez)*np.conj(np.fft.fft(rez))))
        #kor = np.correlate(rez,np.concatenate((rez,rez)))
        #print(kor)
        kor = kor/max(kor)
        x = [-int(len(kor)/2) + i for i in range(len(kor))]
        plt.plot(x,np.concatenate((kor[int(len(kor)/2):],kor[:int(len(kor)/2)])),label=r"$N={}$".format(N))
        #plt.plot([i for i in range(len(rez))],rez,label=r"$\beta={}$".format(beta))
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel(r"$i$",fontsize=14)
    #plt.xticks([i for i in range(1,21)])
    plt.ylabel(r"$C$",fontsize=14)
    plt.title(r"$C(i)$, $\beta=5$",fontsize=13)
    plt.savefig("Figures/avtokor.pdf")
if 0:
    #koliko casa do povprecja
    J=1
    h=1
    beta = 50
    N=10
    nsteps = 1000000
    cakanja = [10000,100000,1000000,5000000]
    for cakaj in cakanja:
        stara = metropolisBrez(cakaj,beta,J,h,N)
        rez = metropolisKor(nsteps,beta,J,h,N,stara)
        suma=0
        povprecja = []
        for i in range(len(rez)):
            suma += rez[i]
            povprecja.append(suma/(i+1))
        plt.plot(list(range(1,len(povprecja)+1)),povprecja,label=r"$n={}$".format(cakaj))
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel(r"$i$",fontsize=14)
    plt.ylabel(r"$\langle E \rangle$",fontsize=14)
    plt.title(r"Tekoča povprečja energije po relaksaciji $\beta=10,N=30$",fontsize=13)
    plt.savefig("Figures/povp2.pdf")   


if 0:
    #odv od tau za maxa
    N=20
    lamb=0
    tr = 10000000
    h=0.1
    y00=np.random.rand(2*N)
    #y00 = np.ones(2*N+2)
    kraj = [i for i in range(N)]
    for tau in [5,10,15,20,25,30]:
        y0 = rPotek2(y00,N,tr,h,lamb,tau)
        rez = rTemp(y0,N,1000000,h,lamb,tau)
        plt.plot(kraj,rez,"-",label=r"$\tau={}$".format(round(tau*0.1,2)))
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.xlabel("i",fontsize=14)
    plt.xticks([i for i in range(1,21)])
    plt.ylabel("T",fontsize=14)
    plt.title(r"Temperaturni profil od $\tau$. $\lambda=0$",fontsize=13)
    plt.savefig("Figures/maxtau3.pdf")
