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
pi = np.pi

@jit(nopython=True)
def A(psi,stanja,z,N):
    U = genU(z)
    for j in range(1,int(N/2)+1):
        psi = delovanjeU(psi,stanja,U,2*j-1-1,N)
    return psi

@jit(nopython=True)
def B(psi,stanja,z,N):
    U = genU(z)
    for j in range(1,int(N/2)+1):
        psi = delovanjeU(psi,stanja,U,2*j-1,N)
    return psi

@jit(nopython=True)
def s2(psi,stanja,z,N):
    return A(B(A(psi,stanja,z*0.5,N),stanja,z,N),stanja,z*0.5,N)

@jit(nopython=True)
def s4(psi,stanja,z,N):
    x1 = 1/(2-2**(1/3))
    x0 = - 2**(1/3)*x1
    return s2(s2(s2(psi,stanja,x1*z,N),stanja,x0*z,N),stanja,x1*z,N)

@jit(nopython=True)
def trotter(y0,stanja,z,N,koraki,metoda=1,ohrani=False):
    n = len(y0)
    resitve = np.ones((koraki+1,n))*0j
    resitve[0]=y0
    """
    resitve=np.copy(y0)
    """
    for i in range(1,koraki+1):
        """
        if metoda==0:
            if ohrani:
                resitve[i]= s2(resitve[i-1],stanja,z,N)
            else:
                resitve = s2(resitve,stanja,z,N)
        """
        resitve[i]= s4(resitve[i-1],stanja,z,N)
        #resitve = s4(resitve,stanja,z,N)
    return resitve

@jit(nopython=True)
def genU(z):
    U = np.zeros((4,4))*1j
    U[0][0]=np.exp(2*z)
    U[1][1]=np.cosh(2*z)
    U[1][2]=np.sinh(2*z)
    U[2][1]=np.sinh(2*z)
    U[2][2]=np.cosh(2*z)
    U[3][3]=np.exp(2*z)
    return np.exp(-z)*U
    #return np.exp(-z)*np.array([[np.exp(2*z),0,0,0],[0,np.cosh(2*z),np.sinh(2*z),0],[0,np.sinh(2*z),np.cosh(2*z),0],[0,0,0,np.exp(2*z)]])

def genZac(N):
    vmesni = [[0],[1]]
    while len(vmesni) != 2**N:
        novi = []
        for konf in vmesni:
            novi.append(konf + [0])
            novi.append(konf + [1])
        vmesni = [novi[i] for i in range(len(novi))]
    return np.array(vmesni)

@jit(nopython=True)
def toBinary(arej):
    rez = 0
    for i in range(1,len(arej)+1):
        rez += arej[-i]*(2**(i-1))
    return rez

def tok(psi,stanja,j,N):
    psii = sigmax(sigmay(psi,stanja,(j+1)%N),stanja,j) - sigmay(sigmax(psi,stanja,(j+1)%N),stanja,j)
    return psii

def celotenTok(psi,stanja,N):
    n = len(psi)
    psii = 1j*np.zeros(n)
    for i in range(N):
        psii += tok(psi,stanja,i,N)
    return psii

def sigmaz(psi,stanja,j):
    n = len(psi)
    psii = 1j*np.zeros(n)
    for i in range(n):
        psii[i] = psi[i]*(-1)**stanja[i][j]
    return psii

def sigmay(psi,stanja,j):
    n = len(psi)
    psii = 1j*np.zeros(n)
    for i in range(n):
        temp = np.copy(stanja[i])
        temp[j] = 1 if stanja[i][j]==0 else 0
        psii[toBinary(temp)] = 1j*psi[i] if stanja[i][j] == 0 else -1j*psi[i]
    return psii
 
def sigmax(psi,stanja,j):
    n = len(psi)
    psii = 1j*np.zeros(n)
    for i in range(n):
        temp = np.copy(stanja[i])
        temp[j] = 1 if stanja[i][j]==0 else 0
        psii[toBinary(temp)] = psi[i]
    return psii
    
def H(psi,stanja,N):
    psii = 1j*np.zeros(2**N)
    for j in range(N):
        psii += sigmax(sigmax(psi,stanja,(j+1)%N),stanja,j)
        psii +=sigmay(sigmay(psi,stanja,(j+1)%N),stanja,j) 
        psii +=sigmaz(sigmaz(psi,stanja,(j+1)%N),stanja,j)
    return psii


def sigmaz2(psi,stanja,j,N,n):
    #najprej na (j+1%N) pol pa še na j
    psii = 1j*np.zeros(n)
    temp = np.copy(stanja[psi])
    temp[(j+1)%N]=(-1)**stanja[psi][(j+1)%N]
    novpsi=toBinary(temp)
    #psii[novpsi]=1
    temp = np.copy(stanja[novpsi])
    temp[j]=(-1)**stanja[novpsi][j]
    psii[toBinary(temp)] = 1
    return psii


def sigmay2(psi,stanja,j,N,n):
    psii = 1j*np.zeros(n)
    temp = np.copy(stanja[psi])
    temp[(j+1)%N] = 1 if stanja[psi][(j+1)%N]==0 else 0
    novpsi=toBinary(temp)
    #psii[novpsi] = 1j if stanja[psi][(j+1)%N]==0 else -1j
    faktor = 1j if stanja[psi][(j+1)%N]==0 else -1j
    temp=np.copy(stanja[novpsi])
    temp[j] = 1 if stanja[novpsi][j]==0 else 0
    psii[toBinary(temp)] = 1j if stanja[novpsi][j]==0 else -1j
    psii[toBinary(temp)]*=faktor
    #psii[novpsi]=0
    return psii


def sigmax2(psi,stanja,j,N,n):
    psii = 1j*np.zeros(n)
    temp = np.copy(stanja[psi])
    temp[(j+1)%N] = 1 if stanja[psi][(j+1)%N]==0 else 0
    novpsi=toBinary(temp)
    #psii[novpsi] = 1
    temp=np.copy(stanja[novpsi])
    temp[j] = 1 if stanja[novpsi][j]==0 else 0
    psii[toBinary(temp)] = 1
    #psii[novpsi]=0
    return psii

def H2(j,stanja,n,N):
    psii=1j*np.zeros(n)
    for i in range(N):
        psii += sigmax2(j,stanja,i,N,n)
        psii +=sigmay2(j,stanja,i,N,n) 
        psii +=sigmaz2(j,stanja,i,N,n)
    return psii
        


@jit(nopython=True)
def tensorConv(a,b):
    if a==0:
        if b==0:
            return 0
        if b==1:
            return 1
    if b==0:
        return 2
    if b==1:
        return 3
    print(1/0)

@jit(nopython=True)
def delovanjeU(psi,stanja,U,j,N):
    prvi = j
    drugi = (j+1)%N
    psii = np.zeros(2**N)+0j*np.zeros(2**N)
    for i in range(2**N):
        trenutni = np.copy(stanja[i])
        trenutni[prvi]=0
        trenutni[drugi]=0
        psii[toBinary(trenutni)] += U[int(tensorConv(0,0))][int(tensorConv(stanja[i][prvi],stanja[i][drugi]))]*psi[i]
        #a=2+3
        trenutni[drugi]=1
        psii[toBinary(trenutni)] += U[int(tensorConv(0,1))][int(tensorConv(stanja[i][prvi],stanja[i][drugi]))]*psi[i]
        trenutni[prvi]=1
        psii[toBinary(trenutni)] += U[int(tensorConv(1,1))][int(tensorConv(stanja[i][prvi],stanja[i][drugi]))]*psi[i]
        trenutni[drugi]=0
        psii[toBinary(trenutni)] += U[int(tensorConv(1,0))][int(tensorConv(stanja[i][prvi],stanja[i][drugi]))]*psi[i]
    return psii

@jit(nopython=True)
def zrebaj(N):
    n = 2**N
    #sigma = np.sqrt(1/(2*n))
    sigma=1
    cifre = np.random.normal(0,sigma,n)+1j*np.random.normal(0,sigma,n)
    cifre = cifre/np.sqrt(np.sum(np.abs(cifre)**2))
    return cifre

def diagonalizacija(N,stanja):
    n = 2**N
    matrika = 1j*np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            #ket = 1j*np.zeros(n)
            #ket[j]=1
            ket = H2(j,stanja,n,N)
            bra = 1j*np.zeros(n)
            bra[i]=1
            rez = np.vdot(bra,ket)
            matrika[i][j]=rez
            matrika[j][i]=np.conjugate(rez)
    return matrika


"""
N=8
stanja=genZac(N)
matrika=diagonalizacija(N,stanja)
print("test")
print(lin.eigh(matrika,eigvals_only=True,eigvals=(0,0))[0])
"""

@jit(nopython=True)
def Z(beta,N,Npsi,delta,stanja,shrani = False,energija=False):
    n = 2**N
    zrebani = np.ones((Npsi,n))*1j
    for i in range(Npsi):
        zrebani[i] = zrebaj(N)
    #zrebani = [zrebaj(N) for i in range(Npsi)]
    rez = 1j*np.zeros(int(0.5*beta/delta))
    """
    if energija:
        rez2 = 1j*np.zeros(int(0.5*beta/delta))
    if shrani:
        shranjene = []
    """
    for i in range(Npsi):
        psi = trotter(zrebani[i],stanja,-delta,N,int(0.5*beta/delta),ohrani=True)
        """
        if energija:
            psi2 = [H(psi[j],stanja,N) for j in range(len(psi))]
        """
        for j in range(1,int(0.5*beta/delta)+1):
            """
            if shrani:
                continue
            """
            rez[j-1] += np.vdot(psi[j],psi[j])
            """
            if energija:
                rez2[j-1] +=np.vdot(psi[j],psi2[j])
            """
        """
        if shrani:
            rez[-1] += np.vdot(psi[int(0.5*beta/delta)],psi[int(0.5*beta/delta)])
            shranjene.append(rez[-1])
        """
    """
    if energija:
        return rez2/rez
    if shrani:
        return shranjene
    """
    return rez/Npsi


if 0:
    #F(beta) od delta
    Nji = [2,4,6]
    for N in Nji:
        print(N)
        stanja = genZac(N)
        psi = zrebaj(N)
        delte = np.linspace(0.001,0.5,20)
        rez = []
        razvoj = trotter(psi,stanja,-delte[0],N,int(10/delte[0]))
        rez.append(-1/20*np.log(np.vdot(razvoj,razvoj)))
        for delta in delte[1:]:
            razvoj = trotter(psi,stanja,-delta,N,int(10/delta))
            rez.append(np.abs(-1/20*np.log(np.vdot(razvoj,razvoj))-rez[0])/np.abs(rez[0]))
        rez[0]=0
        plt.plot(delte,rez,label=r"$N={}$".format(N))
    """
    for N in range(2,8):
        plt.plot(delte,prva[N-2],label=r"$N={}$".format(N))
    """    
    plt.xlabel(r"$\Delta$",fontsize=14)
    plt.ylabel(r"$\delta F$",fontsize=14)
    plt.legend(loc="best")
    plt.grid()
    #plt.yscale("log")
    plt.title(r"$\delta F(\beta=20), N_\psi=1$ za več korakov v imag. času $\Delta$",fontsize=15)
    #plt.savefig("Figures/napake2.pdf")
    
if 0:
    #Z(beta) od Npsi
    Npsi=50
    Nji = [2]
    for N in Nji:
        print(N)
        stanja = genZac(N)
        shranjene = Z(20,N,Npsi,0.1,stanja,shrani=True)
        fji = []
        for i in range(Npsi):
            fji.append(-1/20*np.log(1/(i+1)*shranjene[i]))
        plt.plot(list(range(1,Npsi+1)),fji,label=r"$N={}$".format(N))
    plt.xlabel(r"$N_\psi$",fontsize=14)
    plt.ylabel(r"$F$",fontsize=14)
    #plt.legend(loc="best")
    plt.grid()
    #plt.yscale("log")
    plt.title(r"$F(\beta=20), \Delta=0.1, N=4$ Odvisnost od velikosti vzorca",fontsize=15)
    #plt.savefig("Figures/napake31.pdf")    
if 0:
    #test - unitarnost
    stanja = genZac(8)
    if 1:
        psi = zrebaj(8)
        delta = 0.01
        razvoj = trotter(psi,stanja,-1j*delta,8,int(10/0.01),ohrani=True)
        rez = [np.vdot(razvoj[i],razvoj[i]) for i in range(int(10/0.01)+1)]
        casi = [t*delta for t in range(int(10/0.01)+1)]
        plt.plot(casi,rez)
        plt.xlabel(r"$t$",fontsize=14)
        plt.ylabel(r"$|\psi|^2$",fontsize=14)
        plt.title(r"$N=8$, Ohranjanje norme s časom",fontsize=15)
        plt.grid(True)
        plt.savefig("Figures/napake1.pdf")
    if 0:
        psi = zrebaj(8)
        delte  = np.linspace(0.001,1,20)
        rez = []
        for delta in delte:
            razvoj = trotter(psi,stanja,-1j*delta,4,int(10/delta))
            rez.append(np.vdot(razvoj,razvoj))
        plt.plot(delte,rez)


if 0:
    #F(beta)
    Nji = [2,4,6,8]
    for N in Nji:
        print(N)
        beta = 30
        delta = 0.1
        stanja = genZac(N)
        Zji = Z(beta,N,5,delta,stanja)
        bete = [delta*t*2 for t in range(1,int(0.5*beta/delta)+1)]
        Fji = [-1/bete[i]*np.log(Zji[i]) for i in range(int(0.5*beta/delta))]
        plt.plot(bete,Fji,label=r"$N={}$".format(N))
    plt.xlabel(r"$\beta$",fontsize=14)
    plt.ylabel(r"$F$",fontsize=14)
    plt.legend(loc="best")
    plt.grid()
    #plt.yscale("log")
    plt.title(r"$F(\beta)$, $N_{\psi}=5, \Delta=0.1$",fontsize=15)
    #plt.savefig("Figures/F.pdf")      


if 0:
    #F od n
    Nji = [2,4,6,8,10]
    bete = [5,10,15,20]
    ffji = []
    for beta in bete:
        Fji = []
        for N in Nji:
            print(N)
            delta = 0.1
            stanja = genZac(N)
            Zji = Z(beta,N,1,delta,stanja)
            Fji.append(-1/beta*np.log(Zji[-1]))
        ffji.append(Fji)
        plt.plot(Nji,Fji,"o-",label=r"$\beta={}$".format(beta))
    plt.xlabel(r"$N$",fontsize=14)
    plt.ylabel(r"$F$",fontsize=14)
    plt.legend(loc="best")
    plt.grid()
    #plt.yscale("log")
    plt.title(r"$F(N)$, $N_{\psi}=5, \Delta=0.1$",fontsize=15)
    plt.savefig("Figures/F2.pdf")   

if 0:
    Nji = [2,4,6,8,10]
    x = np.linspace(10,30,50)
    bete = [5,10,15,20]
    barve = ["blue","orange","green","red"]
    ffji = [[-5.743707161028768, -7.536902612478141, -10.570784556697163, -13.43918122962331, -16.544638464173197],[-5.830072487685258, -7.760588666252494, -10.727625028362693, -13.952135549139152, -17.52285356774639],[-5.860551293562613, -7.702057269031907, -10.88640188015602, -14.241038971131715, -17.648857978329595],[-5.9671255508811925, -7.769412497933139, -10.939147270308219, -14.307162093331298, -17.75176740632597]]
    for beta in range(len(bete)):
        plt.plot(Nji,ffji[beta],color=barve[int(beta)],label=r"$\beta={}$".format(bete[beta]))
        k,n = np.polyfit(Nji[1:],ffji[beta][1:],1)
        plt.plot(x,k*x+n*np.ones(50),"--",color=barve[int(beta)])
    plt.xlabel(r"$N$",fontsize=14)
    plt.ylabel(r"$F$",fontsize=14)
    plt.legend(loc="best")
    plt.grid()
    #plt.yscale("log")
    plt.title(r"$F(N)$, $N_{\psi}=5, \Delta=0.1$",fontsize=15)
    plt.savefig("Figures/F3.pdf") 

if 0:
    #povp H
    Nji = [2,4,6,8]
    for N in Nji:
        beta = 10
        delta = 0.1
        stanja = genZac(N)
        Hji = Z(beta,N,5,delta,stanja,energija=True)
        bete = [delta*t*2 for t in range(1,int(0.5*beta/delta)+1)]
        plt.plot(bete,Hji,label=r"$N={}$".format(N))
    plt.xlabel(r"$\beta$",fontsize=14)
    plt.ylabel(r"$E$",fontsize=14)
    plt.legend(loc="best")
    plt.grid()
    #plt.yscale("log")
    plt.title(r"$E(\beta)$, $N_{\psi}=5, \Delta=0.1$",fontsize=15)
    plt.savefig("Figures/E.pdf") 


if 0:
    #H od n
    Nji = [2,4,6,8,10]
    bete = [1,2,3,4]
    for beta in bete:
        Hji = []
        for N in Nji:
            delta = 0.1
            stanja = genZac(N)
            Hji.append(Z(beta,N,5,delta,stanja,energija=True)[-1])
        plt.plot(Nji,Hji,"o-",label=r"$\beta={}$".format(beta))
    plt.xlabel(r"$N$",fontsize=14)
    plt.ylabel(r"$E$",fontsize=14)
    plt.legend(loc="best")
    plt.grid()
    #plt.yscale("log")
    plt.title(r"$E(N)$, $N_{\psi}=5, \Delta=0.1$",fontsize=15)
    plt.savefig("Figures/E2.pdf")  


if 0:
    #H od n + ekstrap
    Nji = [2,4,6,8,10]
    x = np.linspace(10,30,50)
    bete = [1,2,3,4]
    barve = ["blue","orange","green","red"]
    for beta in bete:
        Hji = []
        for N in Nji:
            print(N)
            delta = 0.1
            stanja = genZac(N)
            Hji.append(Z(beta,N,5,delta,stanja,energija=True)[-1])
        plt.plot(Nji,Hji,color=barve[int(beta-1)],label=r"$\beta={}$".format(beta))
        k,n = np.polyfit(Nji[1:],Hji[1:],1)
        plt.plot(x,k*x+n*np.ones(50),"--",color=barve[int(beta-1)])
    plt.xlabel(r"$N$",fontsize=14)
    plt.ylabel(r"$E$",fontsize=14)
    plt.legend(loc="best")
    plt.grid()
    #plt.yscale("log")
    plt.title(r"$E(N)$, $N_{\psi}=5, \Delta=0.1$",fontsize=15)
    plt.savefig("Figures/E2.pdf")  
       
if 0:
    #korelacija spinski
    delta = 0.05
    Nji = [4,6,8,10]
    docasa=5
    Npsi = 5
    ref = 0
    j=0
    #plt.plot([],[])
    for N in Nji:
        print(N)
        zrebani = [zrebaj(N) for i in range(Npsi)]
        casi = [delta*t for t in range(1,int(docasa/delta)+1)]
        korelacije = 1j*np.zeros(int(docasa/delta))
        stanja = genZac(N)
        for zreban in zrebani:
            krneki = celotenTok(zreban,stanja,N)
            ref+=np.vdot(krneki,krneki)
            stanje1 = celotenTok(zreban,stanja,N)
            for i in range(int(docasa/delta)):
                cas = casi[i]
                if cas==delta:
                    stanje2 = trotter(stanje1,stanja,-1j*delta,N,1)
                    stanje3 = celotenTok(stanje2,stanja,N)
                    stanje4 = trotter(zreban,stanja,-1j*delta,N,1)
                    korelacije[0] += np.vdot(stanje4,stanje3)
                    continue
                stanje2 = trotter(stanje2,stanja,-1j*delta,N,1)
                stanje3 = celotenTok(stanje2,stanja,N)
                stanje4=trotter(stanje4,stanja,-1j*delta,N,1)
                korelacije[i] += np.vdot(stanje4,stanje3)
        for i in range(int(docasa/delta)):
            korelacije[i] = korelacije[i]/(ref)
        #korelacije = np.insert(korelacije,0,1)
        plt.plot(casi,korelacije,label=r"$N={}$".format(N))
    plt.xlabel(r"$t$",fontsize=14)
    plt.ylabel(r"$C(t)$",fontsize=14)
    plt.legend(loc="best")
    plt.grid()
    #plt.yscale("log")
    plt.title(r"$\langle J(t) J \rangle, N_{\psi}=5, \Delta=0.05, j=3$",fontsize=15)
    plt.savefig("Figures/korelacijetok11.pdf")  
    
if 0:
    #neki neki spinski kao
    #korelacija spinski
    delta = 0.05
    Nji = [4,6]
    docasa=5
    Npsi = 5
    j=0
    #plt.plot([],[])
    for N in Nji:
        print(N)
        zrebani = [zrebaj(N) for i in range(Npsi)]
        casi = [delta*t for t in range(int(docasa/delta)+1)]
        korelacije = 1j*np.zeros(int(docasa/delta))
        stanja = genZac(N)
        for zreban in zrebani:
            stanje1 = celotenTok(zreban,stanja,N)
            for i in range(int(docasa/delta)):
                cas = casi[i]
                if cas==0:
                    stanje2 = trotter(stanje1,stanja,-1j*delta,N,1)
                    stanje3 = celotenTok(stanje2,stanja,N)
                    stanje4 = trotter(zreban,stanja,-1j*delta,N,1)
                    korelacije[0] += np.vdot(stanje4,stanje3)
                    continue
                stanje2 = trotter(stanje2,stanja,-1j*delta,N,1)
                stanje3 = celotenTok(stanje2,stanja,N)
                stanje4=trotter(stanje4,stanja,-1j*delta,N,1)
                korelacije[i] += np.vdot(stanje4,stanje3)
        for i in range(int(docasa/delta)):
            korelacije[i] = korelacije[i]/Npsi
        korelacije = np.insert(korelacije,0,1)
        integrali = [trapz(korelacije[:i+1],dx=delta) for i in range(len(korelacije))]
        plt.plot(casi,integrali,label=r"$N={}$".format(N))
    plt.xlabel(r"$t$",fontsize=14)
    #plt.ylabel(r"$C(t)$",fontsize=14)
    plt.legend(loc="best")
    plt.grid()
    #plt.yscale("log")
    plt.title(r"$\int_0^t C(s) ds, N_{\psi}=5, \Delta=0.05, j=3$",fontsize=15)
    plt.savefig("Figures/korelacijetok2.pdf")     
    
if 0:
    #korelacija sigme(z)
    delta = 0.05
    Nji = [6]
    docasa=5
    Npsi = 5
    j=0
    #plt.plot([],[])
    for N in Nji:
        print(N)
        zrebani = [zrebaj(N) for i in range(Npsi)]
        casi = [delta*t for t in range(int(docasa/delta)+1)]
        korelacije = 1j*np.zeros(int(docasa/delta))
        stanja = genZac(N)
        for zreban in zrebani:
            stanje1 = sigmaz(zreban,stanja,j)
            for i in range(int(docasa/delta)):
                cas = casi[i]
                if cas==0:
                    stanje2 = trotter(stanje1,stanja,-1j*delta,N,1)
                    stanje3 = sigmaz(stanje2,stanja,j)
                    stanje4 = trotter(zreban,stanja,-1j*delta,N,1)
                    korelacije[0] += np.vdot(stanje4,stanje3)
                    continue
                stanje2 = trotter(stanje2,stanja,-1j*delta,N,1)
                stanje3 = sigmaz(stanje2,stanja,j)
                stanje4=trotter(stanje4,stanja,-1j*delta,N,1)
                korelacije[i] += np.vdot(stanje4,stanje3)
        for i in range(int(docasa/delta)):
            korelacije[i] = korelacije[i]/Npsi
        korelacije = np.insert(korelacije,0,1)
        plt.plot(casi,korelacije,"--",color="red",label=r"$C(t)_1$")
    j=1
    for N in Nji:
        print(N)
        zrebani = [zrebaj(N) for i in range(Npsi)]
        casi = [delta*t for t in range(int(docasa/delta)+1)]
        korelacije = 1j*np.zeros(int(docasa/delta))
        stanja = genZac(N)
        for zreban in zrebani:
            stanje1 = sigmaz(zreban,stanja,j)
            for i in range(int(docasa/delta)):
                cas = casi[i]
                if cas==0:
                    stanje2 = trotter(stanje1,stanja,-1j*delta,N,1)
                    stanje3 = sigmaz(stanje2,stanja,j)
                    stanje4 = trotter(zreban,stanja,-1j*delta,N,1)
                    korelacije[0] += np.vdot(stanje4,stanje3)
                    continue
                stanje2 = trotter(stanje2,stanja,-1j*delta,N,1)
                stanje3 = sigmaz(stanje2,stanja,j)
                stanje4=trotter(stanje4,stanja,-1j*delta,N,1)
                korelacije[i] += np.vdot(stanje4,stanje3)
        for i in range(int(docasa/delta)):
            korelacije[i] = korelacije[i]/Npsi
        korelacije = np.insert(korelacije,0,1)
        plt.plot(casi,korelacije,"--",color="green",label=r"$C(t)_2$")
        #plt.plot(casi,korelacije,label=r"$N={}$".format(N))
    plt.xlabel(r"$t$",fontsize=14)
    #plt.ylabel(r"$C(t)$",fontsize=14)
    #plt.legend(loc="best")
    plt.grid()
    #plt.yscale("log")
    #plt.title(r"$\langle \sigma^z_3(t) \sigma^z_3 \rangle, N_{\psi}=5, \Delta=0.05, j=3$",fontsize=15)
    #plt.savefig("Figures/korelacije3.pdf")  
    
    plt.title(r"Lokalne magnetizacije in korelacije,  $N=6, N_{\psi}=5, \Delta=0.05$",fontsize=15)
if 0:
    #povprecje in korelacija sigme(z)
    delta = 0.05
    N=6
    docasa=5
    Npsi = 5
    stanja = genZac(N)
    temp = 1j*np.zeros(int(5/0.05)+1)
    j=0
    print(len(casi))
    for zreban in zrebani:
        casovni = trotter(zreban,stanja,-1j*delta,N,int(docasa/delta),ohrani=True)
        for i in range(len(temp)):
            cass = sigmaz(casovni[i],stanja,j)
            temp[i] += np.vdot(casovni[i],cass)
    for i in range(len(temp)):
        temp[i]=temp[i]/Npsi
    plt.plot(casi,temp,color="red",label=r"$\sigma_1^z$")
    j=1
    for zreban in zrebani:
        casovni = trotter(zreban,stanja,-1j*delta,N,int(docasa/delta),ohrani=True)
        for i in range(len(temp)):
            cass = sigmaz(casovni[i],stanja,j)
            temp[i] += np.vdot(casovni[i],cass)
    for i in range(len(temp)):
        temp[i]=temp[i]/Npsi
    plt.plot(casi,temp,color="green",label=r"$\sigma_2^z$")
    
    plt.legend(loc="best")
    plt.savefig("Figures/korelacije5.pdf")
if 0:
    #hitrost
    casidiag = []
    casitrot = []
    Nji = [2,4,6,8,10]
    for N in Nji:
        print(N)
        stanja = genZac(N)
        start = time.time()
        a = diagonalizacija(N,stanja)
        casidiag.append(time.time()-start)
        start = time.time()
        a = Z(10,N,5,0.1,stanja,energija=True)
        casitrot.append(time.time()-start)
    plt.xlabel(r"$N$",fontsize=14)
    plt.ylabel(r"$t[s]$",fontsize=14)
    plt.legend(loc="best")
    plt.plot(Nji,casidiag,"o-",label="Diagonalizacija")
    plt.plot(Nji,casitrot,"o-",label="Trotter-Suzuki")
    plt.legend(loc="best")
    plt.grid()
    #plt.yscale("log")
    plt.title(r"Primerjava časovne zahtevnosti",fontsize=15)
    #plt.savefig("Figures/hitrost.pdf")     

if 0:
    #primerjava z diag.
    energijediag = [-6,-7.99999,-11.2111025509279]
    Nji = [2,4,6]
    energijetrot = []
    for N in Nji:
        stanja = genZac(N)
        a = Z(10,N,5,0.1,stanja,energija=True)
        energijetrot.append(a[-1])
    plt.xlabel(r"$N$",fontsize=14)
    plt.ylabel(r"$t$",fontsize=14)
    plt.legend(loc="best")
    plt.plot(Nji,energijediag,"o-",label="Diagonalizacija")
    plt.plot(Nji,energijetrot,"o-",label="Trotter-Suzuki")
    plt.legend(loc="best")
    plt.grid()
    #plt.yscale("log")
    plt.title(r"Primerjava prvih nekaj energij osnovnih stanj",fontsize=15)
    plt.savefig("Figures/hitrost2.pdf")     
    
    
if 0:
    #lepa slikca
    Z = [[0,0],[0,0]]
    levels = np.linspace(-1,1,20)
    barve = plt.get_cmap("viridis")
    CS3 = plt.contourf(Z, levels, cmap=barve)
    plt.clf()
    casi = [0.1,0.2,0.3,0.4,0.5]
    N = 6
    #zacetni = zrebaj(6)
    zacetni = 1j*np.zeros(2**6)
    zacetni[1]=1/np.sqrt(2)
    zacetni[4]=1/np.sqrt(2)
    stanja = genZac(6)
    povprecnesigme = []
    povprecnitokovi = []
    for j in range(6):
        print(j)
        temp = []
        temp2 = []
        for cas in casi:
            casovni = trotter(zacetni,stanja,-1j*0.01,N,int(cas/0.01))
            temp.append(np.vdot(casovni,sigmaz(casovni,stanja,j)))
            temp2.append(np.vdot(casovni,tok(casovni,stanja,j,N)))
        povprecnesigme.append(temp)
        povprecnitokovi.append(temp2)
    for cas in range(len(casi)):
        plt.plot(np.linspace(0,6,300),np.ones(300)*casi[cas],"--",color="k",lw=0.5)
        sekvence = []
        for spin in range(6):
            #sekvence.append(np.abs(0.5*(povprecnesigme[spin][cas]+1)))
            #print(povprecnesigme[spin][cas])
            plt.plot([spin+1],[casi[cas]],"o",markersize=10,color=barve(np.abs(0.5*(povprecnesigme[spin][cas]+1))))
        #plt.scatter([i for i in range(1,7)],[casi[cas] for i in  range(1,7)],s=100,c=sekvence,cmap=barve)
        plt.quiver([i for i in range(1,7)],[casi[cas]+0.05 for i in range(1,7)],[povprecnitokovi[i][cas] for i in range(6)],[0 for i in range(1,7)])
    a = plt.colorbar(CS3)
    a.set_label(r"$\sigma^z$")
    #plt.xlim(0,0.7)
    #plt.ylim(0,0.6)
    plt.ylabel("t[s]")
    plt.title("Časovna odvisnost magnetizacije in spinskega toka")
    plt.savefig("Figures/zadnja4.pdf")

if 0:
    #lepa slikca od temp
    Z = [[0,0],[0,0]]
    levels = np.linspace(-1,1,20)
    barve = plt.get_cmap("viridis")
    CS3 = plt.contourf(Z, levels, cmap=barve)
    plt.clf()
    bete = [1,2,3,4,5]
    N = 6
    zacetni = zrebaj(6)
    stanja = genZac(6)
    sigme = []
    casi = [t*0.05 for t in range(101)] #od 0 do 5
    for beta in bete:
        temp = []
        casovni = trotter(zacetni,stanja,-0.01,N,int(beta/2 / 0.01))
        casovni2 = trotter(casovni,stanja,-1j*0.05,N,int(5/0.05),ohrani=True)
        sigme.append([np.vdot(a,a) for a in casovni2])
    for beta in range(len(bete)):
        plt.plot(np.linspace(0,5,300),np.ones(300)*bete[beta],"--",color="k",lw=0.5)
        for cas in range(len(casi)):
            plt.plot([casi[cas]],[bete[beta]],"o",color=barve(np.abs(0.5*(sigme[beta][cas]+1))))
    a = plt.colorbar(CS3)
    a.set_label(r"$\sigma^z$")
    plt.ylabel("t[s]")
    plt.title("Časovna odvisnost magnetizacije pri več temperaturah, j=0")
    plt.savefig("Figures/zadnjaneki1.pdf")    
"""
def delovanjeU(psi,stanja,U,j,N):
    prvi = j
    drugi = (j+1)%N
    psii = np.ones(2**N)+0j*np.ones(2**N)
    for i in range(2**N):
        temp = 0+0j
        for k in range(2**N):
            if drugi==N-1:
                if np.array_equal(stanja[i][:prvi],stanja[k][:prvi]):
                    temp += U[tensorConv(stanja[i][prvi],stanja[i][drugi])][tensorConv(stanja[k][prvi],stanja[k][drugi])]*psi[k]
            elif np.array_equal(stanja[i][:prvi],stanja[k][:prvi]) and np.array_equal(stanja[i][drugi+1:],stanja[k][drugi+1:]):
                temp += U[tensorConv(stanja[i][prvi],stanja[i][drugi])][tensorConv(stanja[k][prvi],stanja[k][drugi])]*psi[k]
        psii[i]=temp
    return psii
"""



def hamiltonian(n):
    """
   Heisenberg chain of length n without BC.
   """
    N = 2**n
    binaryFactor = np.array([1, 2, 4])
    row = []
    col = []
    value = []
    for j in range(n-1):
        for i in range(N):
            spin1 = (i % binaryFactor[1]) // binaryFactor[0]
            spin2 = (i % binaryFactor[2]) // binaryFactor[1]
            if spin1 != spin2:
                row.append(i)
                col.append(i)
                value.append(-1)
               
                row.append(i)
                value.append(2)
                if spin1 == 1:
                    col.append((i + binaryFactor[0]) % N)
                else:
                    col.append((i - binaryFactor[0]) % N)
            else:
                row.append(i)
                col.append(i)
                value.append(1)
        binaryFactor *= 2
    return scipy.sparse.coo_matrix((value, (row, col)), shape = (N, N), dtype = np.float64)
       
def hamiltonianPeriodic(n):
    """
   Heisenberg chain of length n with periodic BC
   """
    N = 2**n
    binaryFactor = np.array([1, 2, 4])
    row = []
    col = []
    value = []
    for j in range(n):
        for i in range(N):
            if j == n-1:
                spin1 = i // binaryFactor[0]
                spin2 = (i % 2)
                cyclicCorrection = 1
            else:
                spin1 = (i % binaryFactor[1]) // binaryFactor[0]
                spin2 = (i % binaryFactor[2]) // binaryFactor[1]
                cyclicCorrection = 0
            if spin1 != spin2:
                row.append(i)
                col.append(i)
                value.append(-1)
               
                row.append(i)
                value.append(2)
                if spin1 == 1:
                    col.append((i + binaryFactor[0] + cyclicCorrection) % N)
                else:
                    col.append((i - binaryFactor[0] - cyclicCorrection) % N)
            else:
                row.append(i)
                col.append(i)
                value.append(1)
        binaryFactor *= 2
    return scipy.sparse.coo_matrix((value, (row, col)), shape = (N, N), dtype = np.float64)
 
def directDiagBase(hamiltonianMatrix):
    """
   Returns base state energy and coefficients
   """
    base = scipy.sparse.linalg.eigsh(hamiltonianMatrix, which = 'SA')
    return base[0][0], base[1].T[0]






