# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 21:35:07 2018

@author: Admin
"""
import timeit
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
import scipy.sparse
from scipy.optimize import root
from scipy.integrate import quad
from scipy.integrate import romb
from scipy.integrate import complex_ode
from scipy.integrate import simps
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

def difKorak(psi0,h,tau,pot):
    return psi0 + 1j*tau*(1/(2*h*h)*(np.roll(psi0,-1) + np.roll(psi0,1)-2*psi0)-pot*psi0)

def propKorak(psi0,h,tau,pot,K):
    temp = np.copy(psi0)
    final = psi0
    for k in range(1,K+1):
        temp = -0.5*(np.roll(temp,-1)+np.roll(temp,1)-2*temp)/(h*h) + pot*temp
        final = final + (-1j*tau)**k/np.math.factorial(k)*temp
    return final

def uKorak(psi0,h,tau,pot,matrika):
    interm = psi0 + 1j*tau*0.25*((np.roll(psi0,1)+np.roll(psi0,-1)-2*psi0)/(h*h) - 2*pot*psi0)
    resitev = solve_banded((1,1),matrika,interm)
    #resitev = TDMAsolver(matrika[2],matrika[1],matrika[0],interm)
    return resitev
def genPsi0(N,x):
    A = pi**(0.25)*np.sqrt(2**N * np.math.factorial(N))
    hermit = np.zeros(N+1)
    hermit[-1]=1
    psi0 = 1/A*np.polynomial.hermite.Hermite(hermit)(x)*np.exp(-x*x/2)
    psi0[0]=0
    psi0[-1]=0
    return psi0

def narediMatriko(tau,h,l,pot):
    #za implicitno
    pomozni = np.ones(l-1)*(-1j*tau*0.25/(h*h)) +np.zeros(l-1)
    prva = np.insert(pomozni,0,0)
    druga = np.ones(l)*(1+1j*tau*0.5/(h*h)) + 1j*tau*0.5*pot
    tretja = np.append(pomozni,0)
    matrika = np.vstack((prva,druga,tretja))    
    return matrika 

def psi02D(x,y,a):
    return np.exp(-(x-a)*(x-a)/2)*np.exp(-(y)*(y)/2)/np.sqrt(pi)
def pot2D(x,y,lamb):
    return 0.5*(x*x+y*y) +lamb *x*x*y*y

def korak2D(psi0,h,tau,pot,K):
    temp = np.copy(psi0)
    final = psi0
    for k in range(1,K+1):
        temp = -0.5*(np.roll(temp,-1,0)+np.roll(temp,1,0) + np.roll(temp,-1,1) + np.roll(temp,1,1) -4*temp)/(h*h) + pot*temp
        final = final + (-1j*tau)**k/np.math.factorial(k)*temp
    return final


def solver2D(N,h,tau,n,L,lamb,a,korak):
    x = np.linspace(-L,L,int((2*L)/h))
    y = np.linspace(-L,L,int(2*L/h))
    l = len(x)
    evolucija = np.ones((n,l,l))+1j*np.ones((n,l,l))
    psi0 = np.array([[psi02D(i,j,a) for i in x] for j in y]) + 0j*np.ones((l,l))
    psi0[0]=np.zeros(l)
    psi0[-1]=np.zeros(l)
    psi0[:,0]=np.zeros(l)
    psi0[:,-1]=np.zeros(l)
    evolucija[0]= psi0
    
    pot = np.array([[pot2D(i,j,lamb) for i in x] for j in y])
    for i in range(1,n*korak):
        psi0= korak2D(psi0,h,tau,pot,10)
        psi0[0]=np.zeros(l)
        psi0[-1]=np.zeros(l)
        psi0[:,0]=np.zeros(l)
        psi0[:,-1]=np.zeros(l)
        if i%korak == 0:
            print(i)
            evolucija[int(i//korak)] = psi0
    return (x,np.absolute(evolucija)**2)

def solver(N,h,tau,n,L,lamb,metoda,a,K=10,prirastek=0,sunek=0,resonanca=0):
    x = np.linspace(-L,L,int((2*L)/h))
    l = len(x)
    evolucija = np.ones((n,l))+1j*np.ones((n,l))
    evolucija[0] = genPsi0(N,x-a) 
    #evolucija[0]=genPsi0(N,x)
    #pot = 1/2*x*x + lamb*x*x*x*x
    if metoda==2:
        if prirastek != 0:
            pot = 1/2*x*x
        else:
            pot=1/2*x*x + lamb*x*x*x*x
        matrika = narediMatriko(tau,h,l,pot)
        
    for i in range(1,n):
        if prirastek != 0:
            pot = 1/2*x*x + prirastek*i*x*x*x*x
            matrika = narediMatriko(tau,h,l,pot)
        if sunek !=0:
            if i==20:
                pot = 1/2*x*x + sunek* np.exp(-x*x/(2*0.2))
                matrika = narediMatriko(tau,h,l,pot)
            if i==121:
                pot = 1/2*x*x
                matrika = narediMatriko(tau,h,l,pot)
        if resonanca != 0:
            pot = 1/2*(x+6*np.sin((2*pi)/resonanca*i*tau))**2
        if metoda==0:
            evolucija[i] = difKorak(evolucija[i-1],h,tau,pot)
        elif metoda==1:
            evolucija[i] = propKorak(evolucija[i-1],h,tau,pot,K)
        elif metoda==2:
            evolucija[i] = uKorak(evolucija[i-1],h,tau,pot,matrika)
        evolucija[i][0]=0
        evolucija[i][-1]=0
    return (x,np.absolute(evolucija)**2)


if 0:
    #2d
    tau = 0.001
    h = 0.1
    n = 200
    korak = 50
    L = 7
    lamb = 1
    a = 4
    fig = plt.figure()
    temp = solver2D(0,h,tau,n,L,lamb,a,korak)
    rezultati = temp[1]
    maxi = np.amax(rezultati)
    mini = np.amin(rezultati)
    x = temp[0]
    X, Y = np.meshgrid(x,x)
    
    print("evo")
    #potencial = 0.5*x*x + lamb*x*x*x*x
    
    def animiraj(t):
        print(t)
        plt.clf()
        plt.title(r"$\lambda ={}, a=4, t={}$".format(lamb,round(t*tau*korak,5)))
        CS = plt.contourf(X,Y,rezultati[t],levels=np.linspace(mini,maxi,50),cmap="hot")
        CB = plt.colorbar(CS)
        #ax.set_ylim(0,maxy)
    ani = animation.FuncAnimation(fig,animiraj,range(n),interval=100)   
    #plt.show()
    ani.save("anim/Tretjaa4Lamb1.mp4")  
    print("evo")
if 0:
    #vec paketkov
    #animacija druge sunki/lambde
    resonanca = 6.3
    tau = 0.0005
    N = 0
    fig, ax = plt.subplots()
    axi = ax.twinx()
    #lambde = [0,0.4,0.8,1]
    #lambdenakorak = [1/50000,1/25000,1/10000,1/1000]
    #lambdanacas = [i*0.0005 for i in lambdenako
    temp = solver(N,0.01,0.0005,25000,25,0,2,resonanca=6.3)
    rezultati = temp[1]
    x = temp[0]
    maxy = np.amax(rezultati)
    print("evo")
    #potencial = 0.5*x*x + lamb*x*x*x*x
    
    def animiraj(t):
        print(t)
        ax.clear()
        axi.clear()
        ax.set_title(r"$t={}$".format(round(t*0.05,5)))
        ax.plot(x,rezultati[t*100],color="b")
        #ax.set_ylim(0,maxy)
        potencial =1/2*(x+6*np.sin((2*pi)/resonanca*t*100*tau))**2
        axi.plot(x,potencial,"r--")
    ani = animation.FuncAnimation(fig,animiraj,range(250),interval=100)   
    #plt.show()
    ani.save("anim/nektest3.mp4")  
    print("evo")
if 0:
    #animacija druge sunki/lambde
    N = 0
    fig, ax = plt.subplots(2,2,figsize=(10,10))
    axi = [ax[0][0],ax[0][1],ax[1][0],ax[1][1]]
    axii = [axi[0].twinx(),axi[1].twinx(),axi[2].twinx(),axi[3].twinx()]
    rezultati = []
    #lambde = [0,0.4,0.8,1]
    #lambdenakorak = [1/50000,1/25000,1/10000,1/1000]
    #lambdanacas = [i*0.0005 for i in lambdenakorak]
    sunki = [10,30,60,100]
    for n in range(4):
        if n==0:
            temp = solver(N,0.01,0.0005,50000,10,0,2,sunek=sunki[n])
            rezultati.append(temp[1])
            x = temp[0]
            continue
        rezultati.append(solver(N,0.01,0.0005,50000,10,0,2,sunek=sunki[n])[1])
    maxy = [np.amax(i) for i in rezultati]
    print("evo")
    #potencial = 0.5*x*x + lamb*x*x*x*x
    
    def animiraj(t):
        print(t)
        for i in axi:
            i.clear()
        for i in axii:
            i.clear()
        plt.suptitle(r"$t={}$".format(round(t*0.05,5)))
        for i in range(4):
            axi[i].set_title(r"$A={}$".format(round(sunki[i],2)))
            axi[i].plot(x,rezultati[i][100*t],color="b")
            axi[i].set_ylim(0,maxy[i])
            potencial = 0.5*x*x
            if t==1:
                potencial = 0.5*x*x + sunki[i]*np.exp(-x*x/(2*0.2))                
            axii[i].plot(x,potencial,"r--")
    ani = animation.FuncAnimation(fig,animiraj,range(500),interval=100)   
    #plt.show()
    ani.save("anim/DrugaSunek22.mp4")  
    print("evo")
    

if 0:
    #pomoč
    rezultati = solver(0,0.01,0.0001,10000,5,0.1,1)
    a = rezultati[1][0]
    print(np.sum(a)*0.01)
    b = rezultati[1][-1]
    print(np.sum(b)*0.01)

if 0:
    #razni plotti
    skok=1
    lamb=0.2
    tau=0.1
    h=0.1
    resitev = solver(2,h,tau,10,5,lamb,2,10)
    iksi = resitev[0]
    casi = resitev[1][::skok]
    plt.title(r"$|\Psi|^2, h=0.1, \tau =0.1, N=2, \lambda = 0.2$")
    plt.xlabel(r"$x$")
    barva = np.linspace(0,1,len(casi))
    cmap = plt.get_cmap("viridis")
    for i in range(len(casi)):
        print(i)
        if i==0:
            plt.plot(iksi,casi[i],color=cmap(barva[i]),label=r"$n=0$")
            continue
        elif i==len(casi)-1:
            plt.plot(iksi,casi[i],color=cmap(barva[i]),label=r"$n={}$".format(int(len(casi))*skok))
            continue            
        plt.plot(iksi,casi[i],color=cmap(barva[i]))
    plt.legend(loc="best")
    plt.savefig("figures/u2.pdf")
if 0:
    #napaka metod
    h=0.1
    N=2
    lamb=0.2
    L = 10
    cas = 5
    vmesni = solver(N,h,0.0001,int(cas/0.0001),L,lamb,2)
    print("smotle")
    referenca = vmesni[1][-1]
    x = vmesni[0]
    l = len(referenca)
    if 0:
        #odvisnost od tau
        taui = np.linspace(0.0001,0.01,30)
        metoda11 = np.ones(30)
        metoda12= np.ones(30)
        metoda2= np.ones(30)
        for i in range(30):
            print(i)
            temp = solver(N,h,taui[i],int(cas/taui[i]),L,lamb,1,10)[1][-1]
            metoda11[i] = np.sum(np.abs(temp-referenca))/l
            temp = solver(N,h,taui[i],int(cas/taui[i]),L,lamb,1,20)[1][-1]
            metoda12[i] = np.sum(np.abs(temp-referenca))/l
            temp = solver(N,h,taui[i],int(cas/taui[i]),L,lamb,2)[1][-1]
            metoda2[i] = np.sum(np.abs(temp-referenca))/l        
        plt.title(r"Povprečna napaka v odvisnosti od $\tau$. $h=0.1, N=3, \lambda=0.2,T=5, L=5$")
        plt.xlabel(r"$\tau$")
        plt.xlim(0.0001,0.01)
        plt.plot(taui,metoda11,label=r"$K=10$")
        plt.plot(taui,metoda12,label=r"$K=20$")
        plt.plot(taui,metoda2, label="Unitarna")
        plt.legend(loc="best")
        #plt.yscale("log")
        plt.savefig("Figures/napaka1.pdf")
        
    if 1:
        #odvisnost od L
        referenca = referenca[int((10-3)/h):int((10-3)/h)+int(6/h)]
        l = len(referenca)
        lji = [3,3.2,3.4,3.6,3.8,4,4.2,4.4,4.6,4.8,5,5.2,5.4,5.6,5.8,6,6.2,6.4,6.6,6.8,7,7.2,7.4,7.6,7.8,8,8.2,8.4,8.6,8.8,9,9.2,9.4,9.6,9.8,10]
        metoda11 = np.ones(36)
        metoda12= np.ones(36)
        metoda2= np.ones(36)
        for i in range(36):
            print(i)
            temp = solver(N,h,0.0005,10000,lji[i],lamb,1,10)[1][-1][int((lji[i]-3)/h):int((lji[i]-3)/h)+int(6/h)]
            metoda11[i] = np.sum(np.abs(temp-referenca))/l
            temp = solver(N,h,0.0005,10000,lji[i],lamb,1,20)[1][-1][int((lji[i]-3)/h):int((lji[i]-3)/h)+int(6/h)]
            metoda12[i] = np.sum(np.abs(temp-referenca))/l
            temp = solver(N,h,0.0005,10000,lji[i],lamb,2)[1][-1][int((lji[i]-3)/h):int((lji[i]-3)/h)+int(6/h)]
            metoda2[i] = np.sum(np.abs(temp-referenca))/l
            
        plt.title(r"Povprečna napaka v odvisnosti od $L$. $h=0.1, N=3, \lambda=0.2,T=5, \tau=0.0005$")
        plt.xlabel(r"$L$")
        plt.plot(lji,metoda11,label=r"$K=10$")
        plt.plot(lji,metoda12,label=r"$K=20$")
        plt.plot(lji,metoda2, label="Unitarna")
        plt.legend(loc="best")
        #plt.yscale("log")
        plt.savefig("Figures/napaka2.pdf")

if 0:
    lamb = 0.001
    n=2000*3
    #podrobneje druga
    rezultati = solver(0,0.01,0.0005,n,15,lamb,2,10)
    x = rezultati[0]
    zaplottat = rezultati[1][::600]
    cmap = plt.get_cmap("ocean")
    barve = np.linspace(0,0.9,10)
    for i in range(10):
        plt.plot(x,zaplottat[i],color=cmap(barve[i]),label=r"$t={}$".format(round(0.0005*i*600,2)))
    plt.legend(loc="best")    
    plt.title(r"$N=0, a=10, \lambda = {}$".format(lamb))
    plt.savefig("figures/Druga9.pdf")
                

if 0:
    #animacija prve naloge
    N = 0
    fig, ax = plt.subplots(3,2,figsize=(30,20))    
    axi = [ax[0][0],ax[0][1],ax[1][0],ax[1][1],ax[2][0],ax[2][1]]
    axi0 = axi[0].twinx()
    axi1 = axi[1].twinx()
    axi2 = axi[2].twinx()
    axi3 = axi[3].twinx()
    axi4 = axi[4].twinx()
    axi5 = axi[5].twinx()
    axii = [axi0,axi1,axi2,axi3,axi4,axi5]    
    rezultati = []
    #lambde = [0,0.2,0.4,0.6,0.8,1]
    nji = [0,1,2,3,4,5]
    for n in range(6):
        if n==0:
            temp = solver(nji[n],0.01,0.0005,50000,7,0.5,2)
            rezultati.append(temp[1])
            x = temp[0]
            continue
        rezultati.append(solver(nji[n],0.01,0.0005,50000,7,0.5,2)[1])
    maxy = [np.amax(i) for i in rezultati]
    print("evo")
    potencial = 0.5*x*x + 0.5*x*x*x*x
    
    def animiraj(t):
        print(t)
        for i in axi:
            i.clear()
        for i in axii:
            i.clear()
        plt.suptitle(r"$\lambda = {}, t={}$".format(0.5,round(t*0.05,2)),fontsize=30)
        for i in range(6):
            axi[i].set_title(r"$N={}$".format(nji[i]),fontsize=22)
            axi[i].plot(x,rezultati[i][100*t],color="b")
            axi[i].set_ylim(0,maxy[i])
            #potencial = 0.5*x*x+ lambde[i]*x*x*x*x
            axii[i].plot(x,potencial,"r--")
    ani = animation.FuncAnimation(fig,animiraj,range(500),interval=100)   
    #plt.show()
    ani.save("anim/PrvaLambdePol.mp4")
    

    
    
    
    
    
    
    
    
    
