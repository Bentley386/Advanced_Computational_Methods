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

def portret(E,lamb,obhod=False):
    def hamilton(t,y):
        #y = (x,p)
        return np.array([y[1],-y[0]-4*lamb*y[0]**3])
    y0 = np.array([0,np.sqrt(2*E)])
    r = ode(hamilton).set_integrator("dopri5").set_initial_value(y0,0)
    resitve = np.zeros((10001,2))
    h = 0.0005
    resitve[0] = y0
    for i in range(1,10001):
        resitve[i]=r.integrate(r.t+h)
        if obhod and i>200:
            if np.abs(resitve[i][0])<0.2 and resitve[i][1]>0:
                print(i)
                return resitve[:i]
    return resitve
def uKorak(psi0,h,tau,pot,matrika):
    interm = psi0 + 1j*tau*0.25*((np.roll(psi0,1)+np.roll(psi0,-1)-2*psi0)/(h*h) - 2*pot*psi0)
    resitev = solve_banded((1,1),matrika,interm)
    #resitev = TDMAsolver(matrika[2],matrika[1],matrika[0],interm)
    return resitev
def genPsi0(N,x):
    A = pi**(0.25)*np.math.sqrt(2**N * np.math.factorial(N))
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

def solver(N,h,tau,n,L,lamb):
    x = np.linspace(-L,L,int((2*L)/h))
    l = len(x)
    evolucija = np.ones((n,l))+1j*np.ones((n,l))
    evolucija[0] = genPsi0(N,x) 
    #evolucija[0]=genPsi0(N,x)
    pot = 1/2*x*x + lamb*x*x*x*x
    matrika = narediMatriko(tau,h,l,pot)
        
    for i in range(1,n):
        evolucija[i] = uKorak(evolucija[i-1],h,tau,pot,matrika)
        evolucija[i][0]=0
        evolucija[i][-1]=0
    return (x,np.absolute(evolucija)**2)
    
def Psi(x,c):
    n = len(c)
    coef = c*np.sqrt(np.array([1/(2**i * np.math.factorial(i)) for i in range(n)]))
    hermitovi = np.polynomial.hermite.Hermite(coef)
    return 1/(pi**0.25)*hermitovi(x)*np.exp(-x*x/2)
    
def Vij(x,i,j):
    leva = genPsi0(i,x)
    desna = genPsi0(j,x)
    return np.trapz(leva*x*x*x*x*desna,x)

def casovniRazvoj(psi0,vektorji,vrednosti,x,t):
    N = len(vrednosti)
    casovniFaktor = np.array([np.exp(-1j*t*vrednosti[i]) for i in range(N)])
    koeficienti = np.array([np.dot(psi0,vektorji[:,k]) for k in range(N)])
    #koeficienti = np.array([np.trapz(psi0*Psi(x,vektorji[:,i]),x) for i in range(N)])
    koeficienti = koeficienti*casovniFaktor
    vrednost = np.zeros(len(x))+1j*np.zeros(len(x))
    for i in range(N):
        vrednost = vrednost + koeficienti[i]*Psi(x,vektorji[:,i])
    return (x,np.abs(vrednost)**2)

def analiticenH(N,lamb):
    qji = np.zeros((N,N))
    matrika = np.zeros((N,N))
    for i in range(N):
        matrika[i][i] = i + 1/2 
        for j in range(i):
            if abs(i-j)==1:
                qji[i][j]=0.5*np.sqrt(i+j+1)
                qji[j][i]=0.5*np.sqrt(i+j+1)
    return matrika + lamb*np.linalg.matrix_power(qji,4)

def numericenH(N,lamb,x):
    matrika = np.zeros((N,N))
    for i in range(N):
        matrika[i][i] = i + 1/2 + lamb*Vij(x,i,i) 
        for j in range(i):
                el = lamb*Vij(x,i,j)
                matrika[i][j]=el
                matrika[j][i]=el
    return matrika

def pertEn(n,lamb,N):
    qji = np.zeros((N,N))
    for i in range(N):
        for j in range(i):
            if abs(i-j)==1:
                qji[i][j] = 0.5*np.sqrt(i+j+1)
                qji[j][i] = 0.5*np.sqrt(i+j+1)
    qji = lamb*np.linalg.matrix_power(qji,4)

    prvi = qji[n][n]
    drugi = 0
    for k in range(N):
        if k == n:
            continue
        drugi += np.abs(qji[k][n])**2/(n-k)
    return (prvi,drugi)

def pertFun(n,lamb,N,x):
    qji = np.zeros((N,N))
    for i in range(N):
        for j in range(i):
            if abs(i-j)==1:
                qji[i][j] = 0.5*np.sqrt(i+j+1)
                qji[j][i] = 0.5*np.sqrt(i+j+1)
    qji = lamb*np.linalg.matrix_power(qji,4)
    koef = []
    for k in range(N):
        if k==n:
            koef.append(0)
            continue
        koef.append(qji[k][n]/(n-k))
    funk = genPsi0(n,x) + Psi(x,koef)
    return np.abs(funk)**2
"""
def lanczos(zacetna,H,m):
    n = len(H)
    vji = np.ones((n,m))
    vji[:,0] = zacetna
    alfe = np.ones(m)
    bete = np.ones(m-1)
    for i in range(0,m):
        if i==0:
            temp = np.matmul(H,zacetna)
            alfe[0] = np.dot(temp,zacetna)
            w = temp - alfe[0]*zacetna
            continue
        bete[i-1] = lin.norm(w)
        if bete[i-1] == 0:
            raise ValueError("Nekaj se je zjebalo")
        vji[:,i] = w/bete[i-1]
        temp = np.matmul(H,vji[:,i])
        alfe[i] = np.dot(temp,vji[:,i])
        w = temp - alfe[i]*vji[:,i] - bete[i-1]*vji[:,i-1]
    return (alfe,bete,vji)
"""    
def lanczos(zacetna,H):
    N = len(H)
    vji = np.ones((N,N))
    vji[0] = zacetna
    alfe = np.ones(N)
    bete = np.ones(N-1)
    for i in range(0,N):
        if i==0:
            temp = np.matmul(H,zacetna)
            alfe[0] = np.dot(temp,zacetna)
            w = temp - alfe[0]*zacetna
            continue
        bete[i-1] = lin.norm(w)
        if bete[i-1] == 0:
            raise ValueError("Nekaj se je zjebalo")
        vji[i] = w/bete[i-1]
        temp = np.matmul(H,vji[i])
        alfe[i] = np.dot(temp,vji[i])
        w = temp - alfe[i]*vji[i] - bete[i-1]*vji[i-1]
    prehodna= np.transpose(vji)
    return (alfe,bete,prehodna)

def area(v):
    a = 0
    x0,y0 = v[0]
    for i in range(1,len(v)):
        dx = v[i][0]-x0
        dy = v[i][1]-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = v[i][0]
        y0 = v[i][1]
    return a


if 0:
    #primerjava s pertrubacijo - funkcije
    x = np.linspace(-5,5,10000)
    n = 1
    fig, ax = plt.subplots(2,2,True,True)
    H = analiticenH(100,0.01)
    vrednost = lin.eigh(H)[1][n]
    ax[0][0].plot(x,np.abs(Psi(x,vrednost))**2,color="blue",ls="-")
    ax[0][0].plot(x,pertFun(n,0.01,1000,x),color="blue",ls="--")
    ax[0][0].set_title(r"$\lambda = 0.01$")

    H = analiticenH(100,0.025)
    vrednost = lin.eigh(H)[1][n]
    ax[0][1].plot(x,np.abs(Psi(x,vrednost))**2,color="green",ls="-")
    ax[0][1].plot(x,pertFun(n,0.025,1000,x),color="green",ls="--")
    ax[0][1].set_title(r"$\lambda = 0.025$")
    
    H = analiticenH(100,0.05)
    vrednost = lin.eigh(H)[1][n]
    ax[1][0].plot(x,np.abs(Psi(x,vrednost))**2,color="magenta",ls="-")
    ax[1][0].plot(x,pertFun(n,0.05,1000,x),color="magenta",ls="--")
    ax[1][0].set_title(r"$\lambda = 0.05$")

    H = analiticenH(100,0.1)
    vrednost = lin.eigh(H)[1][n]
    ax[1][1].plot(x,np.abs(Psi(x,vrednost))**2,color="cyan",ls="-")
    ax[1][1].plot(x,pertFun(n,0.1,1000,x),color="cyan",ls="--")
    ax[1][1].set_title(r"$\lambda = 0.1$")
    plt.savefig("Figures/pert3.pdf")    

if 0:
    #primerjava s pertrubacijo - energije
    n = 1
    lambde = np.linspace(0,0.5,100)
    numericno = []
    prvired = []
    drugired=[]
    for lamb in lambde:
        H = analiticenH(100,lamb)
        vrednost = lin.eigh(H)[0][n]
        numericno.append(vrednost)
        prvi, drugi = pertEn(n,lamb,1000)
        prvired.append(n+1/2+prvi)
        drugired.append(prvired[-1]+drugi)
    plt.plot(lambde,numericno,label="Numerična vrednost")
    plt.plot(lambde,prvired,label="Prvi red pert.")
    plt.plot(lambde,drugired,label="Drugi red pert.")
    plt.legend(loc="best")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$E$")
    plt.title(r"Primerjava pertrubacijske teorije z numeriko $n=1$")
    a = plt.axes([0.2, 0.2, .25, .25])
    plt.plot(lambde[:20],numericno[:20])
    plt.plot(lambde[:20],prvired[:20])
    plt.plot(lambde[:20],drugired[:20])
    plt.yticks([])
    plt.savefig("Figures/pert2.pdf")
if 1:
    #portret
    #print(89.2/(pi*2*50.5))
    #print(153.0389/(pi*2*100.5))
    #print(259.2/(pi*2*200.5))
    #print(874.7/(pi*2*1000.5)) 0.14
    #print(4929.4/(pi*2*10000.5)) #0.08
    #print(16521.46/(pi*2*50000.5)) 0.05 nisem preveru
    #print(55569.4/(pi*2*100000.5))
    if 1:
        #r(N)
        N = [100*i for i in range(1,1000)]
        rji = []
        for n in N:
            E = n+1/2
            resitev=portret(E,1,True)
            rji.append(area(resitev)/(pi*2*E))
        plt.xlabel(r"$N$")
        plt.ylabel(r"$r$")
        plt.title(r"Delež konvergiranih v odv. od N $\lambda=1$")
        plt.plot(N,rji)
        plt.savefig("Figures/delez.pdf")
    if 0:
        resitev = portret(100000,1,True)
        print(area(resitev))
        #plt.plot(resitev[:,0],resitev[:,1])
    if 0:
        N = 10001
        energije = [n+1/2 for n in range(N)][::1000]
        for en in energije:
            resitev = portret(en,0)
            plt.plot(resitev[:,0],resitev[:,1],color="b",lw=1)
        energije = np.linspace(10000,10100,10)
        cmap = plt.get_cmap("hot")
        barve = np.linspace(0.1,0.9,10)
        for i in range(10):
            resitev = portret(energije[i],1)
            plt.plot(resitev[:,0],resitev[:,1],color=cmap(barve[i]),label=r"$E={}$".format(round(energije[i],2)))
        plt.legend(loc="best")
        plt.title(r"Semiklasični portret $N=1001, \lambda=1$")
        plt.xlabel(r"$x$")
        #plt.axes().set_aspect(1)
        plt.ylabel(r"$p$")
        plt.savefig("Figures/portret5.pdf")

if 0:
    #napaka casnovih razvojev
    L = 5
    h = 0.01
    tau = 0.0001
    x = np.linspace(-L,L,int(2*L/h))
    lambde = np.linspace(0,1,25)
    napake = []
    #psi0 = genPsi0(0,x)
    psi0 = np.zeros(100)
    psi0[0]=1
    for lamb in lambde:
        rezultati1 = solver(0,h,tau,30000,L,lamb)[1][::5000][4]
        H = analiticenH(100,lamb)
        vrednosti, vektorji = lin.eigh(H)
        rezult = casovniRazvoj(psi0,vektorji,vrednosti,x,4*0.5)[1]
        napake.append(np.sum(np.abs(rezultati1-rezult))/len(x))
    plt.title(r"Povprečna razlika valovnih funkcij obeh metod pri $t=2$")
    plt.plot(lambde,napake,".--")
    plt.xlabel(r"$\lambda$")
    plt.savefig("Figures/razvojnapake.pdf")
    
if 0:
    #casovna primerjava
    L = 5
    h = 0.01
    tau = 0.0001
    x = np.linspace(-L,L,int(2*L/h))
    lamb = 0.5
    casi = np.linspace(0.5,5,20)
    hitrost1 = []
    hitrost2 = []
    psi0 = genPsi0(0,x)
    for cas in casi:
        start = time.time()
        rezultati1 = solver(0,h,tau,int(cas/tau),L,lamb)
        hitrost1.append(time.time()-start)
        start = time.time()
        H = analiticenH(100,lamb)
        vrednosti, vektorji = lin.eigh(H)
        rezult = casovniRazvoj(psi0,vektorji,vrednosti,x,cas)[1]
        hitrost2.append(time.time()-start)
    plt.title(r"Čas izvajanja časovnega razvoja do $T$")
    plt.plot(casi,hitrost1,".--",label="Implicitna")
    plt.plot(casi,hitrost2,".--",label="Spektralna")
    plt.legend(loc="best")
    plt.ylabel("t[s]")
    plt.xlabel(r"$T$")
    plt.savefig("Figures/razvojcas.pdf")
if 0:
    #casovni razvoj
    lamb = 2
    L = 5
    h = 0.01
    tau = 0.0001
    x = np.linspace(-L,L,int(2*L/h))
    if 0:
        rezultati1 = solver(0,h,tau,50000,L,lamb)
        for i in range(5):
            plt.plot(x,rezultati1[1][::5000][i],label=r"$t={}$".format(round(i*5000*tau,2)))
    if 0:
        #psi0 = genPsi0(0,x)
        psi0 = np.zeros(50)
        psi0[0]=1
        H = analiticenH(50,lamb)
        vrednosti, vektorji = lin.eigh(H)
        for i in range(1):
            rezult = casovniRazvoj(psi0,vektorji,vrednosti,x,2)[1]
            plt.plot(x,rezult,label=r"$t={}$".format(round(i*0.5,2)))
    plt.title(r"Nkeaj")
    plt.legend(loc="best")
    plt.xlabel(r"$x$")
    #plt.savefig("Figures/razvoj11.pdf")

def jeNot(x,arej,eps):
    for i in arej:
        if np.abs(x-i)<eps:
            return True
    return False
    
def stKonv(arej1,arej2,eps):
    zesprobano = []
    stevilo = 0
    for i in arej1:
        for j in arej2:
            if np.abs(i-j)<eps and i not in zesprobano:
                stevilo += 1
                zesprobano.append(i)
    return stevilo

if 0:
    #lanczos konv(N)
    x=np.linspace(-10,10,10000)
    xx = np.linspace(0,20,1000)
    H = analiticenH(5000,1)
    a = lin.eigh(H)[0][:501]
    Nji = list(range(20,100,1))
    skonvergirane = []
    for N in Nji:
        print(N)
        H = numericenH(N,1,x)
        zacetna = np.zeros(N)
        zacetna[0] = 1
        zacetna[1]=1
        zacetna = zacetna/lin.norm(zacetna)
        alfe, bete, prehodna = lanczos(zacetna,H)
        b = lin.eigh_tridiagonal(alfe,bete)[0]
        skonvergirane.append(stKonv(a,b,0.001))
    plt.title(r"Število konvergiranih energij $\chi_0 = \phi_0 + \phi_1, \lambda=1 \epsilon=0.001$")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$n$")
    plt.plot(Nji,skonvergirane,"o--")
    plt.savefig("Figures/Lanczos31.pdf")
    
    

if 0:
    #lanczos
    x=np.linspace(-10,10,10000)
    xx = np.linspace(0,20,1000)
    H = analiticenH(2000,1)
    a = lin.eigh(H)[0][:41]
    #zacetna = np.random.rand(5000)
    #zacetna = zacetna/lin.norm(zacetna)
    H = numericenH(100,1,x)
    zacetna = np.zeros(100)
    zacetna[0] = 1
    zacetna[1]=1
    zacetna = zacetna/lin.norm(zacetna)
    alfe, bete, prehodna = lanczos(zacetna,H)
    b = lin.eigh_tridiagonal(alfe,bete)[0][:21]
    n = list(range(21))
    plt.title(r"Lastne vrednosti Lanczosa $\chi_0 = \phi_1+\phi_0, N=100, \lambda=1 \epsilon=0.1$")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$E$")
    plt.plot(n,b,"o--",color="magenta")
    for i in a:
        if jeNot(i,b,0.1):
            plt.plot(xx,np.ones(1000)*i,color="green",ls="--",lw=1)
            continue
        plt.plot(xx,np.ones(1000)*i,color="red",ls="--",lw=1)
    plt.savefig("Figures/Lanczos3.pdf")

if 0:
    #E(lambda)
    lambde = np.linspace(0,10,100)
    x = np.linspace(-10,10,10000)
    temp = [[],[],[]]
    for lamb in lambde:
        print(lamb)
        H = analiticenH(500,lamb)
        lastne = lin.eigh(H)[0]
        temp[0].append(lastne[0])
        temp[1].append(lastne[1])
        temp[2].append(lastne[2])
    plt.plot(lambde,temp[0],"-",label=r"$n = 0$")
    plt.plot(lambde,temp[1],"-",label=r"$n = 1$")
    plt.plot(lambde,temp[2],"-",label=r"$n = 2$")

    plt.title(r"$E(\lambda)$")
    plt.ylabel(r"$E$")
    plt.xlabel(r"$\lambda$")
    plt.legend(loc="best")
    plt.savefig("Figures/Eodlamb.pdf")    
if 0: 
    #E(n)
    nji = np.array(range(501))
    x = np.linspace(-10,10,10000)
    lambde = [0,0.25,0.5,0.75,1]
    for lamb in lambde:
        print(lamb)
        if lamb==0:
            plt.plot(nji,nji+1/2*np.ones(501),"-",label=r"$\lambda = 0.0$")
            continue
        H = analiticenH(10000,lamb)
        #zacetna = np.zeros(400)
        #zacetna[0]=1
        #zacetna = np.array([1/20 if i<400 else 0 for i in range(1000)])
        #alfe,bete,prehodna = lanczos(zacetna,H)
        temp = lin.eigh(H)[0]
        plt.plot(nji,temp[:501],"-",label=r"$\lambda = {}$".format(round(lamb,2)))
    plt.title(r"$E(n)$")
    plt.ylabel(r"$E$")
    plt.xlabel(r"$n$")
    plt.legend(loc="best")
    plt.savefig("Figures/Eodn.pdf")
        
if 0:
    #plotti lastnih funkcij in energij
    i = 0
    N = 50
    x = np.linspace(-5,5,5000)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.set_zorder(-9)
    ax.set_zorder(-10)
    lambde = np.linspace(0,1,5)
    for lamb in lambde:
        H = analiticenH(N,lamb)
        resitev = lin.eigh(H)
        funkcija = Psi(x,resitev[1][:,i])
        ax.plot(x,np.ones(5000)*resitev[0][i],"--",lw=1,label=r"$\lambda = {}$".format(round(lamb,2)))
        ax2.plot(x,np.abs(funkcija)**2)
    ax.set_title(r"Osnovno stanje za več vrednosti parametra $\lambda, N=50$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$E$")
    leg = ax.legend(loc="upper right")
    #leg.set_zorder(5)
    plt.savefig("Figures/funkcije0.pdf")
if 0:
    #napaka
    lamb = 5
    x = np.linspace(-10,10,10000)
    if 0:
        #eigh
        H1 = numericenH(50,lamb,x)
        lastne1 = lin.eigh(H1)
        napake1 = []
        for i in range(50):
            napake1.append(np.sum(np.abs(np.matmul(H1,lastne1[1][:,i]) - lastne1[0][i]*lastne1[1][:,i])))
        x = list(range(50))
        plt.plot(x,napake1)
        plt.title(r"Test delovanja funkcije eigh, $N=50, \lambda = 1$")
        plt.ylabel(r"$||Hc - Ec||_1$")
        plt.xlabel("n")
        plt.savefig("Figures/PrviTest.pdf")
    if 1:
        #velikost matrike
        prve = []
        prvee = []
        Nji = list(range(5,51))
        for N in Nji:
            H = numericenH(N,lamb,x)
            HH= analiticenH(N,lamb)
            tempp = lin.eigh(HH)[0]
            temp = lin.eigh(H)[0]
            prve.append(temp[4])
            prvee.append(tempp[4])
        plt.plot(Nji,prve,color="k", ls="-",label="Numericen mat. el.")
        plt.plot(Nji,prvee,color="k", ls="--",label="Analiticen mat. el.")
        plt.title(r"Primerjava pete lastne vrednosti $\lambda = 5$")
        plt.legend(loc="best")
        plt.xlabel(r"$N$")
        #plt.xticks([2,20,40,60,80,100])
        plt.ylabel(r"$E$")
        plt.savefig("Figures/DrugiTest4.pdf")
    if 0:        
        prve = []
        prvee = []
        hji = np.linspace(0.001,0.5,100)
        for h in hji:
            x = np.linspace(-10,10,int(20/h))
            H = numericenH(40,lamb,x)
            prve.append(lin.eigh(H)[0][9])
        plt.plot(hji,prve,color="r", ls="-")
        plt.title(r"Odvisnost devete lastne vrednosti od h $N=40, \lambda = 1$")
        plt.xlabel(r"$h$")
        #plt.xticks([2,20,40,60,80,100])
        plt.ylabel(r"$E$")
        plt.savefig("Figures/TretjiTest.pdf")
#y = Psi(x,lastne[1][0])
#plt.plot(x,np.abs(y)**2)


