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
       
def directDiagBase(hamiltonianMatrix):
    """
   Returns base state energy and coefficients
   """
    base = scipy.sparse.linalg.eigsh(hamiltonianMatrix, which = 'SA')
    return base[0][0], base[1].T[0]

def genZac(N):
    vmesni = [[0],[1]]
    while len(vmesni) != 2**N:
        novi = []
        for konf in vmesni:
            novi.append(konf + [0])
            novi.append(konf + [1])
        vmesni = [novi[i] for i in range(len(novi))]
    return np.array(vmesni)

#converta arej v stevilko
@jit(nopython=True)
def toBinary(arej):
    rez = 0
    for i in range(1,len(arej)+1):
        rez += arej[-i]*(2**(i-1))
    return rez        
            
    

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


def A(Bji,lambde,U,M,N):
    """
    nBji = np.copy(Bji)
    nlambde = np.copy(lambde)
    """
    for j in range(1,int(N/2)+1):
        Bji, lambde = delovanjeVrat(Bji,lambde,U,2*j-2,M)
        #UU=np.kron(np.kron(np.eye(2*j-1-1),U),np.eye(N-2*j))
        #psi = np.matmul(UU,psi)
    return Bji, lambde

def B(Bji,lambde,U,M,N):
    #nBji = np.copy(Bji)
    #nlambde = np.copy(lambde)
    for j in range(1,int((N-1)/2)+1):
        Bji,lambde = delovanjeVrat(Bji,lambde,U,2*j-1,M)
        #UU = np.kron(np.kron(np.eye(2*j-1),U),np.eye(N-1-2*j))
        #psi = np.matmul(UU,psi)
    return Bji, lambde

def s2(Bji,lambde,M,z,N):
    ##nBji = np.copy(Bji)
    #nlambde = np.copy(lambde)
    U1 = genU(z/2)
    U2 = genU(z)
    Bji, lambde = A(Bji,lambde,U1,M,N)
    Bji, lambde = B(Bji,lambde,U2,M,N)
    Bji, lambde = A(Bji,lambde,U1,M,N)
    return Bji, lambde
    #return A(B(A(psi,stanja,z*0.5,N),stanja,z,N),stanja,z*0.5,N)

def s4(Bji, lambde, M, z,N):
    #nBji = np.copy(Bji)
    #nlambde = np.copy(lambde)
    x1 = 1/(2-2**(1/3))
    x0 = - 2**(1/3)*x1
    Bji, lambde = s2(Bji,lambde,M,x1*z,N)
    Bji, lambde = s2(Bji,lambde,M,x0*z,N)
    Bji, lambde = s2(Bji,lambde,M,x1*z,N)
    return Bji,lambde
    #return s2(s2(s2(psi,stanja,x1*z,N),stanja,x0*z,N),stanja,x1*z,N)

#podatki=[]

def trotter(Bji, lambde, M,z,koraki,N,ohrani=False):
    #nBji = np.copy(Bji)
    #nlambde = np.copy(lambde)
    for i in range(1,koraki+1):
        Bji, lambde = s2(Bji, lambde, M,z,N)
        """
        if i%10==0:
            podatki.append([Bji,lambde])
        """
    return Bji, lambde

#zreba koeficiente
@jit(nopython=True)
def zrebaj(N):
    n = 2**N
    #sigma = np.sqrt(1/(2*n))
    sigma=1
    #cifre = np.random.normal(0,sigma,n)
    cifre = np.random.normal(0,sigma,n)+1j*np.random.normal(0,sigma,n)
    cifre = cifre/np.sqrt(np.sum(np.abs(cifre)**2))
    return cifre


def dobiKoef(Aji,n):
    stanja = genZac(n)
    koef = np.zeros(2**n)*1j
    for i in range(2**n):
        matrika = Aji[0][stanja[i][0]]
        for j in range(1,n):
            matrika = np.matmul(matrika,Aji[j][stanja[i][j]])
        koef[i]=matrika
    return koef

def dobiKoef2(Bji,lambde,n):
    stanja = genZac(n)
    koef = np.zeros(2**n)*1j
    for i in range(2**n):
        matrika = np.matmul(Bji[0][stanja[i][0]],np.diag(lambde[0]))
        for j in range(1,n-1):
            matrika = np.matmul(matrika,Bji[j][stanja[i][j]])
            matrika = np.matmul(matrika,np.diag(lambde[j]))
        matrika = np.matmul(matrika,Bji[-1][stanja[i][-1]])
        koef[i]=matrika
    return koef

def prviKorak(psi,N):
    U,s,Vh = svd(np.reshape(psi,(2,N//2)),False) 

    global Napaka
    counter = 0
    for i in range(len(s)):
        if s[i]<10**(-7):
            counter+=1
    if counter>0:
        ss = s.size
        s = s[:ss-counter]
        U = U[:,:ss-counter]
        Vh = Vh[:ss-counter]

    return [[U[0],U[1]],np.matmul(np.diag(s),Vh),len(s),s] #Aja , psi

Napaka=0


def splosniKorak(psi,N,k):
    l = psi.size
    U, s, Vh = svd(np.reshape(psi,(2*k,l//(2*k))),False)
    """
    if odrezi<100:
        global Napaka 
        ss=s.size
        print(ss)
        Napaka = np.sum(np.abs(s[ss-odrezi:])**2)
        s = s[:ss-odrezi]
        U = U[:,:ss-odrezi]
        Vh = Vh[:ss-odrezi]
    """
    global Napaka
    counter = 0
    for i in range(len(s)):
        if s[i]<10**(-7):
            #print("A")
            counter+=1
            s[i]==10**(-5)
    if counter>0:
        ss = s.size
        s = s[:ss-counter]
        U = U[:,:ss-counter]
        Vh = Vh[:ss-counter]
    return [[U[::2],U[1::2]],np.matmul(np.diag(s),Vh),len(s),s] #Aja, psi, k

def zadnjiKorak(psi,N):
    #temp = np.reshape(psi,(2,2))
    return [psi[:,0],psi[:,1]]
    #return [np.transpose(temp[:,0]),np.transpose(temp[:,1])]

def dobiA(psi,n,tudiLamb=False):
    Aji = []
    N=2**n
    for i in range(n):
        if i == 0:
            temp = prviKorak(psi,N)
            if tudiLamb:
                Aji.append([temp[0],temp[-1]])
            else:
                Aji.append(temp[0])
            continue
        elif i==1:
            temp = splosniKorak(temp[1],N,temp[2])
            if tudiLamb:
                Aji.append([temp[0],temp[-1]])
            else:
                Aji.append(temp[0])
            continue
        if i==n-1:
            temp = zadnjiKorak(temp[1],N)
            Aji.append(temp)
            break
        """
        if i==6:
            temp = splosniKorak(temp[1],N,temp[2],odrezi)
        """
        temp = splosniKorak(temp[1],N,temp[2])
        if tudiLamb:
            Aji.append([temp[0],temp[-1]])
        else:
            Aji.append(temp[0])
    return Aji


def AtoT(Aji1,Aji2):
    Tji=[]
    for i in range(1,len(Aji1)-1):
        Tji.append(np.kron(Aji1[i][0],Aji2[i][0])+np.kron(Aji1[i][1],Aji2[i][1]))
    L = np.kron(Aji1[0][0],Aji2[0][0])+np.kron(Aji1[0][1],Aji2[0][1])
    L = np.reshape(L,(1,L.size))
    R = np.kron(Aji1[-1][0],Aji2[-1][0])+np.kron(Aji1[-1][1],Aji2[-1][1])
    R = np.reshape(R,(R.size,1))
    return L, R, Tji





def AtoB(Aji):
    Bji = [[Aji[0][0][0],Aji[0][0][1]]]
    lambde = [Aji[0][1]]
    for i in range(1,len(Aji)-1):
        Bji.append([np.matmul(np.diag(lambde[i-1]**(-1)),Aji[i][0][0]),np.matmul(np.diag(lambde[i-1]**(-1)),Aji[i][0][1])])
        lambde.append(Aji[i][1])
    Bji.append([np.matmul(np.diag(lambde[-1]**(-1)),Aji[-1][0]),np.matmul(np.diag(lambde[-1]**(-1)),Aji[-1][1])])
    return Bji, lambde

def BtoA(Bji, lambde):
    Aji = [[Bji[0][0],Bji[0][1]]]
    for i in range(1,len(Bji)):
        Aji.append([np.matmul(np.diag(lambde[i-1]),Bji[i][0]),np.matmul(np.diag(lambde[i-1]),Bji[i][1])])
    return Aji

def delovanjeVrat(Bji,lambde,U,j,M):
    B00 = U[0][0]*np.matmul(np.matmul(Bji[j][0],np.diag(lambde[j])),Bji[j+1][0])
    B01 = U[1][1]*np.matmul(np.matmul(Bji[j][0],np.diag(lambde[j])),Bji[j+1][1]) + U[1][2]*np.matmul(np.matmul(Bji[j][1],np.diag(lambde[j])),Bji[j+1][0])
    B10 = U[2][1]*np.matmul(np.matmul(Bji[j][0],np.diag(lambde[j])),Bji[j+1][1]) + U[2][2]*np.matmul(np.matmul(Bji[j][1],np.diag(lambde[j])),Bji[j+1][0])
    B11 = U[3][3]*np.matmul(np.matmul(Bji[j][1],np.diag(lambde[j])),Bji[j+1][1])
    #print(np.vstack((np.hstack((B00,B01)),np.hstack((B10,B11)))))
    if j==0:
        B00 = np.reshape(B00,(1,B00.size))
        B01 = np.reshape(B01,(1,B01.size))
        B10 = np.reshape(B10,(1,B10.size))
        B11 = np.reshape(B11,(1,B11.size))

        B00 = np.matmul(B00,np.diag(lambde[j+1]))
        B01 = np.matmul(B01,np.diag(lambde[j+1]))
        B10 = np.matmul(B10,np.diag(lambde[j+1]))
        B11 = np.matmul(B11,np.diag(lambde[j+1]))
        Q = np.zeros((2,len(lambde[j+1])*2))*1j
        Qlevi = np.zeros((2,len(lambde[j+1])))*1j
        Qdesni = np.zeros((2,len(lambde[j+1])))*1j

    elif j==len(Bji)-2:
        B00 = np.reshape(B00,(B00.size,1))
        B01 = np.reshape(B01,(B01.size,1))
        B10 = np.reshape(B10,(B10.size,1))
        B11 = np.reshape(B11,(B11.size,1))
        
        B00 = np.matmul(np.diag(lambde[j-1]),B00)
        B01 = np.matmul(np.diag(lambde[j-1]),B01)
        B10 = np.matmul(np.diag(lambde[j-1]),B10)
        B11 = np.matmul(np.diag(lambde[j-1]),B11)
        Q = np.zeros((len(lambde[j-1])*2,2))*1j
        Qlevi = np.zeros((len(lambde[j-1])*2,1))*1j
        Qdesni = np.zeros((len(lambde[j-1])*2,1))*1j        

    else:
        B00 = np.matmul(np.matmul(np.diag(lambde[j-1]),B00),np.diag(lambde[j+1]))
        B01 = np.matmul(np.matmul(np.diag(lambde[j-1]),B01),np.diag(lambde[j+1]))
        B10 = np.matmul(np.matmul(np.diag(lambde[j-1]),B10),np.diag(lambde[j+1]))
        B11 = np.matmul(np.matmul(np.diag(lambde[j-1]),B11),np.diag(lambde[j+1]))
        Q = np.zeros((len(lambde[j-1])*2,len(lambde[j+1])*2))*1j
        Qlevi = np.zeros((len(lambde[j-1])*2,len(lambde[j+1])))*1j
        Qdesni = np.zeros((len(lambde[j-1])*2,len(lambde[j+1])))*1j 
    Qlevi[::2] = B00
    Qlevi[1::2] = B10
    Qdesni[::2] = B01
    Qdesni[1::2] = B11
    Q[:,::2] = Qlevi
    Q[:,1::2] = Qdesni
    U, s, Vh = svd(Q,False)
    counter = 0
    for i in range(s.size):
        if s[i] < 10**(-7):
            counter+=1
    if counter > 0:
        ss = s.size
        s = s[:ss-counter]
        U = U[:,:ss-counter]
        Vh = Vh[:ss-counter]
    if j==0:
        j0 = U[::2]
        j1 = U[1::2]
        jp0 = np.matmul(Vh[:,::2],np.diag(lambde[j+1]**(-1)))
        jp1 = np.matmul(Vh[:,1::2],np.diag(lambde[j+1]**(-1)))
        lj=s
    elif j==len(Bji)-2:
        j0 = np.matmul(np.diag(lambde[j-1]**(-1)),U[::2])
        j1 = np.matmul(np.diag(lambde[j-1]**(-1)),U[1::2])
        jp0 = Vh[:,::2]
        jp1 = Vh[:,1::2]
        lj=s        
    else:
        j0 = np.matmul(np.diag(lambde[j-1]**(-1)),U[::2])
        j1= np.matmul(np.diag(lambde[j-1]**(-1)),U[1::2])
        jp0 = np.matmul(Vh[:,::2],np.diag(lambde[j+1]**(-1)))
        jp1= np.matmul(Vh[:,1::2],np.diag(lambde[j+1]**(-1)))
        lj=s
    #nBji = np.copy(Bji)
    """
    counter = 0
    for i in range(lj.size):
        if lj[i] < 10**(-7):
            counter+=1
    if counter > 0:
        j0 = j0[:,:lj.size-counter]
        j1 = j1[:,:lj.size-counter]
        jp0 = j0[:lj.size-counter]
        jp1 = jp1[:lj.size-counter]
        lj = lj[:lj.size-counter]
    """
    Bji[j][0]=j0
    Bji[j][1]=j1
    Bji[j+1][0]=jp0
    Bji[j+1][1]=jp1
    #nlambde = np.copy(lambde)
    lambde[j]=lj
    if lambde[j].size>M:
        lambde[j] = lambde[j][:M]
        Bji[j][0] = Bji[j][0][:,:M]
        Bji[j][1] = Bji[j][1][:,:M]
        Bji[j+1][0] = Bji[j+1][0][:M]
        Bji[j+1][1] = Bji[j+1][1][:M]

    return Bji, lambde
    """
    if j==0:
        Bji[j][0] = U[::2]
        Bji[j][1] = U[1::2]
        Bji[j+1][0] = np.matmul(Vh[:,::2],np.diag(lambde[j+1]**(-1)))
        Bji[j+1][1] = np.matmul(Vh[:,1::2],np.diag(lambde[j+1]**(-1)))
        lambde[j]=s
    elif j==len(Bji)-2:
        Bji[j][0] = np.matmul(np.diag(lambde[j-1]**(-1)),U[::2])
        Bji[j][1] = np.matmul(np.diag(lambde[j-1]**(-1)),U[1::2])
        Bji[j+1][0] = Vh[:,::2]
        Bji[j+1][1] = Vh[:,1::2]
        lambde[j]=s        
    else:
        Bji[j][0] = np.matmul(np.diag(lambde[j-1]**(-1)),U[::2])
        Bji[j][1] = np.matmul(np.diag(lambde[j-1]**(-1)),U[1::2])
        Bji[j+1][0] = np.matmul(Vh[:,::2],np.diag(lambde[j+1]**(-1)))
        Bji[j+1][1] = np.matmul(Vh[:,1::2],np.diag(lambde[j+1]**(-1)))
        lambde[j]=s
    if lambde[j].size>M:
        print("a")
        global Napaka
        Napaka+= np.sum(np.abs(lambde[j][M:])**2)
        lambde[j] = lambde[j][:M]
        Bji[j][0] = Bji[j][0][:,:M]
        Bji[j][1] = Bji[j][1][:,:M]
        Bji[j+1][0] = Bji[j+1][0][:M]
        Bji[j+1][1] = Bji[j+1][1][:M]
    """   
    
    
    return Bji, lambde

def BjitoFinal(Bji,lambde):
    #dobi bji in lambde in zracuna normo funkcije
    Aji = BtoA(Bji,lambde)
    L, R, Tji = AtoT(np.conj(Aji),Aji)
    koef = np.matmul(L,Tji[0])
    for i in range(1,len(Tji)):
        koef = np.matmul(koef,Tji[i])
    koef = np.matmul(koef,R)
    return koef


if 0:
    #test s propagatorjem
    n=4
    hammy =hamiltonian(n)
    e0, psi0= directDiagBase(hammy)
    Aji = dobiA(psi0,n,True)
    Bji,lambde=AtoB(Aji)
    Bji, lambde = trotter(Bji,lambde,100,-1j*0.01,100,n)
    Aji = BtoA(Bji,lambde)
    koef = dobiKoef(Aji,n)
    print(koef)
    print(np.exp(-1j*e0)*psi0)

if 0:
    #potreben M v odv. od n
    plt.plot([6,8,10,12,14],[2,13,22,51,94],".:")
    plt.ylabel(r"$M$",fontsize=14)
    plt.xlabel(r"$n$",fontsize=14)
    plt.title(r"Potreben M za $\delta ||\psi||^2 < 0.1$",fontsize=13)
    plt.savefig("rezanje3.pdf")      
    #za 14 je 94, 12 je 51, 10 22 , 8 13, 6 2

if 0:
    #odvisnost od M
    n=14
    psi=zrebaj(n)
    Aji=dobiA(psi,n,True)
    koef0 = dobiKoef(dobiA(psi,n),n)
    #Mji = [i*10 for i in range(1,21)]
    Mji = [i for i in range(90,100)]
    Napaka=0
    napake = []
    Napake = []
    for M in Mji:
        print(M)
        Bji,lambde = AtoB(Aji)
        Bji,lambde = trotter(Bji,lambde,M,-0.1j,100,n)
        koef = dobiKoef2(Bji,lambde,n)
        napake.append(np.abs(np.vdot(koef,koef)-1))
        Napake.append(Napaka)
        Napaka=0
        
        
    plt.plot(Mji,napake,".:",label=r"$\delta ||\psi||^2$")    
    plt.plot(Mji,Napake,".:",label=r"$\sum |\lambda|^2$")    
    plt.legend(loc="best")
    plt.xlabel(r"$M$",fontsize=14)
    plt.title(r"Napaka v odvisnosti od rezanja, $n=10$",fontsize=13)
    #plt.savefig("rezanje2.pdf")    

if 0:
    #E(n)
    nji = [12,14,16,18,20]
    energije = []
    energijeHammy=[]
    fig, ax = plt.subplots()
    for n in nji:
        print(n)
        M=100
        psi = zrebaj(n)
        Aji = dobiA(psi,n,True)
        Bji, lambde = AtoB(Aji)
        Bji,lambde=trotter(Bji,lambde,M,-0.1,50,n)
        vrednosti=[]
        for j in range(10):
            Bji,lambde=trotter(Bji,lambde,M,-0.01,10,n)
            print(Bji[3][0].shape)
            Aji = BtoA(Bji,lambde)
            koef = dobiKoef(Aji,n)
            vrednosti.append(np.log(float(np.abs(np.vdot(psi,koef)))))
        energije.append(-np.polyfit([5+k*0.1 for k in range(1,11)],vrednosti,1)[0])
    fig,ax = plt.subplots()
    ax.plot(nji,energije,"-",color="blue")
    koef = np.polyfit(nji,energije,1)
    x = np.linspace(20,50)
    plt.plot(x,koef[0]*x+koef[1],"-.",color="blue")
    ax.set_xlabel(r"$n$",fontsize=14)
    ax.set_ylabel("$E_0$",fontsize=14)
    ax.set_title(r"Energije osnovnega stanja AFM",fontsize=13)
    ax.legend(loc="best")
    plt.savefig("energije5.pdf")            
if 0:
    #primerjava z diagonalizacijo
    nji = [4,6,8,10,14]
    energije = []
    energijeHammy=[]
    fig, ax = plt.subplots()
    for n in nji:
        print(n)
        M=300
        psi = zrebaj(n)
        Aji = dobiA(psi,n,True)
        Bji, lambde = AtoB(Aji)
        Bji,lambde=trotter(Bji,lambde,M,-0.1,50,n)
        vrednosti=[]
        for j in range(10):
            Bji,lambde=trotter(Bji,lambde,M,-0.01,10,n)
            Aji = BtoA(Bji,lambde)
            koef = dobiKoef(Aji,n)
            vrednosti.append(np.log(float(np.abs(np.vdot(psi,koef)))))
        energije.append(-np.polyfit([5+k*0.1 for k in range(1,11)],vrednosti,1)[0])
        
        hammy = hamiltonian(n)
        energijeHammy.append(directDiagBase(hammy)[0])
    fig,ax = plt.subplots()
    ax.plot(nji,energije,".-",label="TBED")
    ax.plot(nji,energijeHammy,".-",label="Diag.")
    ax2 = ax.twinx()
    ax2.plot(nji,np.abs(np.array(energije)-np.array(energijeHammy)),"--",color="green")
    ax.set_xlabel(r"$n$",fontsize=14)
    ax.set_ylabel("$E_0$",fontsize=14)
    ax.set_title(r"Primerjava z diagonalizacijo",fontsize=13)
    ax.legend(loc="center left")
    plt.savefig("energije4.pdf")    

if 0:
    #fiksen n, fitti za vec beta0
    n = 10
    psi = zrebaj(n)
    Aji0 = dobiA(psi,n,True)
    barve = ["red","purple","brown"]
    bete0=[1,2,3]
    fig,axi = plt.subplots(1,3)
    for beta0 in bete0:
        print(beta0)
        energije = []
        M=200
        Bji, lambde = AtoB(Aji0)
        Bji,lambde=trotter(Bji,lambde,M,-0.1,10*beta0,n)
        vrednosti=[]
        for j in range(10):
            Bji,lambde=trotter(Bji,lambde,M,-0.01,10,n)
            Aji = BtoA(Bji,lambde)
            koef = dobiKoef(Aji,n)
            vrednosti.append(np.log(float(np.abs(np.vdot(psi,koef)))))
        koeff=-np.polyfit([beta0+ k*0.1 for k in range(1,11)],vrednosti,1)
        koef = koeff[0]
        iks = [beta0+k*0.1 for k in range(1,11)]
        axi[beta0-1].plot(iks,np.array(vrednosti),"o",color=barve[beta0-1])
        x = np.linspace(beta0+0.1,beta0+1,100)
        axi[beta0-1].plot(x,-koef*x,"--",color=barve[beta0-1])
        axi[beta0-1].plot(x,-koef*x-koeff[1],color=barve[beta0-1])
        axi[beta0-1].set_title(r"$\beta_0={}$".format(str(beta0)))
        axi[beta0-1].grid(True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle(r"Primerjava fitta s podatki, $n=10$",fontsize=13)
    plt.savefig("energije3.pdf")    

if 0:
    #en od beta0
    nji = [4,6,8,10,12,14]
    for n in nji:
        print(n)
        energije = []
        M=200
        psi = zrebaj(n)
        Aji = dobiA(psi,n,True)
        Bji, lambde = AtoB(Aji)
        Bji,lambde=trotter(Bji,lambde,M,-0.1,10,n)
        for i in range(10):
            vrednosti = []
            for j in range(10):
                Bji,lambde=trotter(Bji,lambde,M,-0.01,10,n)
                Aji = BtoA(Bji,lambde)
                koef = dobiKoef(Aji,n)
                vrednosti.append(np.log(float(np.abs(np.vdot(psi,koef)))))
            energije.append(-np.polyfit([i+1 + k*0.1 for k in range(1,11)],vrednosti,1)[0])
        plt.plot([k for k in range(1,11)],energije,".:",label=r"$n={}$".format(str(n)))
    plt.grid(True)
    plt.xlabel(r"$\beta_0$",fontsize=14)
    plt.ylabel("$E_0$",fontsize=14)
    plt.title(r"Pridobljena energija v odvisnosti od zacetne $\beta_0$",fontsize=13)
    plt.legend(loc="best")
    plt.savefig("energije2.pdf")


if 0:
    #test
    M=200
    n=16
    psi = zrebaj(n)
    Aji =dobiA(psi,n,True)
    Bji, lambde = AtoB(Aji)
    Bji,lambde = trotter(Bji,lambde,M,-0.1j,100,n)
    koef = dobiKoef2(Bji,lambde,n)
    print(np.vdot(koef,koef))
if 0:
    #preverimo normo
    n=10
    psi = zrebaj(n)
    Aji = dobiA(psi,n,True)
    Bji,lambde = AtoB(Aji)
    norme = [np.vdot(psi,psi)-1]
    for i in range(100):
        Bji , lambde = trotter(Bji,lambde,100,-0.1j,10,n)
        koef = dobiKoef2(Bji,lambde,n)
        norme.append(np.vdot(koef,koef)-1)
    
    plt.plot([i for i in range(101)],norme)
    plt.grid(True)
    plt.xlabel("t",fontsize=14)
    plt.ylabel("$||\psi||-1$",fontsize=14)
    plt.title(r"Napaka norme v odv. od časa, $n=10$",fontsize=13)
    plt.tight_layout()
    plt.savefig("norma.pdf")
    

if 0:
    #<sigma_j sigma_k> za vse j in k   kot navadenplot
    def povprecenjk(Aji,L,R,Tji,j,k):
        Vj = np.kron(Aji[j][0],Aji[j][0])-np.kron(Aji[j][1],Aji[j][1])
        Vk = np.kron(Aji[k][0],Aji[k][0])-np.kron(Aji[k][1],Aji[k][1])
        if j==1:
            koef = np.matmul(L,Vj)
        elif k==1:
            koef = np.matmul(L,Vk)
        else:
            koef = np.matmul(L,Tji[0])
        for i in range(1,len(Tji)):
            if i==j-1:
                koef = np.matmul(koef,Vj)
                continue
            if i==k-1:
                koef = np.matmul(koef,Vk)
                continue
            koef = np.matmul(koef,Tji[i])
        return np.matmul(koef,R)
    n=8
     #osnovno stanje
    hammy = hamiltonian(n)
    psi = directDiagBase(hammy)[1]
    Aji = dobiA(psi,n)
    os = [i for i in range(2,n)]
    L, R, Tji = AtoT(np.conj(Aji),Aji)
    korelacije = np.zeros((n-2,n-2))
    for j in range(1,n-1):
        print(j)
        korelacije = []
        for k in range(1,n-1):
            if k==j:
                korelacije.append(1)
                continue
            korelacije.append(povprecenjk(Aji,L,R,Tji,j,k))
        plt.plot(os,korelacije,".:",label=r"$j={}$".format(str(j+1)))
            
    plt.legend(loc="upper center",ncol=3)
    plt.grid(True)
    plt.xlabel("k",fontsize=14)
    plt.ylabel(r"$\langle \sigma^z_j \sigma^z_k \rangle$",fontsize=14)
    plt.title(r"$\langle \sigma^z_j \sigma^z_k \rangle$, $n=8$",fontsize=13)
    plt.savefig("korelacije2.pdf")
    
if 0:
    #<sigma_j sigma_k> za vse j in k   kot contourplot 
    def povprecenjk(Aji,L,R,Tji,j,k):
        """sigmaz
        Vj = np.kron(Aji[j][0],Aji[j][0])-np.kron(Aji[j][1],Aji[j][1])
        Vk = np.kron(Aji[k][0],Aji[k][0])-np.kron(Aji[k][1],Aji[k][1])
        """
        """sigmax
        Vj = np.kron(Aji[j][0],Aji[j][1])+np.kron(Aji[j][1],Aji[j][0])
        Vk = np.kron(Aji[k][0],Aji[k][1])+np.kron(Aji[k][1],Aji[k][0])        
        """
        """sigmaz sigmax"""
        Vj = np.kron(Aji[j][0],Aji[j][0])-np.kron(Aji[j][1],Aji[j][1])
        Vk = np.kron(Aji[k][0],Aji[k][1])+np.kron(Aji[k][1],Aji[k][0])        
        if j==1:
            koef = np.matmul(L,Vj)
        elif k==1:
            koef = np.matmul(L,Vk)
        else:
            koef = np.matmul(L,Tji[0])
        for i in range(1,len(Tji)):
            if i==j-1:
                koef = np.matmul(koef,Vj)
                continue
            if i==k-1:
                koef = np.matmul(koef,Vk)
                continue
            koef = np.matmul(koef,Tji[i])
        return np.matmul(koef,R)

    def povprecenj(Aji,L,R,Tji,j):
        #z x
        Vj = np.kron(Aji[j][0],Aji[j][1])-np.kron(Aji[j][1],Aji[j][0])
        if j==1:
            koef = np.matmul(L,Vj)
        else:
            koef = np.matmul(L,Tji[0])
        for i in range(1,len(Tji)):
            if i==j-1:
                koef = np.matmul(koef,Vj)
                continue
            koef = np.matmul(koef,Tji[i])
        return np.matmul(koef,R) 
    
    n=14
    hammy = hamiltonian(n)
    psi = directDiagBase(hammy)[1]
    Aji = dobiA(psi,n)
    L, R, Tji = AtoT(np.conj(Aji),Aji)
    celekorelacije = []
    for j in range(1,n-1):
        print(j)
        korelacije = []
        for k in range(1,n-1):
            if k==j:
                #korelacije.append(1)
                korelacije.append(povprecenj(Aji,L,R,Tji,j))
                continue
            korelacije.append(povprecenjk(Aji,L,R,Tji,j,k))
        celekorelacije.append(korelacije)
    celekorelacije=np.array(celekorelacije)
    celekorelacije = np.reshape(celekorelacije,(n-2,n-2))
    os = [i for i in range(2,n)]
    L =plt.contourf(os,os,celekorelacije,levels=np.linspace(np.amin(celekorelacije),np.amax(celekorelacije),100),cmap="viridis")
    plt.xlabel("j",fontsize=14)
    plt.ylabel("k",fontsize=14)
    plt.title(r"$\langle \sigma^z_j \sigma^x_k \rangle$, $n=14$",fontsize=13)
    cbar = plt.colorbar(L)
    plt.savefig("korelacije7.png")
if 0:
    #j fiksen korelacije(k) za vec velikosti sistema
    def povprecenjk(Aji,L,R,Tji,j,k):
        Vj = np.kron(Aji[j][0],Aji[j][0])-np.kron(Aji[j][1],Aji[j][1])
        Vk = np.kron(Aji[k][0],Aji[k][0])-np.kron(Aji[k][1],Aji[k][1])
        if j==1:
            koef = np.matmul(L,Vj)
        elif k==1:
            koef = np.matmul(L,Vk)
        else:
            koef = np.matmul(L,Tji[0])
        for i in range(1,len(Tji)):
            if i==j-1:
                koef = np.matmul(koef,Vj)
                continue
            if i==k-1:
                koef = np.matmul(koef,Vk)
                continue
            koef = np.matmul(koef,Tji[i])
        return np.matmul(koef,R)
    
    j = 4
    nji = [8,10,12,14]
    os = list(range(2,8))
    for n in nji:
        print(n)
        hammy = hamiltonian(n)
        psi = directDiagBase(hammy)[1]
        Aji = dobiA(psi,n)
        L, R, Tji = AtoT(np.conj(Aji),Aji)
        korelacije = []
        for k in range(1,7):
            korelacije.append(povprecenjk(Aji,L,R,Tji,j,k))
        korelacije = np.array(korelacije)
        plt.plot(os,np.reshape(korelacije,korelacije.size),".:",label=r"$n={}$".format(str(n)))
            
    plt.legend(loc="best")
    plt.grid(True)
    plt.xlabel("k",fontsize=14)
    plt.ylabel(r"$\langle \sigma^z_1 \sigma^z_k \rangle $",fontsize=14)
    plt.title(r"Korelacije pri fiksnem $j=1$, $\langle \sigma^z_1 \sigma^z_k \rangle$",fontsize=13)            
    plt.savefig("korelacijee1.pdf")


def povprecenj(Aji,L,R,Tji,j):
    Vj = np.kron(np.conj(Aji[j][0]),Aji[j][0])-np.kron(np.conj(Aji[j][1]),Aji[j][1])
    if j==1:
        koef = np.matmul(L,Vj)
    else:
        koef = np.matmul(L,Tji[0])
    for i in range(1,len(Tji)):
        if i==j-1:
            koef = np.matmul(koef,Vj)
            continue
        koef = np.matmul(koef,Tji[i])
    return np.matmul(koef,R)   

def celaMag(Aji,L,R,Tji,n):
    magnetizacije = []
    for j in range(1,n-1):
        magnetizacije.append(povprecenj(Aji,L,R,Tji,j)+0j)
    return magnetizacije


def dobiBExtra(n):
    thete = np.concatenate((np.zeros(n//4),np.ones(n//4)*pi,np.zeros(n//4),np.ones(n//4)*pi))
    #thete = np.concatenate((np.zeros(n//2),np.ones(n//2)*pi))
    Bji = []
    for i in range(n):
        Bji1 = [[np.cos(thete[i]/2)]]
        Bji2 = [[np.sin(thete[i]/2)]]
        Bji.append([Bji1,Bji2])
    return Bji, [np.array([1.0]) for i in range(1,n)]

def maksimalna(x):
    maks = 0
    for i in x:
        if max(i) > maks:
            maks = max(i)
    return maks

if 0:
    #casovni razvoj pol pol verige
    n=100
    steps=20
    koraki=10
    delta=0.1
    M=30
    cmap = plt.get_cmap("brg")
    os = [i for i in range(2,n)]
    Bji, lambde = dobiBExtra(n)
    Aji = Bji
    L, R, Tji = AtoT(np.conj(Aji),Aji)
    mag = np.array(celaMag(Aji,L,R,Tji,n))
    plt.plot(os,np.real(np.reshape(mag,mag.size)),".:",label=r"$t=0$",color=cmap(0))
    for i in range(steps):
        print(i)
        Bji, lambde = trotter(Bji,lambde,M,1j*delta,koraki,n)
        Aji = BtoA(Bji,lambde)
        L, R, Tji = AtoT(np.conj(Aji),Aji)
        mag = np.array(celaMag(Aji,L,R,Tji,n))
        if (i+1)%5==0 and i>0:
            plt.plot(os,np.real(np.reshape(mag,mag.size)),".:",label=r"$t={}$".format((i+1)*koraki*delta),color=cmap((i+1)/steps))
        else:
            plt.plot(os,np.real(np.reshape(mag,mag.size)),".:",color=cmap((i+1)/steps))            
    print(np.sum(np.abs(lambde[3])**2))
    plt.legend(loc="best")
    plt.xlabel(r"$i$",fontsize=14)
    plt.ylabel(r"$\sigma^z$",fontsize=14)
    plt.title(r"Razvoj FM verige, $n=100, M=30, \delta=0.1$")
    plt.savefig("figures/razvojExtra.pdf")


if 1:
    #contour
    n=100
    steps=10
    koraki=100
    delta=0.1
    M=100
    rezultati=[]
    os = [i for i in range(2,n)]
    Bji, lambde = dobiBExtra(n)
    Aji = Bji
    #L, R, Tji = AtoT(np.conj(Aji),Aji)
    #mag = np.array(celaMag(Aji,L,R,Tji,n))
    #rezultati.append(np.real(np.reshape(mag,mag.size)))
    for i in range(steps):
        print(i)
        Bji, lambde = trotter(Bji,lambde,M,1j*delta,koraki,n)
        Aji = BtoA(Bji,lambde)
        #L, R, Tji = AtoT(np.conj(Aji),Aji)
        #mag = np.array(celaMag(Aji,L,R,Tji,n))
        #rezultati.append(np.real(np.reshape(mag,mag.size)))
    
    print(1/0)
    print(np.sum(np.abs(lambde[3])**2))
    rezultati = np.array(rezultati)
    print(rezultati.shape)
    L =plt.imshow(rezultati[::-1],cmap="spring",interpolation="none",extent=[2,n-1,0,steps*koraki*delta])
    plt.gca().set_aspect(4)
    #plt.yticks([0,10,20,30,40],[0,5,10,15,20])
    plt.xlabel(r"$i$",fontsize=14)
    plt.ylabel(r"$t$",fontsize=14)
    plt.title(r"Razvoj FM verige, $n=100, M=30, \delta=0.1$",fontsize=13)
    cbar = plt.colorbar(L)
    plt.tight_layout()
    plt.savefig("figures/razvojExtra2.pdf")    



if 0:
    #animacija do 25 sekun
    steps=250
    koraki=1
    delta=0.1
    M=30
    rezultati=[[],[],[],[]]
    osi = []
    nji = [40,60,80,100]
    for n in nji:
        print(n)
        os = [i for i in range(2,n)]
        osi.append(os)
        Bji, lambde = dobiBExtra(n)
        Aji = Bji
        L, R, Tji = AtoT(np.conj(Aji),Aji)
        mag = np.array(celaMag(Aji,L,R,Tji,n))
        rezultati[int((n-40)/20)].append(np.real(np.reshape(mag,mag.size)))
        for i in range(steps):
            Bji, lambde = trotter(Bji,lambde,M,1j*delta,koraki,n)
            Aji = BtoA(Bji,lambde)
            L, R, Tji = AtoT(np.conj(Aji),Aji)
            mag = np.array(celaMag(Aji,L,R,Tji,n))
            rezultati[int((n-40)/20)].append(np.real(np.reshape(mag,mag.size)))
    
    
    
    fig, ax2 = plt.subplots(2,2,figsize=(15,15))
    axi = [ax2[0][0],ax2[0][1],ax2[1][0],ax2[1][1]]
    barve = ["blue","red","green","orange"]
    
    #plt.savefig("Figures/prikaz2.pdf")
    def animiraj(t):
        print(t)
        for ax in axi:
            ax.clear()
        for i in range(4):
            axi[i].plot(osi[i],rezultati[i][t],".:",color=barve[i])
            axi[i].set_title(r"$n={}$".format(nji[i]))
        plt.suptitle(r"$t={}$".format(round(t*delta,2)))
    ani = animation.FuncAnimation(fig,animiraj,range(251),interval=100)   
    #plt.show()
    ani.save("razvoj2.mp4")  
    print("evo")
    print(1/0)
    
    
if 0:
    #kako dolgo do stene
    nji = [20,30,40,50,60,70,80,90,100]
    kdaj= []
    koraki  = 200
    delta=0.1
    M=30
    for n in nji:
        print(n)
        Bji, lambde = dobiBExtra(n)
        Aji = Bji
        L, R, Tji = AtoT(np.conj(Aji),Aji)
        #mag = povprecenj(Aji,L,R,Tji,1)
        for i in range(koraki):
            Bji, lambde = trotter(Bji,lambde,M,1j*delta,1,n)
            Aji = BtoA(Bji,lambde)
            L, R, Tji = AtoT(np.conj(Aji),Aji)
            mag = np.real(povprecenj(Aji,L,R,Tji,1))
            if mag<0.99:
                kdaj.append((i+1)*delta)
                break
            if i==(koraki-1):
                print("kaj")
    plt.plot(nji,kdaj,"-o")
    plt.xlabel(r"$n$",fontsize=14)
    plt.ylabel(r"$t$",fontsize=14)
    plt.title(r"Potreben čas za vpliv na robu $M=30, \delta=0.1$",fontsize=13)
    plt.savefig("figures/cas.pdf")    

if 0:
    #rezanje
    n=14
    zacetni = zrebaj(n)
    stevila= [i for i in range(65)]
    razlike = [0]
    lambde = [0]
    for stevilo in stevila:
        if stevilo==0:
            continue
        Aji = dobiA(zacetni,n,stevilo)
        novi = dobiKoef(Aji,n)
        razlike.append(np.sum(np.abs(zacetni-novi)**2))
        lambde.append(Napaka)
        Napaka=0
        print(stevilo)
    plt.plot(stevila,razlike,".:",label=r"$|\psi - \psi'|^2$")
    plt.plot(stevila,lambde,".:",label=r"$\Sigma_i |\lambda_i|^2$")
    plt.grid(True)
    plt.xlabel(r"$k$",fontsize=14)
    plt.legend(loc="best")
    plt.title(r"Rezanje $k$ singularnih vrednosti pri $j=7$, $n=14$",fontsize=13)
    plt.savefig("rezanje.pdf")
  




