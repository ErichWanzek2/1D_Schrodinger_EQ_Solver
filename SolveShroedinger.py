"""Schrodinger Equation Solver:
        This is a program that solves the shrodinger equation for a specified potential by
        matrix diagnolization method. The program graphs wavefunctions, graphs energy eigenvalues and
        expectation values of position and momentum.
Written by: Erich Wanzek
Written 2/2/17
Computational Quantum mechanics
University of Notre Dame
"""
##############################################################################################################
##############################################################################################################
import numpy as np
import math
import pylab
import matplotlib.pyplot as plt
import scipy.integrate as integrate
period=10
V0=100
##############################################################################################################
##############################################################################################################
def x_mesh(N,interval):
    """This function generates a mesh of x values
    INPUTS: N(number of steps),interval (tuple (a,b))
    OUTPUTS: xmesh1, xmesh2
    """
    (a,b) = interval
    h = (b-a)/N
    xmesh1=[a]
    for i in range(1,N):
        xmesh1.append(a+i*h)
    xmesh1.append(b)
    xmesh2=xmesh1[1:N]
    
    return xmesh1,xmesh2
##############################################################################################################
def infinite_square_well_potential(x):
    """This function defines the infinite square well potential
    INPUTS:xmesh values 
    OUTPUTS: value of potential
    """
    return 0
##############################################################################################################
def Harmonic_potential(x):
    """"This function defines the harmonic oscillator potenital
    INPUTS: xmesh value
    OUTPUTS: value of potential
    """
    k=1
    return 0.5*k*(x**2)
##############################################################################################################
def quartic_potential(x):
    """"This function defines the quartic potential 
    INPUTS: xmesh value
    OUTPUTS: value of potential"""
    k1=1
    k2=10
    return (k1*x**4)-(k2*x**2)
##############################################################################################################
def sin(x):
    return V0*np.cos(2*np.pi/period*x)
##############################################################################################################   
def Hamiltonian_setup(N,V,interval):
    """This function generates the Hamoltonian matrix for the shrodigner equation that is
        is to be diagnaolized
    INPUTS: N(number of steps)
            V(potential function)
            interval(interval of evalutaiton)
    OUTPUTS: Hamiltonian (NxN numpy matrix)
    """
    (a,b) = interval
    H = np.zeros((N-1,N-1))
    h= (b-a)/N
    for i in range(N-1):
        for j in range(N-1):
            if i==j:
                x=((i+1)*h) + a
                Vi=V(x)
                H[i,j] = Vi + 2/(h**2)
            if i==j+1 or i==j-1:
                H[i,j]=-1/(h**2)
    return H
##############################################################################################################
def calculate_eigenvalues(H):
    """This fucntion calculates the eigenvectors and eigenvalues of the Hamilotonian matrix
    INPUTS: H (NxN hamiltonian matrix)
    OUTPUTS: eigenvalues, eigenvecotrs (numy array of dimesnion N)
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues, eigenvectors

##############################################################################################################
def normalize_eigenvectors(eigenvalues,eigenvectors,xmesh,N,interval):
    """This fucntion normalizes the eigenvectors and also
       produces a probability density fucntion(PDF)
    INPUTS:eigenvectors (eigenvector wavefuncitons)
           xmesh (array of x values)
           N (number of steps)
           interval(interval of evaluation)
    OUTPUTS: pdf (probability density function)
             norm_eigenvactors (normalized eigenvectors)
    """
    (a,b) = interval
    
    pdf=np.zeros((N-1,N-1))
    norm_eigenvectors=np.zeros((N-1,N-1))
    norm_eigenvalues=np.zeros((1,N-1))
    for i in range(N-1):
        pdf[:,i] = np.multiply(eigenvectors[:,i],eigenvectors[:,i])
        normconstant = 1/(integrate.trapz(pdf[:,i],xmesh))**0.5
        pdf[:,i] = (normconstant**2)*pdf[:,i]
        norm_eigenvectors[:,i]= normconstant*eigenvectors[:,i]
        norm_eigenvalues[0,i] = eigenvalues[i]

    return pdf,norm_eigenvectors,norm_eigenvalues
##############################################################################################################
def calculate_expectation_values(eigenvectors,xmesh,N):
    """This fucntion calculates the expectations values for the unitary, space, and momentum operators
       and stores each expectation value for each egienvecotr in an array called the expectation value
       matrix
    INPUTS:eigenvectors: egienvectors in form of numpy array outputed by normalize_eigenvectors function
           xmesh: xmesh grid points in form of numpy array
           N: number of steps
    OUTPUTS:expectation_value_matrix in form of a numpy matrix
    """
    identity=np.ones((1,len(eigenvectors[:,0]))) # Identity matrix for testing

    expectation_value_matrix=np.zeros((N-1,3))   
    for i in range(N-1):
        expectation_value_matrix[i,0] = integrate.trapz((eigenvectors[:,1]*eigenvectors[:,1]),xmesh)
        expectation_value_matrix[i,1] = integrate.trapz((eigenvectors[:,1]*xmesh*eigenvectors[:,1]),xmesh)
        expectation_value_matrix[i,2] = integrate.trapz((eigenvectors[:,1]*np.gradient(eigenvectors[:,1])),xmesh)
    return expectation_value_matrix
##############################################################################################################
def plot_wavefunctions(xmesh,interval,N,V,eigenvectors,pdf): 
    """This fucntion plots the wavefunctions/eigenvectors
       and the probabiltiy density funciton(PDF)
    INPUTS:xmesh: xmesh grid points in form of numpy array
           interval:interval of evaluation
           N:number of steps
           V:specified potential
           eigenvectors:eigenvector in form of numpy array outputted by normalize_eigenvectors functionr
           pdf: probability density function in form of numpy matrix outputed by normalize-eigenvectors fucniton
    OUTPUTS: returns None
    """ 
    for i in range(10):
        psi=[0]
        for val in eigenvectors[:,i]:
            psi.append(val)
        psi.append(0)

        pdfp=[0]
        for val in pdf[:,i]:
            pdfp.append(val)
        pdfp.append(0)

        Vplot=[]

        for i in range(len(xmesh)):
            
            Vplot.append(V(xmesh[i]))
    

        plt.plot(xmesh,psi)
        plt.plot(xmesh,pdfp,'g')
        plt.xlabel('x', fontsize=20, color='black')
        plt.ylabel('psi', fontsize=20, color='black')
        plt.show()

##        plt.plot(xmesh,Vplot,'k')
##        plt.xlabel('x', fontsize=20, color='black')
##        plt.ylabel('V(x)', fontsize=20, color='black')
##        plt.ylim(-5,30)
##        plt.show()
        
    return None
##############################################################################################################
def graph_eigenvalues(N, eigenvalues,interval):
    """This function produces a graph of the anayltical and numerical energy eiganvalues
    INPUTS:N, number of steps
           eigenvalues: numerically calculated eigenvalues
           interval: interval of evaluation
    OUTPUTS:none
    """
    (a,b) =interval
    a=abs(a-b)
    n=[]
    E_numeric=[]
    E_anlytic=[]
    E_anlytic_harm=[]
    
    for i in range(N-1):
        n.append(i)
        E_numeric.append(eigenvalues[0,i])
        E_anlytic.append((((math.pi)**2)*(i**2))/(a**2))
        E_anlytic_harm.append(((1/2)**0.5)*((i)+(1/2)))
    plt.plot(n,E_numeric,'b')
    plt.plot(n,E_anlytic,'g')
    plt.plot(n,E_anlytic_harm,'r')
    plt.xlabel('n', fontsize=20, color='black')
    plt.ylabel('E', fontsize=20, color='black')
    plt.show()
    return None
##############################################################################################################
def graph_expectation_values(N,expectation_value_matrix):
    """This function graphs the expectations value for the unity, position, and momentum operators
        Vs. the nth energy level.
    INPUTS: N numeber of steps
            expectation_value_matric outputed by calculate_expectation_valus funciton
    OUTPUTS: returns None
    """
    n=[]
    unity_op_exp=[]
    position_exp=[]
    momentum_exp=[]

    for i in range(N-1):
        n.append(i)
        unity_op_exp.append(expectation_value_matrix[i,0])
        position_exp.append(expectation_value_matrix[i,1]) 
        momentum_exp.append(expectation_value_matrix[i,2])

    plt.plot(n,unity_op_exp,'b')
    plt.xlabel('nth energy level', fontsize=20, color='black')
    plt.ylabel('unity operator expectation value', fontsize=20, color='b')
    plt.show()

    plt.plot(n,position_exp,'b')
    plt.xlabel('nth energy level', fontsize=20, color='black')
    plt.ylabel('position expectation value', fontsize=20, color='b')
    plt.show()
    
    plt.plot(n,momentum_exp,'b')
    plt.xlabel('nth energy level', fontsize=20, color='black')
    plt.ylabel('momentum expectation value', fontsize=20, color='b')
    plt.show()
    return

##############################################################################################################
def run(N,interval,potential):
    """This function runs all subfunctions
    INPUTS:N
          interval
          potential
    OUTPUTS:Matrix
    """
    xmesh1, xmesh2 = x_mesh(N,interval)
    eigenvalues, eigenvectors = calculate_eigenvalues(Hamiltonian_setup(N,potential,interval))
    pdf, norm_eigenvectors,norm_eigenvalues = normalize_eigenvectors(eigenvalues,eigenvectors,xmesh2,N,interval)
    plot_wavefunctions(xmesh1,interval,N,potential,norm_eigenvectors,pdf)
    expectation_value_matrix = calculate_expectation_values(norm_eigenvectors,xmesh2,N)
    graph_eigenvalues(N,norm_eigenvalues,interval)
    graph_expectation_values(N,expectation_value_matrix)
    return None

##############################################################################################################
##############################################################################################################
run(1000,(-10,10),sin) 
##############################################################################################################
##############################################################################################################


