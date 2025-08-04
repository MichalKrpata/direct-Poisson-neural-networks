 #This file contains the forward Euler solver for the self-regularized rigid body motion and the Crank-Nicolson solver for the energetic self-regularization of rigid body motion

from scipy.optimize import fsolve
from math import *
import numpy as np

import torch

from models.Model import EnergyNet, TensorNet, JacVectorNet
from learn import DEFAULT_folder_name

class RigidBody(object): #Parent Rigid body class
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, T=100, verbose = False):
        """
        The above function is the initialization function for a class that represents a physical system,
        setting up various parameters and variables.
        
        :param Ix: Ix is the moment of inertia about the x-axis. It represents the resistance of an object to changes in its rotational motion about the x-axis
        :param Iy: Iy is the moment of inertia about the y-axis. It represents the resistance of an object to changes in its rotational motion about the y-axis
        :param Iz: Iz is the moment of inertia around the z-axis. It represents the resistance of an object to changes in its rotational motion around the z-axis
        :param d2E: The parameter `d2E` is the Hessian of energy.
        :param mx: The parameter `mx` represents the x-component of the angular momentum
        :param my: The parameter `my` represents the moment of inertia about the y-axis. It is used in the calculations for the rotational dynamics of the system
        :param mz: The parameter `mz` represents the angular momentum in the z-direction
        :param dt: The parameter "dt" represents the time step size for the simulation. It determines how much time elapses between each iteration of the simulation
        :param alpha: The parameter "alpha" represents damping. 
        :param T: T is the temperature in Kelvin, defaults to 100 (optional)
        :param verbose: The `verbose` parameter is a boolean flag that determines whether or not to print out additional information during the initialization of the object. If `verbose` is set to `True`, then additional information will be printed. If `verbose` is set to `False`, then no additional information will be printed, defaults to False (optional)
        """
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz

        self.d2E= d2E

        if Iz > 0 and Iy > 0 and Iz > 0:
            self.Jx = 1/Iz - 1/Iy
            self.Jy = 1/Ix - 1/Iz
            self.Jz = 1/Iy - 1/Ix

        self.mx = mx
        self.my = my
        self.mz = mz

        self.mx0 = mx
        self.my0 = my
        self.mz0 = mz

        self.dt = dt
        self.tau = dt*alpha

        self.hbar = 1.0545718E-34 #reduced Planck constant [SI]
        self.rho = 8.92E+03 #for copper
        self.myhbar = self.hbar * self.rho #due to rescaled mass
        self.kB = 1.38064852E-23 #Boltzmann constant
        self.umean = 4600 #mean sound speed in the low temperature solid (Copper) [SI]
        self.Einconst = pi**2/10 * pow(15/(2* pi**2), 4.0/3) * self.hbar * self.umean * pow(self.kB, -4.0/3) #Internal energy prefactor, Characterisitic volume = 1
        if verbose:
            print("Internal energy prefactor = ", self.Einconst)

        self.sin = self.ST(T) #internal entropy
        if verbose:
            print("Internal entropy set to Sin = ", self.sin, " at T=",T," K")

        self.Ein_init = 1
        self.Ein_init = self.Ein()
        self.sin_init = self.sin
        if verbose:
            print("Initial total energy = ", self.Etot())

        if verbose:
            print("RB set up.")

    def energy_x(self):
        """
        The function calculates the energy of an object in the x-direction.
        :return: the value of 0.5 times the square of the variable self.mx, divided by the variable self.Ix.
        """
        return 0.5*self.mx*self.mx/self.Ix

    def energy_y(self):
        """
        The function calculates the energy of an object in the y-direction.
        :return: the value of the expression 0.5*self.my*self.my/self.Iy.
        """
        return 0.5*self.my*self.my/self.Iy

    def energy_z(self):
        """
        The function calculates the energy of an object rotating around the z-axis.
        :return: the value of the expression 0.5*self.mz*self.mz/self.Iz.
        """
        return 0.5*self.mz*self.mz/self.Iz

    def energy(self):#returns kinetic energy
        """
        The function calculates the kinetic energy of an object based on its mass and moments of inertia.
        :return: the kinetic energy of the object.
        """
        return 0.5*(self.mx*self.mx/self.Ix+self.my*self.my/self.Iy+self.mz*self.mz/self.Iz)

    def omega_x(self):
        """
        The function calculates the angular velocity around the x-axis.
        :return: The value of `self.mx/self.Ix` is being returned.
        """
        return self.mx/self.Ix

    def omega_y(self):
        """
        The function calculates the omega_y value by dividing my by Iy.
        :return: the value of `self.my` divided by `self.Iy`.
        """
        return self.my/self.Iy

    def omega_z(self):
        """
        The function calculates the angular velocity around the z-axis.
        :return: the value of `self.mz/self.Iz`.
        """
        return self.mz/self.Iz

    def m2(self):#returns m^2
        """
        The function calculates the square of the magnitude of a vector.
        :return: the square of the magnitude of a vector.
        """
        return self.mx*self.mx+self.my*self.my+self.mz*self.mz

    def mx2(self):#returns mx^2
        """
        The function mx2 returns the value of mx squared.
        :return: the value of mx^2.
        """
        return self.mx*self.mx

    def my2(self):#returns my^2
        """
        The function `my2` returns the square of the value of `self.my`.
        :return: the square of the value of the variable "my".
        """
        return self.my*self.my

    def mz2(self):#returns mz^2
        """
        The function mz2 returns the square of the value of mz.
        :return: the square of the value of mz.
        """
        return self.mz*self.mz

    def m_magnitude(self):#returns |m|
        """
        The function returns the magnitude of a vector.
        :return: The magnitude of the vector, represented by the variable "m".
        """
        return sqrt(self.m2())

    def Ein(self):#returns normalized internal energy
        """
        The function returns the normalized internal energy.
        :return: the normalized internal energy.
        """
        #return exp(2*(self.sin-1))/self.Iz
        return self.Einconst*pow(self.sin,4.0/3)/self.Ein_init

    def Ein_s(self): #returns normalized derivative of internal energy with respect to entropy (inverse temperature)
        """
        The function returns the normalized derivative of internal energy with respect to entropy.
        :return: the normalized derivative of internal energy with respect to entropy (inverse temperature).
        """
        #return 2*exp(2*(self.sin-1))/self.Iz
        return self.Einconst*4.0/3*pow(self.sin, 1.0/3) / self.Ein_init

    def ST(self, T): #returns entropy of a Copper body with characteristic volume equal to one (Debye), [T] = K
        """
        The function calculates the entropy of a Copper body with a characteristic volume equal to one (Debye) at a given temperature.
        
        :param T: T is the temperature of the Copper body in Kelvin (K)
        :return: the entropy of a Copper body with a characteristic volume equal to one (Debye) at a given temperature T.
        """
        return 2 * pi**2/15 * self.kB * (self.kB/self.hbar *T/self.umean)**3

    def Etot(self):#returns normalized total energy
        """
        The function `Etot` returns the sum of the energy and the input energy.
        :return: the sum of the energy and the input energy, both of which are being calculated by other methods.
        """
        return self.energy() + self.Ein()

    def Sin(self): #returns normalized internal entorpy
        """
        The intenral entropy function returns the normalized internal entropy.
        :return: the normalized internal entropy.
        """
        return self.sin/self.sin_init

    def S_x(self):#kinetic entropy for rotation around x, beta = 1/4Iz
        """
        The function calculates the kinetic entropy for rotation around the x-axis.
        :return: the kinetic entropy for rotation around the x-axis.
        """
        m2 = self.m2()
        return -m2/self.Ix - 0.5*0.25/self.Iz*(m2-self.mx0*self.mx0)**2

    def S_z(self):#kinetic entropy for rotation around z
        """
        The function calculates the kinetic entropy for rotation around the z-axis.
        :return: the kinetic entropy for rotation around the z-axis.
        """
        m2 = self.m2()
        return -m2/self.Iz - 0.5*0.25/self.Iz*(m2-self.mz0*self.mz0)**2

    def Phi_x(self): #Returns the Phi potential for rotation around the x-axis
        """
        The function Phi_x returns the sum of the energy and the S_x potential for rotation around the
        x-axis.
        :return: the sum of the energy and the S_x value.
        """
        return self.energy() + self.S_x()

    def Phi_z(self):
        """
        The function Phi_z returns the sum of the energy and the S_z value.
        :return: the sum of the results of two other functions: `self.energy()` and `self.S_z()`.
        """
        return self.energy() + self.S_z()
        
    def get_L(self, m):
        """
        The function `get_L` returns a 3x3 matrix `L` (Poisson bivector) based on the input parameter `m`.
        
        :param m: The parameter "m" is a scalar value
        :return: The function `get_L` returns a 3x3 numpy array `L` which is calculated using the values of `self.mx`, `self.my`, and `self.mz`.
        """
        L = -1*np.array([[0.0, self.mz, -self.my],[-self.mz, 0.0, self.mx],[self.my, -self.mx, 0.0]])
        return L
        
    def get_E(self, m):
        """
        The function "get_E" returns the energy of an object.
        
        :param m: The parameter "m" is not used in the code snippet provided. It is not clear what it represents or how it is related to the function
        :return: The method `get_E` is returning the result of the method `energy()` called on the object `self`.
        """
        return self.energy()

class RBEhrenfest(RigidBody):#Ehrenfest scheme for the rigid body, Eq. 5.25a from https://doi.org/10.1016/j.physd.2019.06.006, τ=dt
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha):
        """
        The above function is the constructor for the RBEhrenfest class, which is a subclass of another
        class.
        
        :param Ix: The parameter Ix represents the moment of inertia about the x-axis
        :param Iy: The parameter Iy represents the moment of inertia about the y-axis
        :param Iz: The parameter "Iz" represents the moment of inertia about the z-axis
        :param d2E: The parameter "d2E" likely represents the second derivative of the energy with respect to time. It could be used in calculations related to the dynamics or evolution of the system
        :param mx: The parameter "mx" likely represents the x-component of the angular momentum. It could be a value that represents the angular momentumum in the x-direction
        :param my: The parameter "my" represents the angular momentumum in the y-direction
        :param mz: The parameter "mz" represents the angular momentumum in the z-direction. 
        :param dt: dt is the time step size for the simulation. It determines how small the time interval are between each step in the simulation
        :param alpha: The parameter "alpha" is a damping parameter. 
        """
        super(RBEhrenfest, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha)

    def m_new(self, with_entropy = False): #return new m and update RB
        """
        The function `m_new` calculates and returns a new value for the angular momentum `m` based on the
        current values of `mx`, `my`, and `mz`, as well as other variables and matrices.
        
        :param with_entropy: The parameter "with_entropy" is a boolean flag that determines whether or not to include entropy in the calculation of the new angular momentum. If it is set to True, entropy will be included in the calculation. If it is set to False (the default value), entropy will not be included, defaults to False (optional)
        :return: the updated value of the angular momentum vector, `m_new`.
        """
        #calculate
        mOld = [self.mx, self.my, self.mz]
        ω = np.dot(self.d2E, mOld) # (mx/Ix, my/Iy, mz/Iz) = dE/dm = ω
        ham = np.cross(mOld, ω) #m x E_m

        Mreg = np.cross(mOld , np.dot(self.d2E, ham))
        Nreg = np.cross(ham, ω)
        reg = 0.5*self.dt * (Mreg+Nreg)

        m_new = mOld + self.dt*ham + self.dt*reg

        #update
        self.mx = m_new[0]
        self.my = m_new[1]
        self.mz = m_new[2]

        return m_new


class RBESeReCN(RigidBody):#E-SeRe with Crank Nicolson
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha):
        """
        The above function is the constructor for a class called RBESeReCN, which is a subclass of another
        class.
        
        :param Ix: The parameter Ix represents the moment of inertia about the x-axis
        :param Iy: The parameter Iy represents the moment of inertia about the y-axis
        :param Iz: The parameter "Iz" represents the moment of inertia about the z-axis. It is a measure of an object's resistance to changes in its rotational motion about the z-axis
        :param d2E: The parameter "d2E" is the Hessian of energy. 
        :param mx: The parameter "mx" represents the x-component of the m vector
        :param my: The parameter "my" represents the moment of inertia in the y-direction
        :param mz: The parameter "mz" represents the moment of inertia in the z-direction
        :param dt: dt is the time step size for the simulation. It determines the granularity of the simulation and how frequently the system is updated
        :param alpha: The alpha parameter is a constant that determines the weight of the regularization.
        """
        super(RBESeReCN, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha)

    def f(self, mNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the difference between the old and new values of `m` and returns the
        result.
        
        :param mNew: The parameter `mNew` represents the new value of the vector `m` in the function `f`
        :return: a tuple containing the values of `res[0]`, `res[1]`, and `res[2]`.
        """

        mOld = [self.mx, self.my, self.mz]
        dot = np.dot(self.d2E, mOld)
        ham = np.cross(mOld, dot)

        #regularized part t
        dotR = np.dot(self.d2E, ham)
        reg  = np.cross(dotR, mOld)

        #Hamiltionian part t+1
        dotNNew = np.dot(self.d2E, mNew)
        hamNew = np.cross(mNew, dotNNew)

        #regularized part t+1
        dotRNew = np.dot(self.d2E, hamNew)
        regNew  = np.cross(dotRNew, mNew)


        res = mOld - mNew + self.dt/2*(ham + hamNew) #+ self.dt*self.tau/4*(reg + regNew)

        return (res[0], res[1], res[2])

    def m_new(self, with_entropy = False): #return new m and update RB
        """
        The function `m_new` calculates and returns new values for `mx`, `my`, and `mz`, and updates the
        corresponding variables in the class instance.
        
        :param with_entropy: The "with_entropy" parameter is a boolean flag that determines whether or not to include entropy in the calculation of the new angular momentum. If set to True, entropy will be considered in the calculation. If set to False, entropy will not be considered, defaults to False (optional)
        :return: the updated values of mx, my, and mz as a tuple.
        """
        #calculate
        m_new = fsolve(self.f, (self.mx, self.my, self.mz))

        #update
        self.mx = m_new[0]
        self.my = m_new[1]
        self.mz = m_new[2]

        return m_new

class RBIMR(RigidBody):#implicit midpoint
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt):
        """
        The function initializes an instance of the RBIMR class with given parameters.
        
        :param Ix: The moment of inertia about the x-axis
        :param Iy: The parameter "Iy" represents the moment of inertia about the y-axis. It is a measure of an object's resistance to changes in rotation about the y-axis
        :param Iz: The parameter "Iz" represents the moment of inertia about the z-axis. It is a measure of an object's resistance to changes in its rotational motion about the z-axis
        :param d2E: The parameter "d2E" likely represents the second derivative of the energy function. It could be a function or a value that represents the rate of change of energy with respect to time
        :param mx: The parameter "mx" represents the x-component of the moment of inertia
        :param my: The parameter "my" represents the moment of inertia about the y-axis
        :param mz: The parameter "mz" represents the moment of inertia about the z-axis
        :param dt: The parameter "dt" represents the time step or time interval between each iteration or calculation in the RBIMR class.
        """
        super(RBIMR, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, 0.0)

    def f(self, mNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the residual of a given angular field vector `mNew` by using the
        previous angular field vector `mOld` and other variables.
        
        :param mNew: The parameter `mNew` represents the new values of the angular field components `mx`, `my`, and `mz`
        :return: a tuple containing the values of `res[0]`, `res[1]`, and `res[2]`.
        """

        mOld = [self.mx, self.my, self.mz]
        m_mid = [0.5*(mOld[i]+mNew[i]) for i in range(len(mOld))]

        dot = np.dot(self.d2E, m_mid)
        ham = np.cross(m_mid, dot)

        res = mOld - mNew + self.dt*ham

        return (res[0], res[1], res[2])

    def m_new(self, with_entropy = False): #return new m and update RB
        """
        The function `m_new` calculates and returns new values for `mx`, `my`, and `mz`, and updates the
        corresponding variables in the class.
        
        :param with_entropy: The "with_entropy" parameter is a boolean flag that determines whether or not to include entropy in the calculation of the new value of m. If set to True, entropy will be considered in the calculation. If set to False, entropy will not be considered, defaults to False (optional)
        :return: the updated values of mx, my, and mz as a tuple.
        """
        #calculate
        m_new = fsolve(self.f, (self.mx, self.my, self.mz))

        #update
        self.mx = m_new[0]
        self.my = m_new[1]
        self.mz = m_new[2]

        return m_new


class RBRK4(RigidBody):
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, tau):
        """
        The function initializes an instance of the RBRK4 class with given parameters.
        
        :param Ix: The moment of inertia about the x-axis
        :param Iy: The parameter "Iy" represents the moment of inertia about the y-axis. It is a measure of an object's resistance to changes in rotation about the y-axis
        :param Iz: The parameter "Iz" represents the moment of inertia about the z-axis. It is a measure of an object's resistance to changes in its rotational motion about the z-axis
        :param d2E: The parameter "d2E" likely represents the second derivative of the energy function. It could be a function or a value that represents the rate of change of energy with respect to time
        :param mx: The parameter "mx" represents the x-component of the moment of inertia
        :param my: The parameter "my" represents the moment of inertia about the y-axis
        :param mz: The parameter "mz" represents the moment of inertia about the z-axis
        :param dt: The parameter "dt" represents the time step or time interval between each iteration or calculation in the RBRK4 class.
        """
        super(RBRK4, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, 0.0)
        self.tau = tau

    def m_dot(self,m):
        LdH = self.get_L(m)@self.d2E @ m
        M = 0.5*self.get_L(m) @ self.d2E @ LdH
        
        return LdH + self.tau*M



    def m_new(self, with_entropy = False):
        """
        The function `m_new` calculates and returns new values for `mx`, `my`, and `mz`, and updates the
        corresponding variables in the class.
        
        :param with_entropy: The "with_entropy" parameter is a boolean flag that determines whether or not to include entropy in the calculation of the new value of m. If set to True, entropy will be considered in the calculation. If set to False, entropy will not be considered, defaults to False (optional)
        :return: the updated values of mx, my, and mz as a tuple.
        """
        #calculate
        m = [self.mx, self.my, self.mz]
        k1 = self.m_dot(m)
        k2 = self.m_dot(m + self.dt*k1/2)
        k3 = self.m_dot(m + self.dt*k2/2)
        k4 = self.m_dot(m + self.dt*k3)
        m_new = m + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)

        #update
        self.mx = m_new[0]
        self.my = m_new[1]
        self.mz = m_new[2]

        return m_new


class RBESeReFE(RigidBody):#SeRe forward Euler
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha):
        """
        The above function is the constructor for a class called RBESeReFE, which is a subclass of another
        class.
        
        :param Ix: The parameter Ix represents the moment of inertia about the x-axis
        :param Iy: The parameter "Iy" likely represents the moment of inertia about the y-axis. Moment of inertia is a measure of an object's resistance to changes in rotational motion.
        :param Iz: The parameter "Iz" represents the moment of inertia about the z-axis. 
        :param d2E: The parameter "d2E" is the Hessian of energy. 
        :param mx: The parameter "mx" likely represents the x-component of the angular field
        :param my: The parameter "my" likely represents the angular momentum in the y-direction
        :param mz: The parameter "mz" represents the angular field in the z-direction
        :param dt: dt is the time step size for the simulation. It determines the granularity of the simulation and how frequently the calculations are performed
        :param alpha: The alpha parameter is a constant that determines the strength of the regularization term in the RBESeReFE algorithm. 
        """
        super(RBESeReFE, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha)

    def m_new(self, with_entropy = False):
        """
        The function `m_new` calculates the new value of the angular momentum `m` based on the old value
        `mOld` and updates the values of `mx`, `my`, and `mz`, and optionally calculates the new entropy if
        `with_entropy` is True.
        
        :param with_entropy: A boolean parameter that determines whether or not to calculate the new entropy using explicit forward Euler. If set to True, the entropy will be calculated and updated. If set to False, the entropy will not be calculated, defaults to False (optional)
        :return: the updated value of the angular momentum vector, `m`.
        """

        #Construct mOld
        mOld = [self.mx, self.my, self.mz]

        #calculate
        dot = np.dot(self.d2E, mOld)
        ham = np.cross(mOld, dot)

        #regularized part t
        dotR = np.dot(self.d2E, ham)
        reg  = np.cross(dotR, mOld)

        m = mOld + self.dt*ham - self.dt*self.tau/2*reg

        #update
        self.mx = m[0]
        self.my = m[1]
        self.mz = m[2]

        if with_entropy: #calculate new entropy using explicit forward Euler
            sin_new = self.sin+ 0.5*(self.tau-self.dt)*self.dt/self.Ein_s() * ((self.my*self.mz*self.Jx)**2/self.Ix + (self.mz*self.mx*self.Jy)**2/self.Iy + (self.mx*self.my*self.Jz)**2/self.Iz)
            self.sin = sin_new

        return m

class Neural(RigidBody):#SeRe forward Euler
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, method = "without", name = DEFAULT_folder_name, device = "cpu"):
        """
        The function initializes a Neural object with specified parameters and loads a pre-trained neural
        network based on the chosen method.
        
        :param Ix: The parameter Ix represents the x-component of the input vector. It is used in the initialization of the Neural class
        :param Iy: The parameter "Iy" represents the y-component of the moment of inertia of the system. Moment of inertia is a measure of an object's resistance to changes in rotational motion. In this context, it is used to describe the rotational behavior of the system along the y-axis
        :param Iz: Iz is the moment of inertia about the z-axis. It represents the resistance of an object to changes in its rotational motion around the z-axis
        :param d2E: The parameter `d2E` is a function that represents the second derivative of the energy function. It takes as input the current state of the system and returns the second derivative of the energy with respect to the state variables
        :param mx: The parameter `mx` represents the x-component of the angular momentumum 
        :param my: The parameter `my` represents the value of the angular momentumum in the y-direction. It is used in the initialization of the `Neural` class
        :param mz: The parameter `mz` represents the value of the angular momentumum in the z-direction
        :param dt: dt is the time step size used in the simulation. It determines the granularity of the time intervals at which the neural network is updated and the system dynamics are computed
        :param alpha: The parameter "alpha" is a value used in the initialization of the Neural class, damping parameter.
        :param method: The "method" parameter is used to specify the method for calculating the energy and L matrices in the Neural class. It can take one of the following values:, defaults to without (optional)
        :param name: The `name` parameter is a string that represents the folder name where the saved models are located. It is used to load the pre-trained neural network models for energy and L (Lagrangian) calculations. The `name` parameter is used to construct the file paths for loading the models
        """
        super(Neural, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha)
        # Load network
        self.method = method
        if method == "soft":
            self.energy_net = torch.load(name+'/saved_models/soft_jacobi_energy', weights_only=False)  # changed weights_only=False
            self.energy_net.eval()   
            self.L_net = torch.load(name+'/saved_models/soft_jacobi_L', weights_only=False)  # changed weights_only=False
            self.L_net.eval()
        elif method == "without":
            self.energy_net = torch.load(name+'/saved_models/without_jacobi_energy', weights_only=False)  # changed weights_only=False
            self.energy_net.eval()   
            self.L_net = torch.load(name+'/saved_models/without_jacobi_L', weights_only=False)  # changed weights_only=False
            self.L_net.eval()
        elif method == "implicit":
            self.energy_net = torch.load(name+'/saved_models/implicit_jacobi_energy', weights_only=False)  # changed weights_only=False
            self.energy_net.eval()
            self.J_net = torch.load(name+'/saved_models/implicit_jacobi_J', weights_only=False)  # changed weights_only=False
            self.J_net.eval()
            def L_net(z):
                L = -1*torch.tensor([[[0.0, z[2], -z[1]],[-z[2], 0.0, z[0]],[z[1], -z[0], 0.0]]])
                return L
            self.L_net = L_net
        else:
            raise Exception("Unkonown method: ", method)

        self.device = device
        self.energy_net.to(self.device)
        if hasattr(self, 'L_net') and isinstance(self.L_net, torch.nn.Module): self.L_net.to(self.device)
        if hasattr(self, 'J_net'): self.J_net.to(self.device)

    # Get gradient of energy from NN
    def neural_zdot(self, z):
        """
        The function `neural_zdot` calculates the Hamiltonian of a neural network given a set of input
        values.
        
        :param z: The parameter `z` is the input to the `neural_zdot` function. It is a tensor or array that represents the input data for the neural network.
        :return: the hamiltonian, which is a numpy array.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        En = self.energy_net(z_tensor)

        E_z = torch.autograd.grad(En.sum(), z_tensor, only_inputs=True)[0]
        E_z = torch.flatten(E_z)

        if self.method == "soft" or self.method == "without":
            L = self.L_net(z_tensor).detach().cpu().numpy()[0]
            hamiltonian = np.matmul(L, E_z.detach().cpu().numpy())
        else:
            J, cass = self.J_net(z_tensor)
            J = J.detach().cpu().numpy()
            hamiltonian = np.cross(J, E_z.detach().cpu().numpy())

        return hamiltonian

    def f(self, mNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the difference between two sets of values and returns the result.
        
        :param mNew: The parameter `mNew` represents the new values of `mx`, `my`, and `mz`
        :return: a tuple containing the values of `res[0]`, `res[1]`, and `res[2]`.
        """

        mOld = [self.mx, self.my, self.mz]

        zdo = self.neural_zdot(mOld)
        zd = self.neural_zdot(mNew)

        res = mOld - mNew + self.dt/2*(zdo + zd)

        return (res[0], res[1], res[2])

    def get_cass(self, z):
        """
        The function `get_cass` takes in a parameter `z`, converts it to a tensor, passes it through a
        neural network `J_net`, and returns the output `cass` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the neural network `J_net`. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32` and `requires_grad` set to `True`. The `requires_grad` flag
        :return: the value of `cass` as a NumPy array.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        J, cass = self.J_net(z_tensor)
        return cass.detach().cpu().numpy()

    def get_L(self, z):
        """
        The function `get_L` takes a tensor `z`, passes it through a neural network `L_net`, and returns the
        output `L` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `L_net` neural network. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will be computed for this
        :return: the value of L, which is obtained by passing the input z through the L_net neural network and converting the result to a numpy array.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        L = self.L_net(z_tensor).detach().cpu().numpy()[0]
        return L

    def get_E(self, z):
        """
        The function `get_E` takes a parameter `z`, converts it to a tensor, passes it through a neural
        network called `energy_net`, and returns the resulting energy value.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `energy_net` neural network. It is converted to a tensor using `torch.tensor` and is set to have a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will
        :return: the value of E, which is the output of the energy_net model when given the input z.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        E = self.energy_net(z_tensor).detach().cpu().numpy()[0]
        return E

    # original implementation of m_new
    """def m_new(self, with_entropy = False): #return new m and update RB
        ""
        The function `m_new` calculates new values for `mx`, `my`, and `mz` using the `fsolve` function and
        updates the corresponding variables.
        
        :param with_entropy: The parameter "with_entropy" is a boolean flag that determines whether or not to include entropy in the calculation of the new value of m. If it is set to True, entropy will be considered in the calculation. If it is set to False, entropy will not be considered, defaults to False (optional)
        :return: the updated values of mx, my, and mz as a tuple.
        ""
        #calculate
        m_new = fsolve(self.f, (self.mx, self.my, self.mz))

        # update
        self.mx = m_new[0]
        self.my = m_new[1]
        self.mz = m_new[2]

        return m_new"""
    
    def _hamiltonian(self, z_tensor):
        z_tensor.requires_grad_(True)
        En = self.energy_net(z_tensor).squeeze(0)
        
        E_z = torch.autograd.grad(En.sum(), z_tensor, create_graph=True)[0]
        
        if self.method == "soft" or self.method == "without":
            L = self.L_net(z_tensor).squeeze(0)
            hamiltonian = torch.matmul(L, E_z.unsqueeze(-1)).squeeze(-1)
        else: # "implicit"
            J, cass = self.J_net(z_tensor)
            J = J.squeeze(0)
            cass = cass.squeeze(0)
            hamiltonian = torch.cross(J, E_z, dim=-1)
            
        return hamiltonian
    
    def m_new(self, with_entropy=False, solver_iterations=100, tol=1.5e-8):
        m_old = torch.tensor([self.mx, self.my, self.mz], dtype=torch.float32, device=self.device)
        m_new = m_old.clone()

        zd_old = self._hamiltonian(m_old)

        for _ in range(solver_iterations):
            m_prev = m_new.clone()

            zd_new = self._hamiltonian(m_prev)
            m_new = m_old + 0.5 * self.dt * (zd_old + zd_new)

            diff = torch.norm(m_new - m_prev)
            denom = torch.norm(m_prev) + 1e-12
            rel_error = diff / denom

            if rel_error.item() < tol:
                break

        m_new_np = m_new.detach().cpu().numpy()

        self.mx = m_new_np[0]
        self.my = m_new_np[1]
        self.mz = m_new_np[2]

        return m_new_np
    
    # won't use
    """def _solver_step_gpu(self, z_old_gpu, solver_iterations=4):
        ""
        Performs one step of the implicit midpoint solver on the GPU.
        This is a stateless function.
        
        :param z_old_gpu: The current state tensor on the GPU.
        :return: The next state tensor on the GPU.
        ""
        z_new_gpu = z_old_gpu.clone()

        for _ in range(solver_iterations):
            z_mid_gpu = 0.5 * (z_old_gpu + z_new_gpu)
            hamiltonian_mid = self._hamiltonian(z_mid_gpu)
            z_new_gpu = z_old_gpu + self.dt * hamiltonian_mid
            
        return z_new_gpu"""

class RBNeuralIMR(Neural):#implicit midpoint rule
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, method = "without", name = DEFAULT_folder_name, device = "cpu"):
        """
        The above function is the constructor for a class called RBNeuralIMR, which is a subclass of another
        class.
        
        :param Ix: The parameter Ix represents the moment of inertia about the x-axis. It is a measure of an object's resistance to changes in its rotational motion about the x-axis
        :param Iy: The parameter "Iy" represents the moment of inertia about the y-axis. 
        :param Iz: The parameter "Iz" represents the moment of inertia about the z-axis. 
        :param d2E: The parameter "d2E" represents the second derivative of the energy function. 
        :param mx: The parameter "mx" represents the x-component of the angular momentum. It is used in the RBNeuralIMR class initialization to define the initial angular momentum state of the system
        :param my: The parameter "my" represents the angular momentum in the y-direction. It is used in the RBNeuralIMR class initialization to set the initial angular momentum of the system
        :param mz: The parameter "mz" represents the angular momentum in the z-direction
        :param dt: dt is the time step size for the simulation. It determines the granularity of the simulation, with smaller values resulting in more accurate but slower simulations
        :param alpha: The alpha parameter is a constant that determines the strength of the damping in the system. It is used in the calculation of the damping force in the RBNeuralIMR class
        :param method: The "method" parameter is used to specify the method to be used for the calculation. The default value is set to "without", defaults to without (optional)
        :param name: The name parameter is used to specify the folder name where the results of the RBNeuralIMR class will be saved. If no name is provided, it will use the DEFAULT_folder_name
        """
        super(RBNeuralIMR, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, method = method, name = name, device=device)

    def f(self, mNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the difference between the old and new values of `m` and adds the
        product of the time step `dt` and the derivative of `z` to it.
        
        :param mNew: The parameter `mNew` represents the new values of `mx`, `my`, and `mz`
        :return: a tuple containing three values: res[0], res[1], and res[2].
        """

        mOld = [self.mx, self.my, self.mz]
        m_mid = [0.5*(mOld[i]+mNew[i]) for i in range(len(mOld))]

        zd = self.neural_zdot(m_mid)

        res = mOld - mNew + self.dt*zd

        return (res[0], res[1], res[2])


    def m_new(self, with_entropy = False, solver_iterations=100, tol=1.5e-8):
        m_old = torch.tensor([self.mx, self.my, self.mz], dtype=torch.float32, device=self.device)
        m_new = m_old.clone()

        for _ in range(solver_iterations):
            m_prev = m_new.clone()
            m_mid = 0.5 * (m_old + m_prev)
            m_mid.requires_grad_(True)

            hamiltonian = self._hamiltonian(m_mid)
            m_new = m_old + self.dt * hamiltonian

            diff = torch.norm(m_new - m_prev)
            denom = torch.norm(m_prev) + 1e-12
            rel_error = diff / denom

            if rel_error.item() < tol:
                break
        
        m_new_np = m_new.detach().cpu().numpy()
        self.mx, self.my, self.mz = m_new_np.tolist()
        return m_new_np


class HeavyTopCN(RigidBody): #Crank-Nicolson
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, Mgl, init_rx,  init_ry,  init_rz):
        """
        The function initializes the HeavyTopCN class with given parameters. CN stands for the Crank-Nicholson method.
        
        :param Ix: The moment of inertia about the x-axis
        :param Iy: The parameter Iy represents the moment of inertia about the y-axis of a heavy top
        :param Iz: The parameter "Iz" represents the moment of inertia of the heavy top around the z-axis. 
        :param d2E: The parameter "d2E" represents the second derivative of the energy function. It is a function that calculates the second derivative of the energy with respect to the angles of rotation
        :param mx: The mx parameter represents the x-component of the angular momentumum vector
        :param my: The parameter "my" represents the mass of the object along the y-axis in the HeavyTopCN class
        :param mz: The parameter `mz` represents the z-component of the angular momentumum vector
        :param dt: dt is the time step size for the simulation. It determines how small the time intervals are between each update of the system
        :param alpha: The parameter alpha is a constant used in the calculation of the time derivative of the angular momentumum. It is typically a small value that determines the accuracy and stability of the numerical integration method used to solve the equations of motion
        :param Mgl: The parameter Mgl represents the potential energy due to gravity for the heavy top system. It is used in the Hamiltonian equation to calculate the total energy of the system
        :param init_rx: The initial x-coordinate of the position vector r
        :param init_ry: The parameter `init_ry` represents the initial value for the y-coordinate of the vector `r`. It is used in the initialization of the `HeavyTopCN` class
        :param init_rz: The parameter `init_rz` represents the initial value of the z-coordinate of the vector `r`. It is used in the initialization of the `HeavyTopCN` class
        """
        super(HeavyTopCN, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha)
        self.Mgl = Mgl #Hamiltonian = 1/2 M I^{-1} M + Mgl r . chi
        self.chi = np.array((0.0, 0.0, 1.0))
        self.r = np.array((init_rx, init_ry, init_rz))

    def get_E(self, m):
        """
        The function `get_E` calculates the total energy of an object by adding the energy of the object's
        parent class and the product of the object's mass, gravitational acceleration, and the dot product
        of the object's position vector and a given vector.
        
        :param m: The parameter `m` represents the mass of the object
        :return: the sum of the energy calculated by the parent class (using the `energy()` method) and the dot product of `self.r` and `self.chi`, multiplied by `self.Mgl`.
        """
        return super(HeavyTopCN,self).energy() + self.Mgl*np.dot(self.r, self.chi)

    def get_L(self, m):
        """
        The function `get_L` returns a 6x6 numpy array `L` based on the input parameter `m` and the
        attributes `self.mz`, `self.my`, `self.mx`, and `self.r`.
        
        :param m: The parameter `m` is not defined in the code snippet you provided. It is a variable that represents the moment of inertia.
        :return: The function `get_L` returns a numpy array `L` which is a 6x6 matrix.
        """
        L = np.array([
            [0.0, -self.mz, self.my, 0.0, -self.r[2], self.r[1]],
            [self.mz, 0.0, -self.mx, self.r[2], 0.0, -self.r[0]],
            [self.my, -self.mx, 0.0, -self.r[1], self.r[0], 0.0],
            [0.0, -self.r[2], self.r[1], 0.0, 0.0, 0.0],
            [self.r[2], 0.0, -self.r[0], 0.0, 0.0, 0.0],
            [-self.r[1], self.r[0], 0.0, 0.0, 0.0, 0.0]])
        return L

    def f(self, mrnew):#defines the function f zero of which is sought
        """
        The function `f` calculates the residuals for a given set of input variables.
        
        :param mrnew: The parameter `mrnew` is a tuple containing the values for `mNew` and `rNew`
        :return: a tuple containing the values of `m_res[0]`, `m_res[1]`, `m_res[2]`, `r_res[0]`, `r_res[1]`, and `r_res[2]`.
        """
        mNew = (mrnew[0], mrnew[1], mrnew[2])
        rNew = (mrnew[3], mrnew[4], mrnew[5])
        mOld = np.array((self.mx, self.my, self.mz))
        rOld = self.r

        m_dot = np.dot(self.d2E, mOld)
        m_ham = np.cross(mOld, m_dot)
        m_r = np.cross(rOld, self.Mgl*self.chi)
        r_m = np.cross(rOld, m_dot)

        #Hamiltionian part t+1
        m_dotNew = np.dot(self.d2E, mNew)
        m_hamNew = np.cross(mNew, m_dotNew)
        m_rNew = np.cross(rNew, self.Mgl*self.chi)
        r_mNew = np.cross(rNew, m_dotNew)

        m_res = mOld - mNew + self.dt/2*(m_ham + m_r + m_hamNew + m_rNew)
        r_res = rOld - rNew + self.dt/2*(r_m + r_mNew)

        #return (res[0], res[1], res[2]) 
        return (m_res[0], m_res[1], m_res[2], r_res[0], r_res[1], r_res[2])

    def m_new(self, with_entropy = False): #return new m and update RB
        """
        The function `m_new` calculates new values for `mx`, `my`, `mz`, and `r` using the `fsolve` function
        and returns the updated values.
        
        :param with_entropy: The parameter "with_entropy" is a boolean flag that determines whether or not to include entropy in the calculation. If it is set to True, entropy will be included in the calculation. If it is set to False, entropy will not be included, defaults to False (optional)
        :return: a tuple containing the updated values of (self.mx, self.my, self.mz) and self.r.
        """
        #calculate
        (self.mx, self.my, self.mz, self.r[0], self.r[1], self.r[2]) = fsolve(self.f, (self.mx, self.my, self.mz, self.r[0], self.r[1], self.r[2]))

        #update
        #self.mx = m_new[0]
        #self.my = m_new[1]
        #self.mz = m_new[2]
        #self.r = r_new

        return ((self.mx, self.my, self.mz), self.r)

class HeavyTopIMR(HeavyTopCN): #implicit midpoint rule
    #def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, Mgl, init_rx,  init_ry,  init_rz):
    #    super(HeavyTopIMR, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, Mgl, init_rx,  init_ry,  init_rz)

    def f(self, mrnew):#defines the function f zero of which is sought
        """
        The function `f` calculates the residuals for a given set of inputs and returns them as a tuple.
        
        :param mrnew: The parameter `mrnew` is a list or tuple containing the following elements:
        :return: a tuple containing the values of `m_res[0]`, `m_res[1]`, `m_res[2]`, `r_res[0]`, `r_res[1]`, and `r_res[2]`.
        """
        mNew = (mrnew[0], mrnew[1], mrnew[2])
        rNew = (mrnew[3], mrnew[4], mrnew[5])
        mOld = np.array((self.mx, self.my, self.mz))
        rOld = self.r
        m_mid = [0.5*(mOld[i]+mNew[i]) for i in range(len(mOld))]
        r_mid = [0.5*(rOld[i]+rNew[i]) for i in range(len(rOld))]


        m_dot = np.dot(self.d2E, m_mid)
        m_ham = np.cross(m_mid, m_dot)
        m_r = np.cross(r_mid, self.Mgl*self.chi)
        r_m = np.cross(r_mid, m_dot)

        m_res = mOld - mNew + self.dt*(m_ham + m_r)
        r_res = rOld - rNew + self.dt*r_m 

        #return (res[0], res[1], res[2]) 
        return (m_res[0], m_res[1], m_res[2], r_res[0], r_res[1], r_res[2])



class HeavyTopNeural(HeavyTopCN):
    def __init__(self, Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, Mgl, init_rx, init_ry, init_rz, method = "without", name = DEFAULT_folder_name):
        """
        The function initializes a HeavyTopNeural object with specified parameters and loads a neural
        network model based on the chosen method.
        
        :param Ix: The parameter Ix represents the moment of inertia of the heavy top about the x-axis
        :param Iy: The parameter "Iy" represents the moment of inertia about the y-axis of a heavy top.
        :param Iz: Iz is the moment of inertia about the z-axis. 
        :param d2E: The parameter `d2E` represents the second derivative of the energy function. 
        :param mx: The parameter `mx` represents the x-component of the angular momentum of the heavy top
        :param my: The parameter `my` represents the angular momentum in the y-axis of a heavy top.
        :param mz: The parameter `mz` represents the angular momentum of the heavy top along the z-axis
        :param dt: dt is the time step size for the simulation. It determines how small the time intervals are between each iteration of the simulation
        :param alpha: A damping parameter.
        :param Mgl: Mgl is the product of the mass of the heavy top and the acceleration due to gravity. It represents the gravitational potential energy of the system
        :param init_rx: The parameter `init_rx` represents the initial value of the x-coordinate of the heavy top's center of mass
        :param init_ry: The parameter `init_ry` represents the initial value of the y-component of the angular velocity of the heavy top. It is used to initialize the simulation of the heavy top's motion
        :param init_rz: The parameter `init_rz` represents the initial value of the rotation about the z-axis (yaw) for the HeavyTopNeural object. It is used to specify the initial orientation of the heavy top
        :param method: The "method" parameter is used to specify the method for solving the equations of motion in the HeavyTopNeural class. It can take one of the following values:, defaults to without (optional)
        :param name: The `name` parameter is a string that represents the name of the folder where the saved models are located. It is used to load the pre-trained neural network models for energy and angular momentum calculations
        """
        super(HeavyTopNeural, self).__init__(Ix, Iy, Iz, d2E, mx, my, mz, dt, alpha, Mgl, init_rx, init_ry, init_rz)
        # Load network
        self.method = method
        if method == "soft":
            self.energy_net = torch.load(name+'/saved_models/soft_jacobi_energy', weights_only=False)  # changed weights_only=False
            self.energy_net.eval()   
            self.L_net = torch.load(name+'/saved_models/soft_jacobi_L', weights_only=False)  # changed weights_only=False
            self.L_net.eval()
        elif method == "without":
            self.energy_net = torch.load(name+'/saved_models/without_jacobi_energy', weights_only=False)  # changed weights_only=False
            self.energy_net.eval()   
            self.L_net = torch.load(name+'/saved_models/without_jacobi_L', weights_only=False)  # changed weights_only=False
            self.L_net.eval()
        elif method == "implicit":
            raise Exception("Implicit solver not yet implemented for HT.")
            self.energy_net = torch.load(name+'/saved_models/implicit_jacobi_energy', weights_only=False)  # changed weights_only=False
            self.energy_net.eval()
            self.J_net = torch.load(name+'/saved_models/implicit_jacobi_J', weights_only=False)  # changed weights_only=False
            self.J_net.eval()
            def L_net(z):
                L = -1*torch.tensor([[[0.0, z[2], -z[1]],[-z[2], 0.0, z[0]],[z[1], -z[0], 0.0]]])
                return L
            self.L_net = L_net
        else:
            raise Exception("Unkonown method: ", method)
        self.device = next(self.energy_net.parameters()).device
        if self.device.type == "cuda":
            torch.cuda.current_device()

    # Get gradient of energy from NN
    def neural_zdot(self, z):
        """
        The `neural_zdot` function calculates the Hamiltonian of a neural network given a set of input
        values.
        
        :param z: The parameter `z` is the input to the `neural_zdot` function. It is expected to be a numerical value or an array-like object that can be converted to a tensor
        :return: The function `neural_zdot` returns the variable `hamiltonian`.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        En = self.energy_net(z_tensor)

        E_z = torch.autograd.grad(En.sum(), z_tensor, only_inputs=True)[0]
        E_z = torch.flatten(E_z)

        if self.method == "soft" or self.method == "without":
            L = self.L_net(z_tensor).detach().cpu().numpy()[0]
            hamiltonian = np.matmul(L, E_z.detach().cpu().numpy())
        else:
            raise Exception("Implicit not implemented for HT yet.")
            J, cass = self.J_net(z_tensor)
            J = J.detach().numpy()
            hamiltonian = np.cross(J, E_z.detach().numpy())

        return hamiltonian

    def f(self, mrNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the difference between the old and new values of `mr` and adds the
        average of the neural network outputs for the old and new values multiplied by the time step.
        
        :param mrNew: mrNew is a list containing the new values for the variables mx, my, mz, rx, ry, and rz
        :return: the result of the calculation, which is stored in the variable "res".
        """

        mOld = [self.mx, self.my, self.mz]
        rOld = self.r
        mrOld = np.concatenate([mOld, rOld])
        mNew = [mrNew[0], mrNew[1], mrNew[2]]
        rNew = [mrNew[3], mrNew[4], mrNew[5]]

        zdo = self.neural_zdot(mrOld)
        zd = self.neural_zdot(mrNew)

        res = np.array(mrOld) - np.array(mrNew) + self.dt/2*(zdo + zd)

        return res

    def get_cass(self, z):
        """
        The function `get_cass` takes in a parameter `z`, converts it to a tensor, passes it through a
        neural network `J_net`, and returns the output `cass` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the neural network. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will be computed for this tensor during backprop
        :return: the value of `cass` as a NumPy array.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        J, cass = self.J_net(z_tensor)
        return cass.detach().cpu().numpy()

    def get_L(self, z):
        """
        The function `get_L` takes a tensor `z`, passes it through a neural network `L_net`, and returns the
        output `L` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `L_net` neural network. It is converted to a tensor using `torch.tensor` and is set to have a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will
        :return: the value of L, which is obtained by passing the input z through the L_net neural network and converting the result to a numpy array.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        L = self.L_net(z_tensor).detach().cpu().numpy()[0]
        return L

    def get_E(self, z):
        """
        The function `get_E` takes a parameter `z`, converts it to a tensor, passes it through a neural
        network called `energy_net`, and returns the resulting energy value.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `energy_net` neural network. It is converted to a tensor using `torch.tensor` and is set to have a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will
        :return: the value of E, which is the output of the energy_net model when given the input z.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        E = self.energy_net(z_tensor).detach().cpu().numpy()[0]
        return E

    #@tf.function
    def m_new(self, with_entropy = False): #return new m and update RB
        """
        The function `m_new` calculates new values for `mx`, `my`, `mz`, and `r` based on the current values
        and returns the updated values.
        
        :param with_entropy: The `with_entropy` parameter is a boolean flag that determines whether or not to include entropy in the calculation. If `with_entropy` is set to `True`, entropy will be included in the calculation. If `with_entropy` is set to `False` (default), entropy will not be included, defaults to False (optional)
        :return: The function `m_new` returns a tuple containing the updated values of `mx`, `my`, `mz` (momefield components) and `r` (position vector).
        """
        #calculate
        mr_new = fsolve(self.f, (self.mx, self.my, self.mz, self.r[0], self.r[1], self.r[2]))

        # update
        self.mx = mr_new[0]
        self.my = mr_new[1]
        self.mz = mr_new[2]
        self.r = [mr_new[3], mr_new[4], mr_new[5]]

        return ((self.mx, self.my, self.mz), self.r)

# The `HeavyTopNeuralIMR` class is a subclass of `HeavyTopNeural` that defines a function `f` which
# calculates the difference between old and new values of `mr` and adds the product of `dt` and the
# derivative of `z` with respect to `mr`. IMR stands for the Implicit Midpoint Rule
class HeavyTopNeuralIMR(HeavyTopNeural):
    def f(self, mrNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the difference between the old and new values of `mr` and adds the
        product of `dt` and the derivative of `z` with respect to `mr`.
        
        :param mrNew: The parameter `mrNew` is a list or array containing the new values for `mx`, `my`, `mz`, and `r`
        :return: the value of the variable "res".
        """

        mOld = [self.mx, self.my, self.mz]
        rOld = self.r
        mrOld = np.concatenate([mOld, rOld])
        mr_mid = 0.5*(np.array(mrNew)+mrOld)

        zd = self.neural_zdot(mr_mid)

        res = np.array(mrOld) - np.array(mrNew) + self.dt*zd

        return res

class Particle3DCN(object): #Crank-Nicolson
    def __init__(self, M, dt, alpha, init_rx,  init_ry,  init_rz, init_mx, init_my, init_mz):
        """
        The function initializes the variables M, dt, alpha, init_rx, init_ry, init_rz, init_mx, init_my,
        and init_mz.
        
        :param M: The parameter M represents the mass of the particle in the Hamiltonian equation
        :param dt: dt is the time step size. It determines the size of each time step in the simulation
        :param alpha: The alpha parameter represents the strength of the harmonic potential in the Hamiltonian. It determines how tightly the particle is confined in space. A larger value of alpha corresponds to a stronger confinement
        :param init_rx: The initial x-coordinate of the position vector
        :param init_ry: The parameter `init_ry` represents the initial y-coordinate of the position vector `r`. It is used to specify the initial position of the particle in the y-direction
        :param init_rz: The parameter `init_rz` represents the initial position in the z-direction. It is used to initialize the position vector `self.r` in the `__init__` method of the class
        :param init_mx: The initial momentum in the x-direction
        :param init_my: The parameter `init_my` represents the initial momentum in the y-direction
        :param init_mz: The parameter `init_mz` represents the initial value of the momentum component in the z-direction
        """
        self.M = M #Hamiltonian = 1/2 p^2/M + 1/2 alpha r^2
        self.r = np.array((init_rx, init_ry, init_rz))
        self.p = np.array((init_mx, init_my, init_mz))
        self.alpha = alpha
        self.dt = dt

    def get_E(self, m):
        """
        The function `get_E` calculates the total energy `E` based on the input `m` and some constants `M`
        and `alpha`.
        
        :param m: The parameter `m` is a list or tuple containing six elements. The elements represent the values of `m[0]`, `m[1]`, `m[2]`, `m[3]`, `m[4]`, and `m[5]`, where the first three give the position while the latter three position
        :return: The function `get_E` returns the value of the expression `0.5*(m[3]**2 + m[4]**2 + m[5]**2)/self.M + 0.5 *self.alpha * (m[0]**2 + m[1]**2 + m[2]**2)`.
        """
        return 0.5*(m[3]**2 + m[4]**2 + m[5]**2)/self.M + 0.5 *self.alpha * (m[0]**2 + m[1]**2 + m[2]**2)

    def get_L(self, m = (0.0, 0.0, 0.0)):
        """
        The function `get_L` returns a 6x6 numpy array representing a transformation matrix.
        
        :param m: The parameter `m` is a tuple with three elements representing the x, y, and z coordinates respectively. The default value for `m` is (0.0, 0.0, 0.0)
        :return: The function `get_L` returns a 6x6 numpy array `L` with the following values:
        """
        L = np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]])
        return L

    def f(self, rpNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the residual of a given set of input parameters `rpNew` by performing a
        series of mathematical operations.
        
        :param rpNew: The parameter `rpNew` is a list or array containing the new values of `r` and `p`. It should have a length of 6, where the first 3 elements represent the new values of `r` and the last 3 elements represent the new values of `p`
        :return: a tuple containing the values of `rpres[0]`, `rpres[1]`, `rpres[2]`, `rpres[3]`, `rpres[4]`, and `rpres[5]`.
        """
        rpOld = np.array([self.r[0], self.r[1], self.r[2], self.p[0], self.p[1], self.p[2]])
        #dEOld = np.concatenate([self.p/self.M, self.alpha*self.r])
        #rmdot = self.get_L(rpOld).dot(dEOld)
        rmdot = np.concatenate([self.p/self.M, -self.alpha*self.r])
        #print("rmdot=",rmdot)

        #Hamiltionian part t+1
        #dENew = np.concatenate([np.array(rpNew[0:3])/self.M, self.alpha*np.array(rpNew[3:6])])
        #rmdotNew = self.get_L(rpNew).dot(dENew)
        rmdotNew = np.concatenate([rpNew[3:6]/self.M, -self.alpha*rpNew[0:3]])

        rpres = rpOld-rpNew + self.dt/2*(rmdot+rmdotNew)

        return (rpres[0], rpres[1], rpres[2], rpres[3], rpres[4], rpres[5]) 
        #return rpres

    def m_new(self): #return new r and p
        """
        The function `m_new` calculates new values for `r` and `p` using the `fsolve` function and returns
        the updated values.

        :return: The function `m_new` returns a tuple containing two tuples. The first tuple contains the values `(rx, ry, rz)` and the second tuple contains the values `(px, py, pz)`.
        """
        #calculate
        #(self.r[0], self.r[1], self.r[2], self.p[0], self.p[1], self.p[2]) = fsolve(self.f, (self.r[0], self.r[0], self.r[0], self.p[0], self.p[1], self.p[2]))
        (rx, ry, rz, px, py, pz) = fsolve(self.f, (self.r[0], self.r[1], self.r[2], self.p[0], self.p[1], self.p[2]))
        self.r[0] = rx
        self.r[1] = ry
        self.r[2] = rz
        self.p[0] = px
        self.p[1] = py
        self.p[2] = pz
        #forward Euler:
        #rpOld = np.concatenate([self.r, self.p])
        #dEOld = np.concatenate([self.p/self.M, self.alpha*self.r])
        #print("deOld = ", dEOld)
        #rp = rpOld + self.dt*self.get_L(rpOld).dot(dEOld)
        #rp = rpOld + self.dt*self.get_L(rpOld).dot(np.ones(6))
        #self.r = rp[0:3]
        #self.p = rp[3:6]
        #print(self.r)
        #print(self.p)

        return ((rx, ry, rz), (px, py, pz))

class Particle3DIMR(Particle3DCN):
    def f(self, rpNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the residual of a given input `rpNew` by performing a series of
        mathematical operations.
        
        :param rpNew: The parameter `rpNew` is a numpy array that represents the new values of position and momentum. It has the following structure:
        :return: a tuple containing the values of `rpres[0]`, `rpres[1]`, `rpres[2]`, `rpres[3]`, `rpres[4]`, and `rpres[5]`.
        """
        rpOld = np.array([self.r[0], self.r[1], self.r[2], self.p[0], self.p[1], self.p[2]])
        rp_mid = 0.5*(np.array(rpNew)+rpOld)
        rmdot = np.concatenate([rp_mid[3:6]/self.M, -self.alpha*rp_mid[0:3]])

        rpres = rpOld-rpNew + self.dt*rmdot

        return (rpres[0], rpres[1], rpres[2], rpres[3], rpres[4], rpres[5]) 

class Particle3DNeural(Particle3DCN):
    def __init__(self, M, dt, alpha, init_rx,  init_ry,  init_rz, init_mx, init_my, init_mz, method = "without", name = DEFAULT_folder_name):
        """
        The function initializes a Particle3DNeural object with specified parameters and loads a neural
        network based on the chosen method.
        
        :param M: The parameter M represents the mass of the particle
        :param dt: dt is the time step size for the simulation. It determines how much time elapses between each iteration of the simulation
        :param alpha: The parameter "alpha" is a value used in the calculation of the forces acting on the particle. It represents the strength of the forces relative to the mass of the particle. 
        :param init_rx: The parameter `init_rx` is the initial x-coordinate of the particle's position
        :param init_ry: The parameter `init_ry` represents the initial value for the y-component of the particle's position. It is used to initialize the particle's position in the y-direction
        :param init_rz: The parameter `init_rz` represents the initial value of the z-coordinate of the particle's position. It is used to initialize the particle's position in the z-direction
        :param init_mx: The parameter `init_mx` is the initial x-component of the particle's angular momentum
        :param init_my: The parameter `init_my` is the initial y-component of the particle's angular momentum 
        :param init_mz: The parameter `init_mz` represents the initial value of the z-component of the particle's angular momentum. It is used to initialize the particle's angular momentum in the `__init__` method of the `Particle3DNeural` class
        :param method: The "method" parameter is used to specify the method for solving the equations of motion for the Particle3DNeural object. There are three possible values for this parameter:, defaults to without (optional)
        :param name: The `name` parameter is a string that represents the name of the folder where the saved models are located. It is used to load the pre-trained neural network models for energy and L (angular momentum) calculations
        """
        super(Particle3DNeural, self).__init__(M, dt, alpha, init_rx, init_ry, init_rz, init_mx, init_my, init_mz)
        # Load network
        self.method = method
        if method == "soft":
            self.energy_net = torch.load(name+'/saved_models/soft_jacobi_energy', weights_only=False)  # changed weights_only=False
            self.energy_net.eval()   
            self.L_net = torch.load(name+'/saved_models/soft_jacobi_L', weights_only=False)  # changed weights_only=False
            self.L_net.eval()
        elif method == "without":
            self.energy_net = torch.load(name+'/saved_models/without_jacobi_energy', weights_only=False)  # changed weights_only=False
            self.energy_net.eval()   
            self.L_net = torch.load(name+'/saved_models/without_jacobi_L', weights_only=False)  # changed weights_only=False
            self.L_net.eval()
        elif method == "implicit":
            raise Exception("Implicit solver not yet implemented for P3D.")
            self.energy_net = torch.load(name+'/saved_models/implicit_jacobi_energy', weights_only=False)  # changed weights_only=False
            self.energy_net.eval()
            self.J_net = torch.load(name+'/saved_models/implicit_jacobi_J', weights_only=False)  # changed weights_only=False
            self.J_net.eval()
            def L_net(z):
                L = -1*torch.tensor([[[0.0, z[2], -z[1]],[-z[2], 0.0, z[0]],[z[1], -z[0], 0.0]]])
                return L
            self.L_net = L_net
        else:
            raise Exception("Unkonown method: ", method)
        self.device = next(self.energy_net.parameters()).device
        if self.device.type == "cuda":
            torch.cuda.current_device()

    # Get gradient of energy from NN
    def neural_zdot(self, z):
        """
        The function `neural_zdot` calculates the Hamiltonian of a neural network model given a set of
        input parameters.
        
        :param z: The parameter `z` is a tensor representing the input to the neural network. It is of type `torch.Tensor` and has a shape determined by the specific neural network architecture being used
        :return: the Hamiltonian, which is a scalar value representing the energy of the system.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        En = self.energy_net(z_tensor)

        E_z = torch.autograd.grad(En.sum(), z_tensor, only_inputs=True)[0]
        E_z = torch.flatten(E_z)

        if self.method == "soft" or self.method == "without":
            L = self.L_net(z_tensor).detach().cpu().numpy()[0]
            hamiltonian = np.matmul(L, E_z.detach().cpu().numpy())
        else:
            raise Exception("Implicit not implemented for P3D yet.")
            J, cass = self.J_net(z_tensor)
            J = J.detach().numpy()
            hamiltonian = np.cross(J, E_z.detach().numpy())

        return hamiltonian

    def f(self, rpNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the residual of a given input `rpNew` by subtracting it from `rpOld` and
        adding the product of `self.dt/2` and the sum of `zdo` and `zd`.
        
        :param rpNew: The parameter `rpNew` is a numpy array that represents the new values of `r` and `p`
        :return: the value of the variable "res".
        """

        rpOld = np.concatenate([self.r, self.p])

        zdo = self.neural_zdot(rpOld)
        zd = self.neural_zdot(rpNew)

        res = np.array(rpOld) - np.array(rpNew) + self.dt/2*(zdo + zd)

        return res

    def get_cass(self, z):
        """
        The function `get_cass` takes in a parameter `z`, converts it to a tensor, passes it through a
        neural network `J_net`, and returns the output `cass` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the neural network. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will be computed for this tensor during backprop
        :return: the value of `cass` as a NumPy array.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        J, cass = self.J_net(z_tensor)
        return cass.detach().cpu().numpy()

    def get_L(self, z):
        """
        The function `get_L` takes a tensor `z`, passes it through a neural network `L_net`, and returns the
        output `L` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `L_net` neural network. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will be computed with respect
        :return: the value of L, which is obtained by passing the input z through the L_net neural network and converting the result to a numpy array.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        L = self.L_net(z_tensor).detach().cpu().numpy()[0]
        return L

    def get_E(self, z):
        """
        The function `get_E` takes a parameter `z`, converts it to a tensor, passes it through a neural
        network called `energy_net`, and returns the resulting energy value.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `energy_net` neural network. It is converted to a tensor using `torch.tensor` and is set to have a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will
        :return: the value of E, which is the output of the energy_net model when given the input z.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        E = self.energy_net(z_tensor).detach().cpu().numpy()[0]
        return E

    #@tf.function
    def m_new(self): #return new r and p
        """
        The function `m_new` calculates new values for `r` and `p` using the `fsolve` function and returns
        them as a tuple.

        :return: a tuple containing the updated values of `self.r` and `self.p`.
        """
        #calculate
        (self.r[0], self.r[1], self.r[2], self.p[0], self.p[1], self.p[2])= fsolve(self.f, (self.r[0], self.r[1], self.r[2], self.p[0], self.p[1], self.p[2]))

        # update
        #self.r = rpNew[0:3]
        #self.p = rpNew[3:5]

        return ((self.r[0], self.r[1], self.r[2]), (self.p[0], self.p[1], self.p[2]))

class Particle3DNeuralIMR(Particle3DNeural):
    def f(self, rpNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the residual between the old and new values of `rp` and the time
        derivative of `z` using the midpoint method.
        
        :param rpNew: rpNew is a numpy array that represents the new values of the variables r and p
        :return: the value of the variable "res".
        """
        rpOld = np.concatenate([self.r, self.p])
        rp_mid = 0.5*(np.array(rpNew)+rpOld)

        zd = self.neural_zdot(rp_mid)

        res = np.array(rpOld) - np.array(rpNew) + self.dt*zd

        return res

class Particle3DKeplerIMR(Particle3DIMR):
    def f(self, rpNew):#defines the function f zero of which is sought
        """
        The function f calculates the residual of a given set of inputs and returns it as a tuple.
        
        :param rpNew: The parameter `rpNew` is a numpy array that represents the new values of position and momentum. It has a shape of (6,) and contains the following elements:
        :return: a tuple containing the values of `rpres[0]`, `rpres[1]`, `rpres[2]`, `rpres[3]`, `rpres[4]`, and `rpres[5]`.
        """
        rpOld = np.array([self.r[0], self.r[1], self.r[2], self.p[0], self.p[1], self.p[2]])
        rp_mid = 0.5*(np.array(rpNew)+rpOld)
        r_mid = rp_mid[0:3]
        rmdot = np.concatenate([rp_mid[3:6]/self.M, -self.alpha*r_mid/(np.dot(r_mid, r_mid)**(1.5)+1.0e-06)])

        rpres = rpOld-rpNew + self.dt*rmdot

        return (rpres[0], rpres[1], rpres[2], rpres[3], rpres[4], rpres[5]) 

class Particle2DIMR(object): #Implicit midpont rule
    def __init__(self, M, dt, alpha, init_rx,  init_ry, init_mx, init_my, zeta):
        """
        The function initializes the variables M, dt, alpha, init_rx, init_ry, init_mx, init_my, and zeta.
        
        :param M: The parameter M represents the mass of the particle in the Hamiltonian
        :param dt: dt is the time step size. It determines the size of each time step in the simulation
        :param alpha: The parameter alpha represents the strength of the harmonic potential in the Hamiltonian. It determines how tightly the particle is confined to the potential well. A larger value of alpha corresponds to a stronger confinement
        :param init_rx: The initial x-coordinate of the position vector
        :param init_ry: The parameter `init_ry` represents the initial y-coordinate of the position vector. It is used to specify the initial position of the particle in the y-direction
        :param init_mx: The initial momentum in the x-direction
        :param init_my: The parameter `init_my` represents the initial momentum in the y-direction
        :param zeta: The parameter zeta represents the damping coefficient in the system. It determines the rate at which the system loses energy due to damping. A higher value of zeta leads to faster energy dissipation and damping of the system.
        """
        self.M = M #Hamiltonian = 1/2 p^2/M + 1/2 alpha r^2
        self.r = np.array((init_rx, init_ry))
        self.p = np.array((init_mx, init_my))
        self.alpha = alpha
        self.dt = dt
        self.zeta = zeta

    def get_E(self, m):
        """
        The function `get_E` calculates the total energy `E` based on the input `m` and the class attributes `M` and `alpha`.
        
        :param m: The parameter `m` is a list or tuple containing four elements. The first two elements (`m[0]` and `m[1]`) represent the x and y components of the position vector, while the last two elements (`m[2]` and `m[3]`) represent the momentum.
        :return: the value of the expression 0.5*(m[2]**2 + m[3]**2)/self.M + 0.5 *self.alpha * (m[0]**2 + m[1]**2).
        """
        return 0.5*(m[2]**2 + m[3]**2)/self.M + 0.5 *self.alpha * (m[0]**2 + m[1]**2)

    def get_L(self, m = (0.0, 0.0, 0.0)):
        """
        The function `get_L` returns a 4x4 numpy array representing a transformation matrix.
        
        :param m: The parameter `m` is a tuple with three elements representing the x, y, and z coordinates respectively. The default value for `m` is (0.0, 0.0, 0.0)
        :return: a 4x4 numpy array called L.
        """
        L = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0]])
        return L

    def f(self, rpNew):#defines the function f zero of which is sought
        """
        The function f calculates the residual of the difference between the old and new values of r and p,
        taking into account various factors such as mass, dissipation, and time step.
        
        :param rpNew: The parameter `rpNew` is a numpy array containing the new values of position and momentum. It has the form `[x_new, y_new, px_new, py_new]`
        :return: a tuple containing the values of `rpres[0]`, `rpres[1]`, `rpres[2]`, and `rpres[3]`.
        """
        rpOld = np.array([self.r[0], self.r[1], self.p[0], self.p[1]])
        rp_mid = 0.5*(np.array(rpNew)+rpOld)
        rmdot = np.concatenate([rp_mid[2:4]/self.M, -self.alpha*rp_mid[0:2]])
        rmdot += -self.zeta*np.array((0.0, 0.0, rp_mid[2], rp_mid[3])) #dissipation

        rpres = rpOld-rpNew + self.dt*rmdot
        return (rpres[0], rpres[1], rpres[2], rpres[3]) 

    def m_new(self): #return new r and p
        """
        The function `m_new` returns new values for `r` and `p` by solving a system of equations using the `fsolve` function.
        :return: a tuple containing two tuples. The first tuple contains the values of `rx` and `ry`, and the second tuple contains the values of `px` and `py`.
        """
        (rx, ry, px, py) = fsolve(self.f, (self.r[0], self.r[1], self.p[0], self.p[1]))
        self.r[0] = rx
        self.r[1] = ry
        self.p[0] = px
        self.p[1] = py
        return ((rx, ry), (px, py))

class Particle2DNeural(Particle2DIMR):
    def __init__(self, M, dt, alpha, init_rx,  init_ry, init_mx, init_my, zeta, method = "without", name = DEFAULT_folder_name):
        """
        The function initializes a Particle2DNeural object with specified parameters and loads a neural
        network based on the chosen method.
        
        :param M: The parameter M represents the mass of the particle
        :param dt: dt is the time step size for the simulation. It determines how much time elapses between each iteration of the simulation
        :param alpha: Provides the strength of external field.
        :param init_rx: The parameter `init_rx` represents the initial x-coordinate of the particle's position
        :param init_ry: The parameter `init_ry` represents the initial y-coordinate of the particle in a 2D space. It is used to specify the starting position of the particle along the y-axis
        :param init_mx: The parameter `init_mx` represents the initial x-component of the momentum of the particle
        :param init_my: The parameter `init_my` represents the initial y-coordinate of the momentum of the particle in a 2D system. It is used to initialize the momentum of the particle along the y-axis
        :param zeta: A friction coefficient in the equations.
        :param method: The "method" parameter is used to specify the method for solving the equations of motion in the Particle2DNeural class. It can take one of the following values:, defaults to without (optional)
        :param name: The `name` parameter is a string that represents the name of the folder where the saved models are located. It is used to load the pre-trained neural network models for energy and L (Lagrangian) calculations. 
        """
        super(Particle2DNeural, self).__init__(M, dt, alpha, init_rx, init_ry, init_mx, init_my, zeta)
        # Load network
        self.method = method
        if method == "soft":
            self.energy_net = torch.load(name+'/saved_models/soft_jacobi_energy', weights_only=False)  # changed weights_only=False
            self.energy_net.eval()   
            self.L_net = torch.load(name+'/saved_models/soft_jacobi_L', weights_only=False)  # changed weights_only=False
            self.L_net.eval()
        elif method == "without":
            self.energy_net = torch.load(name+'/saved_models/without_jacobi_energy', weights_only=False)  # changed weights_only=False
            self.energy_net.eval()   
            self.L_net = torch.load(name+'/saved_models/without_jacobi_L', weights_only=False)  # changed weights_only=False
            self.L_net.eval()
        elif method == "implicit":
            raise Exception("Implicit solver not yet implemented for P3D.")
            self.energy_net = torch.load(name+'/saved_models/implicit_jacobi_energy', weights_only=False)  # changed weights_only=False
            self.energy_net.eval()
            self.J_net = torch.load(name+'/saved_models/implicit_jacobi_J', weights_only=False)  # changed weights_only=False
            self.J_net.eval()
            def L_net(z):
                L = -1*torch.tensor([[[0.0, z[2], -z[1]],[-z[2], 0.0, z[0]],[z[1], -z[0], 0.0]]])
                return L
            self.L_net = L_net
        else:
            raise Exception("Unkonown method: ", method)
        self.device = next(self.energy_net.parameters()).device
        if self.device.type == "cuda":
            torch.cuda.current_device()

    # Get gradient of energy from NN
    def neural_zdot(self, z):
        """
        The function `neural_zdot` calculates the Hamiltonian for a given input `z` using neural networks.
        
        :param z: The parameter `z` is a tensor representing the input to the neural network. It is of type `torch.Tensor` and has a shape determined by the specific neural network architecture being used.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        En = self.energy_net(z_tensor)

        E_z = torch.autograd.grad(En.sum(), z_tensor, only_inputs=True)[0]
        E_z = torch.flatten(E_z)

        if self.method == "soft" or self.method == "without":
            L = self.L_net(z_tensor).detach().cpu().numpy()[0]
            hamiltonian = np.matmul(L, E_z.detach().cpu().numpy())
        else:
            raise Exception("Implicit not implemented for P2D yet.")
            J, cass = self.J_net(z_tensor)
            J = J.detach().numpy()
            hamiltonian = np.cross(J, E_z.detach().numpy())
        return hamiltonian

    def f(self, rpNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the residual between the old and new values of `rp` and the time
        derivative of `z`.
        
        :param rpNew: The parameter `rpNew` represents the new values of `r` and `p` that are being passed to the function `f`
        :return: the value of the variable "res".
        """
        rpOld = np.concatenate([self.r, self.p])
        rp_mid = 0.5*(np.array(rpNew)+rpOld)

        zd = self.neural_zdot(rp_mid)
        res = np.array(rpOld) - np.array(rpNew) + self.dt*zd
        return res

    def get_cass(self, z):
        """
        The function `get_cass` takes in a parameter `z`, converts it to a tensor, passes it through a
        neural network `J_net`, and returns the output `cass` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the neural network. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will be computed for this tensor during backprop
        :return: The function `get_cass` returns the value of `cass` as a NumPy array.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        J, cass = self.J_net(z_tensor)
        return cass.detach().cpu().numpy()

    def get_L(self, z):
        """
        The function `get_L` takes a tensor `z`, passes it through a neural network `L_net`, and returns the
        output `L` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `L_net` neural network. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32`. 
        :return: the value of L, which is obtained by passing the input z through the L_net neural network and converting the result to a numpy array.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        L = self.L_net(z_tensor).detach().cpu().numpy()[0]
        return L

    def get_E(self, z):
        """
        The function `get_E` takes a parameter `z`, converts it to a tensor, passes it through a neural
        network called `energy_net`, and returns the resulting energy value.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `energy_net` neural network. It is converted to a tensor using `torch.tensor` and is set to have a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will
        :return: the value of E, which is the output of the energy_net model when given the input z.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        E = self.energy_net(z_tensor).detach().cpu().numpy()[0]
        return E

    def m_new(self): #return new r and p
        """
        The function `m_new` calculates new values for `r` and `p` using the `fsolve` function and returns
        the updated values.
        :return: a tuple containing two tuples. The first tuple contains the values of `self.r[0]` and `self.r[1]`, and the second tuple contains the values of `self.p[0]` and `self.p[1]`.
        """
        #calculate
        (self.r[0], self.r[1], self.p[0], self.p[1])= fsolve(self.f, (self.r[0], self.r[1], self.p[0], self.p[1]))

        return ((self.r[0], self.r[1]), (self.p[0], self.p[1]))

class ShivamoggiIMR(object):
    def __init__(self, M, dt, alpha, init_rx,  init_ry, init_rz, init_u):
        """
        The function initializes the variables M, dt, alpha, init_rx, init_ry, init_rz, and init_u.
        
        :param M: The parameter M represents the mass of the system. It is used in the Hamiltonian equation to calculate the kinetic energy term, which is given by 1/2 p^2/M, where p is the momentum of the system
        :param dt: dt is the time step size used in numerical integration methods to update the system's state. It determines the granularity of the simulation and affects the accuracy and stability of the calculations. Smaller values of dt result in more accurate but slower simulations, while larger values of dt can lead to faster but less accurate
        :param alpha: The parameter alpha represents the strength of the harmonic potential in the Hamiltonian. It determines how tightly the particle is confined in the potential well. A larger value of alpha corresponds to a stronger confinement
        :param init_rx: The initial x-coordinate of the particle's position
        :param init_ry: The parameter `init_ry` represents the initial value of the y-coordinate of the position vector
        :param init_rz: The parameter `init_rz` represents the initial value of the z-coordinate of the position vector
        :param init_u: The `init_u` parameter represents the initial momentum of the system. It is a vector that specifies the initial momentum in each direction (x, y, z)
        """
        self.M = M #Hamiltonian = 1/2 p^2/M + 1/2 alpha r^2
        self.u = init_u
        self.x = np.array((init_rx, init_ry, init_rz))
        self.alpha = alpha
        self.dt = dt

    def get_E(self, m):
        """
        The function `get_E` calculates the value of E using the formula E = m[3]**2 + m[0]**2 - m[2]**2.
        
        :param m: The parameter `m` is a list or tuple containing four elements
        :return: the value of m[3]**2 + m[0]**2 - m[2]**2.
        """
        return m[3]**2 + m[0]**2 - m[2]**2

    def get_UV(self, m):
        """
        The function `get_UV` takes a list `m` as input and returns two tuples (U and V) based on the values in `m`.
        
        :param m: The parameter `m` is a list containing four elements: `u`, `x`, `y`, and `z`
        :return: The function `get_UV` returns a tuple of two tuples. The first tuple contains three values: 0.0, 2*u*(x+z), and 0.0. The second tuple contains three values: x, 0, and -z.
        """
        u = m[0]
        x = m[1]
        y = m[2]
        z = m[3]
        return (0.0, 2*u*(x+z), 0.0), (x, 0, -z)

    def get_L(self, m = (0.0, 0.0, 0.0, 0.0)):
        """
        The function `get_L` calculates and returns a 4x4 matrix L based on the input parameter m.
        
        :param m: The parameter `m` is a tuple of four values `(m[0], m[1], m[2], m[3])`
        :return: a 4x4 numpy array called L.
        """
        U, V = self.get_UV(m)
        L = np.array([
            [0.0, -U[0], -U[1], -U[2]],
            [U[0], 0.0, -V[2], V[1]],
            [U[1], V[2], 0.0, -V[0]],
            [U[2], -V[1], V[0], 0.0]])/(m[0]+m[3])
        return L

    def f(self, mNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the residual of a given input `mNew` by performing a series of
        mathematical operations.
        
        :param mNew: The parameter `mNew` is a list or array containing four values
        :return: a tuple containing the values of mres[0], mres[1], mres[2], and mres[3].
        """
        mOld = np.array([self.u, self.x[0], self.x[1], self.x[2]])
        m_mid = 0.5*(np.array(mNew)+mOld)
        mdot = np.array([-m_mid[0]*m_mid[2], m_mid[3]*m_mid[2], m_mid[3]*m_mid[1]-m_mid[0]**2, m_mid[1]*m_mid[2]])

        mres = mOld-mNew + self.dt*mdot
        return (mres[0], mres[1], mres[2], mres[3]) 

    def m_new(self): #return new r and p
        """
        The function `m_new` returns new values for `u`, `x[0]`, `x[1]`, and `x[2]` by solving the equation
        `f(u, x[0], x[1], x[2]) = 0` using the `fsolve` function.
        :return: a tuple containing the values of u, x, y, and z.
        """
        (u, x, y, z) = fsolve(self.f, (self.u, self.x[0], self.x[1], self.x[2]))
        self.u = u
        self.x[0] = x
        self.x[1] = y
        self.x[2] = z
        return (u, x, y, z)


class ShivamoggiIMR(object):
    def __init__(self, M, dt, alpha, init_rx,  init_ry, init_rz, init_u):
        """
        The function initializes the variables M, dt, alpha, init_rx, init_ry, init_rz, and init_u.
        
        :param M: The parameter M represents the mass of the system. It is used in the Hamiltonian equation to calculate the kinetic energy term, which is given by 1/2 p^2/M, where p is the momentum of the system
        :param dt: dt is the time step size used in numerical integration methods to update the system's state. It determines the granularity of the simulation and affects the accuracy and stability of the calculations. Smaller values of dt result in more accurate but slower simulations, while larger values of dt can lead to faster but less accurate
        :param alpha: A parameter in the Shivamoggi equations
        :param init_rx: The initial x-coordinate of the particle's position
        :param init_ry: The parameter `init_ry` represents the initial value of the y-coordinate of the position vector
        :param init_rz: The parameter `init_rz` represents the initial value of the z-coordinate of the position vector
        :param init_u: The `init_u` parameter represents the initial momentum of the system. It is a vector that specifies the initial momentum in each direction (x, y, z)
        """
        self.M = M #Hamiltonian = 1/2 p^2/M + 1/2 alpha r^2
        self.u = init_u
        self.x = np.array((init_rx, init_ry, init_rz))
        self.alpha = alpha
        self.dt = dt

    def get_E(self, m):
        """
        The function `get_E` calculates the value of E using the formula E = m[3]**2 + m[0]**2 - m[2]**2.
        
        :param m: The parameter `m` is a list or tuple containing four elements
        :return: the value of m[3]**2 + m[0]**2 - m[2]**2.
        """
        return m[3]**2 + m[0]**2 - m[2]**2

    def get_UV(self, m):
        """
        The function `get_UV` takes a list `m` as input and returns a tuple of two tuples, where the first
        tuple is calculated based on the values of `u`, `x`, and `z` from `m`, and the second tuple is
        calculated based on the value of `x` and `z` from `m`.
        
        :param m: The parameter `m` is a list containing four elements: `u`, `x`, `y`, and `z`
        :return: The function `get_UV` returns a tuple of two tuples. The first tuple contains three values: 0.0, 2*u*(x+z), and 0.0. The second tuple contains three values: x, 0, and -z.
        """
        u = m[0]
        x = m[1]
        y = m[2]
        z = m[3]
        return (0.0, 2*u*(x+z), 0.0), (x, 0, -z)

    def get_L(self, m = (0.0, 0.0, 0.0, 0.0)):
        """
        The function `get_L` calculates and returns a 4x4 matrix L based on the input parameter m.
        
        :param m: The parameter `m` is a tuple of four values `(m[0], m[1], m[2], m[3])`
        :return: The function `get_L` returns a 4x4 numpy array.
        """
        U, V = self.get_UV(m)
        L = np.array([
            [0.0, -U[0], -U[1], -U[2]],
            [U[0], 0.0, -V[2], V[1]],
            [U[1], V[2], 0.0, -V[0]],
            [U[2], -V[1], V[0], 0.0]])/(m[0]+m[3])
        return L

    def f(self, mNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the residual of a given input `mNew` by performing a series of
        mathematical operations.
        
        :param mNew: The parameter `mNew` is a list or array containing the new values for `u`, `x[0]`, `x[1]`, and `x[2]`
        :return: a tuple containing the values of mres[0], mres[1], mres[2], and mres[3].
        """
        mOld = np.array([self.u, self.x[0], self.x[1], self.x[2]])
        m_mid = 0.5*(np.array(mNew)+mOld)
        mdot = np.array([-m_mid[0]*m_mid[2], m_mid[3]*m_mid[2], m_mid[3]*m_mid[1]-m_mid[0]**2, m_mid[1]*m_mid[2]])

        mres = mOld-mNew + self.dt*mdot
        return (mres[0], mres[1], mres[2], mres[3]) 

    def m_new(self): #return new r and p
        """
        The function `m_new` returns new values for `u`, `x[0]`, `x[1]`, and `x[2]` by solving the equation
        `f(u, x[0], x[1], x[2]) = 0` using the `fsolve` function.
        :return: a tuple containing the values of u, x, y, and z.
        """
        (u, x, y, z) = fsolve(self.f, (self.u, self.x[0], self.x[1], self.x[2]))
        self.u = u
        self.x[0] = x
        self.x[1] = y
        self.x[2] = z
        return (u, x, y, z)

class ShivamoggiNeural(ShivamoggiIMR):
    def __init__(self, M, dt, alpha, init_rx,  init_ry, init_rz, init_u, method = "without", name = DEFAULT_folder_name):
        """
        The function initializes a ShivamoggiNeural object with specified parameters and loads the
        appropriate neural network models based on the chosen method.
        
        :param M: The parameter M represents the mass of the system. It is a scalar value
        :param dt: The parameter `dt` represents the time step size in the simulation. It determines how much time elapses between each iteration of the simulation
        :param alpha: The parameter "alpha" is a value used in the ShivamoggiNeural class initialization. It
        is a scalar value that represents a coefficient used in the calculations performed by the class. The specific purpose and meaning of this parameter may depend on the context and implementation of the Shivamoggi equations
        :param init_rx: The parameter `init_rx` is the initial x-coordinate of the position vector. It represents the starting position of the object in the x-direction
        :param init_ry: The parameter `init_ry` represents the initial value for the y-coordinate of the position vector. It is used in the initialization of the `ShivamoggiNeural` class
        :param init_rz: The parameter `init_rz` represents the initial value for the z-coordinate of the position vector. It is used in the initialization of the `ShivamoggiNeural` class
        :param init_u: The parameter `init_u` represents the initial value of the variable `u` in the ShivamoggiNeural class. It is used to initialize the state of the system
        :param method: The "method" parameter in the code snippet refers to the method used for solving the Shivamoggi equations. There are three possible options:, defaults to without (optional)
        :param name: The `name` parameter is a string that represents the folder name where the saved models are located. It is used to load the pre-trained neural network models for energy and L (Lagrangian) calculations. The `name` parameter is used to construct the file paths for loading the models
        """
        super(ShivamoggiNeural, self).__init__(M, dt, alpha, init_rx,  init_ry, init_rz, init_u)
        # Load network
        self.method = method
        if method == "soft":
            self.energy_net = torch.load(name+'/saved_models/soft_jacobi_energy', weights_only=False)  # changed weights_only=False
            self.energy_net.eval()   
            self.L_net = torch.load(name+'/saved_models/soft_jacobi_L', weights_only=False)  # changed weights_only=False
            self.L_net.eval()
        elif method == "without":
            self.energy_net = torch.load(name+'/saved_models/without_jacobi_energy', weights_only=False)  # changed weights_only=False
            self.energy_net.eval()   
            self.L_net = torch.load(name+'/saved_models/without_jacobi_L', weights_only=False)  # changed weights_only=False
            self.L_net.eval()
        elif method == "implicit":
            raise Exception("Implicit solver not yet implemented for Shivamoggi.")
            self.energy_net = torch.load(name+'/saved_models/implicit_jacobi_energy', weights_only=False)  # changed weights_only=False
            self.energy_net.eval()
            self.J_net = torch.load(name+'/saved_models/implicit_jacobi_J', weights_only=False)  # changed weights_only=False
            self.J_net.eval()
            def L_net(z):
                L = -1*torch.tensor([[[0.0, z[2], -z[1]],[-z[2], 0.0, z[0]],[z[1], -z[0], 0.0]]])
                return L
            self.L_net = L_net
        else:
            raise Exception("Unkonown method: ", method)
        self.device = next(self.energy_net.parameters()).device
        if self.device.type == "cuda":
            torch.cuda.current_device()

    # Get gradient of energy from NN
    def neural_zdot(self, z):
        """
        The function `neural_zdot` calculates the Hamiltonian of a neural network given a set of input
        values.
        
        :param z: The parameter `z` is a tensor representing the input to the neural network. It is of type `torch.Tensor` and has a shape determined by the dimensions of the input data
        :return: The function `neural_zdot` returns the variable `hamiltonian`.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        En = self.energy_net(z_tensor)

        E_z = torch.autograd.grad(En.sum(), z_tensor, only_inputs=True)[0]
        E_z = torch.flatten(E_z)

        if self.method == "soft" or self.method == "without":
            L = self.L_net(z_tensor).detach().cpu().numpy()[0]
            hamiltonian = np.matmul(L, E_z.detach().cpu().numpy())
        else:
            raise Exception("Implicit not implemented for HT yet.")
            J, cass = self.J_net(z_tensor)
            J = J.detach().numpy()
            hamiltonian = np.cross(J, E_z.detach().numpy())
        return hamiltonian

    def f(self, mNew):#defines the function f zero of which is sought
        """
        The function `f` calculates the difference between the old and new values of `m` and adds the
        product of the time step and the derivative of `z` to it.
        
        :param mNew: The parameter `mNew` represents the new values of `m` that are passed to the function `f`
        :return: the difference between the old values and the new values, plus the product of the time step and the derivative of the neural network with respect to the midpoint of the old and new values.
        """
        mOld = np.array([self.u, self.x[0], self.x[1], self.x[2]])
        m_mid = 0.5*(np.array(mNew)+mOld)

        zd = self.neural_zdot(m_mid)
        res = np.array(mOld) - np.array(mNew) + self.dt*zd
        return res

    def get_cass(self, z):
        """
        The function `get_cass` takes in a parameter `z`, converts it to a tensor, passes it through a
        neural network `J_net`, and returns the output `cass` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the neural network. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will be computed for this tensor during backprop
        :return: the value of `cass` as a NumPy array.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        J, cass = self.J_net(z_tensor)
        return cass.detach().cpu().numpy()

    def get_L(self, z):
        """
        The function `get_L` takes a tensor `z`, passes it through a neural network `L_net`, and returns the
        output `L` as a numpy array.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `L_net` neural network. It is converted to a tensor using `torch.tensor` with a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will be computed with respect
        :return: the value of L, which is obtained by passing the input z through the L_net neural network and converting the result to a numpy array.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        L = self.L_net(z_tensor).detach().cpu().numpy()[0]
        return L

    def get_E(self, z):
        """
        The function `get_E` takes a parameter `z`, converts it to a tensor, passes it through a neural
        network called `energy_net`, and returns the resulting energy value.
        
        :param z: The parameter `z` is a numerical input that is used as an input to the `energy_net` neural network. It is converted to a tensor using `torch.tensor` and is set to have a data type of `torch.float32`. The `requires_grad=True` argument indicates that gradients will
        :return: the value of E, which is the output of the energy_net model when given the input z.
        """
        z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True, device=self.device)
        E = self.energy_net(z_tensor).detach().cpu().numpy()[0]
        return E

    def m_new(self): #return new r and p
        """
        The function `m_new` returns the values of `u`, `x[0]`, `x[1]`, and `x[2]` after solving the
        equation `f` using the `fsolve` function.

        :return: a tuple containing the values of self.u, self.x[0], self.x[1], and self.x[2].
        """
        #calculate
        (self.u, self.x[0], self.x[1], self.x[2])= fsolve(self.f, (self.u, self.x[0], self.x[1], self.x[2]))

        return (self.u, self.x[0], self.x[1], self.x[2])
