'''

    Name:               Problem Set 2 - Question 1, CHEM 5641

    Description:        Complete the search of a, b values for trial wave function Types 2 and 3 for the helium atom
                        (i.e. find the optimal values of a, b that gives the lowest energy for the helium atom). Also,
                        repeat what we did for trial wave function Type 1 to find the optimal a value and the corresponding
                        energy value. 

    Author:             John Ferrier, NEU Physics

'''

#Import Modules
import numpy as np
from scipy import special
import matplotlib.pyplot as plt


class HeliumWaveFunction():

    def __init__( self,
                max_R = 8.,
                Resolution = 50 ):
        '''

            Inputs:
                max_r           - Maximum radius to consider                    ( float )
                Resolution      - Resolution of steps                           ( int )
                    (higher = more accurate & more computationally intense)


            Builds ψ_i as (if only using 1 r):

                  ψ_i( 0, 0, 0 ) ______ ψ_i( 0, 0, π )
                             r /      /|
             ψ_i( R_0, 0, 0 ) /______/ |
                             |       | | ψ_i( 0, 2π, π )         where ψ_i = R( r )*Y_nlm( θ, φ )
                           θ |       | /
                             |_______|/
           ψ_i( R_0, 2π, 0 )    φ    ψ_i( R_0, 2π, π )


           Builds Ψ as:

                                 ______                     ______
                               /      /|                  /      /|
                              /______/ |                 /______/ |
                  Ψ   = c_1 *|       | |   + ... + c_i *|       | |
                             |  ψ_1  | /                |  ψ_i  | /
                             |_______|/                 |_______|/

        '''

        #Initialize variables
        self.Z          = 2                                             #Atomic Number
        self.resolution = Resolution                                    #Steps to make in linspace
        self.n          = []                                            #Array of n values for spherical harmonics n >= 0
        self.m          = []                                            #Array of m values for spherical harmonics -m <= n <= m
        self.max_R      = max_R

        #Buld r linspace
        self.r1         = np.linspace( 0.0001, self.max_R, self.resolution )           #Array of r1 from 0.0001 to max_R (must not be zero due to nucleus)
        self.r2         = np.linspace( 0.0002, self.max_R+0.0001, self.resolution )    #Array of r2 Shifted from r1 to avoid dividing by 0

        #Build scalar range for wave functions
        self.ζ1         = np.linspace( 0.0001, 5., self.resolution )
        self.ζ2         = np.linspace( 0.0001, 5., self.resolution )

        #Build the meshes
        self.R1, self.R2 = np.meshgrid( self.r1, self.r2 )              #Returns three 2D meshes
        
        #Build Derivative constants
        self.dr         = self.r1[1] - self.r1[0]                       #Change in r1
        self.dV         = self.dr**2                                    #Change in volume V. Don't forget the ( r_1^2 r_2^2 ) for integration

        #Define final lists for Ψ
        self.ψ          = []                                            #List of basis functions such that ψ_i( r )
        self.Ψ          = []                                            #4D Tensor of Ψ such that Ψ( r1, r2 ) = Σ c_i*ψ_i

        #Build n and m arrays
        self._build_n_m()

        #Build Energies
        self._buildEnergies()

    #Utilizes self.Z to build the m and n arrays
    def _build_n_m( self ):
        
        self.n.append(0)
        self.m.append(0)

    #Build Potential Energy function V( r, θ, φ ) 
    def _V( self, Ψ ):
        
        temp_V = np.conjugate( Ψ )*Ψ*( ( self.Z/self.r1 ) + ( self.Z/self.r2 ) - ( 1./( self.r1 - self.r2 ) ) )*4.*np.pi*( self.r1**2 )*( self.r2**2 )*self.dV
        return np.sum( temp_V )

    #Apply Kinetic Energy operator T( r, θ, φ )
    def _T( self, Ψ, ζ, R ):
            
        #Only radial changes
        return np.sum( ( ζ**(3./2.) )*( ζ**2 )*np.exp( -ζ*R )*np.conjugate( Ψ )*Ψ*4.*np.pi*( R**2 )*self.dV/self.dr )

    #Normalization factor
    def _N( self, ζ ):
        return -8*np.pi*( ζ**2 )*self._Harmonic()

    #Build Radial Component of Ψ
    def _Radial( self, R, ζ ):
        return 2.*( ζ**(3./2.) )*np.exp( -ζ*R )

    #Build Harmonic component of Ψ
    def _Harmonic( self ): 
        return 1./( 2.*np.sqrt( np.pi ) )

    #Builds a tensor of values for multiple m and n components over all φ and θ
    def _buildWaveFunction( self, ζ1, ζ2 ):
        
        #Build first electron wave function with ζ1
        ψ1      = np.array( self._Radial( self.r1, ζ1 )*self._Harmonic() )

        #Build second electron wave function with ζ2
        ψ2      = np.array( self._Radial( self.r2, ζ2 )*self._Harmonic() )

        #Add values
        return ( ψ1 + ψ2, ψ1, ψ2 ) 
        
    #Build the Associated Energies spectrum
    def _buildEnergies( self ):

        E       = np.zeros( (len( self.ζ1), len( self.ζ2 )) )

        for ζ1 in range( len( self.ζ1 ) ):

            for ζ2 in range( len( self.ζ2 ) ):

                #Build wave function
                Ψ, ψ1, ψ2   = self._buildWaveFunction( self.ζ1[ζ1], self.ζ2[ζ2] )
                N1          = self._N(  self.ζ1[ζ1] )
                N2          = self._N(  self.ζ2[ζ2] )
                T1          = self._T( ψ1, self.ζ1[ζ1], self.r1 )
                T2          = self._T( ψ2, self.ζ2[ζ2], self.r2 )
                V           = self._V( Ψ )
                E[ζ1][ζ2]   = T1/N1 + T2/N2 + V/( N1 + N2 )

        E_min   = np.amin( E )

        #plt.contourf( self.ζ1, self.ζ2, E, 200, cmap = 'magma' )
        #plt.scatter( min_c1, min_c2, c = '#03FDFC' )
        #plt.title( "ψ_type Energy Field" )
        #plt.xlabel('C1')
        #plt.ylabel('C2')
        #plt.show()

        print( f"Minimum energy = {E_min} " ) #\nAt point (C1, C2) = ({min_c1}, {min_c2})" )


atom    = HeliumWaveFunction()
