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
import matplotlib.pyplot as plt

'''
Build Functions for Energy and Normalization
'''

#Normalization Factor <ψ|ψ>
def N( A, B, waveType = 1 ):
    
    if waveType == 1:
        n   = 1./( 2.*A**6. )

    elif waveType == 2:
        n   = ( A**2. + 6.*A*B + B**2. )*( A**4. + 14.*( A**2. )*B**2. + B**4. )/( ( A**3. )*( B**3. )*( A + B )**6. )

    elif waveType == 3:
        n   = ( 8.*A**2. - 5.*A*B + B**2. )/( 16.*( A**3. )*( ( A - B )**5. ) )
    
    return n

#Kinetic Energy <ψ|T|ψ>
def T( A, B, waveType = 1 ):
    
    if waveType == 1:
        t   = 1./( 2.*A**4. )

    elif waveType == 2:
        t   = ( 1./6. )*( 3./( A*B**3. ) + 3./( B*A**3. ) - ( 384.*A**2. )/( ( A + B )**6. ) + 384.*A/( ( A + B )**5. ) )

    elif waveType == 3:
        t   = -( -8.*A**3. + 7.*( A**2. )*B - 4.*A*B**2. + B**3. )/( 16.*( A**3. )*( A - B )**4. )
    
    return t

#Kinetic Energy <ψ|V|ψ>
def V( A, B, Z, waveType = 1 ):
    
    if waveType == 1:
        v   = ( 5. - 16.*Z )/( 16.*A**5. )

    elif waveType == 2:
        v   = (-1./( ( A**3. )*( B**3. )*( A + B )**5. ) )*( Z*( A**2. + 6.*A*B + B**2. )*( A**4. + 14.*( A**2. )*( B**2. ) + B**4. ) - A*B*( A**4. + 5.*( A**3. )*B + 28.*( A**2. )*( B**2. ) + 5.*A*B**3. + B**4. ) )

    elif waveType == 3:
        v   = ( ( 5. - 16.*Z )*A**2. + 4.*( Z - 1 )*A*B + B**2. )/( 16.*( A**3 )*( ( A - B )**4. ) )

    return v

#Total Energy <ψ|H|ψ>/<ψ|ψ>
def E( A, B, Z, waveType = 1 ):
    return ( T( A, B, waveType ) + V( A, B, Z, waveType ) )/N( A, B, waveType )

'''
Main Part of code
'''

#Define Exact Energies
ExE         = np.array( [ -0.527592, -2.90372, -7.27991, -13.6556, -22.0310, -32.4062, -44.7814, -59.1566, -75.5317, -93.9068 ] )

#Define the Atomic type
Z           = 2

#Define Wavetypes
ψ_types     = [ 1, 2, 3 ]

#Define Ranges for a and b (b is shifted to avoid division by 0 errors)
a           = np.linspace( 0.00001, 4.0, num = 2000 )
b           = np.linspace( 0.000011, 4.0001, num = 2000 )

#Build Mesh grid of a and b
A, B        = np.meshgrid( a, b )

#Define list for energy meshes to minimize
Energies    = []
min_Energy  = []
PctError    = []
min_ab      = []

#Get Exact Answer for Helium
ExactE      = ExE[ Z-1 ]

#Initialize the plotting figures
fig, axs    = plt.subplots( 1, len( ψ_types ) )
fig.suptitle( "ψ_type Energy Landscapes" )


#Determine minimum energies for ψ_type = [ 1, 2, 3 ]
for i in range( len( ψ_types ) ):

    #Calculate and append the Energies
    Energies.append( E( A, B, Z, ψ_types[i] ) )

    #Find the minimum Energy
    E_min   = np.amin( Energies[-1] )

    #Find the A, B values of the minimum Energy
    result  = np.where( Energies[-1] == E_min )
    min_a   = a[ result[1][0] ]
    min_b   = b[ result[0][0] ]

    #Append Minimum Values
    min_ab.append( np.array( [ min_a, min_b ] ) )

    #Append the minimum Energy
    min_Energy.append( E_min )

    #Calculate and Append the error in the energy
    PctError.append( 100.*np.abs( ExactE - E_min )/np.abs( ExactE ) )

    #Build Plot
    axs[i].contourf( A, B, Energies[i], 200, cmap = 'magma' )
    axs[i].scatter( min_ab[-1][0], min_ab[-1][1], c = '#03FDFC' )
    axs[i].set_title( f"ψ_type {ψ_types[i]} Energy Field" )
    axs[i].set( xlabel = 'a', ylabel = 'b' )

    #Output Information
    print( f"ψ_type = {ψ_types[i]}\nMinimum Calculated Energy = {E_min}\nExact Minimum Energy = {ExactE}\nPercent Error = {round( PctError[-1], 2 )}%\nMinimum Point (a, b) = ( {min_ab[-1][0]}, {min_ab[-1][1]} )\n" )

#Determine Which wave function type is best for Helium
min_PCT = np.inf
min_ψ   = 0
for i in range( len( PctError ) ):

    #Check if the current percent error is less than the previous one
    if PctError[i] < min_PCT:
        min_PCT = PctError[i]
        min_ψ   = i + 1

#Print out which ψ type is best for Helium
print( f"The best ψ type for Z = {Z} is ψ_type {min_ψ}" )

#Show the last energy landscape with a contour plot
plt.show()
