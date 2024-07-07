from PSO import run_evolution
import numpy as np

#electorde positions [um]
PosX = np.array([10000, 10000])
PosY = np.array([-5500, -5500]) 
PosZ = np.array([1500, -1500])

#Pulse duration [ms]
Dur = 0.2

file = 'test' #filename
generation_limit = 2 #number of iterations


run_evolution(PosX, PosY, PosZ, Dur, file, generation_limit)