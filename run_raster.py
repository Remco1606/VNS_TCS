from model import GeneticAlgorithm
import numpy as np
import matplotlib.pyplot as plt

#Electrode coordinates [um] 
PosX = [10000, 10000]
PosY = [-5500, -5500]
PosZ = [1500, -1500]

#Stimulion waveform amplitudes [[waveform_1], [waveform_2], ... [waveform_i]] needs to be a even number of amplitude changes per waveform
waveform = np.array([[-7400, 0], [-7400,0]])

# Pulse Duration
Dur = 0.2

#filename
filename = 'test'

#call the function that generates the data for the excitation map
Excitation, zero_coordinate, cordY, cordZ = GeneticAlgorithm(PosX, PosY, PosZ, waveform, Dur)


#create excitation map
m = []
color = []
fitness = 0
for i in range(len(Excitation)):
     #print(str(df.loc[i].at['Excitation']))
      
    if Excitation[i] == 1:
        m.append('^')
        color.append('green')
    else:
        m.append('o')
        color.append('red')
   
    plt.scatter(x = cordY[i], y = cordZ[i], marker = m[i], color =color[i])
    
plt.savefig(r'images/ExcitationMap_'+str(filename)+'.png')
plt.close('all')