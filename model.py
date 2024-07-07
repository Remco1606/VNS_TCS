
from neuron import h
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

h.load_file('stdgui.hoc')
h.load_file('interpxyz.hoc')

# units are mv and mA

dedx = 1 # dedx  gradient in v/m

def init_model():
    h.load_file('MRGaxon.hoc')          # import the MRG model. In this file the paramaters of the neuron can be altered.

    for sec in h.allsec():
        sec.insert('xtra')              # Insert 'xtra' in de sections. This couples the location of the extracellular stimulation to the potential at the nerve section

    h.define_shape()        # crestes cordinates of the nodes in 3D
    h.grindaway()           # in interpxyz.hoc, determines interpolated locations of nodes

    for sec in h.allsec():
        for seg in sec:
            #seg.xtra._ref_ex = seg._ref_e_extracellular
            h.setpointer(sec(seg.x)._ref_e_extracellular, 'ex', sec(seg.x).xtra)        # Pointer so extracellular can call correctly on xtra

    v = {}
    Stimules = {}
    for sec in h.allsec():
        v[str(sec)] = h.Vector().record(sec(0.5)._ref_v)                                # Setup recording vector
        Stimules[str(sec)] = h.Vector().record(sec(0.5)._ref_e_extracellular)
    tvec = h.Vector().record(h._ref_t)                                                  # Setup time vector


    #print("Init_model complete")

    return v, tvec, Stimules

def calcesI(x, y, z, i, sigma_e = 2.76e-07):   # x,y,z are electrode coordinates, in this section the transfer resistance is calculated for the electrodes 
    for sec in h.allsec():
        for seg in sec:
            r = np.sqrt((x - seg.x_xtra)**2 + y**2 + (z - seg.z_xtra)**2)
            rx = (1e-3)/(4*np.pi*sigma_e*r)
            if i == 1:
                seg.rx1_xtra = rx
            elif i == 2:
                seg.rx2_xtra = rx
            elif i == 3:
                seg.rx3_xtra = rx
            elif i == 4:
                seg.rx4_xtra = rx
            elif i == 5:
                seg.rx5_xtra = rx
            elif i == 6:
                seg.rx6_xtra = rx    



#this function starts the neuron simulation
def Stimulation(v, paterns, dt, t0, tstop, Del, Dur, genome): 
   
   if len(genome) > 0 :
       amp1 = genome[0]
   else: 
       amp1 = np.zeros(paterns)
   if len(genome) > 1 :
       amp2 = genome[1]
   else: 
       amp2 = np.zeros(paterns)
   if len(genome) > 2 :
       amp3 = genome[2]
   else: 
       amp3 = np.zeros(paterns)
   if len(genome) > 3 :
       amp4 = genome[3]
   else: 
       amp4 = np.zeros(paterns)
   if len(genome) > 4 :
       amp5 = genome[4]
   else: 
       amp5 = np.zeros(paterns)
   if len(genome) > 5 :
       amp6 = genome[5]
   else: 
       amp6 = np.zeros(paterns)

   h.dt = dt
   total_steps = int((tstop - t0) / dt)
   delay_steps = int(Del / dt)
   pattern_steps = int(Dur / dt / paterns)
   end_steps = int(total_steps - delay_steps - (Dur/dt))

######## here the waveforms vector is made so that for every timestep there is a amplitude 
   data1 = np.repeat(amp1, pattern_steps)
   data2 = np.repeat(amp2, pattern_steps)
   data3 = np.repeat(amp3, pattern_steps)
   data4 = np.repeat(amp4, pattern_steps)
   data5 = np.repeat(amp5, pattern_steps)
   data6 = np.repeat(amp6, pattern_steps) 
   
   data1 = np.concatenate((np.zeros(delay_steps), data1, np.zeros(end_steps)))
   data2 = np.concatenate((np.zeros(delay_steps), data2, np.zeros(end_steps)))
   data3 = np.concatenate((np.zeros(delay_steps), data3, np.zeros(end_steps)))
   data4 = np.concatenate((np.zeros(delay_steps), data4, np.zeros(end_steps)))
   data5 = np.concatenate((np.zeros(delay_steps), data5, np.zeros(end_steps)))
   data6 = np.concatenate((np.zeros(delay_steps), data6, np.zeros(end_steps)))
   
   t = np.linspace(t0, tstop, total_steps) # Vector with time stamp
   
   global Stim1, Stim2, Stim3, Stim4, Stim5, Stim6, stim_time
   h.tstop = tstop
    
   Stim1=h.Vector(data1)
   Stim2=h.Vector(data2)
   Stim3=h.Vector(data3)
   Stim4=h.Vector(data4)
   Stim5=h.Vector(data5)
   Stim6=h.Vector(data6)
   stim_time = h.Vector(t)
    
   Stim1.play(h._ref_is1_xtra,stim_time, 1)
   Stim2.play(h._ref_is2_xtra,stim_time, 1)
   Stim3.play(h._ref_is3_xtra,stim_time, 1)
   Stim4.play(h._ref_is4_xtra,stim_time, 1)
   Stim5.play(h._ref_is5_xtra,stim_time, 1)
   #Stim6.play(h._ref_is6_xtra,stim_time, 1)

   h.run()


# This function finds for two electrodes what the threhold current is when you stimulate with both of them
def FindStimulation(PosX, PosY, PosZ, genome, Dur, electrodes):
    amp1 = np.array(genome[0])
    amp2 = np.array(genome[1])
    #amp3 = genome[2]
    #amp4 = genome[3]
    #amp5 = genome[4]
    dt = 0.001
    Del = 0.5 # delay until start stimulation
    t0 = 0
    tstop = max([3, Del+Dur+2])
    paterns = np.size(amp1) 
    mult = 1
    v, tvec, Stimules = init_model()
    i = 0
    exit = False
    flag = 0
    for k in range(np.size(PosX)):
            calcesI(PosX[k], PosY[k], PosZ[k], (k+1))      #negative sign becuase we move the electrode and not the axon
    
    while i < 34 and not exit:

        Stimulation(v, paterns, dt, t0, tstop, Del, Dur, np.array([amp1*mult, amp2*mult]))#, amp3, amp4)
        recdat = np.array(v['node[20]'])
        if np.max(recdat) > 0:
            if flag == 1:
                exit = True
            else:
                mult = mult - 0.025
        elif np.max(recdat) <= 0:
           flag = 1
           mult = mult + 0.03
           #mult = mult + ((10**-(np.floor(np.log10(np.max(amp1)))))*100)

        i += 1

    #print('mult = ' + str(mult))
    for j in range(electrodes):
        genome[j] *= mult
        genome[j] = np.round(genome[j])
    
    return genome



# This is the main function that loops over the different raster points and calles on neuron to check if the fiber is activated or not

def GeneticAlgorithm(PosX, PosY, PosZ, genome, Dur):
    dt = 0.001 #timestep
    Del = 0.5 # delay until start stimulation
    t0 = 0  #time that the simulation starts
    tstop = max([3, Del+Dur+2]) # time the simulation stops
    paterns = np.size(genome[0])    #amount of amplitude changes
    
    v, tvec, Stimules = init_model()        #Functions where NEURON model is initialized

    #Grid points
    walkY = np.array([-2500, -2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000, 2500])
    walkZ = np.array([-2500, -2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000, 2500])

    # generate array to keep track of stimulation location
    cordY = np.zeros(np.size(walkY)*np.size(walkZ))                           
    cordZ = np.zeros(np.size(walkY)*np.size(walkZ))

    Excitation = np.zeros(np.size(walkY)*np.size(walkZ))    # Vector that keeps track if point is activated or not 
    colist = ['walkY', 'walkZ', 'tvec', 'V', 'Wave', 'Stimules']
    Nodes = pd.DataFrame(columns = colist)
    
    start_time = time.time()
    counter = 0
    for i in range(np.size(walkY)):
        for j in range(np.size(walkZ)):
            for k in range(np.size(PosX)):
                calcesI(PosX[k], (PosY[k]-walkY[i]), (PosZ[k]-walkZ[j]), (k+1))      #calculate the transfer resistance for every electrode to the fiber, in practice we move the electrode and not the fiber
            Stimulation(v, paterns, dt, t0, tstop, Del, Dur, genome)                   # Calculate the membrane potential
            recdat = np.array(v['node[20]'])                                            # save the membrane potential over time at node[20] in recdat vector
            Excitation[counter] = (1 if np.max(recdat) > 0 else 0)                      #if the membrane potential at node[20] get higher then 0 we register that as a action potential created
            #Uncommend below and comment above if the PSO optimization code is used
            #Excitation[counter] = (((np.sqrt((cordZ[j]/100)**2 + (cordY[i]/100)**2))**3+10) if np.max(recdat) > 0 else 0)

            if (walkY[i] == 0 and walkZ[j] == 0):      #This statment is only here for implemantation with genetic algorithm.
                zero_coordinate = counter
                
            cordY[counter] = walkY[i]
            cordZ[counter] = walkZ[j]
            counter = counter +1
    end_time = time.time()

    
    return Excitation, zero_coordinate, cordY, cordZ

