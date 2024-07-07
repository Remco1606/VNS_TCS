Files:

model.py -> the code that creates the activation raster using NEURON

run_raster.py -> simple code to create a raster where you can change the amount of electrodes, their locations and the stimulating waveform

PSO.py -> the code that tuns contains the PSO
run_PSO.py -> Code to simply run the PSO with the variable you commonly want to change



If the particle swarm optimization is used in the model file the excitation should be uncommented so that it calculates the fitness value and not just gives back 1 or 0

The Partice Swarm Optimization which is run_PSO.py only works on a linux server for now due to the parallel processing.