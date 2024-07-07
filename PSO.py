from model import GeneticAlgorithm 
from model import FindStimulation
from neuron import h
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool
from copy import deepcopy

#Create waveform vector where one vector is filled with vectors for each electrode and their stimulation waveform
def generate_genome(paterns, electrodes, th):
    Genome = [None]*electrodes #The input to the function so in our case a series of amplitudes that is fed to a electrode
    Amplitudes = np.zeros(paterns)
    amp_choices = [-2*th, -th, -.5*th, 0, 0.5*th, th, 2*th]         #amplitude that can be randomely chosen out of
    for i in range(electrodes):
        for j in range(paterns):
            Amplitudes[j] = np.random.choice(amp_choices)
        Genome[i] = Amplitudes.copy()

    return Genome

#Call function that optimizes a waveform vector so it just activates the target, so the threshold waveform.
def GeneOptimizerWrapper(args):
    PosX, PosY, PosZ, individual, Dur, electrodes = args
    return FindStimulation(PosX, PosY, PosZ, individual, Dur, electrodes)

# This function makes it so optimization can be done parallel optimizing availible cores
def GenomeOptimizer(PosX, PosY, PosZ, Population, Dur, electrodes, population_size, cores):

    # Prepare arguments for parallel execution
    args_list = [(PosX, PosY, PosZ, Population[i], Dur, electrodes) for i in range(population_size)]

    # Run GeneticAlgorithm in parallel
    with Pool(cores) as pool:
        Population = pool.map(GeneOptimizerWrapper, args_list)

    return Population

# Generate the complete set of waveforms for the optimization algorithm
def generate_population(population_size, paterns, electrodes, PosX, PosY, PosZ, Dur, th):      # Make a group of genomes which becomes a genration
    Population = [None]*population_size # a group of genomes which make up a generation
    for i in range(population_size):
        Population[i] = generate_genome(paterns, electrodes, th)
    print('Starting population = ' + str(Population))    
    return Population

#Call on the function to generate the activation raster
def GeneticAlgorithmWrapper(args):
    PosX, PosY, PosZ, individual, Dur = args
    return GeneticAlgorithm(PosX, PosY, PosZ, individual, Dur)

#Calculate the fitness value for ech waveform set 
def fitness_parallel(Excitation, zero_coordinate):
    penalty = 810000#100000     #If the target is not stimulated a penalty is added
    fitness = np.sum(Excitation) 
    if Excitation[zero_coordinate] == 0:
       fitness += penalty  # Penalty
        
    return fitness

# Linelength is added to the fitness value to promote or demote waveform vectors based on the complexity of the waveform if the orginal fitness value is the same as a other result
def Line_length(Fitness, Population, population_size):
    max_penalty = 5000
    linelength = np.zeros(population_size)
    for i in range(population_size):
        pop = np.concatenate(Population[i])
        for j in range(np.size(pop)):
            if j == 0:
                linelength[i] = abs(pop[j])
            else:
                linelength[i] = linelength[i] + abs(pop[j]-pop[j-1])
    
    linelength = np.round((linelength - np.min(linelength)) / (np.max(linelength) - np.min(linelength)),2)      # normalize linelength between 0 and 1
    Fitness += linelength * max_penalty

    return Fitness

# The output of this function is the fitness bvalue for each waveform vector. The fitness values, and therefore the activation rasters are parallel processed.
def fitness(population_size, Population, PosX, PosY, PosZ, Dur, file, cores):
    Fitness = np.zeros(population_size)

    # Prepare arguments for parallel execution
    args_list = [(PosX, PosY, PosZ, Population[i], Dur) for i in range(population_size)]

    # Run GeneticAlgorithm in parallel
    with Pool(cores) as pool:
        excitation_results = pool.map(GeneticAlgorithmWrapper, args_list)

    # Compute fitness for each result
    for i, (Excitation, zero_coordinate, y, z) in enumerate(excitation_results):
        Fitness[i] = np.round(fitness_parallel(Excitation, zero_coordinate), 2)

    Fitness = Line_length(Fitness, Population, population_size)
        
    #Normal fitness funcion
    enumerated_arr = list(enumerate(Fitness))
    probabilities = 1 / np.array(Fitness, dtype=float)     # in this case because a lower fitness is better we want the probability to be reversed of the fitness becuase then a lower fitness has a higher chance
    fitness_sorted = sorted(enumerated_arr, key=lambda x: x[1])

    probabilities /= probabilities.sum()    #probabilities must add up to 1
    #print('Probabilities = ' + str(probabilities))
    print('Fitness original = ' + str(sorted(list(enumerate(Fitness)), key=lambda x: x[1])))

    return Fitness, fitness_sorted, probabilities


# This is the main loop and the function you call to start the PSO
def run_evolution(PosX, PosY, PosZ, Dur, file, generation_limit = 3000,  Population = None, Personal_Best = None, Fitness_Personal_Best = None, Swarm_best = None, Swarm_best_fitness = None):
	
    th = -7400  # The threshold value of a single electrode. The idea of TCS is that this th with the corresponding pulse duration is split over multiple electrodes
    begin_time = time.time()
    timestr = time.strftime("%Y_%m_%d-%H%M")
    print('start = ' + str(timestr))
    solution = 0    #variable to check if the solution is found
    paterns =10     #number of amplitude changes in a waveform per electrode
    electrodes = 2  #number of electrodes
    population_size = 8 #Number of waveform vectors, this is the amound of genomes (or for PSO particles) that are beign compared each generation. must be dividable by two
    
    #!!!! change to the number of course toy have availible !!!
    cores = 4 # number of cores availible for parralel processing 
    
    
    lowest_genome = []      #Keep track of lowest genome and therefore waveform
    lowest_fitness = []        #keep track of the lowest fitness value  

    w = np.arange(0.9, 0.2, (0.2-0.9)/generation_limit) # The amount of inlfuence the current waveform has on the waveform generated for the next iteration. Should decrease with the amount of iterations to converge to the global best
    c1 = 2  # amount of randomness introduced over the personal best vlaue
    c2 = 2  # amount of randomnes introduced over the global best value 

    time_begin = time.time()

    #Generate starting population
    if Population == None:
        print(0)
        Population = generate_population(population_size, paterns, electrodes, PosX, PosY, PosZ, Dur, th)
        print(1)
        Population = GenomeOptimizer(PosX, PosY, PosZ, Population, Dur, electrodes, population_size, cores)

    print('population created')

    #Start loop of iterations
    for i in range(generation_limit):                               #If the maximum amount of iterations has been done stop code
              
        time_begin = time.time()
        print("Processing new generation : " + str(i))
        
        #calculates the fintness value for each genome (paritcle, waveform vector)
        Fitness, fitness_sorted, probabilities = fitness(population_size, Population, PosX, PosY, PosZ,Dur, file, cores)

        #check if a better global best has neem found
        if i == 0 and Personal_Best == None:
            Swarm_best_fitness = fitness_sorted[0][1]
            Fitness_Personal_Best = Fitness
            Personal_Best = Population
            Swarm_best = Population[fitness_sorted[0][0]]
        

        #keep track of the iterations and their performance
        OriginalFitness = sorted(list(enumerate(Fitness)), key=lambda x: x[1])
        lowest_genome.append(Population[OriginalFitness[0][0]])
        print("Current Lowest Genome = " + str(Population[OriginalFitness[0][0]]))
        lowest_fitness.append(Fitness[OriginalFitness[0][0]])
        old_population = Population        
        if solution == 1: break
        
        #Update the generation
        new_Population = [None]*population_size
        print('fitness_sorted = ' + str(fitness_sorted[0][1]) + ' Swarm_best_Fitness = ' + str(Swarm_best_fitness))
        if fitness_sorted[0][1] < Swarm_best_fitness:
            Swarm_best_fitness = fitness_sorted[0][1]
            Swarm_best = Population[fitness_sorted[0][0]]
            print('swarm best = ' + str(Swarm_best))

        #Check if a better personal best is found
        sb = np.concatenate(Swarm_best)
        for j in range(population_size):
            if Fitness[j] < Fitness_Personal_Best[j]:
                Fitness_Personal_Best[j] = Fitness[j]
                Personal_Best[j] = Population[j]
            
            #update the genomes next waveform vector according to the formula of PSO found in thesis     
            pop = np.concatenate(Population[j])
            pb = np.concatenate(Personal_Best[j])
            r1 = np.round(np.random.uniform(0, 1), 1)
            r2 = np.round(np.random.uniform(0, 1), 1)
            update = np.round((w[i]*pop + c1*r1*(pb - pop) + c2*r2*(sb - pop)))
            update = pop + update
            update = np.split(update, electrodes)
            #print("update = " + str(update))
            Population[j] = update

        #Optimize the new genome so it at least stimulatates the target point
        Population = GenomeOptimizer(PosX, PosY, PosZ, Population, Dur, electrodes, population_size, cores)

        #print('Population = ' + str(Population))
        print('time one generation = ' + str(time.time() - time_begin))
    
    end_time = time.time()
	

    #print and save the data
    print('run time = ' + str(end_time-begin_time))
    print('last Population = ' + str(old_population))
    print('Personal Best = ' + str(Personal_Best) )
    print('personal best fitness = ' + str(Fitness_Personal_Best))
    print('swarm best = ' + str(Swarm_best))
    print('swarm best fitness = ' + str(Swarm_best_fitness))
    if solution == 1:
        print("Soulution found, see last generation")
        #print("Solution = " + str(old_population[indic_solution]))
    else:
       print("Solution not found last run population is population")
    timestr = time.strftime("%Y_%m_%d-%H%M")
    df = pd.DataFrame({'Population': old_population, 'Fitness': Fitness})
    df.to_csv(r'images/FinalZ_'+str(file)+'.csv', index=None, header=True)
    Tracker = pd.DataFrame({'Fitness': lowest_fitness, 'Genome': lowest_genome})    
    Tracker.to_csv(r'images/Best_Fitness_FinalZ'+str(file)+'.csv', index=None, header=True)    
        
    
       