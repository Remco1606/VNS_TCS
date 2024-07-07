from model import GeneticAlgorithm 
from model import FindStimulation
from neuron import h
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool
from copy import deepcopy

def generate_genome(paterns, electrodes, PosX, PosY, PosZ, Dur, th, th0):
    Genome = [None]*electrodes #The input to the function so in our case a series of amplitudes that is fed to a electrode
    Amplitudes = np.zeros(paterns)
    alpha = 1
    beta = 3
    amp_choices = [th, -th, 0]
    amp_choices0 = [th0, -th0, 0]
    for i in range(electrodes):
        if PosZ[i] == 0:
            for j in range(paterns):
                #Amplitudes[j] = np.random.choice([0, -14000])
                Amplitudes[j] = np.random.choice(amp_choices0)
            
        else:    
            for j in range(paterns):
                #Amplitudes[j] = np.random.choice([0, -14000])
                Amplitudes[j] = np.random.choice(amp_choices)
           
        Genome[i] = Amplitudes.copy()

    return Genome

def GeneOptimizerWrapper(args):
    PosX, PosY, PosZ, individual, Dur, electrodes = args
    return FindStimulation(PosX, PosY, PosZ, individual, Dur, electrodes)

def GenomeOptimizer(PosX, PosY, PosZ, Population, Dur, electrodes, population_size, cores):

    # Prepare arguments for parallel execution
    args_list = [(PosX, PosY, PosZ, Population[i], Dur, electrodes) for i in range(population_size)]

    # Run GeneticAlgorithm in parallel
    with Pool(cores) as pool:
        Population = pool.map(GeneOptimizerWrapper, args_list)

    return Population

def generate_population(population_size, paterns, electrodes, PosX, PosY, PosZ, Dur, th, th0):      # Make a group of genomes which becomes a genration
    Population = [None]*population_size # a group of genomes which make up a generation
    #Population[0] = [np.array([ -7000, -7000,  -7000,  -7000,  -7000, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, -7000, -7000,  -7000, -7000, -7000])]
    for i in range(population_size):
        Population[i] = generate_genome(paterns, electrodes, PosX, PosY, PosZ, Dur, th, th0)
    print('Starting population = ' + str(Population))    
    return Population

def GeneticAlgorithmWrapper(args):
    PosX, PosY, PosZ, individual, Dur, file = args
    return GeneticAlgorithm(PosX, PosY, PosZ, individual, Dur, file)

def fitness_parallel(Excitation, zero_coordinate):
    #Excitation, zero_coordinate = individual
    #print('excitation = ' + str(Excitation))
    #print('zero coordinate = ' + str(zero_coordinate))
    penalty = 350000#100000
    fitness = np.sum(Excitation)
    if Excitation[zero_coordinate] == 0:
       fitness += penalty  # Penalty
        
    return fitness

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


def linear_scaling_fitness(fitness):
    # Find the minimum and maximum fitness values in the population
    #print('fitness before' +str(fitness))
    max_val = max(fitness)
    fitness = [max_val - val for val in fitness]
    #print('fitness after' +str(fitness))
    fmultiple = 1.2
    min_fitness = min(fitness)
    max_fitness = max(fitness)
    avg_fitness = np.average(fitness)
    if min_fitness > (fmultiple*avg_fitness-max_fitness):
        delta = max_fitness-avg_fitness
        a = (fmultiple - 1)*avg_fitness/delta 
        b = avg_fitness * (max_fitness - fmultiple*avg_fitness)/delta
    else:
        delta = avg_fitness-min_fitness
        a = avg_fitness/delta
        b = -min_fitness*avg_fitness/delta

    # Scale the fitness values
    scaled_fitness = [np.round((a * i + b),2) for i in fitness]
    
    return scaled_fitness


def fitness(population_size, Population, PosX, PosY, PosZ, Dur, file, cores):
    Fitness = np.zeros(population_size)

    # Prepare arguments for parallel execution
    args_list = [(PosX, PosY, PosZ, Population[i], Dur, file + i) for i in range(population_size)]

    # Run GeneticAlgorithm in parallel
    with Pool(cores) as pool:
        excitation_results = pool.map(GeneticAlgorithmWrapper, args_list)

    # Compute fitness for each result
    for i, (Excitation, zero_coordinate) in enumerate(excitation_results):
        Fitness[i] = np.round(fitness_parallel(Excitation, zero_coordinate), 2)

    #Fitness = Line_length(Fitness, Population, population_size)
    # Sorting, etc., remains the same
    #Scaled Fitness function        # linear fitness scaling asumes the higher the fitness the better so in the function Fitness is inversed, therefore this is not neccecary for probabilities
    scaled_fitness = linear_scaling_fitness(Fitness) 
    enumerated_arr = list(enumerate(scaled_fitness))
    probabilities = np.array(scaled_fitness, dtype=float)
    fitness_sorted = sorted(enumerated_arr, key=lambda x: x[1], reverse=True)

    #Normal fitness funcion
    #enumerated_arr = list(enumerate(Fitness))
    #probabilities = 1 / np.array(Fitness, dtype=float)     # in this case because a lower fitness is better we want the probability to be reversed of the fitness becuase then a lower fitness has a higher chance
    #fitness_sorted = sorted(enumerated_arr, key=lambda x: x[1])

    probabilities /= probabilities.sum()    #probabilities must add up to 1
    #print('Probabilities = ' + str(probabilities))
    print('Fitness original = ' + str(sorted(list(enumerate(Fitness)), key=lambda x: x[1])))

    return Fitness, fitness_sorted, probabilities


def selection_function(population_size, probabilities):
    selected_genome1 = np.random.choice(population_size, replace=False, p=probabilities)     #poulation_size -1 because random takes 0 into the possabilities
    selected_genome2 = np.random.choice(population_size, replace=False, p=probabilities)
    selected_genomes = [selected_genome1, selected_genome2]    
    #print('selected genomes = ' + str(selected_genomes))
    return selected_genomes

def single_point_crossover(parent1, parent2, electrodes, paterns):           #instead of braking the individual amplite paterns up I also switch the amplites for a complete electrode depending on chance
    #begin = time.time()
    parent1_flat = np.concatenate(parent1)
    parent2_flat = np.concatenate(parent2)
    type_crossover = np.random.choice([1,2,3,4])

    if type_crossover == 1: #single point crossover
        p = np.random.randint(1, np.size(parent1_flat))
        Genome1 = np.concatenate((parent1_flat[:p],parent2_flat[p:]))
        Genome2 = np.concatenate((parent2_flat[:p],parent1_flat[p:]))
    elif type_crossover == 2:    #Dubble point crossover
        p1, p2 = np.random.randint(1, np.size(parent1_flat), 2)
        while p2 <= p1:
            p1, p2 = np.random.randint(1, np.size(parent1_flat), 2)
        #print('p1, p1 = ' + str(p1) + str(p2))
        Genome1 = np.concatenate((parent1_flat[:p1],parent2_flat[p1:p2],parent1_flat[p2:]))
        Genome2 = np.concatenate((parent2_flat[:p1],parent1_flat[p1:p2],parent2_flat[p2:]))
    elif type_crossover == 3: #inversion
        p1, p2 = np.random.randint(1, np.size(parent1_flat), 2)
        while p2 <= p1:
            p1, p2 = np.random.randint(1, np.size(parent1_flat), 2)
        Genome1 = np.concatenate((parent1_flat[:p1],parent1_flat[p1:p2][::-1],parent1_flat[p2:]))
        Genome2 = np.concatenate((parent2_flat[:p1],parent2_flat[p1:p2][::-1],parent2_flat[p2:]))
    else:   #Do nothing
        Genome1 = parent1_flat
        Genome2 = parent2_flat
    #print('Genome1 = ' + str(Genome1))
    Genome1 = np.where(Genome1 > 10000, 10000, Genome1)
    Genome2 = np.where(Genome2 > 10000, 10000, Genome2)
    Genome1 = np.split(Genome1, electrodes)
    Genome2 = np.split(Genome2, electrodes)
    #print('singlepoitcrossover time = ' + str(begin-time.time()))
    return Genome1

def mutation(genome, paterns, th, th0): 
    #for j in range(len(genome)):
    #    genome[j] *= np.random.normal(1,0.16)
    #    genome[j] = np.round(genome[j])
    random_number = np.random.randint(0, 1000)
	
    if random_number < 125:
        num_mutations = np.random.randint(1, len(genome) + 1)  # Random number of mutations
        mutated_indices = np.random.choice(len(genome), num_mutations, replace=False).astype(int)
        amp_choices = [th, -th, 0]
        amp_choices0 = [th0, -th0, 0]
            #print(str(mutated_indices))
        for j in range(np.size(mutated_indices)):
            indices_reverse = np.random.choice(paterns)
            #if mutated_indices[j] == 1:
            #    genome[mutated_indices[j]][indices_reverse] = np.random.choice(amp_choices0)
            #else:
            genome[mutated_indices[j]][indices_reverse] = np.random.choice(amp_choices)
            #if genome[mutated_indices[j]][indices_reverse] == 0:
            #    genome[mutated_indices[j]][indices_reverse] = -14000
            #else:
            #    genome[mutated_indices[j]][indices_reverse] = 0

    
    #genome[0] = np.where(genome[0] < 0, th, genome[0])      #th is already negative so no -th
    #genome[1] = np.where(genome[1] < 0, th0, genome[1])
    #genome[2] = np.where(genome[2] < 0, th, genome[2])
            
    return genome

def make_new_population(population_size, Population, Fitness, fitness_sorted, Elites, electrodes, paterns, probabilities, th, th0): ### probably the elites are changed because the value is not copied but its a pointer
    NewPopulation = []
    #print('fitness_sorted' + str(fitness_sorted))
    min_indices = [fitness_sorted[i][0] for i in range(Elites)]
    #print('min_indices' + str(min_indices))
    for i in range(Elites):
        #print(str(min_indices[i]))
        equal = False
        for j in range(Elites):
            if np.array_equal(Population[j], Population[min_indices[i]]) == True:
                equal = True

        if equal == False:
            NewPopulation.append(deepcopy(Population[min_indices[i]]))
        else:
            NewPopulation.append(deepcopy(Population[i]))
        #print(str(NewPopulation))
    
    for i in range(int((population_size-Elites))):#/2)):                  #minus Elites becasue these you want to keep and /2 because the loop generates 2 new genomes
        parents = selection_function(population_size, probabilities)
        exit = False
        j = 0
        while j < 4 and not exit:
            if abs(Fitness[parents[0]] - Fitness[parents[1]]) <= 4 :                   #Reduce same type of paterns getting more and more. maximum tries of finding different partner
                parents = selection_function(population_size, probabilities)
                j = j+1
            else:
                exit = True
        offspring_a = single_point_crossover(Population[parents[0]], Population[parents[1]], electrodes, paterns)
        offspring_a = mutation(offspring_a, paterns, th, th0)
        #offspring_b = mutation(offspring_b, paterns, th)
        NewPopulation.append(offspring_a)
        #NewPopulation.append(offspring_b)
        
    return NewPopulation

def run_evolution(PosX, PosY, PosZ, Dur, file, generation_limit = 3000, fitness_limit = 1, Population = None):
	
    th = -7400
    th0 = -4200
    begin_time = time.time()
    timestr = time.strftime("%Y_%m_%d-%H%M")
    print('start = ' + str(timestr))
    solution = 0
    paterns = 10
    electrodes = 2
    population_size = 80 #must be dividable by two
    Elites = 8 # must be dividable by two
    cores = 46 # number of cores availible for parralel processing
    lowest_genome = []
    lowest_fitness = []

    if Elites % 2 != 0:
        print("Error: Elites is not even.")
    if population_size % 2 != 0:
        print("Error: Elites is not even.")
    time_begin = time.time()

    if Population == None:
        Population = generate_population(population_size, paterns, electrodes, PosX, PosY, PosZ, Dur, th, th0)
        Population = GenomeOptimizer(PosX, PosY, PosZ, Population, Dur, electrodes, population_size, cores)
    #else:
        #Population = make_new_population(population_size, Population, fitness_sorted, Elites, electrodes, paterns, probabilities)
        #Population = GenomeOptimizer(PosX, PosY, PosZ, Population, Dur, electrodes, population_size, cores)

    #print('time genome optimezer = ' + str((time.time()) - time_begin))
    #print('optimized population = ' + str(Population))
    print('population created')
    for i in range(generation_limit):                               #If the maximum amount of iterations has been done stop code
              
        time_begin = time.time()
        print("Processing new generation : " + str(i))
        #print(str(Population))
        #print('population = : ' + str(Population))
        Fitness, fitness_sorted, probabilities = fitness(population_size, Population, PosX, PosY, PosZ,Dur, file, cores)
        OriginalFitness = sorted(list(enumerate(Fitness)), key=lambda x: x[1])
        lowest_genome.append(Population[OriginalFitness[0][0]])
        print('Current Lowest Genome = ' + str(Population[OriginalFitness[0][0]]))
        lowest_fitness.append(Fitness[OriginalFitness[0][0]])
        print('Fitness sorted = ' + str(fitness_sorted))
        #for j in range(np.size(Fitness)):
        #    if Fitness[j] <= fitness_limit:        #if the solution has been found stop code
        #        solution = 1
        #        indic_solution = j
                
        #    if solution == 1: break
        old_population = Population        
        if solution == 1: break

        Population = make_new_population(population_size, Population, Fitness, fitness_sorted, Elites, electrodes, paterns, probabilities, th, th0)
        Population = GenomeOptimizer(PosX, PosY, PosZ, Population, Dur, electrodes, population_size, cores)
        print('time one generation = ' + str(time.time() - time_begin))
    
    end_time = time.time()
	
    print('run time = ' + str(end_time-begin_time))
    print('last Population = ' + str(old_population))
    if solution == 1:
        print("Soulution found, see last generation")
        #print("Solution = " + str(old_population[indic_solution]))
    else:
       print("Solution not found last run population is population")
    timestr = time.strftime("%Y_%m_%d-%H%M")
    df = pd.DataFrame({'Population': old_population, 'Fitness': Fitness})
    df.to_csv(r'images/Fianl3_'+str(timestr)+'.csv', index=None, header=True)
    Tracker = pd.DataFrame({'Fitness': lowest_fitness, 'Genome': lowest_genome})    
    Tracker.to_csv(r'images/Best_Fitness_Final3'+str(timestr)+'.csv', index=None, header=True)    
        
    
       