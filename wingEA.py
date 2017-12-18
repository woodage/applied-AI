__author__ = 'robbie'
# Exercise 6.2 — Wingdesign
import random
from functools import reduce
AMOUNT_GENERATIONS = 1000
AMOUNT_INDIVIDUALS = 30

def rouletteWheelSelection(individuals):
    pass

def truncateSelection(individuals, topXIndividuals):
    # Get all the fitness results.
    fintessResults = [fitness(individual) for individual in individuals]
    # Create a list that will contain top individuals.
    selection = []
    for _ in range(topXIndividuals):
        for individual in individuals:
            # Check if the getPhenotype is the best fitnessResult.
            if fitness(individual) == max(fintessResults):
                # Add to selection.
                selection.append(individual)
                # Remove from result otherwise we get the same getPhenotype each time.
                fintessResults.remove(max(fintessResults))
                # Go for next best individual.
                break
    # return the selected individuals.
    return selection

def tournamentSelection(individuals):
    # List that will contain all winners of tournament.
    winners = []
    # Define the tournament candidates as the individuals.
    tournamentCandidates = list(individuals)
    # Check if there are tournament candidates left in the game.
    while len(tournamentCandidates) > 0:
        candidates = []
        # Select 2 candidates and remove from tournament Candidate list.
        for _ in range(2):
            randomIndex = random.randint(0, len(tournamentCandidates) - 1)
            candidates.append(tournamentCandidates[randomIndex])
            tournamentCandidates.remove(tournamentCandidates[randomIndex])
        # The best one become winners.. (Lower is better for now).
        if fitness(candidates[0]) > fitness(candidates[1]):
            winners.append(candidates[0])
        else:
            winners.append(candidates[1])
    # Return winners.
    return winners

def mutate(individuals):
    for individu in individuals:
        # Choose random gen to modify.
        genIndex = random.randint(0, len(individuals[0].genes) - 1)
        # Choose random bit of the gen to flip.
        bitIndex = random.randint(0, len(individu.genes[genIndex]) - 1)
        # Flip it.
        individu.genes[genIndex][bitIndex] = 1 if individu.genes[genIndex][bitIndex] == 0 else 0
        individu.genotype = individu.genes
    return individuals

def crossover(selection):
    # List that will contain children.
    children = []
    # List that will contain all individuals that will reproduce.
    becomingParents = list(selection)
    while len(becomingParents) > 1:
        # Choose a father.
        father = becomingParents[random.randint(0, len(becomingParents) - 1)]
        # Remove the father from the becomingParents list.
        becomingParents.remove(father)
        # Choose a mother.
        mother = becomingParents[random.randint(0, len(becomingParents) - 1)]
        # Remove the mother from the becomingParents list.
        becomingParents.remove(mother)
        # Calculate the half of amount of genes from individual.
        half = int(len(father.genes) / 2)
        # Get new child.
        child = WingDesign()
        # Assign half of the genes from father and half of the genes from mother.
        child.genes = father.genes[half:] + mother.genes[:half]
        child.genotype = child.genes
        children.append(child)
    return mutate(children)

def fitness(individual):
    v = individual.getPhenotype()
    lift = ((v['A'] - v['B']) ** 2) + ((v['C'] - v['D']) ** 2) - (( v['A'] - 30) ** 3) - ((v['C'] - 40) ** 3)
    return lift

def bitfield(n):
    return list(map(int, list('{0:06b}'.format(n))))

def get_bit_int(n):
    out = 0
    for bit in n:
        out = (out << 1) | bit
    return out

"""Object that is representive as wing design."""
class WingDesign(object):
    def __init__(self):
        self.genes = []
        # Generate 4 genes that will represent a value between 0 and 63.
        for _ in range(0, 4):
            self.genes.append(bitfield(random.randint(0, 64)))
        self.genotype = self.genes

    # The getPhenotype will return a dictionary with the vars as key.
    def getPhenotype(self):
        return { 'A' : get_bit_int(self.genotype[0]), 'B' : get_bit_int(self.genotype[1]), 'C' : get_bit_int(self.genotype[2]), 'D' : get_bit_int(self.genotype[3])}

results = []
# Run program 100 times to get best result.
for run in range(100):
    print(run)
    # Init some population.
    population = [WingDesign() for _ in range(AMOUNT_INDIVIDUALS)]
    # Evolve till the max amount generations is reached.
    for i in range(AMOUNT_GENERATIONS):
        # Selection of population.
        selection = truncateSelection(population, int(len(population) / 2))
        # Define new population. Recombine and mutate.
        population = selection + crossover(selection)
        while len(population) < AMOUNT_INDIVIDUALS:
            # Get more children if the population is to small.
            population += crossover(selection)
        population = population[:AMOUNT_INDIVIDUALS]
    # Get the best fitness of the last generation.
    bestFitness = max([fitness(individu) for individu in population])

    # Check witch individu has the best fitness.
    for individu in population:
        if fitness(individu) == bestFitness:
            results.append(individu)
            break

finalFitness = max([fitness(individu) for individu in results])
for individu in results:
    if fitness(individu) == finalFitness:
        print(individu.getPhenotype()['A'])
        print(individu.getPhenotype()['B'])
        print(individu.getPhenotype()['C'])
        print(individu.getPhenotype()['D'])