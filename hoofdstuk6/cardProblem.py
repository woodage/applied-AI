__author__ = 'robbie'
# Exercise 6.1 — Card problem
import random
from functools import reduce
TARGET_PILE_0 = 36
TARGET_PILE_1 = 360
AMOUNT_GENERATIONS = 1000
AMOUNT_INDIVIDUALS = 30

def rouletteWheelSelection(individuals):
    pass

def truncateSelection(individuals, topXIndividuals, cards):
    # Get all the fitness results.
    fintessResults = [fitness(cards, individual) for individual in individuals]
    # Create a list that will contain top individuals.
    selection = []
    for _ in range(topXIndividuals):
        for individual in individuals:
            # Check if the getPhenotype is the best fitnessResult. (Lower is better..)
            if fitness(cards, individual) == min(fintessResults):
                # Add to selection.
                selection.append(individual)
                # Remove from result otherwise we get the same getPhenotype each time.
                fintessResults.remove(min(fintessResults))
                # Go for next best individual.
                break
    # return the selected individuals.
    return selection

def tournamentSelection(individuals, cards):
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
        if fitness(cards, candidates[0]) < fitness(cards, candidates[1]):
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

def crossover(selection, amountcards = 10):
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
        child = CardsForPile(amountcards)
        # Assign half of the genes from father and half of the genes from mother.
        child.genes = father.genes[half:] + mother.genes[:half]
        child.genotype = child.genes
        children.append(child)
    return mutate(children)

def fitness(cards, individual, test = False):
     # The first pile sum to a number as close as possible to 36.
    Pile_0 = [cards[i] for i in individual.getPhenotype()]
    # The other pile multiply to a number as close as possible to 360.
    remainingIndexes = list(filter(lambda x: x not in individual.getPhenotype(), list(range(len(cards)))))
    Pile_1 = [cards[i] for i in remainingIndexes]
    # Calculate each pile.
    sumOfPile_0 = sum([card.number for card in Pile_0])
    if len(Pile_1):
        multiplyOfPile_1 = reduce(lambda x,y: x * y, [card.number for card in Pile_1])
    else:
        multiplyOfPile_1 = 0
    # Return the amount of distance from the expectation and current result.
    sum_error = ( TARGET_PILE_0 - sumOfPile_0) / TARGET_PILE_0
    multiply_error = (multiplyOfPile_1 - TARGET_PILE_1) / TARGET_PILE_1
    combined_error = abs(sum_error) + abs(multiply_error)
    return combined_error

"""A card object. Has number as attribute."""
class Card(object):
    def __init__(self, number):
        self.number = number

"""Object that will return the card indexes for Pile0. The remaining cards ben be for Pile1"""
class CardsForPile(object):
    def __init__(self, amountCards):
        self.genes = []
        # Generate 10 genes that contains 'maybe' an index of the card list.
        for i in range(0, amountCards):
            amountOfOnes = random.randint(0, amountCards)
            # Each gen contains a list of zero's and one's. If there are more one's then zero's, the index of the gen in the list will be equal on the card number to divide.
            self.genes.append([1 for _ in range(0, amountOfOnes)] + [0 for _ in range(0, amountCards - amountOfOnes)])
        self.genotype = self.genes

    # The getPhenotype will return a list with card indexes that can be assigned to a pile.
    def getPhenotype(self):
        results = []
        for i in range(len(self.genotype)):
            # Check if there are more one's then zero's.
            if sum(self.genotype[i]) > 5:
                # Push the index of the gen as result.
                results.append(i)
        return results

isGoalAchieved = False
# You have 10 cards numbered from 1 to 10.
cards = [Card(x) for x in range(1, 11)]
# Define the amount of generations.
# Create population.
population = [CardsForPile(len(cards)) for _ in range(AMOUNT_INDIVIDUALS)]
# Evolve.
for i in range(AMOUNT_GENERATIONS):
    print("GENERATION " + str(i) + " with population of " + str(len(population)))
    # Selection of population.
    selection = truncateSelection(population, int(len(population) / 2), cards)
    # Termination. Is goal achieved?
    for individu in selection:
        if fitness(cards, individu) == 0:
            isGoalAchieved = True
    if not isGoalAchieved:
        # Replace some losers with children of the winning selection.
        losers = list(filter(lambda x: x not in selection, population))
        del losers[-len(crossover(selection)):]
        # Define new population
        population = selection + mutate(losers) + crossover(selection)
        if len(population) < AMOUNT_INDIVIDUALS:
            population += crossover(selection)[AMOUNT_INDIVIDUALS - len(population):]
        population = population[:AMOUNT_INDIVIDUALS]
    else:
        print("After" + str(i) + " amount of generations, The goal has been achieved.")
        break
# Get the best fitness of the last generation.
bestFitness = min([fitness(cards, individu) for individu in population])

# Check wich individu has the best fitness.
for individu in population:
    if fitness(cards, individu) == bestFitness:
        print("After " + str(i) +" amount of generations, The closest individu has fitness off : " + str(fitness(cards, individu)))
        Pile_0 = [cards[i] for i in individu.getPhenotype()]
        # The other pile multiply to a number as close as possible to 360.
        remainingIndexes = list(filter(lambda x: x not in individu.getPhenotype(), list(range(len(cards)))))
        Pile_1 = [cards[i] for i in remainingIndexes]
        print("With cards for pile 0 : " + str(individu.getPhenotype()) + " sum("+str(sum([sum([card.number for card in Pile_0])]))+")")
        print("With cards for pile 1 : " + str(remainingIndexes) + " multiply(" + str(reduce(lambda x,y: x * y, [card.number for card in Pile_1])) + ")")
        break