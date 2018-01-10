__author__ = 'robbie'
import random
import urllib.request, json
import sqlite3
import copy

conn = sqlite3.connect("cities.db")
c = conn.cursor()

cities = ['Amersfoort',
'Baarn',
'De Bilt',
'Bunnik',
'Bunschoten',
'Eemnes',
'Houten',
'IJsselstein',
'Leusden',
'Lopik',
'Montfoort',
'Nieuwegein',
'Oudewater',
'Renswoude',
'Rhenen',
'De Ronde Venen',
'Soest',
'Stichtse Vecht',
'Utrecht',
'Utrechtse Heuvelrug',
'Veenendaal',
'Vianen',
'Wijk bij Duurstede',
'Woerden',
'Woudenberg',
'Zeist']

# All the indexes of the cities.
indexes = [i for i in range(0, len(cities))]

def insert_distance_in_to_database():
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            from_city = cities[i].replace(" ", "+")
            to_city = cities[j].replace(" ", "+")
            miDistance = 0
            str_url = "https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins="+str(from_city)+"&destinations="+str(to_city)+"&key=AIzaSyAh1or4PKmF9c3A_WXvjXB7yHwCvPOKqKk"
            with urllib.request.urlopen(str_url) as url:
                data = json.loads(url.read().decode())
                miDistance = data['rows'][0]['elements'][0]['distance']['text']
                kmDistance = float(miDistance.replace(" mi", "")) * 1.609344
                # Insert a row of data
                c.execute("INSERT INTO `distance_between_cities` VALUES ('"+from_city+"','"+to_city+"',"+ str(kmDistance) +")")
                # Save (commit) the changes
                conn.commit()
    conn.close()

def fitness(individu):
    totalDistance = 0.0
    # Calculate distance between each city in the phenotype list.
    for i in range(len(individu.getPhenotype())):
        if i != 0:
            fromCity = individu.getPhenotype()[i - 1].replace(" ", "+")
            toCity = individu.getPhenotype()[i].replace(" ", "+")
            distance = 0.0
            for row in c.execute("select `distance` from distance_between_cities WHERE `to`='"+toCity+"' AND `from`='"+fromCity+"'"):
                if row is None:
                    for row1 in c.execute("select `distance` from `distance_between_cities` WHERE  `to`='"+fromCity+"' and `from`='"+toCity+"'"):
                        distance = row1[0]
                else:
                    distance = row[0]
            totalDistance += distance
    # Return the result of km.
    return totalDistance

def truncateSelection(individuals, topXIndividuals):
    # Get all the fitness results.
    fintessResults = [fitness(individual) for individual in individuals]
    # Create a list that will contain top individuals.
    selection = []
    for _ in range(topXIndividuals):
        for individual in individuals:
            # Check if the getPhenotype is the best fitnessResult.
            if fitness(individual) == min(fintessResults):
                # Add to selection.
                selection.append(individual)
                # Remove from result otherwise we get the same getPhenotype each time.
                fintessResults.remove(min(fintessResults))
                # Go for next best individual.
                break
    # Return the selected individuals.
    return selection

"""We use order based crossover in this example. """
def crossover(population):
    # Define a list with children to return.
    children = []
    # For all the fathers...
    for i in range(len(population)):
        # Make with everyone a child...
        for j in range(i + 1, len(population)):
            # Get child.
            child = Individu()
            # Assign mother genes to the child.
            child.genes = copy.copy(population[j].genes)
            # Get halve a gen from the father.
            father_genes_for_crossover = copy.copy(population[i].genes[0:int(len(population[i].genes) / 2)])
            # Get the indexes where the father selected gen values are stored in the mother genes.
            mother_indexes_free_for_father = [population[j].genes.index(g) for g in father_genes_for_crossover]
            mother_indexes_free_for_father.sort()
            # Assign the father genes to the child.
            for k in mother_indexes_free_for_father:
                child.genes[k] = father_genes_for_crossover[0]
                father_genes_for_crossover.pop(0)
            # Set the child in the children list.
            children.append(child)
    return children

""" We use insertion mutate in this example. """
def mutate(population):
    # Mutate each individu in the population.
    for individu in population:
        # Get random indexes from the genes.
        random_index_0 = random.randint(0, len(individu.genes) - 1)
        random_index_1 = random.randint(0, len(individu.genes) - 1)
        # Indexes will not be the same.
        while random_index_0 == random_index_1:
            random_index_1 = random.randint(0, len(individu.genes) - 1)
        # Create a temp value.
        tmp = individu.genes[random_index_0]
        individu.genes[random_index_0] = individu.genes[random_index_1]
        individu.genes[random_index_1] = tmp
    return population

class Individu(object):

    def __init__(self):
        copyIndexes = copy.copy(indexes)
        random.shuffle(copyIndexes)
        self.genes = copyIndexes

    def getPhenotype(self):
        return [cities[i] for i in self.genes]

indi = Individu()
AMOUNT_INDIVIDUALS = 30
AMOUNT_GENERATIONS = 100

population = [Individu() for _ in range(0, AMOUNT_INDIVIDUALS)]

for current_generation in range(AMOUNT_GENERATIONS):
    print("CALCULATING GENERATION " + str(current_generation))
    best_population = truncateSelection(population, int(len(population) / 2))
    kids = crossover(best_population)
    population = best_population + mutate(kids)[:int(len(population) / 2)]
    print(min([fitness(individual) for individual in population]))
    print(len(population))

c.close()