__author__ = 'robbie'
import random
import urllib.request, json
import sqlite3
import copy
import pygame
from pygame import *

colors = [(244, 122, 66),(181, 244, 65),(65, 244, 181), (65, 190, 244),(65, 65, 244),(181, 65, 244), (244, 65, 151), (244, 65, 67)]
color_red = (178, 35, 19)
class BackgroundGUI(pygame.sprite.Sprite):
    def __init__(self, image_file, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

class GUI(object):

    def __init__(self):
        # Initialise PyGame
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.Font('OpenSans-Regular.ttf', 16)
        #self.screen = pygame.display.set_mode((528, 615))
        self.screen = pygame.display.set_mode((800, 615))
        pygame.display.set_caption('Travelling Salesmen Problem.')
        pygame.mouse.set_visible(True)
        self.background = BackgroundGUI('kaartNederland.jpg', [0,0])
        self.screen.blit(self.background.image, self.background.rect)
        pygame.display.update()

    def displayRoute(self, individual, color, generation_number):
        self.screen.fill((0,0,0))
        self.background = BackgroundGUI('kaartNederland.jpg', [0,0])
        self.screen.blit(self.background.image, self.background.rect)
        city_location_on_map = {'Groningen, NL': (428, 82),
                                'Leeuwarden, NL': (328, 88),
                               'Assen, NL': (426, 124),
                               'Zwolle, NL': (377, 236),
                               'Lelystad, NL': (291, 232),
                               'Arnhem, NL': (351, 342),
                               'Utrecht, NL': (247, 323),
                               'Haarlem, NL': (183, 263),
                               'Den Haag, NL': (137, 324),
                               'Middelburg, NL': (48, 444),
                               'Den Bosch, NL': (272, 406),
                               'Maastricht, NL': (324, 583)}

        # Draw the lines.
        for i in range(len(individual.getPhenotype())):
            # Draw line to the previous city.
            if i != 0:
                location_previous = city_location_on_map[individual.getPhenotype()[i - 1]]
                location_current = city_location_on_map[individual.getPhenotype()[i]]
                pygame.draw.line(self.screen, color, [location_previous[0], location_previous[1]],[location_current[0], location_current[1]], 4)

        # Draw some text.
        label = self.font.render("GENERATIE : "+ str(generation_number), False, color_red)
        self.screen.blit(label, (550, 140))

        # Draw some text.
        label = self.font.render("OPTIMALE ROUTE : km(" + str(int(fitness(individual))) + ")", False, color_red)
        self.screen.blit(label, (550, 180))
        for i in range(len(individual.getPhenotype())):
            label = self.font.render(individual.getPhenotype()[i], False, color)
            self.screen.blit(label, (550, 200 + (i * 20)))
        pygame.display.update()

def insert_distance_in_to_database():
    for i in range(len(cities)):
        for j in range(i + 1, len(cities)):
            from_city = cities[i].replace(" ", "+")
            to_city = cities[j].replace(" ", "+")
            miDistance = 0
            str_url = "https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&origins="+str(from_city)+"&destinations="+str(to_city)+"&key=AIzaSyAh1or4PKmF9c3A_WXvjXB7yHwCvPOKqKk"
            with urllib.request.urlopen(str_url) as url:
                data = json.loads(url.read().decode())
                print(str_url)
                print(data)
                miDistance = data['rows'][0]['elements'][0]['distance']['text']
                kmDistance = float(miDistance.replace(" mi", "")) * 1.609344
                from_city = cities[i].replace("+", " ")
                to_city = cities[j].replace("+", " ")
                # Insert a row of data
                c.execute("INSERT INTO `distance_between_cities` VALUES ('"+from_city+"','"+to_city+"',"+ str(kmDistance) +")")
                # Save (commit) the changes
                conn.commit()
    conn.close()

""" Return the fitness result from the given individual. Lower is better. """
def fitness(individual):
    total_distance = 0.0
    # Calculate distance between each city in the phenotype list.
    for i in range(len(individual.getPhenotype())):
        if i != 0:
            from_city = individual.getPhenotype()[i - 1]
            to_city = individual.getPhenotype()[i]
            for row in c.execute("select `distance` from distance_between_cities WHERE `to`='"+to_city+"' AND `from`='"+from_city+"'"):
                 total_distance += copy.copy(row[0])

            for row1 in c.execute("select `distance` from `distance_between_cities` WHERE  `to`='"+from_city+"' and `from`='"+to_city+"'"):
                total_distance += copy.copy(row1[0])
    # Return the result of km.
    return total_distance

""" Truncate selection will return the best individuals based on the lowest fitness. """
def truncate_selection(individuals, top_X_Individuals):
    # Get all the fitness results.
    fintess_results = [fitness(individual) for individual in individuals]
    # Create a list that will contain top individuals.
    selection = []
    for _ in range(top_X_Individuals):
        for individual in individuals:
            # Check if the getPhenotype is the best fitnessResult.
            if fitness(individual) == min(fintess_results):
                # Add to selection.
                selection.append(individual)
                # Remove from result otherwise we get the same getPhenotype each time.
                fintess_results.remove(min(fintess_results))
                # Go for next best individual.
                break
    # Return the selected individuals.
    return selection

""" We use order based crossover in this example. """
def crossover(population):
    # Define a list with children to return.
    children = []
    # For all the fathers...
    for i in range(len(population)):
        # Make with everyone a child...
        for j in range(i + 1, len(population)):
            # Get child.
            child = Individual()
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
    for individual in population:
        # Get random indexes from the genes.
        random_index_0 = random.randint(0, len(individual.genes) - 1)
        random_index_1 = random.randint(0, len(individual.genes) - 1)
        # Indexes will not be the same.
        while random_index_0 == random_index_1:
            random_index_1 = random.randint(0, len(individual.genes) - 1)
        # Create a temp value.
        tmp = individual.genes[random_index_0]
        individual.genes[random_index_0] = individual.genes[random_index_1]
        individual.genes[random_index_1] = tmp
    return population

""" Class that will represent the indivudual. """
class Individual(object):

    def __init__(self):
        copyIndexes = copy.copy(indexes)
        random.shuffle(copyIndexes)
        self.genes = copyIndexes

    def getPhenotype(self):
        return [cities[i] for i in self.genes]

""" Return the best individual from the last generation. """
def getBestIndividualFromEA(amount_individuals, amount_generations):
    # Init a gui.
    gui = GUI()
    # Start line color for drawing on GUI.
    lineColor = (0, 0, 0)
    # Generate a random population.
    population = [Individual() for _ in range(0, amount_individuals)]
    # Loop each amount generation.
    for current_generation in range(amount_generations):
        print("CALCULATING GENERATION " + str(current_generation))
        # Get the best population based on their fitness.
        best_population = truncate_selection(population, int(len(population) / 2))
        # Perform a crossover with the best population. This will return kids that can be seen as new individuals.
        kids = crossover(best_population)
        # Create the new population.
        population = best_population + mutate(kids)[:int(len(population) / 2)]
        # Calculate the best fitness.
        bestFitness = min([fitness(individual) for individual in population])
        # Loop each individual from last generation.
        for individual in population:
            # Check if the fitness of the individual matches with the bestFitness.
            if fitness(individual) == bestFitness:
                gui.displayRoute(individual, lineColor, current_generation)
                lineColor = random.choice(colors)

    # Calculate the best fitness.
    bestFitness = min([fitness(individual) for individual in population])
    # Loop each individual from last generation.
    for individual in population:
        # Check if the fitness of the individual matches with the bestFitness.
        if fitness(individual) == bestFitness:
            # Return individual.
            return individual

# Constants for the algorithm.
AMOUNT_INDIVIDUALS = 40
AMOUNT_GENERATIONS = 100
# Make connection.
conn = sqlite3.connect("cities.db")
c = conn.cursor()
# Cities.
cities = ['Groningen, NL','Leeuwarden, NL','Assen, NL','Zwolle, NL','Lelystad, NL','Arnhem, NL','Utrecht, NL','Haarlem, NL','Den Haag, NL','Middelburg, NL','Den Bosch, NL','Maastricht, NL']
# All the indexes of the cities.
indexes = [i for i in range(0, len(cities))]
# Get best individual from algorithm.
best_individual = getBestIndividualFromEA(AMOUNT_INDIVIDUALS, AMOUNT_GENERATIONS)
# Print status.
print(best_individual.getPhenotype())
print(fitness(best_individual))
# Close database connection.
c.close()