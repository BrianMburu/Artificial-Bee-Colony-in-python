import random
import datetime
import pandas as pd
import numpy as np
import networkx as nx

class ArtificialBeeColony:
    def __init__(self, G, num_bees, max_iterations):
        self.G = G
        self.num_bees = num_bees
        self.max_iterations = max_iterations
        
        self.current_population = [self.generate_possible_solution() for i in range(self.num_bees)]
        self.current_best_solution = self.current_population[0]  # initialize best solution as the first element of the population
        self.population_size = len(self.current_population)
        self.num_employeed_bees = self.population_size // 2
        self.num_onlooker_bees = self.population_size - self.num_employeed_bees
        self.test_data = []
        self.test_cases = 0
    
    """
    Function to Compute fitness value.
    """ 
    def evaluate_fitness(self, path, eps=0.9):
        fitness = 0.0
        
        for i in range(1, len(path)):
            total_distance = 0
            curr_node = path[i-1]
            next_node = path[i]
            if self.G.has_edge(curr_node, next_node):
                fitness += self.G[curr_node][next_node]['weight']
            else:
                fitness += 0
        fitness = np.power(abs(fitness + eps), 2)
        return fitness

    def apply_random_neighborhood_structure(self, path):
        """
        This function applies the neighborhood structure to find a new solution.
        It randomly swaps two nodes in the path.
        """
        new_path = path.copy()
        node1, node2 = random.sample(path, 2)
        node1_index = path.index(node1)
        node2_index = path.index(node2)
        new_path[node1_index], new_path[node2_index] = new_path[node2_index], new_path[node1_index]
        
        return new_path

    def sort_population_by_fitness(self, population):
        """
        This function sorts the population of paths based on their fitness (the total weight of the edges in the path)
        """
        return sorted(population, key=lambda x: self.evaluate_fitness(x), reverse=True)

    def choose_solution_with_probability(self, population, probability_list):
        """
        This function selects a solution from the population based on the probability list.
        """
        random_value = random.random()
        cumulative_probability = 0.0
        for i in range(len(population)):
            cumulative_probability += probability_list[i]
            if random_value <= cumulative_probability:
                return population[i]

    def generate_possible_solution(self):
        """
        This function generates a random solution (a random path) in the graph
        """
        nodes = list(self.G.nodes)
        start = nodes[0]
        end = nodes[-1]
        samples = list(nx.all_simple_paths(self.G, start, end))
        for i in range(len(samples)):
            if len(samples[i]) != len(nodes):
                extra_nodes = [node for node in nodes if node not in samples[i]]
                random.shuffle(extra_nodes)
                samples[i] = samples[i] + extra_nodes

        sample_node = random.choice(samples)
        return sample_node
    
    def run(self, patience=10):
        gen_fitness = np.zeros(self.max_iterations)
        patience_counter = 0
        for iteration in range(self.max_iterations):
            # Employed Bee phase
            for i in range(self.num_employeed_bees):
                current_solution = self.current_population[i]
                new_solution = self.apply_random_neighborhood_structure(current_solution)
                new_solution_cost = self.evaluate_fitness(new_solution)
                current_solution_cost = self.evaluate_fitness(current_solution)

                if new_solution_cost > current_solution_cost:
                    self.current_population[i] = new_solution
                    self.test_cases += 1

            #Onlooker Bee phase
            probability_list = [1.0 / self.evaluate_fitness(solution) for solution in self.current_population]
            probability_list = [probability / sum(probability_list) for probability in probability_list]

            for i in range(self.num_onlooker_bees):
                selected_solution = self.choose_solution_with_probability(self.current_population, probability_list)
                new_solution = self.apply_random_neighborhood_structure(selected_solution)
                new_solution_cost = self.evaluate_fitness(new_solution)
                selected_solution_cost = self.evaluate_fitness(selected_solution)

                if new_solution_cost > selected_solution_cost:
                    selected_solution_index = self.current_population.index(selected_solution)
                    self.current_population[selected_solution_index] = new_solution
                    self.test_cases += 1

            # Scout Bee phase
            current_population = self.sort_population_by_fitness(self.current_population)
            current_fitness_value = self.evaluate_fitness(self.current_best_solution)
            
            if self.evaluate_fitness(self.current_population[0]) > current_fitness_value:
                self.current_best_solution = self.current_population[0]

            # If the best solution does not change for a certain number of iterations, generate a new random solution
            gen_fitness[iteration] = current_fitness_value
            
            if iteration > 0:
                if gen_fitness[iteration]==gen_fitness[iteration-1]:
                    patience_counter += 1
                    
            if patience_counter >= patience:
                self.current_population[-1] = self.generate_possible_solution()
                patience_counter = 0
            
            self.test_data.append([iteration, current_fitness_value, self.test_cases])
        
            last_node = list(self.G.nodes)[-1]
        last_node_index = self.current_best_solution.index(last_node) + 1
        return self.current_best_solution[ : last_node_index], current_fitness_value
    
if __name__ == "__main__":
    """ Example Setup """
    Gn = nx.DiGraph()
    
    #Add nodes to the graph
    for i in range(11):
        Gn.add_node(i)

    #Add Weighted nodes to the graph
    edges = [(0, 1,{'weight': 1}), (1, 3,{'weight': 2}), (1, 2,{'weight': 1}),(2, 4,{'weight': 2}),
            (3, 2,{'weight': 2}),(3, 4,{'weight': 1}),(3, 5,{'weight': 2}),(3, 7,{'weight': 4}),
            (4, 5,{'weight': 1}),(4, 6,{'weight': 2}),(5, 7,{'weight': 2}),(5, 8,{'weight': 3}),
            (6, 7,{'weight': 1}),(7, 9,{'weight': 2}),(8, 10,{'weight': 2}),(9, 10,{'weight': 1})]

    Gn.add_edges_from(edges)
    abc = ArtificialBeeColony(Gn, num_bees = 53, max_iterations=500)

    start = datetime.datetime.now()
    best_path, best_fitness = abc.run(patience = 12)
    end = datetime.datetime.now()

    abc_time = end - start

    abc_test_data = pd.DataFrame(abc.test_data, columns = ["iterations","fitness_value","test_cases"])

    print("Optimal path: ", best_path)
    print("Optimal path cost: ", best_fitness)
    print("ABC total Exec time => ", abc_time.total_seconds())
    abc_test_data.to_csv("abc_test_data_results.csv")