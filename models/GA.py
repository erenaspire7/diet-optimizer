import os, sys

sys.path.insert(0, os.getcwd())

import json
import random
import numpy as np


from chromosome.index import Chromosome


class GeneticModel:
    def __init__(self):
        self.population_size = 100
        self.elitism_count = int(self.population_size * 0.2)
        self.crossover_rate = 0.9
        self.mutation_rate = 0.9
        self.mutation_magnitude = 50
        self.stagnation_limit = 10000

        # can be "guassian" | "uniform"
        self.mutation_type = "uniform"

        # extra_params
        self.nutritional_weight = 0.1
        self.gram_upper_bound = 15000
        self.cost_weight = 0.9

        self.population = [
            Chromosome(self.gram_upper_bound, self.cost_weight, self.nutritional_weight)
            for _ in range(self.population_size)
        ]

        self.performance = []
        self.best_fitness = -np.inf
        self.best_individual = None
        self.generation = 0

    def log(self, fitness_score, override=False):
        print(f"Generation: {self.generation}, Best Fitness: {fitness_score:.4f}")
        if self.generation == 1 or self.generation % 10 == 0 or override:
            self.performance.append(
                {"generation": self.generation, "fitness": float(fitness_score)}
            )

    def train(self, max_generations=10_000):
        generations_without_improvement = 0

        for _ in range(max_generations):
            self.generation += 1

            fitness_scores = np.array(
                [entry.fitness_function() for entry in self.population]
            )

            best_idx = np.argmax(fitness_scores)
            self.log(fitness_scores[best_idx])

            if self.best_fitness < fitness_scores[best_idx]:
                self.best_fitness = fitness_scores[best_idx]
                self.best_individual = self.population[best_idx]
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= self.stagnation_limit:
                self.log(fitness_scores[best_idx], True)
                break

            parents = self.population[np.argsort(fitness_scores)[-self.elitism_count :]]
            new_population = list(parents)

            for _ in range(0, len(self.population) - self.elitism_count):
                parent_1, parent_2 = (
                    parents[random.randint(0, len(parents) - 1)],
                    parents[random.randint(0, len(parents) - 1)],
                )
                child = self.crossover(parent_1, parent_2)
                new_population.append(self.mutate(child))

            self.population = new_population

    def get_min_max(self, idx):
        min_i = min(item.params[idx].weight for item in self.population)
        max_i = max(item.params[idx].weight for item in self.population)

        return min_i, max_i

    def mutate(self, chromosome: Chromosome):
        for idx, item in enumerate(chromosome.params):
            if random.random() <= self.mutation_rate:
                if self.mutation_type == "guassian":
                    # guassian mutation
                    item.weight += np.random.normal(0, self.mutation_magnitude)

                else:
                    min_i, max_i = self.get_min_max(idx)

                    # normal mutation
                    if random.random() <= 0.5:
                        item.weight = random.uniform(item.weight, max_i)
                    else:
                        item.weight = random.uniform(min_i, item.weight)

        return chromosome

    def crossover(self, parent_1: Chromosome, parent_2: Chromosome):
        point = random.randint(0, 77)

        child = Chromosome(
            self.gram_upper_bound, self.cost_weight, self.nutritional_weight
        )

        for idx in range(point):
            child.params[idx].weight = parent_1.params[idx].weight

        for idx in range(point, 77):
            child.params[idx].weight = parent_2.params[idx].weight

        return child

    def save_results(self):
        with open(f"ga-diet-optimizer-v1.json", "w+") as json_file:
            json.dump(
                {
                    "best_individual": self.best_individual.to_dict(),
                    "best_fitness": float(self.best_fitness),
                    "performance": self.performance,
                    "params": {
                        "population_size": self.population_size,
                        "elitism_count": self.elitism_count,
                        "crossover_rate": self.crossover_rate,
                        "stagnation_limit": self.stagnation_limit,
                        "nutritional_weight": self.nutritional_weight,
                        "gram_upper_bound": self.gram_upper_bound,
                        "cost_weight": self.cost_weight,
                        "mutation_rate": self.mutation_rate,
                        "mutation_magnitude": self.mutation_magnitude,
                        "mutation_type": self.mutation_type,
                    },
                },
                json_file,
            )

    def run(self):
        self.train()
        self.save_results()
