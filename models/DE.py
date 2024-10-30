import os, sys

sys.path.insert(0, os.getcwd())

import json
import copy
import random
import numpy as np


from chromosome.index import Chromosome


class DifferentialModel:
    def __init__(self):
        self.population_size = 100
        self.F = 0.8  # Scaling factor
        self.CR = 0.8  # Crossover rate
        self.stagnation_limit = 10000

        # extra_params
        self.nutritional_weight = 0.4
        self.gram_upper_bound = 20_000
        self.cost_weight = 0.6

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

    def train(self, max_generations=100_000):
        generations_without_improvement = 0

        for _ in range(max_generations):
            self.generation += 1

            next_population = []
            gen_best_fitness = -np.inf
            gen_best_individual = None

            for i in range(self.population_size):
                r1, r2, r3 = random.sample(range(self.population_size), 3)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = random.sample(range(self.population_size), 3)

                mutant = self.mutate(
                    self.population[i],
                    self.population[r1],
                    self.population[r2],
                    self.population[r3],
                )

                trial = self.crossover(self.population[i], mutant)
                next_individual, fitness = self.selection(self.population[i], trial)

                if fitness > gen_best_fitness:
                    gen_best_fitness = fitness
                    gen_best_individual = next_individual

                next_population.append(next_individual)

            self.population = next_population

            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_individual = copy.deepcopy(gen_best_individual)
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= self.stagnation_limit:
                self.log(self.best_fitness, True)
                break

            self.log(self.best_fitness)

    def mutate(
        self, target: Chromosome, r1: Chromosome, r2: Chromosome, r3: Chromosome
    ):
        mutant = copy.deepcopy(target)
        max_val = max(w.weight for w in mutant.params)

        for idx, item in enumerate(mutant.params):
            weight = r1.params[idx].weight + self.F * (
                r2.params[idx].weight - r3.params[idx].weight
            )

            if weight < 0:
                item.weight = random.randint(0, max_val + 1)

        return mutant

    def crossover(self, target: Chromosome, mutant: Chromosome):
        trial = copy.deepcopy(target)

        for idx, item in enumerate(trial.params):
            if random.random() <= self.CR:
                item.weight = mutant.params[idx].weight

        return trial

    def selection(self, target: Chromosome, trial: Chromosome):
        target_fitness = target.fitness_function()
        trial_fitness = trial.fitness_function()

        if trial_fitness > target_fitness:
            return trial, trial_fitness
        else:
            return target, target_fitness

    def run(self):
        self.train()
        self.save_results()

    def save_results(self):
        with open(f"de-diet-optimizer-v1.json", "w+") as json_file:
            json.dump(
                {
                    "best_individual": self.best_individual.to_dict(),
                    "best_fitness": float(self.best_fitness),
                    "performance": self.performance,
                    "params": {
                        "population_size": self.population_size,
                        "differential_weight": self.F,
                        "crossover_rate": self.CR,
                        "stagnation_limit": self.stagnation_limit,
                        "nutritional_weight": self.nutritional_weight,
                        "gram_upper_bound": self.gram_upper_bound,
                        "cost_weight": self.cost_weight,
                    },
                },
                json_file,
            )
