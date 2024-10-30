import os, sys

sys.path.insert(0, os.getcwd())

from chromosome.index import Chromosome
import numpy as np
import random
import json
import copy


class ParticleSwarmOptimizationModel:
    def __init__(self):
        self.population_size = 10

        # extra_params
        self.nutritional_weight = 0.8
        self.gram_upper_bound = 4000
        self.cost_weight = 0.2

        self.population = [
            Chromosome(self.gram_upper_bound, self.cost_weight, self.nutritional_weight)
            for _ in range(self.population_size)
        ]

        self.performance = []
        self.best_fitness = -np.inf
        self.best_individual = None
        self.generation = 0

        self.inertia_weight = 0.4
        self.cognitive_weight = 2.0
        self.social_weight = 1.5

        # Initialize personal best positions and fitness
        self.personal_best_positions = [individual for individual in self.population]
        self.personal_best_fitness = np.array(
            [entry.fitness_function() for entry in self.population]
        )

        # Initialize velocities properly
        self.velocities = np.zeros(
            (self.population_size, len(self.population[0].params))
        )

    def log(self, fitness_score, override=False):
        print(f"Generation: {self.generation}, Best Fitness: {fitness_score:.4f}")
        if self.generation == 1 or self.generation % 10 == 0 or override:
            self.performance.append(
                {"generation": self.generation, "fitness": float(fitness_score)}
            )

    def train(self, max_generations=100_000):
        fitness_scores = np.array(
            [entry.fitness_function() for entry in self.population]
        )

        best_idx = np.argmax(fitness_scores)
        self.best_individual = self.population[best_idx]
        self.best_fitness = fitness_scores[best_idx]

        for _ in range(max_generations):
            self.generation += 1

            for i, particle in enumerate(self.population):
                current_fitness = particle.fitness_function()

                # Update personal best
                if current_fitness > self.personal_best_fitness[i]:
                    self.personal_best_positions[i] = copy.deepcopy(particle)
                    self.personal_best_fitness[i] = current_fitness

                # Update global best
                if current_fitness > self.best_fitness:
                    self.best_individual = copy.deepcopy(particle)
                    self.best_fitness = current_fitness

                for j, param in enumerate(particle.params):
                    r1, r2 = random.uniform(0, 1), random.uniform(0, 1)

                    self.velocities[i][j] = (
                        self.inertia_weight * self.velocities[i][j]
                        + self.cognitive_weight
                        * r1
                        * (
                            self.personal_best_positions[i].params[j].weight
                            - param.weight
                        )
                        + self.social_weight
                        * r2
                        * (self.best_individual.params[j].weight - param.weight)
                    )

                    # Update position (weight)
                    param.weight += self.velocities[i][j]

            self.log(self.best_fitness)

    def save_results(self):
        with open(f"pso-diet-optimizer-v1.json", "w+") as json_file:
            json.dump(
                {
                    "best_individual": self.best_individual.to_dict(),
                    "best_fitness": float(self.best_fitness),
                    "performance": self.performance,
                    "params": {
                        "population_size": self.population_size,
                        "inertia_weight": self.inertia_weight,
                        "cognitive_weight": self.cognitive_weight,
                        "social_weight": self.social_weight,
                        "nutritional_weight": self.nutritional_weight,
                        "gram_upper_bound": self.gram_upper_bound,
                        "cost_weight": self.cost_weight,
                    },
                },
                json_file,
            )

    def run(self):
        self.train()
        self.save_results()
