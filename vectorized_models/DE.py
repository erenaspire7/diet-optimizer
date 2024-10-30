import os
import sys

sys.path.insert(0, os.getcwd())

import json
import numpy as np
from data.index import Data
import pandas as pd
from chromosome.index import Chromosome


class VectorizedDifferentialModel:
    def __init__(self):
        self.population_size = 100
        self.F = 0.8  # Scaling factor
        self.CR = 0.8  # Crossover rate
        self.stagnation_limit = 10000

        self.gram_upper_bound = 5_000
        self.nutritional_weight = 0.4
        self.cost_weight = 0.6

        # Initialize data structures
        self.data = Data()
        self.item_names = self.data.get_items()
        self.num_params = len(self.item_names)

        # Create vectorized population array [population_size x num_params]
        self.population = np.random.uniform(
            1, self.gram_upper_bound, (self.population_size, self.num_params)
        )

        # Track best solution
        self.best_fitness = -np.inf
        self.best_weights = None
        self.performance = []
        self.generation = 0

        # Cache nutritional data for faster access
        self.cached_data = self._cache_nutritional_data()
        self.optimal_values = self.data.get_optimal_annual_intake().iloc[0].values
        self.ideal_cost = self.data.get_idealized_cost()

    def _cache_nutritional_data(self) -> dict:
        """Cache nutritional values per gram for each item."""
        cache = {}
        for name in self.item_names:
            df = self.data.get_nutritional_value(name)
            base_weight = df["weight (grams)"].iloc[0]
            nutritional_values = (
                df.drop(["Item per $1", "weight (grams)"], axis=1).iloc[0].astype(float)
            )
            cache[name] = {
                "per_gram": nutritional_values / base_weight,
                "cost_per_gram": 1 / base_weight,  # cost ratio per gram
            }
        return cache

    def compare_nutritional_values(self, nutritional_values: np.ndarray) -> np.ndarray:
        """Vectorized comparison of nutritional values against optimal values."""
        # Calculate differences [batch_size x num_nutrients]
        differences = nutritional_values - self.optimal_values

        # Binary check if greater [batch_size x num_nutrients]
        is_greater = (differences > 0).astype(float)

        # Normalize differences to get relative magnitudes
        max_vals = np.maximum(self.optimal_values, 1)  # Avoid division by zero
        relative_diffs = differences / max_vals

        # Combine components
        combined_score = is_greater * (1 - relative_diffs)

        # Return mean score for each individual [batch_size]
        return np.mean(combined_score, axis=1)

    def calculate_batch_fitness(self, weights: np.ndarray) -> np.ndarray:
        """Vectorized fitness calculation for multiple individuals."""
        # Each row represents an individual's weights
        batch_size = weights.shape[0]

        # Initialize arrays for nutritional values and costs
        nutritional_values = np.zeros((batch_size, len(self.optimal_values)))
        total_costs = np.zeros(batch_size)

        # Calculate nutritional values and costs for all items
        for idx, item_name in enumerate(self.item_names):
            item_data = self.cached_data[item_name]
            # Broadcasting weights across nutritional values
            nutritional_contribution = np.outer(weights[:, idx], item_data["per_gram"])
            nutritional_values += nutritional_contribution
            total_costs += weights[:, idx] * item_data["cost_per_gram"]

        nutritional_reward = self.nutritional_weight * self.compare_nutritional_values(
            nutritional_values
        )

        # Calculate cost reward (vectorized)
        cost_reward = self.cost_weight * (
            self.ideal_cost / (self.ideal_cost + np.abs(self.ideal_cost - total_costs))
        )

        fitness = cost_reward + nutritional_reward

        return fitness

    def mutate(self, population: np.ndarray) -> np.ndarray:
        """Vectorized mutation operation."""
        # Generate random indices for mutation
        r1, r2, r3 = (
            np.random.choice(
                self.population_size, size=self.population_size, replace=False
            )
            for _ in range(3)
        )

        # Create mutant vectors
        mutants = population[r1] + self.F * (population[r2] - population[r3])

        # taking a trick from GA
        mask = mutants < 0
        max_val = np.max(mutants)
        mutants[mask] = np.random.uniform(0, max_val + 1, size=np.sum(mask))
        return mutants

    def crossover(self, population: np.ndarray, mutants: np.ndarray) -> np.ndarray:
        """Vectorized crossover operation."""
        mask = np.random.random((self.population_size, self.num_params)) < self.CR
        return np.where(mask, mutants, population)

    def train(self, max_generations: int = 100_000):
        generations_without_improvement = 0

        for _ in range(max_generations):
            self.generation += 1

            # Generate mutants and perform crossover
            mutants = self.mutate(self.population)
            trials = self.crossover(self.population, mutants)

            # Calculate fitness for current population and trials
            population_fitness = self.calculate_batch_fitness(self.population)
            trial_fitness = self.calculate_batch_fitness(trials)

            # Selection (vectorized)
            better_mask = trial_fitness > population_fitness
            self.population[better_mask] = trials[better_mask]

            # Update best solution
            gen_best_idx = np.argmax(population_fitness)
            gen_best_fitness = population_fitness[gen_best_idx]

            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_weights = self.population[gen_best_idx].copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Log progress
            self.log(self.best_fitness)

            # Check for stagnation
            if generations_without_improvement >= self.stagnation_limit:
                break

    def log(self, fitness_score: float, override: bool = False):
        print(f"Generation: {self.generation}, Best Fitness: {fitness_score:.4f}")
        if self.generation == 1 or self.generation % 10 == 0 or override:
            self.performance.append(
                {"generation": self.generation, "fitness": float(fitness_score)}
            )

    def create_best_chromosome(self) -> Chromosome:
        """Create a Chromosome object from the best weights."""
        best_chromosome = Chromosome(
            self.gram_upper_bound, self.cost_weight, self.nutritional_weight
        )
        for idx, (name, weight) in enumerate(zip(self.item_names, self.best_weights)):
            best_chromosome.params[idx].name = name
            best_chromosome.params[idx].weight = float(weight)
        return best_chromosome

    def run(self):
        self.train()
        self.save_results()

    def save_results(self):
        best_chromosome = self.create_best_chromosome()
        with open("de-diet-optimizer-v1.json", "w+") as json_file:
            json.dump(
                {
                    "best_individual": best_chromosome.to_dict(),
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
