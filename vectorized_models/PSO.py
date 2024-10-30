import os, sys

sys.path.insert(0, os.getcwd())

from chromosome.index import Chromosome
from data.index import Data
import numpy as np
import json


class VectorizedPSO:
    def __init__(self):
        self.population_size = 100
        self.inertia_weight = 0.7
        self.cognitive_weight = 2.0
        self.social_weight = 2.5
        self.stagnation_limit = 10000

        self.min_bound = 0

        # Optimization weights
        self.gram_upper_bound = 1_000
        self.nutritional_weight = 0.55
        self.cost_weight = 0.45

        self.data = Data()
        self.item_names = self.data.get_items()
        self.num_params = len(self.item_names)

        # Initialize population matrix [population_size x num_params]
        self.positions = np.random.uniform(
            1, self.gram_upper_bound, (self.population_size, self.num_params)
        )

        self.velocities = np.zeros((self.population_size, self.num_params))

        # Track personal and global bests
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = np.zeros(self.population_size)
        self.best_fitness = -np.inf
        self.best_weights = None

        # Performance tracking
        self.performance = []
        self.generation = 0

        # Cache nutritional data and optimal values
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
        relative_diffs = np.abs(differences) / max_vals

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

        return cost_reward + nutritional_reward

    def update_velocities_and_positions(self):
        """Vectorized update of velocities and positions."""
        # Generate random coefficients for entire population
        r1 = np.random.uniform(0, 1, (self.population_size, 1))
        r2 = np.random.uniform(0, 1, (self.population_size, 1))

        # Update velocities (vectorized)
        self.velocities = (
            self.inertia_weight * self.velocities
            + self.cognitive_weight
            * r1
            * (self.personal_best_positions - self.positions)
            + self.social_weight * r2 * (self.best_weights - self.positions)
        )

        # Update positions
        self.positions += self.velocities

        # rebound functionality
        lower_violation = self.positions < self.min_bound

        self.positions = np.where(
            lower_violation,
            self.min_bound + (self.min_bound - self.positions),
            self.positions,
        )
        self.velocities = np.where(lower_violation, -self.velocities, self.velocities)

    def train(self, max_generations: int = 100_000):
        fitness_scores = self.calculate_batch_fitness(self.positions)
        self.personal_best_fitness = fitness_scores.copy()

        # Initialize global best
        best_idx = np.argmax(fitness_scores)
        self.best_fitness = fitness_scores[best_idx]
        self.best_weights = self.positions[best_idx].copy()

        generations_without_improvement = 0

        for _ in range(max_generations):
            self.generation += 1

            self.update_velocities_and_positions()

            current_fitness = self.calculate_batch_fitness(self.positions)

            # Update personal bests (vectorized)
            better_mask = current_fitness > self.personal_best_fitness
            self.personal_best_fitness[better_mask] = current_fitness[better_mask]
            self.personal_best_positions[better_mask] = self.positions[better_mask]

            gen_best_idx = np.argmax(current_fitness)
            gen_best_fitness = current_fitness[gen_best_idx]

            if gen_best_fitness > self.best_fitness:
                self.best_fitness = gen_best_fitness
                self.best_weights = self.positions[gen_best_idx].copy()
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
        with open("pso-diet-optimizer-v1.json", "w+") as json_file:
            json.dump(
                {
                    "best_individual": best_chromosome.to_dict(),
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
