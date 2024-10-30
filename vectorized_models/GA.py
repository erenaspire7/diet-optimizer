import os, sys
import json
import numpy as np

sys.path.insert(0, os.getcwd())

from chromosome.index import Chromosome
from data.index import Data


class VectorizedGeneticModel:
    def __init__(self):
        self.population_size = 100
        self.chromosome_size = 77
        self.elitism_count = int(self.population_size * 0.2)
        self.crossover_rate = 0.8
        self.mutation_rate = 0.8
        self.mutation_magnitude = 100
        self.stagnation_limit = 10000
        self.mutation_type = "uniform"  # can be "gaussian" | "uniform"

        # Extra params
        self.nutritional_weight = 0.1
        self.gram_upper_bound = 15000
        self.cost_weight = 0.9

        # Initialize data structures
        self.data = Data()
        self.item_names = self.data.get_items()
        self.num_params = len(self.item_names)

        # Create vectorized population array [population_size x num_params]
        self.population = np.random.uniform(
            1, self.gram_upper_bound, (self.population_size, self.num_params)
        )

        # Track best solution
        self.performance = []
        self.best_fitness = -np.inf
        self.best_individual_weights = None
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

        return cost_reward + nutritional_reward

    def vectorized_mutation(self, population_matrix: np.ndarray) -> np.ndarray:
        """Apply mutation to entire population at once"""
        mutation_mask = np.random.random(population_matrix.shape) <= self.mutation_rate

        if self.mutation_type == "gaussian":
            mutations = np.random.normal(
                0, self.mutation_magnitude, size=population_matrix.shape
            )
            population_matrix[mutation_mask] += mutations[mutation_mask]
        else:  # uniform mutation
            min_vals = np.min(population_matrix, axis=0)
            max_vals = np.max(population_matrix, axis=0)

            # Randomly choose between upper and lower bound mutations
            upper_mask = np.random.random(population_matrix.shape) <= 0.5
            lower_mask = ~upper_mask

            # Apply uniform mutations
            mutations_upper = np.random.uniform(
                population_matrix, np.broadcast_to(max_vals, population_matrix.shape)
            )
            mutations_lower = np.random.uniform(
                np.broadcast_to(min_vals, population_matrix.shape), population_matrix
            )

            population_matrix[mutation_mask & upper_mask] = mutations_upper[
                mutation_mask & upper_mask
            ]
            population_matrix[mutation_mask & lower_mask] = mutations_lower[
                mutation_mask & lower_mask
            ]

        return population_matrix

    def vectorized_crossover(self, parents: np.ndarray) -> np.ndarray:
        """Perform crossover on selected parents to create new population"""
        num_children = self.population_size - self.elitism_count
        children = np.zeros((num_children, self.chromosome_size))

        for i in range(num_children):
            # Randomly select parents
            parent_indices = np.random.choice(len(parents), size=2, replace=True)
            parent_1 = parents[parent_indices[0]]
            parent_2 = parents[parent_indices[1]]

            # Generate random crossover point
            point = np.random.randint(0, self.chromosome_size)

            # Create child using vectorized operations
            child = np.concatenate([parent_1[:point], parent_2[point:]])
            children[i] = child

        return children

    def create_best_chromosome(self) -> Chromosome:
        """Create a Chromosome object from the best weights."""
        best_chromosome = Chromosome(
            self.gram_upper_bound, self.cost_weight, self.nutritional_weight
        )
        for idx, (name, weight) in enumerate(zip(self.item_names, self.best_weights)):
            best_chromosome.params[idx].name = name
            best_chromosome.params[idx].weight = float(weight)
        return best_chromosome

    def log(self, fitness_score: float, override: bool = False) -> None:
        print(f"Generation: {self.generation}, Best Fitness: {fitness_score:.4f}")
        if self.generation == 1 or self.generation % 10 == 0 or override:
            self.performance.append(
                {"generation": self.generation, "fitness": float(fitness_score)}
            )

    def train(self, max_generations: int = 100_000) -> None:
        generations_without_improvement = 0

        for _ in range(max_generations):
            self.generation += 1

            # Vectorized fitness calculation for entire population
            population_fitness = self.calculate_batch_fitness(self.population)

            # Find best individual
            best_idx = np.argmax(population_fitness)
            best_fitness = population_fitness[best_idx]
            self.log(best_fitness)

            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_weights = self.population[best_idx].copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            if generations_without_improvement >= self.stagnation_limit:
                self.log(best_fitness, True)
                break

            sorted_indices = np.argsort(population_fitness)
            parents = self.population[sorted_indices[-self.elitism_count :]]

            children = self.vectorized_crossover(parents)
            new_population = np.vstack([parents, children])

            self.population = self.vectorized_mutation(new_population)

    def save_results(self) -> None:
        best_chromosome = self.create_best_chromosome()
        with open("ga-diet-optimizer-v1.json", "w+") as json_file:
            json.dump(
                {
                    "best_individual": best_chromosome.to_dict(),
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

    def run(self) -> None:
        self.train()
        self.save_results()
