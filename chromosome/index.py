import os, sys

sys.path.insert(0, os.getcwd())

import random
from data.index import Data
import pandas as pd
import numpy as np


class NutritionalItem:
    def __init__(self, name, gram_upper_bound):
        self.name = name
        self.weight = random.uniform(1, gram_upper_bound)

    def calculate_nutritional_value(self):
        data_df = Data()
        mini_df = data_df.get_nutritional_value(self.name)

        ratio = self.weight / mini_df["weight (grams)"].iloc[0]

        # ratio is the actual cost in USD
        self.cost = ratio

        for col in mini_df.columns[2:]:
            mini_df.loc[:, col] = mini_df[col].astype(float) * ratio

        mini_df = mini_df.drop(["Item per $1", "weight (grams)"], axis=1)

        return mini_df

    def to_dict(self):
        return {"name": self.name, "weight": self.weight}


class Chromosome:
    def __init__(self, gram_upper_bound, cost_weight, nutritional_weight):
        self.params = [
            NutritionalItem(name, gram_upper_bound) for name in Data().get_items()
        ]
        self.cost_weight = cost_weight
        self.nutritional_weight = nutritional_weight

    def compare_nutritional_values(self, df1, df2):
        vals1 = df1.iloc[0]
        vals2 = df2.iloc[0]

        # Calculate differences
        differences = vals1 - vals2
        is_greater = (differences > 0).astype(float)

        max_vals = vals2.replace(0, 1)
        relative_diffs = differences / max_vals

        closeness_score = 1 / (1 + np.abs(relative_diffs))
        combined_score = is_greater * closeness_score

        return combined_score.mean()

    def to_dict(self):
        return {"params": [item.to_dict() for item in self.params]}

    def fitness_function(self):
        optimal_values = Data().get_optimal_annual_intake()

        ideal_cost = Data().get_idealized_cost()

        combined_df = (
            pd.concat([item.calculate_nutritional_value() for item in self.params])
            .sum()
            .to_frame()
            .T  
        )

        total_cost = sum(item.cost for item in self.params)

        cost_reward = self.cost_weight * (
            ideal_cost / (ideal_cost + abs(ideal_cost - total_cost))
        )

        nutritional_reward = self.nutritional_weight * self.compare_nutritional_values(
            combined_df, optimal_values
        )

        return cost_reward + nutritional_reward
