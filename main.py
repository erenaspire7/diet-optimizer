# from vectorized_models.DE import VectorizedDifferentialModel
from vectorized_models.GA import VectorizedGeneticModel

# from vectorized_models.PSO import VectorizedPSO

if __name__ == "__main__":
    model = VectorizedGeneticModel()
    model.run()
