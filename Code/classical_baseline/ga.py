import time

import numpy as np
from jmetal.operator import BinaryTournamentSelection, BitFlipMutation
from TestingProblem import TCO
import pandas as pd
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import BitFlipMutation, SPXCrossover
from jmetal.util.termination_criterion import StoppingByEvaluations
import argparse
from CustomObserver import CuBasicObserver
import os

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int)
parser.add_argument('p', type=int)
parser.add_argument('name',type=str)
args = parser.parse_args()
num_experiment = args.n
population_size = args.p
file_name = args.name

if file_name == "elevator_two":
        df = pd.read_csv("../data/"+"elevator"+".csv")
        data_array = []
        data_array.append(list(df["cost"]))
        data_array.append(list(df["input_div"]))
elif file_name == "elevator_three":
        df = pd.read_csv("../data/" + "elevator" + ".csv")
        data_array = []
        data_array.append(list(df["cost"]))
        data_array.append(list(df["pcount"]))
        data_array.append(list(df["dist"]))
else:
        df = pd.read_csv("../data/" + file_name + ".csv")
        data_array = []
        data_array.append(list(df["time"]))
        data_array.append(list(df["rate"]))

length = len(df)
problem = TCO(length, data_array, file_name)

if file_name == "paintcontrol":
        max_evaluation = 400000
elif file_name == "iofrol":
        max_evaluation = 400000
elif file_name == "elevator_two" or file_name == "elevator_three":
        max_evaluation = 400000
elif file_name == "gsdtsr":
        max_evaluation = 400000
algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=population_size,
        offspring_population_size=population_size,
        mutation=BitFlipMutation(1.0 / problem.number_of_bits),
        crossover=SPXCrossover(1.0),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluation),
    )

# Create and attach the custom observer
frequency = population_size
observer = CuBasicObserver(frequency)
algorithm.observable.register(observer)

# Run the algorithm
algorithm.run()

if not os.path.exists("GA" + "/" + file_name + "/" + str(population_size) + "/" + str(num_experiment)):
        os.makedirs("GA" + "/" + file_name  + "/" + str(population_size) + "/" + str(num_experiment))
observer.export_gen("GA" + "/" + file_name + "/" + str(population_size) + "/" + str(num_experiment)+"/"+"generation_data.csv")
