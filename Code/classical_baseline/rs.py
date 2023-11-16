import time

import numpy as np
from jmetal.operator import BinaryTournamentSelection, BitFlipMutation
from TestingProblem import TCO
import pandas as pd
from jmetal.algorithm.multiobjective.random_search import RandomSearch
from jmetal.operator import BitFlipMutation, SPXCrossover
from jmetal.problem import OneMax
from jmetal.util.observer import PrintObjectivesObserver, BasicObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from TestingProblem import OneMax1
import argparse
from CustomObserver import CustomObserver, CuBasicObserver
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int)
parser.add_argument('p', type=int)
parser.add_argument('name',type=str)
parser.add_argument('mu', type=float)
parser.add_argument('cross', type=float)
args = parser.parse_args()
num_experiment = args.n
population_size = args.p
file_name = args.name
mu = args.mu
crossover = args.cross

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
algorithm = RandomSearch(
        problem=problem,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluation),
)

# Create and attach the custom observer
frequency = population_size
observer = CuBasicObserver(frequency)
algorithm.observable.register(observer)

# Run the algorithm
algorithm.run()

if not os.path.exists("RS" + "/" + file_name + "/" + str(num_experiment)):
        os.makedirs("RS" + "/" + file_name  + "/" + str(num_experiment))
observer.export_gen("RS" + "/" + file_name + "/" + str(num_experiment)+"/"+"evolution_data.csv")