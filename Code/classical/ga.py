import time

import numpy as np
from jmetal.operator import BinaryTournamentSelection, BitFlipMutation
from TestingProblem import TCO
import pandas as pd
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import BitFlipMutation, SPXCrossover
from jmetal.problem import OneMax
from jmetal.util.observer import PrintObjectivesObserver, BasicObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from TestingProblem import OneMax1
import argparse
from CustomObserver import CustomObserver
import os

parser = argparse.ArgumentParser()
parser.add_argument('n', type=int)
parser.add_argument('p', type=int)
parser.add_argument('name',type=str)
args = parser.parse_args()
num_experiment = args.n
population_size = args.p
file_name = args.name

case_study = file_name
if file_name == "elevator_two" or file_name == "elevator_three":
        file_name = "elevator"

df = pd.read_csv("../../data/"+file_name+".csv", dtype={"time": float, "rate": float})
length = len(df)
print(length)
problem = TCO(length, df, case_study)
# np.set_printoptions(linewidth=500)

# algorithm = GeneticAlgorithm(
#     problem=problem,
#     population_size=100,  # 10
#     offspring_population_size=100, #10
#     mutation=BitFlipMutation(probability=0.9),
#     crossover=SPXCrossover(probability=0.9), # 0.9, 20
#     selection=BinaryTournamentSelection(),
#     termination_criterion=StoppingByEvaluations(1000)  # 500
# )
# algorithm = GeneticAlgorithm(
#     problem=problem,
#     population_size=10,
#     offspring_population_size=10,
#     mutation=BitFlipMutation(1.0 / problem.number_of_bits),
#     crossover=SPXCrossover(1.0),
#     termination_criterion=StoppingByEvaluations(max_evaluations=100),
# )
# problem = OneMax1(number_of_bits=512)
algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=population_size,#40
        offspring_population_size=population_size,#40
        mutation=BitFlipMutation(1.0 / problem.number_of_bits),
        crossover=SPXCrossover(1.0),
        termination_criterion=StoppingByEvaluations(max_evaluations=5000*population_size),
    )

# algorithm.observable.register(observer=BasicObserver(100))
# Create and attach the custom observer
# Create and attach the custom observer
frequency = population_size
observer = CustomObserver(frequency)
algorithm.observable.register(observer)

# Run the algorithm
algorithm.run()

# Export captured values to a CSV file
# if not os.path.exists("GA" + "/" + file_name + "/" + str(num_experiment)):
#         os.makedirs("GA" + "/" + file_name + "/" + str(num_experiment))
# observer.export_to_csv("GA" + "/" + file_name + "/" + str(num_experiment)+"/"+"evolution_data.csv")
# observer.export_plot("GA" + "/" + file_name + "/" + str(num_experiment)+"/"+"fitness_evolution.png")

# if not os.path.exists("GA" + "/" + file_name + "_three" + "/" + str(num_experiment)):
#         os.makedirs("GA" + "/" + file_name + "_three" + "/" + str(num_experiment))
# observer.export_to_csv("GA" + "/" + file_name + "_three" + "/" + str(num_experiment)+"/"+"evolution_data.csv")
# observer.export_plot("GA" + "/" + file_name + "_three" + "/" + str(num_experiment)+"/"+"fitness_evolution.png")

if not os.path.exists("GA" + "/" + file_name + "_two" + "/" + str(num_experiment)):
        os.makedirs("GA" + "/" + file_name + "_two" + "/" + str(num_experiment))
observer.export_to_csv("GA" + "/" + file_name + "_two" + "/" + str(num_experiment)+"/"+"evolution_data.csv")
observer.export_plot("GA" + "/" + file_name + "_two" + "/" + str(num_experiment)+"/"+"fitness_evolution.png")





# begin = time.time()
# algorithm.run()
# end = time.time()
# result = algorithm.get_result()
#
# exe_time = end - begin
#
# print("Algorithm: {}".format(algorithm.get_name()))
# print("Problem: {}".format(problem.name()))
# print("Solution: " + result.get_binary_string())
# print("Fitness:  " + str(result.objectives[0]))
# print("Computing time: {}".format(algorithm.total_computing_time))

# algorithm = GeneticAlgorithm(
#     problem=problem,
#     population_size=population_size,
#     offspring_population_size=population_size,
#     mutation=BitFlipMutation(probability=mutation_probability),
#     crossover=SinglePointCrossover(probability=1.0),
#     selection=BinaryTournamentSelection(),
#     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
# )

# algorithm.run()

# result = algorithm.get_result()
# result_int = result.variables
# solution = []
# for l in range(len(result_int)):
#     solution.append(dec2bin(result_int[l],len(config_dic['inputID'])))
#
# #get count_times
# valid_input = config_dic['valid_input']
# pt = config_dic['p']
# count = []
# for l in range(len(solution)):
#     count_times = 0
#     for i in range(len(valid_input)):
#         if valid_input[i] == solution[l] and pt[l] > 0:
#             count_times += 1
#     count.append(count_times*100)
#
# # GenerateUnitTest.generateUnitTestClass(config_dic['module_name'], config_dic['inputID'], config_dic['outputID'], config_dic['num_qubit'], result_int, solution, count, config_dic['program_folder'],ROOT)
#
#
# solution.insert(0, 'Solution')
# obj = [-result.objectives[0][0]]
#
# obj.insert(0,'Fitness')
# total_time = [algorithm.total_computing_time]
# total_time.insert(0, 'Total time')
#
# for l in range(len(solution)):
#     solution_sheet.cell(row=1,column=l+1,value=solution[l])
# for l in range(len(obj)):
#     solution_sheet.cell(row=2,column=l+1,value=obj[l])
# for l in range(len(total_time)):
#     solution_sheet.cell(row=3,column=l+1,value=total_time[l])
#
# wb.save(config_dic['program_folder']+ ROOT +config_dic['module_name']+'_result.xlsx')
#
# print("\n")
# print('Computing time: ', algorithm.total_computing_time)
# print('File '+config_dic['module_name']+'_result.xlsx'+' is generated.')

