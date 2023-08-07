import csv
import time
from jmetal.util.observer import Observer
from jmetal.core.solution import Solution
import logging
LOGGER = logging.getLogger("jmetal")
from matplotlib import pyplot as plt

# class CustomObserver(Observer):
#     def __init__(self, frequency):
#         self.frequency = frequency
#         self.evaluation_number = []
#         self.fitness_values = []
#         self.computing_time = []
#         self.solutions = []
#
#     def update(self, *args, **kwargs):
#         # algorithm = args[0].algorithm
#         computing_time = kwargs["COMPUTING_TIME"]
#         evaluations = kwargs["EVALUATIONS"]
#         solutions = kwargs["SOLUTIONS"]
#
#         if (evaluations % self.display_frequency) == 0 and solutions:
#             if type(solutions) == list:
#                 fitness = solutions[0].objectives
#             else:
#                 fitness = solutions.objectives
#
#             LOGGER.info(
#                 "Evaluations: {} \n Best fitness: {} \n Computing time: {}".format(evaluations, fitness, computing_time)
#             )
#     def export_to_csv(self, file_name):
#         with open(file_name, "w", newline="") as file:
#             writer = csv.writer(file)
#             writer.writerow(["Evaluation Number", "Fitness Value", "Computing Time", "Solution"])
#             writer.writerows(
#                 zip(self.evaluation_number, self.fitness_values, self.computing_time, self.solutions)
#             )

class CustomObserver(Observer):
    def __init__(self, frequency: int = 1, algorithm: str = "ga") -> None:
        """Show the number of evaluations, the best fitness and the computing time.
        :param frequency: Display frequency."""

        self.display_frequency = frequency
        self.algorithm = algorithm
        self.solution_list = []
        self.fitness_list = []
        self.computing_time_list = []
        self.evaluation_list = []

    def update(self, *args, **kwargs):
        computing_time = kwargs["COMPUTING_TIME"]
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]

        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives

            LOGGER.info(
                "Evaluations: {} \n Best fitness: {} \n Computing time: {}".format(evaluations, fitness, computing_time)
            )
        solution_bit = []
        for value in solutions[0].variables:
            if value:
                solution_bit.append(1)
            else:
                solution_bit.append(0)
        self.solution_list.append(solution_bit)
        self.computing_time_list.append(computing_time)
        self.fitness_list.append(fitness[0])
        self.evaluation_list.append(evaluations)
        # with open(self.file_name, "w", newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerow(["Evaluation Number", "Fitness Value", "Computing Time", "Solution"])
        #     writer.writerows(
        #         zip(evaluations, fitness, computing_time, solutions)
        #     )
    def export_to_csv(self, file_name):
        with open(file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Evaluation Number", "Fitness Value", "Computing Time", "Solution"])
            writer.writerows(
                zip(self.evaluation_list, self.fitness_list, self.computing_time_list, self.solution_list)
            )
    def export_plot(self, file_name):
        plt.clf()
        if self.algorithm == "ga":
            evaluation_list_gen = [self.evaluation_list[i]/100 for i in range(len(self.evaluation_list))]
        else:
            evaluation_list_gen = [self.evaluation_list[i] for i in range(len(self.evaluation_list))]
        plt.plot(evaluation_list_gen, self.fitness_list)
        plt.savefig(file_name)
        # with open(file_name, "w", newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerow(["Evaluation Number", "Fitness Value", "Computing Time", "Solution"])
        #     writer.writerows(
        #         zip(self.evaluation_list, self.fitness_list, self.computing_time_list, self.solution_list)
        #     )

class CuBasicObserver(Observer):
    def __init__(self, frequency: int = 1) -> None:
        """Show the number of evaluations, the best fitness and the computing time.
        :param frequency: Display frequency."""

        self.display_frequency = frequency
        self.evaluation_list = []
        self.fitness_list = []
        self.computing_time_list = []

    def update(self, *args, **kwargs):
        computing_time = kwargs["COMPUTING_TIME"]
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]

        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives

            LOGGER.info(
                "Evaluations: {} \n Best fitness: {} \n Computing time: {}".format(evaluations, fitness, computing_time)
            )
            self.evaluation_list.append(evaluations)
            self.fitness_list.append(fitness)
            self.computing_time_list.append(computing_time)
    def export_to_csv(self, file_name):
        with open(file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Evaluation Number", "Fitness Value", "Computing Time"])
            writer.writerows(
                zip(self.evaluation_list, self.fitness_list, self.computing_time_list)
            )
    def export_plot(self, file_name):
        plt.clf()
        evaluation_list_gen = [self.evaluation_list[i] for i in range(len(self.evaluation_list))]
        plt.plot(evaluation_list_gen, self.fitness_list)
        plt.savefig(file_name)

