import random
from Parameters import POPSIZE, MAXEVAL, UpdatePeriod
from Program import Program
from SemanticLibrary import SemLibrary
from StatisticFile import StatisticFile
from global_state import evaluation  # variable globale à définir

class Population:
    def __init__(self):
        self.generation = 0
        self.program_list = [Program() for _ in range(POPSIZE)]
        for program in self.program_list:
            program.execute()
        self.best_pro = self.program_list[0]
        self.update_best_pro()
        self.sem_lib = SemLibrary()

    def update_best_pro(self):
        for program in self.program_list:
            if program.fitness < self.best_pro.fitness:
                self.best_pro = program.copy()  # supposons une méthode de copie

    def reproduction(self):
        new_programs = []

        for current_program in self.program_list:
            offspring = current_program.copy()

            if random.random() < 0.5:
                ind = random.randint(0, POPSIZE - 1)
                r2 = self.program_list[ind]
                if r2 == current_program:
                    r2 = self.program_list[(ind + 1) % POPSIZE]
                offspring.pro_de_mutate(self.best_pro, r2)
            else:
                offspring.mutate_and_divide(self.sem_lib, offspring.input_semantics, offspring.target_semantics)
                offspring.generate_name()

            offspring.execute()
            new_programs.append(offspring)

        # Tournament selection of size 2
        for i in range(POPSIZE):
            if new_programs[i].fitness <= self.program_list[i].fitness:
                self.program_list[i] = new_programs[i]

        self.update_best_pro()

    def evolve(self, statistic_file: StatisticFile):
        global evaluation
        self.generation = 0
        while True:
            if self.generation % 50 == 0:
                print(f"{self.generation} : {evaluation} -> {self.best_pro.fitness}")
                print(self.best_pro.name)

            self.reproduction()
            self.generation += 1

            if self.generation % UpdatePeriod == 0:
                self.sem_lib.update_library()
                print(self.sem_lib.count_items())

            statistic_file.procedure_record(evaluation, self.best_pro.fitness)

            if evaluation > MAXEVAL or self.best_pro.fitness < 1e-4:
                print(f"{self.generation} : {evaluation} -> {self.best_pro.fitness}")
                print(self.best_pro.name)
                break
