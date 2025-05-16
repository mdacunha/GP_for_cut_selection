import sys
import time
from Data import read_train_data, read_test_data, norm_flag
from Population import Population
from StatisticFile import StatisticFile

def main(run: int, job: int, base: str):
    iterative = 30  # total number of independent runs
    SF = StatisticFile()
    SF.init_file(run, job, base)

    SF.problem_init(iterative)

    global norm_flag
    if 10 <= run < 13:  # Indices of Airfoil, BHouse, Tower
        norm_flag = True
    else:
        norm_flag = False

    read_train_data(run, job)
    SF.independent_run_init(run, job)

    suc = 0
    suc_R2 = 0
    eva_times = 0
    train_err = 1e6
    test_err = 1e6
    test_R2 = 0
    prog_size = 0

    start = time.time()
    seed = (1 + run) * (job + 1)
    import random
    random.seed(seed)

    from global_state import evaluation  # à définir quelque part
    evaluation = 0

    pop = Population()
    pop.evolve(SF)

    df_tra_time = time.time() - start

    if pop.bestPro.fitness < 1e-4:
        suc = 1

    eva_times = evaluation
    train_err = pop.bestPro.fitness
    prog_size = pop.bestPro.count_program_length()
    time_elapsed = df_tra_time

    read_test_data(run, job)
    pop.bestPro.execute_test_program()
    test_err = pop.bestPro.fitness
    test_R2 = pop.bestPro.get_test_R2()

    if test_R2 > 0.999:
        suc_R2 = 1

    SF.independent_run_record(run, job, suc, suc_R2, eva_times, train_err, test_err, test_R2, prog_size, pop.bestPro.name, time_elapsed)
    SF.problem_overall_record(run)

if __name__ == "__main__":
    run = int(sys.argv[1])
    job = int(sys.argv[2])
    base = sys.argv[3]
    main(run, job, base)
