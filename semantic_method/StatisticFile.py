import numpy as np
import os

class StatisticFile:
    def __init__(self):
        self.conv_internal = 50000
        self.last_internal = 0

        # Chemins pour les fichiers
        self.overallAdd = ""
        self.fitness_convAdd = ""
        self.fitness_timesAdd = ""
        self.train_errorAdd = ""
        self.test_errorAdd = ""
        self.test_R2Add = ""
        self.prog_sizeAdd = ""
        self.example_progAdd = ""
        self.trainTimeAdd = ""

        # Statistiques par job
        self.suc = None
        self.sucR2 = None
        self.train_error = None
        self.test_error = None
        self.testR2 = None
        self.eva_times = None
        self.prog_size = None
        self.trainTime = None

    def init_file(self, run: int, job: int, base: str):
        self.overallAdd       = os.path.join(base, f"overall_{run}_{job}.txt")
        self.fitness_convAdd  = os.path.join(base, f"fitness_conv_{run}_{job}.txt")
        self.fitness_timesAdd = os.path.join(base, f"fitness_time_{run}_{job}.txt")
        self.train_errorAdd   = os.path.join(base, f"train_error_{run}_{job}.txt")
        self.test_errorAdd    = os.path.join(base, f"test_error_{run}_{job}.txt")
        self.test_R2Add       = os.path.join(base, f"test_R2_{run}_{job}.txt")
        self.prog_sizeAdd     = os.path.join(base, f"prog_size_{run}_{job}.txt")
        self.example_progAdd  = os.path.join(base, f"example_prog_{run}_{job}.txt")
        self.trainTimeAdd     = os.path.join(base, f"trainTime_{run}_{job}.txt")

        for filepath in [
            self.overallAdd, self.fitness_convAdd, self.fitness_timesAdd,
            self.train_errorAdd, self.test_errorAdd, self.test_R2Add,
            self.prog_sizeAdd, self.example_progAdd, self.trainTimeAdd
        ]:
            with open(filepath, 'w') as f:
                pass

    def problem_init(self, iterations: int):
        self.suc         = np.zeros(iterations)
        self.sucR2       = np.zeros(iterations)
        self.train_error = np.zeros(iterations)
        self.test_error  = np.zeros(iterations)
        self.testR2      = np.zeros(iterations)
        self.eva_times   = np.zeros(iterations)
        self.prog_size   = np.zeros(iterations)
        self.trainTime   = np.zeros(iterations)

    def independent_run_init(self, run: int, job: int):
        self.last_internal = 0

        with open(self.fitness_convAdd, 'a') as f:
            f.write(f"{run}\t{job}\n")
        for filepath in [
            self.fitness_timesAdd, self.train_errorAdd, self.test_errorAdd,
            self.test_R2Add, self.prog_sizeAdd, self.trainTimeAdd
        ]:
            with open(filepath, 'a') as f:
                f.write(f"{run}\t{job}\t")
        with open(self.example_progAdd, 'a') as f:
            f.write(f"{run}\t{job}\n")

    def procedure_record(self, eval_val: float, fitness: float):
        with open(self.fitness_convAdd, 'a') as f:
            recordeval = int(eval_val) // self.conv_internal
            if self.last_internal == 0:
                f.write(f"{self.last_internal}\t{fitness}\n")
            for i in range(self.last_internal + 1, recordeval + 1):
                f.write(f"{i * self.conv_internal}\t{fitness}\n")
            self.last_internal = recordeval

    def independent_run_record(self, run, job, suc_flag, suc_R2,
                               f_times, train_err, test_err,
                               test_R2, prog_siz, example_prog, train_time):

        self.suc[job]         = suc_flag
        self.sucR2[job]       = suc_R2
        self.eva_times[job]   = f_times
        self.train_error[job] = train_err
        self.test_error[job]  = test_err
        self.testR2[job]      = test_R2
        self.prog_size[job]   = prog_siz
        self.trainTime[job]   = train_time

        print(f"{run}\t{job}\tsuc:\t{self.suc[:job+1].mean():.4f}"
              f"\tsucR2:\t{self.sucR2[:job+1].mean():.4f}"
              f"\teva_times:\t{self.eva_times[:job+1].mean():.4f}"
              f"\ttrain_error:\t{self.train_error[:job+1].mean():.4f}"
              f"\ttest_error:\t{self.test_error[:job+1].mean():.4f}"
              f"\ttest_R2:\t{self.testR2[:job+1].mean():.4f}"
              f"\tprog_size:\t{self.prog_size[:job+1].mean():.4f}"
              f"\ttrain_time:\t{self.trainTime[:job+1].mean():.4f}")

        with open(self.fitness_timesAdd, 'a') as f:
            f.write(f"{f_times}\n")

        self.procedure_record(f_times, train_err)

        with open(self.train_errorAdd, 'a') as f:
            f.write(f"{train_err}\n")
        with open(self.test_errorAdd, 'a') as f:
            f.write(f"{test_err}\n")
        with open(self.test_R2Add, 'a') as f:
            f.write(f"{test_R2}\n")
        with open(self.prog_sizeAdd, 'a') as f:
            f.write(f"{prog_siz}\n")
        with open(self.example_progAdd, 'a') as f:
            f.write(f"{example_prog}\n\n")
        with open(self.trainTimeAdd, 'a') as f:
            f.write(f"{train_time}\n")

    def problem_overall_record(self, run: int):
        with open(self.overallAdd, 'a') as f:
            f.write(
                f"{run}\tsuc:\t{self.suc.mean():.4f}"
                f"\tsuc_R2:\t{self.sucR2.mean():.4f}"
                f"\teva_times:\t{self.eva_times.mean():.4f}"
                f"\ttrain_error:\t{self.train_error.mean():.4f}"
                f"\ttest_error:\t{self.test_error.mean():.4f}"
                f"\ttest_R2:\t{self.testR2.mean():.4f}"
                f"\tprog_size:\t{self.prog_size.mean():.4f}"
                f"\ttrain_time:\t{self.trainTime.mean():.4f}\n"
            )
