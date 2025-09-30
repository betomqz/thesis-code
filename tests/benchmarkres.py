import pandas as pd
import shutil

from pathlib import Path
from dataclasses import dataclass
from counters import Counters

# Directory to store results
PARENT_DIR = Path('benchmarks/')
RESULTS_CSV = PARENT_DIR.joinpath('results.csv')
LOGGER_FORMAT = '%(asctime)s %(name)s %(funcName)s %(levelname)s: %(message)s'

@dataclass(frozen=True)
class BenchRes():
    solver: str
    n: int
    hessian: str
    time: float
    f_final: float
    grad_norm: float
    cntr : Counters
    success: bool

    def get_result_as_row(self):
        '''Returns a dictionary with the values to be inserted as a row to de database'''
        row = self.__dict__.copy()        # Copy so we can modify it
        row.pop('cntr', None) # We don't care about the object itself
        row['f_evals'] = self.cntr.f_evals
        row['g_evals'] = self.cntr.g_evals
        row['c_evals'] = self.cntr.c_evals
        row['j_evals'] = self.cntr.j_evals
        return row

    def write_to_csv(self):
        '''Writes a row to the csv results'''
        row = self.get_result_as_row()
        row_df = pd.DataFrame([row])
        if RESULTS_CSV.exists():
            row_df.to_csv(RESULTS_CSV, mode='a', header=False, index=False)
        else:
            row_df.to_csv(RESULTS_CSV, mode='w', header=True, index=False)

def clean_outputs():
    '''Cleans the parent directory from all benchmarks'''
    shutil.rmtree(PARENT_DIR)
    PARENT_DIR.mkdir(parents=True)

