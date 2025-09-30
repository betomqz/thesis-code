import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Use LaTeX and font size 16
sns.set_theme()
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 16,
    'font.family': 'serif'
})
sns.set_context('talk')

PARENT_DIR = Path('benchmarks/')
RESULTS_CSV = PARENT_DIR.joinpath('results.csv')
DEST_DIR = Path.cwd().parent.joinpath('thesis-latex/figs')

df = pd.read_csv(RESULTS_CSV)
df['time_log'] = np.log2(df['time'])

def _plot_pivoted(pivoted_time, normal=True):
    title = 'normal' if normal else 'log'

    plt.figure(figsize=(10, 6 if normal else 7))
    pivoted_time.plot(marker="o", logy=False, ax=plt.gca())
    plt.xlabel("$n$")
    if normal:
        plt.ylabel("Segundos")
    else:
        plt.ylabel(r'Segundos ($\log_2$)')
    fontsize = 14
    if normal:
        plt.legend(loc='upper left', fontsize=fontsize)
    else:
        plt.legend(bbox_to_anchor=(0.5, -0.2), fontsize=fontsize, loc='upper center', ncol=3)
        # plt.legend(loc='upper left', fontsize=fontsize, ncol=2)
        # plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', ncol=1)
    plt.tight_layout()
    plt.savefig(f"{DEST_DIR.joinpath('benchmarks')}/{title}.pdf", bbox_inches='tight')
    plt.close()

def log_time():

    grouped = df.groupby(['solver', 'hessian', 'n'], as_index=False).median()

    pivoted_time = grouped.pivot_table(
        index="n", 
        columns=["solver", "hessian"], 
        values="time_log"
    )
    _plot_pivoted(pivoted_time, False)

def normal_time():

    grouped = df.groupby(['solver', 'hessian', 'n'], as_index=False).median()

    pivoted_time = grouped.pivot_table(
        index="n", 
        columns=["solver", "hessian"], 
        values="time"
    )
    _plot_pivoted(pivoted_time, True)



def gen_all():
    log_time()
    normal_time()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate benchmark graphs for the thesis')
    parser.add_argument(
        '--log',
        action='store_true',
        help='Only generate the graph using log time'
    )
    parser.add_argument(
        '--normal',
        action='store_true',
        help='Only generate the graph using normal time'
    )
    args = parser.parse_args()

    if args.log:
        log_time()
    elif args.normal:
        normal_time()
    else:
        gen_all()

