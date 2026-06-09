import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

PARENT_DIR = Path('logs/experiments/')
RESULTS_CSV = PARENT_DIR.joinpath('results.csv')
DEST_DIR = Path.cwd().parent.joinpath('thesis-latex/figs')

COMPARISON_DIR = Path('logs/comparison/')
COMPARISON_CSV = COMPARISON_DIR.joinpath('results.csv')
COMPARISON_DEST = DEST_DIR.joinpath('comparison')

METHOD_LABEL = {
    'SciPyAttack': 'L-BFGS-B',
    'OptimusAttack': 'LS-SQP',
}

# COL1 = '2AB0FF'
# COL2 = '00675F' # dark green
# COL2 = 'FF7582' # redish
# COL2 = '54BD58' # light green

df = pd.read_csv(RESULTS_CSV)
df['log2_c'] = np.log2(df['c'])

## Source-target heatmaps
def source_target_heatmap_success(formulation='szegedy', norm='L2'):
    filt_df = df[(df['formulation'] == formulation) & (df['norm'] == norm)]
    hm_succ = filt_df.groupby(['a','t'])['success'].mean().unstack(fill_value=0)
    _source_target_heatmap(hm_succ, f'success-{formulation}-{norm}')

def source_target_heatmap_nits(formulation='szegedy', norm='L2'):
    filt_df = df[(df['formulation'] == formulation) & (df['norm'] == norm)]
    hm_nits = filt_df.groupby(['a','t'])['nits'].mean().unstack(fill_value=0)
    _source_target_heatmap(hm_nits, f'nits-{formulation}-{norm}')

def source_target_heatmap_min_dist(formulation='szegedy', norm='L2', attacker_name='SciPyAttack'):
    filt_df = df[
        (df['formulation'] == formulation) &
        (df['norm'] == norm) &
        (df['attacker_name'] == attacker_name) &
        (df['success'] == True)
    ]
    hm_dist = filt_df.groupby(['a','t'])['distance'].min().unstack(fill_value=0)
    _source_target_heatmap(hm_dist, f'min-dist-{formulation}-{norm}') # should probably add the attacker_name

def _source_target_heatmap(data, title):
    # plt.figure(figsize=(7, 6))
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt='.2f')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$a$')
    # plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{DEST_DIR.joinpath('heatmaps')}/{title}.pdf", bbox_inches='tight')
    plt.close()

def c_vs_distance_plot(formulation='szegedy', norm='L2'):
    filt_df = df[(df['formulation'] == formulation) & (df['norm'] == norm)]

    dist_grouped = filt_df.groupby(['log2_c'], as_index=False)['distance'].mean()
    sucs_grouped = filt_df.groupby(['log2_c'], as_index=False)['success'].mean()

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Plot mean distance on primary y-axis
    sns.lineplot(data=dist_grouped, x='log2_c', y='distance', marker='o', ax=ax1, label='Distancia promedio', color=f'tab:blue', legend=False)
    ax1.set_xlabel(r'$\log_2(c)$')
    ax1.set_ylabel('Distancia')
    ax1.tick_params(axis='y')

    # Create secondary y-axis
    ax2 = ax1.twinx()
    sns.lineplot(data=sucs_grouped, x='log2_c', y='success', marker='s', ax=ax2, label='Porcentaje de éxito', color=f'tab:orange', legend=False)
    ax2.set_ylabel('Éxito')
    ax2.tick_params(axis='y')

    # plt.title(f'Mean Distance and Success Rate vs c ({formulation}, {norm})')
    ax1.grid(True)
    ax2.grid(False)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2)

    plt.tight_layout()
    plt.savefig(f"{DEST_DIR.joinpath('cdist')}/{formulation}-{norm}.pdf", bbox_inches='tight')
    plt.close()

def c_vs_dist_scatter(norm='L2'):
    filt_df = df[(df['formulation'] == 'szegedy') | (df['formulation'] == 'carlini')]
    filt_df = filt_df[filt_df['norm'] == norm]

    g = sns.relplot(
        data=filt_df,
        x='log2_c',
        y='distance',
        hue='success',
        hue_order=[True, False],
        style='success',
        style_order=[True, False],
        col='formulation',
        aspect=1.2,
        alpha=0.9
    )

    g.set_axis_labels(r'$\log_2(c)$', 'Distancia')
    g._legend.set_title('Resultado')
    g._legend.texts[0].set_text('Éxito')
    g._legend.texts[1].set_text('Fallo')
    g.axes.flat[0].set_title('Szegedy')
    g.axes.flat[1].set_title('Carlini')
    g.savefig(f"{DEST_DIR.joinpath('cdist')}/{norm}-scatter.pdf", bbox_inches='tight')
    plt.close()

## Comparison: LS-SQP vs L-BFGS-B on a fixed (a=3, t=8) singleton task
def comparison_image_grid():
    '''
    2-row x N-col PDF showing the adversarial example each method produced for
    each c. Rows: L-BFGS-B (top) and LS-SQP (bottom). One column per c value
    found in the comparison CSV. Source/target reference is in
    `comparison_source_target`.
    '''
    cdf = pd.read_csv(COMPARISON_CSV)
    cs = sorted(cdf['c'].unique())
    methods = ['SciPyAttack', 'OptimusAttack']

    fig, axes = plt.subplots(2, len(cs), figsize=(2.5 * len(cs), 5.5))
    for i, method in enumerate(methods):
        for j, c in enumerate(cs):
            fname = COMPARISON_DIR.joinpath(
                f'{method}-L2-szegedy-3-8-{c:.2f}.png'
            )
            img = mpimg.imread(fname)
            axes[i, j].imshow(img)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(f'$c = {c:g}$')
            if j == 0:
                axes[i, j].set_ylabel(METHOD_LABEL[method])

    plt.tight_layout()
    COMPARISON_DEST.mkdir(parents=True, exist_ok=True)
    plt.savefig(COMPARISON_DEST.joinpath('comparison-grid.pdf'),
                bbox_inches='tight')
    plt.close()

def gen_heatmaps():
    # Successful attacks
    # source_target_heatmap_success(formulation='szegedy', norm='L2')
    # source_target_heatmap_success(formulation='carlini', norm='L2')
    # source_target_heatmap_success(formulation='szegedy', norm='L1')
    # source_target_heatmap_success(formulation='carlini', norm='L1')
    # source_target_heatmap_success(formulation='szegedy', norm='LINF')
    # source_target_heatmap_success(formulation='carlini', norm='LINF')

    # number of iterations needed
    source_target_heatmap_nits(formulation='szegedy', norm='L2')
    source_target_heatmap_nits(formulation='carlini', norm='L2')
    source_target_heatmap_nits(formulation='szegedy', norm='L1')
    source_target_heatmap_nits(formulation='carlini', norm='L1')
    source_target_heatmap_nits(formulation='szegedy', norm='LINF')
    source_target_heatmap_nits(formulation='carlini', norm='LINF')

    # minimum distance
    source_target_heatmap_min_dist(formulation='szegedy', norm='L2')
    source_target_heatmap_min_dist(formulation='carlini', norm='L2')
    source_target_heatmap_min_dist(formulation='szegedy', norm='L1')
    source_target_heatmap_min_dist(formulation='carlini', norm='L1')
    source_target_heatmap_min_dist(formulation='szegedy', norm='LINF')
    source_target_heatmap_min_dist(formulation='carlini', norm='LINF')

def gen_c_vs_dist():
    c_vs_distance_plot(formulation='szegedy', norm='L2')
    c_vs_distance_plot(formulation='carlini', norm='L2')
    c_vs_distance_plot(formulation='szegedy', norm='L1')
    c_vs_distance_plot(formulation='carlini', norm='L1')
    c_vs_distance_plot(formulation='szegedy', norm='LINF')
    c_vs_distance_plot(formulation='carlini', norm='LINF')

    c_vs_dist_scatter(norm='L2')
    c_vs_dist_scatter(norm='L1')
    c_vs_dist_scatter(norm='LINF')

def gen_comparison():
    comparison_image_grid()

# Generate all plots available
def gen_all():
    gen_heatmaps()
    gen_c_vs_dist()
    gen_comparison()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate graphs for the thesis')
    parser.add_argument(
        '--hm',
        action='store_true',
        help='Only generate the heatmaps available'
    )
    parser.add_argument(
        '--cdist',
        action='store_true',
        help='Only generate the images of c vs the distance'
    )
    parser.add_argument(
        '--comparison',
        action='store_true',
        help='Only generate the LS-SQP vs L-BFGS-B comparison artifacts'
    )
    args = parser.parse_args()


    if args.hm:
        gen_heatmaps()
    elif args.cdist:
        gen_c_vs_dist()
    elif args.comparison:
        gen_comparison()
    else:
        gen_all()

