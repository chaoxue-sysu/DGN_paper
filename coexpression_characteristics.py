# -*- coding: utf-8 -*-
# @Author: Xue Chao
# @Time: 2024/04/26 16:10
# @Function: Co-expression network analysis.
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn
from pyecharts import options as opts
from pyecharts.charts import Graph
from scipy.stats import pearsonr

from util import cluster_df, make_dir
from associated_cell_types import get_cell_type_abbr_cate
from para import output_DIR, Common_results_dir


def __extract_z_col(df, cell_abbrs=None, suffix='.z'):
    df = df[[d for d in df.columns if d.endswith(suffix)]]
    df.columns = df.columns.map(lambda x: '.'.join(x.split('.')[:-1]))
    if cell_abbrs is not None:
        df.columns = df.columns.map(lambda x: cell_abbrs[x])
    return df


def plot_coexpression_corr(expr_path, out_fn):
    cell_abbrs, cell_cate = get_cell_type_abbr_cate()
    r_cut = 0.2
    df = pd.read_table(expr_path, header=0, index_col=0)
    df = __extract_z_col(df, cell_abbrs)
    sort_cols = cluster_df(df.corr()).columns
    df = df[sort_cols]
    cells = df.columns.values
    corr = np.corrcoef(df.values.T)
    nodes, links, categories = [], [], []
    for i in np.arange(0, corr.shape[0] - 1):
        for j in np.arange(i + 1, corr.shape[0]):
            r = corr[i, j]
            if r > r_cut:
                links.append({"source": cells[i], "target": cells[j], "value": r})
    cates = sorted(set(cell_cate.values()))
    cate_idxs = {cates[i]: i for i in range(len(cates))}
    categories = [{'name': c} for c in cates]
    for i in np.arange(0, corr.shape[0]):
        category = cate_idxs[cell_cate[cells[i]]]
        show = True
        nodes.append({'name': cells[i], 'category': category, 'label': {'show': show, 'formatter': '{b}'}})
    nodes = sorted(nodes, key=lambda x: x['category'], reverse=True)
    fn = os.path.basename(expr_path)
    graph = (
        Graph(init_opts=opts.InitOpts(width="1400px", height="1200px"))
        .add("", nodes, links, categories, layout='circular',  # circular
             is_draggable=True,
             repulsion=10,
             linestyle_opts=opts.LineStyleOpts(width=0.5, curve=0.2, color='source'),
             label_opts=opts.LabelOpts(font_family='Arial', color='black'),
             is_rotate_label=True,
             )
        .set_global_opts(title_opts=opts.TitleOpts(title=out_fn), legend_opts=opts.LegendOpts(
            textstyle_opts=opts.TextStyleOpts(font_family='Arial', color='black'), orient='vertical')
                         )
    )
    graph.render(f"{Common_results_dir}/paper_supp/{out_fn}.html")

    seaborn.clustermap(df.corr())
    plt.savefig(f"{Common_results_dir}/paper_supp/{out_fn}.heatmap.png")


def plot_k_factor(data_path, fig_path):
    df = pd.read_table(data_path, index_col=0).T
    df['log(sample_size)'] = np.log2(df['sample_size'])
    r, p = pearsonr(df['log(sample_size)'], df['k_factor'])
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    seaborn.regplot(x='log(sample_size)', y='k_factor', data=df, ci=95, line_kws={'color': 'red'}, ax=ax,
                    scatter_kws={'s': 15})
    sx = 0.95 * (np.max(df['log(sample_size)']) - np.min(df['log(sample_size)'])) + np.min(df['log(sample_size)'])
    sy = 0.95 * (np.max(df['k_factor']) - np.min(df['k_factor'])) + np.min(df['k_factor'])
    plt.text(sx, sy, r'$R' + f'=${r:.2f}, ' + r'$p' + f'=${p:.2g}', ha='right', va='center')
    plt.xlabel(r'$log_{2}(Sample\,size)$')
    plt.ylabel(r'$k$')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(fig_path)


def plot_k_factor_edge(data_path, fig_path):
    df = pd.read_table(data_path, index_col=0).T
    df['log(sample_size)'] = np.log2(df['sample_size'])
    r, p = pearsonr(df['log(sample_size)'], df['mean_abs_r'])
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    seaborn.regplot(x='log(sample_size)', y='mean_abs_r', data=df, ci=95, line_kws={'color': 'red'}, ax=ax,
                    scatter_kws={'s': 15})
    sx = 0.95 * (np.max(df['log(sample_size)']) - np.min(df['log(sample_size)'])) + np.min(df['log(sample_size)'])
    sy = 0.95 * (np.max(df['mean_abs_r']) - np.min(df['mean_abs_r'])) + np.min(df['mean_abs_r'])
    plt.text(sx, sy, r'$R' + f'=${r:.2f}, ' + r'$p' + f'=${p:.2g}', ha='right', va='center')
    plt.xlabel(r'$log_{2}(Sample\,size)$')
    plt.ylabel(r'$Mean(|r|)$')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(fig_path)


def plot_corr_dist(expr_degree_paths):
    cell_abbrs, cell_cate = get_cell_type_abbr_cate()
    dfs = []
    for expr_p, degree_p, type in expr_degree_paths:
        out_path = f'{os.path.dirname(expr_p)}/degree_expr_spearman.corr.xlsx'
        edf = __extract_z_col(pd.read_table(expr_p, header=0, index_col=0), cell_abbrs)
        ddf = __extract_z_col(pd.read_table(degree_p, header=0, index_col=0), cell_abbrs)
        genes = np.intersect1d(edf.index, ddf.index)
        edf = edf.loc[genes, :]
        ddf = ddf.loc[genes, :]
        cells = np.intersect1d(edf.columns.values, ddf.columns.values)
        rs = []
        for c in cells:
            ## non-zero genes correlation
            x, y = edf[c].values, ddf[c].values
            # non_zero= x!=0
            # x,y=x[non_zero],y[non_zero]
            r, p = scipy.stats.pearsonr(x, y)
            rs.append(r)
        df = pd.DataFrame({'cell_type': cells, 'r': rs})
        df.to_excel(out_path, index=False)
        df['Dataset'] = type
        dfs.append(df)
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    fdf = pd.concat(dfs, axis=0, ignore_index=True)
    seaborn.kdeplot(fdf, x='r', hue='Dataset', fill=True, ax=ax)
    # seaborn.histplot(fdf,x='spearman_r',hue='Dataset', kde=True, bins=50)
    median = np.percentile(fdf['r'], 50)
    ax.axvline(x=median, color='r', linestyle='--', linewidth=1)
    ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1] * 1.1])
    ax.text(median, plt.ylim()[1] * 0.9, f'Median={median:.2f} ', color='black', ha='right')

    plt.xlabel("Pearson's R")
    plt.tight_layout()
    plt.show()
    pass


def plot_edge_dist(corr_path, k_path, fig_path):
    cdf = pd.read_table(corr_path)
    kdf = pd.read_table(k_path, index_col=0)
    kdf = kdf.T.sort_values(by=['sample_size']).T
    all_val = np.power(cdf.loc[:, kdf.columns].values, kdf.loc['k_factor', :].values).flatten()
    arr = np.sort(all_val)
    indices = np.linspace(0, len(arr) - 1, cdf.shape[0], dtype=int)
    bg_val = arr[indices]
    need_axes_num = kdf.shape[1]
    sub_axes_unit = 3
    n = int(np.sqrt(need_axes_num) - 0.0001) + 1
    plot_ncol = n
    plot_nrow = int(need_axes_num / plot_ncol - 0.0001) + 1
    fig, axes = plt.subplots(plot_nrow, plot_ncol, figsize=(plot_ncol * sub_axes_unit, plot_nrow * sub_axes_unit))
    k = 0
    top_edge_n = int(0.95 * len(bg_val))
    bg_val = bg_val[:top_edge_n]
    max_bg_num = np.max(bg_val)
    for i in range(plot_nrow):
        for j in range(plot_ncol):
            if plot_nrow == 1:
                ax = axes[j]
            else:
                if plot_ncol == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]
            if k < need_axes_num:
                cell = kdf.columns[k]
                k_fac = kdf.loc['k_factor', cell]
                ss = int(kdf.loc['sample_size', cell])
                val = np.sort(cdf.loc[:, cell].values)
                k_val = np.power(val, k_fac)
                val = val[:top_edge_n]
                k_val = k_val[:top_edge_n]
                max_num = max([max_bg_num, np.max(np.concatenate((val, k_val)))])
                ax.plot(val, bg_val, '.', label='raw')
                ax.plot(k_val, bg_val, '.', label='k-adjusted')
                ax.plot([0, max_num], [0, max_num], '-', c='black')
                ax.set_title(f'{cell}\ncell={ss};k={k_fac:.3f}')
                ax.legend()
            else:
                plt.delaxes(ax)
            k += 1
    plt.tight_layout()
    make_dir(os.path.dirname(fig_path))
    plt.savefig(fig_path)


def main():
    ks = ['hs_gtex_gse97930_fc', 'tms_global', 'simulate']
    for i in range(len(ks)):
        k = ks[i]
        dpath = f'{output_DIR}/{k}/intermediate/degree.k_adjusted.REZ.webapp.genes.txt.gz'
        plot_coexpression_corr(dpath, k)

        k_path = f'{output_DIR}/{k}/intermediate/k_factor.txt'
        edge_sample_plot_path = f'{output_DIR}/{k}/intermediate/edge_sample_size.scatter.png'
        plot_k_factor_edge(k_path, edge_sample_plot_path)
        k_sample_plot_path = f'{output_DIR}/{k}/intermediate/k_sample_size.scatter.png'
        plot_k_factor(k_path, k_sample_plot_path)

        corr_path = f'{output_DIR}/{k}/intermediate/resample_corr.txt.gz'
        edge_dist_plot_path = f'{output_DIR}/{k}/intermediate/edge.dist.png'
        plot_edge_dist(corr_path, k_path, edge_dist_plot_path)
    #
    ks = ['hs_gtex_gse97930_fc', 'tms_global']
    vs = ['Human', 'Mouse']
    datas = []
    for i in range(len(ks)):
        k = ks[i]
        v = vs[i]
        dir = f'{output_DIR}/{k}/intermediate'
        datas.append([f'{dir}/expr.mean.REZ.webapp.genes.txt.gz', f'{dir}/degree.k_adjusted.REZ.webapp.genes.txt.gz',
                      v])
    plot_corr_dist(datas)


if __name__ == '__main__':
    main()
