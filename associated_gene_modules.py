# -*- coding: utf-8 -*-
# @Author: Xue Chao
# @Time: 2024/04/16 12:36
# @Function: Analyze phenotype-associated module
import os
import re

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from pyecharts.charts import Graph
from pyecharts import options as opts

from util import KGGSEE, log, cluster_df, __heatmap_color_rstyle, jaccard, go_enrich_plot, DGN_API
from associated_cell_types import get_gwas_cate, get_gwas_meta, heatmap_annot2, \
    get_cell_type_abbr_cate, heatmap_two_marker
from para import output_DIR, GWAS_meta_xlsx, UNSELECT_PHENOS, hs_GTEx_GSE97930_fc_dir, TMS_global_dir


def gene_module_annotation_table(dgn_dirs, tags, output_dir, by_p=False):
    phenos = sorted(set([x.split('.')[0] for x in os.listdir(dgn_dirs[0]) if x.endswith('.log')]))
    phenos = [p for p in phenos if p not in UNSELECT_PHENOS]
    data = []
    show_max_term = 3
    top_module_n = 3
    # annot_cols = [['GO:BP','GO:CC','GO:MF'], ['KEGG']]
    # annot_tags=['GO','KEGG']

    # annot_cols = [['GO:BP'], ['GO:CC'], ['GO:MF'], ['KEGG']]
    # annot_tags=['GO:BP','GO:CC','GO:MF', 'KEGG']
    annot_cols = [['GO:BP']]
    annot_tags = ['GO:BP']
    row_1 = ['.']
    for i in range(len(tags)):
        row_1 += [tags[i] for j in range(len(annot_cols))]
    row_2 = ['phenotype_abbr']
    for i in range(len(tags)):
        row_2 += annot_tags
    data += [row_1, row_2]
    for phe in phenos:
        no_file = np.array(
            [not os.path.isfile(f'{dgn_dirs[i]}/{phe}.assoc_gene_module_anno.xlsx') for i in range(len(dgn_dirs))])
        if np.all(no_file):
            continue
        row = [phe]
        for i in range(len(dgn_dirs)):
            dp = f'{dgn_dirs[i]}/{phe}.assoc_gene_module_anno.xlsx'
            if not os.path.isfile(dp):
                print(f'warning: no {dgn_dirs[i]}/{phe} assoc module.')
                for j in range(len(annot_cols)):
                    row.append('/')
                continue
            ## top 1 assoc cells
            kg = KGGSEE(f'{dgn_dirs[i]}/{phe}')
            cell_df = kg.assoc_cell_df('degree.k_adjusted.txt.gz')
            top_cell = cell_df.loc[0, 'TissueName']
            # print(f'{phe} top cells: {top_cell}')
            df = pd.read_excel(dp, index_col=0)
            # max_module_num=df.shape[0]
            # if df.shape[0]<max_module_num:
            #     max_module_num=df.shape[0]
            # df=df.loc[df.index[:max_module_num],df.columns]
            df['cell'] = df.index.map(lambda x: '_'.join(str(x).split('_')[:-1]).strip())
            # df=df.loc[df['cell']==top_cell,:]
            df = df.sort_values(by=['p'])
            for j in range(len(annot_cols)):
                cols = annot_cols[j]
                terms = {}
                temrs_p = {}
                for col in cols:
                    ci = 0
                    for idx in df.index:  # [:top_module_n]
                        if ci >= top_module_n:
                            break
                        term_str = str(df.loc[idx, col])
                        mid_pv = df.loc[idx, 'p']
                        if term_str == '.':
                            continue
                        arr = term_str.split(';')[:1]
                        for a in arr:
                            a = a.strip()
                            p_value = 0
                            p_value_match = re.search(r'P=([\d\.]+(?:e[+-]?\d+)?)', a)
                            if p_value_match:
                                p_value = -np.log10(float(p_value_match.group(1)))
                            p_value = -np.log10(mid_pv)
                            # p_value=mid_pv
                            if a not in terms:
                                terms[a] = 0
                                temrs_p[a] = 0
                            terms[a] += 1
                            if temrs_p[a] < p_value:
                                temrs_p[a] = p_value
                        ci += 1
                if by_p:
                    terms = temrs_p
                max_num = len(terms)
                if max_num > show_max_term:
                    max_num = show_max_term
                row.append('; '.join(sorted(terms.keys(), key=lambda x: -terms[x])[:max_num]))
        data.append(row)
    f_name = 'gene_module_annotation.xlsx'
    if by_p:
        f_name = 'gene_module_annotation.most_sig_p.xlsx'
    fdf = pd.DataFrame(data)
    fdf.to_excel(f'{output_dir}/{f_name}', index=False, header=False)

    ## fine table

    df = fdf.copy()
    df = df.loc[df.index[1:], df.columns[:2]]
    df.columns = df.loc[df.index[0], :]
    df = df.loc[df.index[1:], :]
    term_col = df.columns[1]
    for i in df.index:
        for j in df.columns[1:]:
            term = str(df.loc[i, j]).strip()
            if term == '.' or term == '' or term == '/':
                continue
            if not pd.isna(df.loc[i, j]):
                arr = []
                for x in term.split(';'):
                    tv = re.sub(r'\([^\)]*\)', '', x).strip()
                    pv = re.search(r'P=([\d\.]+(?:e[+-]?\d+)?)', x).group(1)
                    arr.append(f'{tv} (P={pv})')
                new_term = ', '.join(arr)
                df.loc[i, j] = new_term
    gwas_meta = get_gwas_meta(['phenotype', 'category_plot'])
    df['Phenotype'] = df['phenotype_abbr'].map(lambda x: gwas_meta[x][0])
    df['Abbr'] = df['phenotype_abbr']
    df['cate'] = df['Abbr'].map(lambda x: gwas_meta[x][1])
    df = df.sort_values(by=['cate'])
    df = df[['Phenotype', 'Abbr', term_col]]
    df.to_excel(f'{output_dir}/{f_name.split(".xlsx")[0]}.fine.xlsx', index=False)


def combine_module(dgn_dirs, tags, output_dir):
    name_abbr, abbr_cate = get_cell_type_abbr_cate()
    phenos = sorted(set([x.split('.')[0] for x in os.listdir(dgn_dirs[0]) if x.endswith('.log')]))
    phenos = [p for p in phenos if p not in UNSELECT_PHENOS]
    for i in range(len(dgn_dirs)):
        with pd.ExcelWriter(f'{output_dir}/associated_gene_module.{tags[i]}.xlsx') as bw:
            for phe in phenos:
                dp = f'{dgn_dirs[i]}/{phe}.assoc_gene_module_anno.xlsx'
                if not os.path.isfile(dp):
                    if not os.path.isfile(dp):
                        print(f'warning: no {phe} assoc module.')
                    continue
                df = pd.read_excel(dp)
                df = df.sort_values(by=['p'])
                mids = []
                cts = []
                for idx in df.index:
                    raw_mid = df.loc[idx, 'module_id']
                    arr = raw_mid.split('_')
                    raw_id = arr[-1]
                    raw_ct = '_'.join(arr[:-1])
                    ct = name_abbr[raw_ct]
                    mids.append(f'{phe}:{ct}_{raw_id}')
                    cts.append(ct)
                df['module_id'] = mids
                df['cell_type'] = cts
                df = df[[c for c in df.columns if c not in ['p.global_adj_fdr']]]
                df.to_excel(bw, sheet_name=f'{phe}', index=False)


def phenotype_similarity_by_module(dgn_dir, use_max_module=False):
    phenos = sorted(set([x.split('.')[0] for x in os.listdir(dgn_dir) if x.endswith('.log')]))
    phenos = [p for p in phenos if p not in UNSELECT_PHENOS]
    pheno_genes = {}
    pheno_modules = {}
    for pheno in phenos:
        kggsee_prefix = f'{dgn_dir}/{pheno}'
        assoc_module_path = f'{kggsee_prefix}.assoc_gene_module.txt'
        if not os.path.isfile(assoc_module_path):
            log(f'warning: no {pheno} associated module file.')
            continue
        df = pd.read_table(assoc_module_path, header=0, index_col=0)
        df = df.loc[df['p.adj_fdr'] <= 0.1, :]
        df = df.sort_values(by=['p'])
        if df.shape[0] < 1:
            log(f'warning: no {pheno} significant associated modules.')
            continue
        module_genes = set()
        for mid in df.index:
            module_genes = module_genes.union(set(str(df.loc[mid, 'module_gene']).split(',')))
        pheno_genes[pheno] = module_genes
        pheno_modules[pheno] = [str(df.loc[mid, 'module_gene']).split(',') for mid in df.index]
    eff_phenos = sorted(pheno_genes.keys())
    jac_mat = np.zeros(shape=(len(eff_phenos), len(eff_phenos)))
    for i in range(len(eff_phenos)):
        for j in range(i, len(eff_phenos)):
            if use_max_module:
                jis = []
                for m1 in pheno_modules[eff_phenos[i]]:
                    for m2 in pheno_modules[eff_phenos[j]]:
                        jis.append(jaccard(m1, m2))
                ji = max(jis)
            else:
                ji = jaccard(pheno_genes[eff_phenos[i]], pheno_genes[eff_phenos[j]])
            jac_mat[i, j] = ji
            jac_mat[j, i] = ji
    jac_df = pd.DataFrame(jac_mat, columns=eff_phenos, index=eff_phenos)
    ## sort by category and cluster inner category.
    gwas_cate = get_gwas_cate()
    sorted_gwas_cate = ['Brain', 'Immune/Blood', 'Others']
    sorted_cell_cate = ['Brain', 'Immune/Blood', 'Others']
    sorted_cols = []
    sorted_rows = []
    df = jac_df
    n_x, n_y = df.shape
    area_coor_idx = [[0, 0]]
    for i in range(len(sorted_gwas_cate)):
        sg = sorted_gwas_cate[i]
        sc = sorted_cell_cate[i]
        cols = [c for c in df.columns if gwas_cate[c] == sg]
        rows = [x for x in df.index if gwas_cate[x] == sc]
        if len(cols) > 0 and len(rows) > 0:
            sodf = cluster_df(df.loc[:, cols].loc[rows, :])
            sorted_cols += sodf.columns.tolist()
            sorted_rows += sodf.index.tolist()
        else:
            sorted_cols += cols
            sorted_rows += rows
        last_row, last_col = area_coor_idx[-1]
        area_coor_idx.append([len(rows) + last_row, len(cols) + last_col])
    df = df.loc[:, sorted_cols].loc[sorted_rows, :]
    area_coor = []
    for i in range(1, len(area_coor_idx) - 1):
        x, y = area_coor_idx[i]
        if x >= n_x or y >= n_y:
            continue
        area_coor.append([df.index[x], df.columns[y]])

    row_cate = df.index.map(lambda x: gwas_cate[x])
    col_cate = df.columns.map(lambda x: gwas_cate[x])

    zero_mat = np.zeros(shape=df.shape)
    annot1_df = pd.DataFrame(zero_mat > 1, index=df.index, columns=df.columns)
    heatmap_annot2(df, annot1_df, annot1_df, row_cate, col_cate, area_coor, '', '', max_abs_r=1,
                   color_title='Jaccard index',
                   annot_legend_show=False)

    #
    # seaborn.clustermap(jac_df,cmap=__heatmap_color_rstyle_single(),annot=True,fmt='.2f',linewidths=0.5, linecolor='gray')
    # plt.show()


def __load_module_genes(dgn_prefix, is_sig_gene=False):
    col_key = 'module_gene'
    if is_sig_gene:
        col_key = 'assoc_gene'
    assoc_module_path = f'{dgn_prefix}.assoc_gene_module.txt'
    if not os.path.isfile(assoc_module_path):
        return None
    df = pd.read_table(assoc_module_path, header=0, index_col=0)
    df = df.loc[df['p.adj_fdr'] <= 0.1, :]
    df = df.sort_values(by=['p'])
    module_genes = {}
    for mid in df.index:
        module_genes[mid] = set(str(df.loc[mid, col_key]).split(','))
    return module_genes


def __map_mid_abbr(mid: str, abbr_dict: {}):
    arr = mid.split('_')
    cell = '_'.join(arr[:-1])
    id = arr[-1]
    return f'{abbr_dict[cell]}_{id}'


def get_two_module_jaccard_index_df(dgn_prefix1, dgn_prefix2, only_assoc_gene=False):
    pheno1 = os.path.basename(dgn_prefix1)
    pheno2 = os.path.basename(dgn_prefix2)
    mg1 = __load_module_genes(dgn_prefix1, only_assoc_gene)
    mg2 = __load_module_genes(dgn_prefix2, only_assoc_gene)
    if mg1 is None or mg2 is None:
        log(f'is non')
        return
    mids1 = sorted(mg1.keys())
    mids2 = sorted(mg2.keys())
    jac_mat = np.zeros(shape=(len(mids1), len(mids2)))
    for i in range(len(mids1)):
        for j in range(len(mids2)):
            ji = jaccard(mg1[mids1[i]], mg2[mids2[j]])
            jac_mat[i, j] = ji
    jac_df = pd.DataFrame(jac_mat, index=mids1, columns=mids2)
    cell_abbrs, cell_cate = get_cell_type_abbr_cate()
    jac_df.columns = jac_df.columns.map(lambda x: f'{pheno2}:{__map_mid_abbr(x, cell_abbrs)}')
    jac_df.index = jac_df.index.map(lambda x: f'{pheno1}:{__map_mid_abbr(x, cell_abbrs)}')
    return jac_df


def plot_pheno_assoc_module_corr(dgn_prefix1, dgn_prefix2, fig_dir):
    pheno1 = os.path.basename(dgn_prefix1)
    pheno2 = os.path.basename(dgn_prefix2)
    all_mgenes = get_two_module_jaccard_index_df(dgn_prefix1, dgn_prefix2, False)
    sig_mgenes = get_two_module_jaccard_index_df(dgn_prefix1, dgn_prefix2, True)
    heatmap_two_marker(all_mgenes, sig_mgenes, color_title='Jaccard index',
                       x_title=f'Associated modules of {pheno2}',
                       y_title=f'Associated module of {pheno1}',
                       save_fig=f'{fig_dir}/{pheno1}-{pheno2}.Jaccard.heatmap.png')
    for i in all_mgenes.index:
        for j in all_mgenes.columns:
            all_mgenes.loc[i, j] = f'{all_mgenes.loc[i, j]}({sig_mgenes.loc[i, j]})'
    all_mgenes.index.name = '.'
    all_mgenes.to_excel(f'{fig_dir}/{pheno1}-{pheno2}.Jaccard.table.xlsx')

    # zero_mat=np.zeros(shape=df.shape)
    # annot1_df=pd.DataFrame(zero_mat>1,index=df.index,columns=df.columns)
    # heatmap_annot2(df,annot1_df,annot1_df,None,None,None,'','',max_abs_r=1,color_title='Jaccard index',
    #                annot_legend_show=False,x_rev_title=f'Associated modules of {pheno2}',
    #                y_rev_title=f'Associated module of {pheno1}',legend_show=False)
    # seaborn.clustermap(jac_df,cmap=__heatmap_color_rstyle_single(),annot=True,fmt='.2f',linewidths=0.5, linecolor='gray')
    # plt.show()


def plot_two_sim_module_both(dgn_prefix1, module_id1, dgn_prefix2, module_id2, expr_path, intermediate_dir, out_tag):
    mg1 = __load_module_genes(dgn_prefix1)
    mg2 = __load_module_genes(dgn_prefix2)
    ag1 = __load_module_genes(dgn_prefix1, True)
    ag2 = __load_module_genes(dgn_prefix2, True)

    pheno1_name = os.path.basename(dgn_prefix1)
    pheno2_name = os.path.basename(dgn_prefix2)

    common_module_genes = mg1[module_id1].union(mg2[module_id2])
    assoc_genes1 = ag1[module_id1]
    assoc_genes2 = ag2[module_id2]
    assoc_common_genes = assoc_genes1.intersection(assoc_genes2)
    unique_assoc_genes1 = assoc_genes1.difference(assoc_common_genes)
    unique_assoc_genes2 = assoc_genes2.difference(assoc_common_genes)
    all_assoc_genes = assoc_genes1.union(assoc_genes2)
    dgn_api = DGN_API()
    cell_name = os.path.basename(expr_path).split('.')[0]
    cell_corr = dgn_api.get_module_corr(common_module_genes, [expr_path], [cell_name], intermediate_dir)
    corr = cell_corr[cell_name]
    ## plot go enrichment
    go_enrich_plot(sorted(common_module_genes), f"{output_DIR}/analysis/combine/{out_tag}.merge.module_go_enrich.png")

    seaborn.clustermap(corr, cmap=__heatmap_color_rstyle(), vmin=-1, vmax=1)
    plt.show()

    raw_cor_mat = corr.values
    m_genes = corr.columns.tolist()
    cor_mat = np.abs(raw_cor_mat)
    cor_mat[np.diag_indices(cor_mat.shape[0])] = 0
    degree = np.nansum(cor_mat, axis=0)
    node_size = np.power(5, degree / np.max(degree)) * 2 + 0.5
    node_size = node_size * 3
    nodes = []
    links = []
    categories = [{'name': 'UnSig. genes'}, {'name': 'Sig. genes in both'}, {'name': f'Sig. genes in {pheno1_name}'},
                  {'name': f'Sig. genes in {pheno2_name}'}]
    per_cut = np.percentile(degree, 75)
    top_degree_cut = sorted(degree)[-5]
    for i in range(cor_mat.shape[0]):
        gene_name = m_genes[i]
        if not np.any(cor_mat[i, :] >= 0.6):
            continue
        label_color = 'black'
        font_weight = 'normal'
        category = 0
        show = True
        # if degree[i] >= per_cut:
        #     show = True
        if degree[i] >= top_degree_cut:
            label_color = 'red'
            font_weight = 'bolder'
        if gene_name in assoc_common_genes:
            category = 1
        if gene_name in unique_assoc_genes1:
            category = 2
        if gene_name in unique_assoc_genes2:
            category = 3
        # nodes.append({'name': m_genes[i], "symbolSize": node_size[i], 'category': category,
        #               'label': {'show': show}})  # 'value':expr_mean[i],
        nodes.append(opts.GraphNode(name=m_genes[i], symbol_size=node_size[i], category=category, value=degree[i],
                                    # itemstyle_opts=opts.ItemStyleOpts(color=node_color),
                                    label_opts=opts.LabelOpts(is_show=show, color=label_color,
                                                              font_weight=font_weight)))
    nodes = sorted(nodes, key=lambda x: x.opts['category'])
    for i in range(cor_mat.shape[0] - 1):
        for j in range(i + 1, cor_mat.shape[0]):
            w = cor_mat[i, j]
            if w < 0.6:
                continue
            edge_color = '#E1E1E1'
            if m_genes[i] in all_assoc_genes or m_genes[j] in all_assoc_genes:
                edge_color = '#EE6666'
            # links.append({"source": m_genes[i], "target": m_genes[j], "value": w})
            links.append(opts.GraphLink(source=m_genes[i], target=m_genes[j], value=w,
                                        linestyle_opts=opts.LineStyleOpts(color=edge_color, curve=0.2, width=0.7)))
    # 创建Graph对象
    graph = (
        Graph(init_opts=opts.InitOpts(width="1800px", height="1200px"))
        .add("", nodes, links, categories, layout='circular',  # circular,force
             is_draggable=True,
             repulsion=10,
             # linestyle_opts=opts.LineStyleOpts(width=0.5, curve=0.2),  # 设置边的样式
             # label_opts=opts.LabelOpts(font_family='Arial', color='black'),
             is_rotate_label=True,
             )
        .set_global_opts(title_opts=opts.TitleOpts(title=f"{module_id1}-{module_id2}"), legend_opts=opts.LegendOpts(
            textstyle_opts=opts.TextStyleOpts(font_family='Arial', color='black'), orient='vertical'))
        .set_colors(['#73C0DE', '#EE6666', '#EA7CCC', '#91CC75'])
    )

    graph.render(f"{output_DIR}/analysis/combine/{out_tag}.merge.module_network.html")
    cor_mat[np.diag_indices(cor_mat.shape[0])] = 1


if __name__ == '__main__':
    # 1. module annotation analysis
    gwas_conf = GWAS_meta_xlsx
    fig_dir = f'{output_DIR}/analysis/combine'
    sc_types = ['hs_gtex_gse97930_fc', 'tms_global']
    dgn_dirs = [f'{output_DIR}/{k}/result' for k in sc_types]
    expr_dirs = [hs_GTEx_GSE97930_fc_dir, TMS_global_dir]
    gene_module_annotation_table(dgn_dirs, ['DGN.hs', 'DGN.mm'], fig_dir, True)
    combine_module(dgn_dirs, ['hs', 'mm'], fig_dir)

    # 2. module analysis
    sc_types = ['hs_gtex_gse97930_fc']
    dgn_dirs = [f'{output_DIR}/{k}/result' for k in sc_types]
    for i in range(len(dgn_dirs)):
        dgn_dir = dgn_dirs[i]
        dgn_par_dir = os.path.dirname(dgn_dir)
        expr_dir = expr_dirs[i]
        phenotype_similarity_by_module(dgn_dir)
        plot_pheno_assoc_module_corr(f'{dgn_dir}/BIP', f'{dgn_dir}/SCZ', fig_dir)
        plot_pheno_assoc_module_corr(f'{dgn_dir}/UC', f'{dgn_dir}/RA')

        plot_two_sim_module_both(f'{dgn_dir}/BIP', 'Ex1_1', f'{dgn_dir}/SCZ', 'Ex1_1',
                                 f'{expr_dir}/Ex1.txt.gz', f'{dgn_par_dir}/intermediate', 'bip_scz_ex1')
        plot_two_sim_module_both(f'{dgn_dir}/UC', 'Immune(DC)_3', f'{dgn_dir}/RA', 'Immune(DC)_4',
                                 f'{expr_dir}/Immune(DC).txt.gz', f'{dgn_par_dir}/intermediate', 'uc_ra_dc2')
