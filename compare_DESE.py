# -*- coding: utf-8 -*-
# @Author: Xue Chao
# @Time: 2024/06/11 20:14
# @Function: Compare DGN to DESE in estimating associated cells and genes.
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats
import seaborn
from matplotlib import pyplot as plt
from scipy.stats import stats

from associated_genes import PhenoGene, compare_assoc_gene_AUC_group_plot, compare_assoc_gene_venn
from associated_cell_types import get_gwas_cate, get_cell_type_abbr_cate
from para import output_DIR, UNSELECT_PHENOS, CELL_CATE_colors, GWAS_ROOT, GWAS_case_dir
from util import KGGSEE, log, kggsee_dese, LOCAL_DIR, batch_run_function

class Compare_DGN_DESE:
    def __init__(self,dgn_outdirs,dgn_tags):
        self.dgn_outdirs=dgn_outdirs
        self.dgn_tags=dgn_tags
        self.expr_tags={'DGN':'degree.k_adjusted.txt.gz','DESE':'expr.mean.txt.gz'}
        self.methods=['DGN','DESE']

    def load_cell_p(self,kggsee_prefix,adj_p=True):
        ks=KGGSEE(kggsee_prefix)
        dfs=[]
        for m in self.methods:
            if adj_p:
                cell_p=ks.assoc_cell_adj_p(self.expr_tags[m])
            else:
                cell_p = ks.assoc_cell_raw_p(self.expr_tags[m])
            sdf=pd.DataFrame(cell_p,index=[m])
            dfs.append(sdf)
        df=pd.concat(dfs,axis=0).T
        return df

    def load_cell_p_full(self,kggsee_prefix):
        ks=KGGSEE(kggsee_prefix)
        dfs=[]
        for m in self.methods:
            cell_p=ks.assoc_cell_df(self.expr_tags[m])
            sdf=cell_p
            sdf=sdf.loc[:,[c for c in sdf.columns if c!='Median(IQR)SigVsAll']]
            sdf.columns=['cell_type','p','p.adj']
            sdf.index=sdf['cell_type']
            sdf=sdf[['p','p.adj']]
            sdf.columns = sdf.columns.map(lambda x:f'{x}({m})')
            dfs.append(sdf)
        df=pd.concat(dfs,axis=1)
        return df

    def compare_cell_p_corr_dist(self,output_dir):
        name_abbr,abbr_cate=get_cell_type_abbr_cate()
        adj_p=False
        sc_rs={}
        dfs=[]
        gwas_cate=get_gwas_cate()
        for sc_tag,dgn_dir in zip(self.dgn_tags,self.dgn_outdirs):
            phenos=sorted(set([f.split('.')[0] for f in os.listdir(dgn_dir) if f.endswith('.log')]))
            phenos = [p for p in phenos if p not in UNSELECT_PHENOS]
            rs=[]
            sig_dfs=[]
            all_sig,both_sig=0,0
            with pd.ExcelWriter(f'{output_dir}/associated_cell_types.{sc_tag}.xlsx') as bw:
                for pheno in phenos:
                    kp=f'{dgn_dir}/{pheno}'
                    cell_p=self.load_cell_p(kp,adj_p)
                    min_p=-np.log10(cell_p.min(axis=0))
                    sig_dfs.append(pd.DataFrame(min_p,columns=[pheno]))
                    cell_p_v=cell_p.values
                    r,p=scipy.stats.spearmanr(cell_p_v[:, 0], cell_p_v[:, 1])
                    rs.append(r)
                    full_cell_p=self.load_cell_p_full(kp)
                    full_cell_p.index.name='cell_type'
                    full_cell_p=full_cell_p.sort_values(by=['p.adj(DGN)','p(DGN)'])
                    full_cell_p.index=full_cell_p.index.map(lambda x:name_abbr[x])
                    notes=[]
                    adj_p_cut=0.05
                    for c in full_cell_p.index:
                        nt='Unsig.'
                        dgn_p=full_cell_p.loc[c,'p.adj(DGN)']
                        dese_p=full_cell_p.loc[c,'p.adj(DESE)']
                        if dgn_p<adj_p_cut and dese_p<adj_p_cut:
                            nt='Both Sig.'
                            both_sig+=1
                        if dgn_p >= adj_p_cut and dese_p < adj_p_cut:
                            nt = 'DESE Sig. Only'
                        if dgn_p < adj_p_cut and dese_p >= adj_p_cut:
                            nt = 'DGN Sig. Only'
                        notes.append(nt)
                        if dgn_p<adj_p_cut:
                            all_sig+=1
                    full_cell_p['note']=notes
                    full_cell_p.to_excel(bw,sheet_name=f'{pheno}')
            log(f'{sc_tag}: all sig={all_sig}; both sig={both_sig}; uniq sig={all_sig-both_sig}')
            sig_fdf=pd.concat(sig_dfs,axis=1).T
            max_p=sig_fdf.max().max()
            sig_fdf['Category']=sig_fdf.index.map(lambda x:gwas_cate[x])
            sig_fdf['dif']=sig_fdf['DGN']-sig_fdf['DESE']
            fig, ax = plt.subplots(figsize=(3.8, 3.6))
            seaborn.scatterplot(sig_fdf,x='DESE',y='DGN',hue='Category',ax=ax,palette=CELL_CATE_colors,hue_order=sorted(set(gwas_cate.values())))
            ax.plot([0,max_p],[0,max_p],'-',c='black',lw=0.8)
            plt.tight_layout()
            plt.show()
            plt.close()
            # print(sig_fdf.sort_values(by=['dif'])
            sc_rs[sc_tag]=rs
            df=pd.DataFrame({'r':rs},index=phenos)
            df['Phenotype']=df.index
            df['Category']=df.index.map(lambda x:gwas_cate[x])
            df=df.sort_values(by=['Category','r'])
            df['index']=np.arange(df.shape[0])
            fig, ax = plt.subplots(figsize=(4.0, 3.6))
            ij=0
            log(f'{sc_tag}:mean r={np.nanmean(df["r"])}')
            for cate,gdf in df.groupby(by='Category'):
                c=CELL_CATE_colors[ij]
                ax.bar(gdf['index'],gdf['r'],label=cate,alpha=0.7,color=c,edgecolor=c,width=0.7)
                ij+=1
            ax.set_xticks(df['index'])
            ax.set_xticklabels(df['Phenotype'],rotation=90)
            ax.set_ylabel(f'Spearman\'s R')
            ax.set_ylim((0,1.4))
            ax.legend()
            plt.tight_layout()
            plt.show()
            plt.close()
            # dfs.append(df)
            # one scRNA-seq one xlsx file

    def __gene_note(self,p_dgn,p_dese,p_cutoff):
        tag_val=['DGN Sig. only','DESE Sig. only','Both Sig.','Unsig.']
        v=3
        if p_dgn<=p_cutoff and p_dese<=p_cutoff:
            v=2
        elif p_dgn<=p_cutoff and p_dese>p_cutoff:
            v=0
        elif p_dgn>p_cutoff and p_dese<=p_cutoff:
            v=1
        return v,tag_val[v]


    def compare_gene_p(self,output_dir,db_local_dir,db_ecs_dir,db_ehe_dir):
        ## TO-DO: compare
        adj_p=False
        sc_rs={}
        dfs=[]
        gwas_cate=get_gwas_cate()
        for sc_tag,dgn_dir in zip(self.dgn_tags,self.dgn_outdirs):
            phenos=sorted(set([f.split('.')[0] for f in os.listdir(dgn_dir) if f.endswith('.log')]))
            phenos = [p for p in phenos if p not in UNSELECT_PHENOS]
            rs=[]
            sig_dfs=[]
            with pd.ExcelWriter(f'{output_dir}/associated_genes.{sc_tag}.xlsx') as bw:
                combine_pheno_genes = {}
                combine_tag_df={}
                combine_ncbi_df=[]
                combine_tag_auc={}
                for pheno in phenos:
                    ncbi_df = pd.read_table(f'{db_local_dir}/{pheno}.tsv')
                    gene_pmids={}
                    for ni in ncbi_df.index:
                        gene_pmids[ncbi_df.loc[ni,'gene']]=ncbi_df.loc[ni,'count']
                    ecs_ks=KGGSEE(f'{db_ecs_dir}/{pheno}')
                    ecs_df = ecs_ks.cond_assoc_genes_df(index_col=0)
                    ecs_p={}
                    for ni in ecs_df.index:
                        ecs_p[ni]={'CondiECSP':ecs_df.loc[ni,'CondiECSP'],'GeneScore':ecs_df.loc[ni,'GeneScore']}
                    ehe_ks = KGGSEE(f'{db_ehe_dir}/{pheno}')
                    ehe_p = ehe_ks.cond_sig_assoc_gene_herit()
                    genes_dfs=[]
                    sig_genes={}
                    kp=f'{dgn_dir}/{pheno}'
                    ks=KGGSEE(kp)
                    dfs={}
                    sig_p_cutoff=1
                    for m in self.methods:
                        df=ks.cond_assoc_genes_df(self.expr_tags[m],index_col=0)
                        df=df.loc[~df.index.duplicated(),:]
                        dfs[m]=df
                        sig_p_cutoff=ks.gene_based_p_cutoff()
                        genes_dfs.append([df,'CondiECSP'])
                    spec_cols=['GeneScore','CondiECSP']
                    com_df=dfs['DGN'].loc[:,[x for x in dfs['DGN'].columns if x not in spec_cols]]
                    sort_idx=com_df.index.values
                    for m in self.methods:
                        dfs[m] = dfs[m].loc[sort_idx, :]
                    for scol in spec_cols:
                        com_df[f'{scol}(Raw)']=[ecs_p[comi][scol] for comi in com_df.index]
                        for m in self.methods:
                            com_df[f'{scol}({m})'] = dfs[m].loc[:, scol]
                    #herit
                    com_df['CondiHeritEHE']=com_df.index.map(lambda x:ehe_p[x] if x in ehe_p else np.nan)
                    note_tags=[]
                    note_v=[]
                    for g in com_df.index:
                        tag_v,tag=self.__gene_note(com_df.loc[g,f'CondiECSP(DGN)'],com_df.loc[g,f'CondiECSP(DESE)'],sig_p_cutoff)
                        note_tags.append(tag)
                        note_v.append(tag_v)
                    com_df[f'note']=note_tags
                    com_df[f'note_val'] = note_v
                    com_df=com_df.sort_values(by=['note_val','CondiECSP(DGN)'])
                    com_df=com_df.loc[:,[x for x in com_df.columns if x not in ['note_val']]]
                    com_df.index.name='Gene'
                    com_df['Pubmed count']=com_df.index.map(lambda x:gene_pmids[x])
                    com_df.to_excel(bw,sheet_name=f'{pheno}')
                    ## do
                    db_name='pubmed'
                    ncbi_df['gene'] = ncbi_df['gene'].map(lambda x: f'{pheno}:{x}')
                    combine_ncbi_df.append(ncbi_df)
                    sort_genes_dfs=[]
                    for i in range(len(self.methods)):
                        tag=self.methods[i]
                        df, p_col = genes_dfs[i]
                        df['unique_p'] = df[p_col]
                        df['unique_gene'] = df.index.map(lambda x: f'{pheno}:{x}')
                        df.sort_values(by='unique_p', inplace=True, ignore_index=True)
                        sort_genes_dfs.append(df)
                        if tag not in combine_tag_df:
                            combine_tag_df[tag]=[]
                        combine_tag_df[tag].append(df)
                        sig_genes[tag] = set(df.loc[df['unique_p'] < sig_p_cutoff, 'unique_gene'].values.tolist())
                    tags=self.methods
                    fig, tag_auc = PhenoGene(db_name).eval_ROC_gene_based(sort_genes_dfs, ncbi_df,
                                                                          tags, 5, title=pheno,plot_fig=False)
                    for i in range(len(tags)):
                        tag=tags[i]
                        if tag not in combine_tag_auc:
                            combine_tag_auc[tag]=[]
                        combine_tag_auc[tag].append(tag_auc[tag])
                    for tag in sig_genes.keys():
                        if tag not in combine_pheno_genes:
                            combine_pheno_genes[tag] = []
                        combine_pheno_genes[tag] += sorted([f'{pheno}:{sg}' for sg in sig_genes[tag]])
                ## plot auc volint
                auc_df=pd.DataFrame(combine_tag_auc)
                fig = compare_assoc_gene_AUC_group_plot(auc_df,two_side=True,figsize=(3.5,3.5))
                fig.savefig(f'{output_dir}/compare_gene_AUC_group.{sc_tag}.png')
                plt.close()
                ddfs=[]
                for m in tags:
                    ddf=pd.concat(combine_tag_df[m],axis=0,ignore_index=True)
                    ddf=ddf.sort_values(by=['unique_p'])
                    ddfs.append(ddf)
                fn_df=pd.concat(combine_ncbi_df,axis=0,ignore_index=True)
                fig, tag_auc = PhenoGene(db_name).eval_ROC_gene_based(ddfs, fn_df,
                                                                      tags, 5, title='', plot_fig=True)
                fig.savefig(f'{output_dir}/compare_gene_ROC.{sc_tag}.png')
                for tag in tags:
                    combine_pheno_genes[tag]=set(combine_pheno_genes[tag])
                compare_assoc_gene_venn(combine_pheno_genes, f'{output_dir}/compare_gene_venn.{sc_tag}.png', show_size=False)

def check_gene(gene_name,cells:[]):
    score_dir=f'{output_DIR}/hs_gtex_gse97930_fc/intermediate'
    expr_path=f'{score_dir}/expr.mean.REZ.webapp.genes.txt.gz'
    degree_path=f'{score_dir}/degree.k_adjusted.REZ.webapp.genes.txt.gz'
    tag_path={'expr':expr_path,'degree':degree_path}
    data={}
    tags=['expr','degree']
    for tag in tags:
        data[tag]=[]
        df=pd.read_table(tag_path[tag],index_col=0)
        if cells is None:
            cells=[c[:-2] for c in df.columns if c.endswith('.z')]
        for cell in cells:
            val=df.loc[gene_name,f'{cell}.mean']
            z=df.loc[gene_name,f'{cell}.z']
            data[tag].append(z)
    res_df=pd.DataFrame(data,index=cells)
    print(res_df)


def degree_expr_dif(dgn_dirs):
    for dgn_dir in dgn_dirs:
        expr_path=f'{dgn_dir}/intermediate/degree.k_adjusted.REZ.webapp.genes.txt.gz'
        degree_path=f'{dgn_dir}/intermediate/expr.mean.REZ.webapp.genes.txt.gz'
        expr_df=pd.read_table(expr_path,index_col=0)
        degree_df = pd.read_table(degree_path, index_col=0)
        expr_df=expr_df.loc[:,[x for x in expr_df.columns if x.endswith('.z')]]
        degree_df = degree_df.loc[:, [x for x in degree_df.columns if x.endswith('.z')]]
        expr_df.columns=expr_df.columns.map(lambda x:x[:-2])
        degree_df.columns = degree_df.columns.map(lambda x: x[:-2])
        common_cells=expr_df.columns.intersection(degree_df.columns)
        common_genes=expr_df.index.intersection(degree_df.index)
        expr_df=expr_df.loc[common_genes,common_cells]
        degree_df = degree_df.loc[common_genes, common_cells]
        expr_dif=expr_df-degree_df
        degree_dif=degree_df-expr_df
        expr_dif.index.name='Gene'
        degree_dif.index.name='Gene'
        expr_dif.to_csv(f'{dgn_dir}/intermediate/expr.z_dif.txt.gz',sep='\t')
        degree_dif.to_csv(f'{dgn_dir}/intermediate/degree.z_dif.txt.gz', sep='\t')

def __run_dese_dif(cmd):
    os.system(cmd)


def run_dif_dese():
    gwas_dir=GWAS_case_dir
    exclude_gwas=UNSELECT_PHENOS
    gwas=[f.split('.')[0] for f in os.listdir(gwas_dir)]
    gwas=[g for g in gwas if g not in exclude_gwas]
    gwas_para=[{'abbr':g,'gwas_path':f'{gwas_dir}/{g}.gwas.tsv.gz'} for g in gwas]
    sc_types = ['hs_gtex_gse97930_fc', 'tms_global']
    kggsee_dir=f'{LOCAL_DIR}/lib/kggsee'
    vcf_ref= f'{LOCAL_DIR}/lib/gty/EUR.hg19.vcf.gz'
    kggsee_jar=f'{kggsee_dir}/kggsee.jar'
    resource_dir=f'{kggsee_dir}/resources'
    multi_correct_method='benfdr'
    fwer=0.05
    top_n_gene=1000
    nt=8
    chr_col='chr'
    bp_col='bp'
    p_col='p'
    cmds=[]
    for i in range(len(sc_types)):
        sc_tag=sc_types[i]
        gene_score=' '.join([f'{output_DIR}/{sc_tag}/intermediate/{x}.z_dif.txt.gz' for x in ['expr','degree']])
        for gp in gwas_para:
            gwas_file=gp['gwas_path']
            out_prefix=f'{output_DIR}/{sc_tag}/dif_result_norez/{gp["abbr"]}'
            cmd=kggsee_dese(gwas_file, gene_score, out_prefix, kggsee_jar, resource_dir, multi_correct_method, fwer, top_n_gene, nt,
                        chr_col, bp_col, p_col,
                        ref_genome='hg19', remove_hla=True, java_path='java', jvm_gb='80',
                        vcf_ref=vcf_ref, keep_ref=None, saved_ref=None, run_rez=False)
            cmds.append([cmd,])
    batch_run_function(__run_dese_dif, cmds, int(sys.argv[1]))
    pass

def compare_z_dist(dgn_dirs,sc_tags):
    i=-1
    for dgn_dir in dgn_dirs:
        i+=1
        expr_path=f'{dgn_dir}/intermediate/degree.k_adjusted.REZ.webapp.genes.txt.gz'
        degree_path=f'{dgn_dir}/intermediate/expr.mean.REZ.webapp.genes.txt.gz'
        expr_df=pd.read_table(expr_path,index_col=0)
        degree_df = pd.read_table(degree_path, index_col=0)
        expr_df=expr_df.loc[:,[x for x in expr_df.columns if x.endswith('.z')]]
        degree_df = degree_df.loc[:, [x for x in degree_df.columns if x.endswith('.z')]]
        expr_df.columns=expr_df.columns.map(lambda x:x[:-2])
        degree_df.columns = degree_df.columns.map(lambda x: x[:-2])
        common_cells=expr_df.columns.intersection(degree_df.columns)
        common_genes=expr_df.index.intersection(degree_df.index)
        expr_df=expr_df.loc[common_genes,common_cells]
        degree_df = degree_df.loc[common_genes, common_cells]
        expr_val=np.sort(expr_df.values.flatten())
        degree_val=np.sort(degree_df.values.flatten())
        seaborn.kdeplot(expr_val,fill=True, label='Z (Expression)')
        seaborn.kdeplot(degree_val,fill=True, label='Z (Degree)')
        plt.legend()
        plt.xlabel(f'Z score')
        plt.title(f'{sc_tags[i]}')
        plt.show()
        plt.close()
        plt.scatter(x=expr_val, y=degree_val)
        min_x=min((np.min(degree_val), np.min(expr_val)))
        max_x = max((np.max(degree_val), np.max(expr_val)))
        plt.plot([min_x,max_x], [min_x,max_x], 'r--')
        plt.xlabel('Data1 Quantiles')
        plt.ylabel('Data2 Quantiles')
        plt.title('QQ Plot Between Two Data Sets')
        ks_stat, p_value = stats.ks_2samp(expr_val, degree_val)
        print(f"KS Statistic: {ks_stat:.4f}, P-value: {p_value:.4f}")
        plt.xlabel(f'Z (Expression)')
        plt.ylabel(f'Z (Degree)')
        plt.title(f'{sc_tags[i]}')
        plt.show()


def compare_dese():
    db_local_dir = f'{GWAS_ROOT}/gold_case/gene_based/pubmed'
    db_ecs_dir= f'{GWAS_ROOT}/gold_case/gene_based/ecs'
    db_ehe_dir = f'{GWAS_ROOT}/gold_case/gene_based/ehe'
    fig_dir = f'{output_DIR}/analysis/combine'
    sc_types=['hs_gtex_gse97930_fc','tms_global']
    sc_tags=['Human','Mouse']
    dgn_dirs=[f'{output_DIR}/{k}/result' for k in sc_types]
    cdd = Compare_DGN_DESE(dgn_dirs,sc_tags)
    cdd.compare_cell_p_corr_dist(fig_dir)
    cdd.compare_gene_p(fig_dir,db_local_dir,db_ecs_dir,db_ehe_dir)
    dgn_root_dirs = [f'{output_DIR}/{k}' for k in sc_types]
    degree_expr_dif(dgn_root_dirs)
    run_dif_dese()
    compare_z_dist(dgn_root_dirs,sc_tags)

if __name__ == '__main__':
    compare_dese()