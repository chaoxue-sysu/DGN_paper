# -*- coding: utf-8 -*-
# @Author  : Xue Chao
# @Time    : 2023/04/17 19:32
# @Function:
import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
import requests
import scipy
import seaborn
from matplotlib import pyplot as plt
import seaborn as sns
from venn import venn
import xml.dom.minidom as xmldom

from para import GWAS_ROOT, GWAS_case_dir, GWAS_meta_xlsx, output_DIR, UNSELECT_PHENOS, COLORS_gene_compare_auc
from util import make_dir, read_line, log, get_gene_alias,  replace_with_gene_symbol_of_df, pvalue_adjust, \
    run_command, LOCAL_DIR, batch_shell_plus

class NCBI:
    def __init__(self):
        self.sess=requests.session()
        pass
    def request_ncbi(self,url):
        res=self.sess.post(url)
        xobj=xmldom.parseString(res.text)
        count=xobj.documentElement.getElementsByTagName("Count")[0].firstChild.data
        pmids=[]
        for pid in xobj.documentElement.getElementsByTagName("Id"):
            pmids.append(str(pid.firstChild.data).strip())
        return count,pmids

    def single_trait_gene(self,traits,genes):
        geneTerm='+OR+'.join([f'({gene}[tiab])' for gene in genes])
        traitTerm='+OR+'.join([f'({t}[tiab])' for t in traits])
        base_url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?'
        query=f'db=pubmed&term=(({traitTerm})+AND+({geneTerm})+AND+(gene[tiab]+OR+genes[tiab]+OR+mRNA[tiab]+OR+protein[tiab]+OR+proteins[tiab]+OR+transcription[tiab]+OR+transcript[tiab]+OR+transcripts[tiab]+OR+expressed[tiab]+OR+expression[tiab]+OR+expressions[tiab]+OR+locus[tiab]+OR+loci[tiab]+OR+SNP[tiab]))&datetype=edat&retmax=100'
        url=f'{base_url}{query}'
        count=0
        pmids=[]
        while True:
            try:
                count,pmids=self.request_ncbi(url)
                break
            except:
                log(f'except! waiting for retrying ...')
                time.sleep(30)
                continue
        return str(count).strip(),pmids

    def batch_trait_gene(self,traits,genes,out_path):
        '''
        :param genes:
        :param out_path:
        :return:
        '''
        alias_genes=get_gene_alias(genes)
        if alias_genes is None:
            return None
        gene_alias_map={}
        for i in range(len(alias_genes)):
            gene_alias_map[genes[i]]=alias_genes[i]
        first = False
        egenes=[]
        if os.path.isfile(out_path):
            try:
                df=pd.read_table(out_path,header=0)
                egenes = list(df['gene'])
            except:
                egenes=[]
        else:
            first=True
        search_genes=[]
        for g in genes:
            if g not in egenes:
                search_genes.append(g)
        log(f'all: {len(genes)}, exist: {len(egenes)}, remain: {len(search_genes)}')
        with open(out_path,'a') as bw:
            if first:
                bw.write('\t'.join(['gene','count','pmids'])+'\n')
            i=0
            for g in search_genes:
                i+=1
                c,pids=self.single_trait_gene(traits,gene_alias_map[g])
                bw.write('\t'.join([g,c,','.join(pids)])+'\n')
                log(f'({i}/{len(search_genes)}) {g}: {c}')
                time.sleep(2)

class KGGSEE:
    def __init__(self, prefix):
        self.prefix = prefix
        self.kggsee_jar = f'{LOCAL_DIR}/lib/kggsee/kggsee.jar'
        self.resource_dir = f'{LOCAL_DIR}/lib/kggsee/resources'
        self.ref_eur_gty = f'{LOCAL_DIR}/lib/gty/EUR.hg19.vcf.gz'

    def ecs_run(self,gwas_file,remove_region='hla',run=False):
        make_dir(os.path.dirname(self.prefix))
        ## default
        para = f'''
            --resource {self.resource_dir}
            --sum-file {gwas_file}
            --out {self.prefix}      
            --multiple-testing benfdr
            --p-value-cutoff 0.05
            --top-gene 1000
            --buildver hg19
            --nt 8
            --chrom-col chr
            --pos-col bp
            --p-col p
            --filter-maf-le 0.05
            --gene-finemapping
            --db-gene refgene
            --no-gz
            --vcf-ref {self.ref_eur_gty}
            --no-plot
        '''
        if remove_region == 'hla':
            para += f' --regions-out chr6:27477797-34448354'
        para = re.sub('\s+', ' ', para)
        cmd = f'java -Xmx80G -jar {self.kggsee_jar} {para}'
        if run:
            run_command(cmd)
        return cmd

    def ehe_run(self,gwas_file,phenotype_model='qualitative',remove_region='hla',run=False):
        make_dir(os.path.dirname(self.prefix))
        ## default
        para = f'''
            --gene-herit
            --gene-assoc-condi
            --resource {self.resource_dir}
            --sum-file {gwas_file}
            --out {self.prefix}      
            --multiple-testing benfdr
            --p-value-cutoff 0.05
            --top-gene 1000
            --buildver hg19
            --nt 8
            --chrom-col chr
            --pos-col bp
            --p-col p
            --filter-maf-le 0.05
            --db-gene refgene
            --no-gz
            --vcf-ref {self.ref_eur_gty}
            --no-plot
        '''
        if phenotype_model=='qualitative':
            para+=f' --case-col n_case --control-col n_ctrl'
        if phenotype_model=='quantitative':
            para += f' --nmiss-col n_eff'
        if remove_region == 'hla':
            para += f' --regions-out chr6:27477797-34448354'
        para = re.sub('\s+', ' ', para)
        cmd = f'java -Xmx80G -jar {self.kggsee_jar} {para}'
        if run:
            run_command(cmd)
        return cmd

    def gene_based_p_cutoff(self):
        p_cut = None
        i = 0
        for line in read_line(f'{self.prefix}.log'):
            if 'Significance level of p value cutoffs for the overall error rate' in line:
                i += 1
                p_cut = float(line.strip().split(':')[-1].strip())
        return p_cut

    def ecs_cond_assoc_genes(self,gene_score_name=''):
        Pcut = self.gene_based_p_cutoff()
        gp = f'{self.prefix}{gene_score_name}.gene.assoc.condi.txt'
        genes = []
        if os.path.isfile(gp):
            df = pd.read_table(gp)
            genes = list(df.loc[df['CondiECSP'] < Pcut, 'Gene'].values)
        return genes

    def ecs_assoc_genes(self,gene_score_name=''):
        gp = f'{self.prefix}{gene_score_name}.gene.assoc.condi.txt'
        genes = []
        if os.path.isfile(gp):
            df = pd.read_table(gp)
            genes = list(df['Gene'])
        return genes

    def ecs_cond_assoc_genes_df(self,gene_score_name=''):
        Pcut = self.gene_based_p_cutoff()
        gp = f'{self.prefix}{gene_score_name}.gene.assoc.condi.txt'
        df=None
        if os.path.isfile(gp):
            rdf = pd.read_table(gp)
            df = rdf.loc[rdf['CondiECSP'] < Pcut, :]
        return df

    def ecs_assoc_genes_df(self,gene_score_name=''):
        gp = f'{self.prefix}{gene_score_name}.gene.assoc.condi.txt'
        df=None
        if os.path.isfile(gp):
            df = pd.read_table(gp)
        return df


def getTPRandFPR(trueGenes,falseGenes,predTrueGenes,predFalseGenes):
    TP=len(set(predTrueGenes).intersection(set(trueGenes)))
    TN=len(set(predFalseGenes).intersection(set(falseGenes)))
    FP=len(set(predTrueGenes).intersection(set(falseGenes)))
    FN=len(set(predFalseGenes).intersection(set(trueGenes)))
    TPR,FPR=0,0
    if TP+FN!=0:
        TPR=TP/(TP+FN)
    if FP+TN!=0:
        FPR=FP/(FP+TN)
    return TPR,FPR,TP,len(predTrueGenes)

class PhenoGene:
    def __init__(self,db_name):
        self.db_name=db_name

    def access_db(self,phenotypes:[],out_tsv,genes):
        '''
        从相应数据库获取数据并处理成标准格式
        :return:
        '''
        support_dbs=['pubmed']
        if self.db_name not in support_dbs:
            raise Exception(f'Database: {self.db_name} is not supported! Supported list: {", ".join(support_dbs)}')
        make_dir(os.path.dirname(out_tsv))
        if self.db_name=='pubmed':
            NCBI().batch_trait_gene(phenotypes, genes, out_tsv)

    def eval_ROC_gene_based(self,sort_genes_dfs, ncbi_df, tags, libCutoff=5, title='', plot_step=1,print_info=False,plot_fig=True):
        colors=COLORS_gene_compare_auc
        tag_auc = {}
        if ncbi_df is None:
            return False, tag_auc
        db = ncbi_df
        if db.shape[0] == 0:
            return False, tag_auc
        trueGeneNum = db.loc[db['count'] >= libCutoff,].index.size
        rocVals = {}
        ppVal = {}
        k=-1
        for df in sort_genes_dfs:
            k+=1
            specDB = db.loc[[i for i in db.index if db.loc[i, 'gene'] in list(df['unique_gene'].values)],]
            trueGenes = list(specDB.loc[specDB['count'] >= libCutoff, 'gene'].values)
            falseGenes = [g for g in specDB['gene'] if g not in trueGenes]
            tag = tags[k]
            rocVals[tag] = []
            for i in np.arange(0,df.shape[0],plot_step):
                predTrueGenes = list(df.loc[df.index[:i], 'unique_gene'].values)
                predFalseGenes = list(df.loc[df.index[i:], 'unique_gene'].values)
                TPR, FPR, TP, PP = getTPRandFPR(trueGenes, falseGenes, predTrueGenes, predFalseGenes)
                rocVals[tag].append([TPR, FPR])
                # rocVals[tag].append([TP, PP])
            PP = df.index.size
            TP = len(set(df['unique_gene'].values).intersection(trueGenes))
            ppVal[tag] = [PP, TP]
        if print_info:
            print(f'all：{db.index.size}; positive：{trueGeneNum}；ratio：{trueGeneNum / db.index.size:.3f}')
        kc=-1
        for k in tags:
            kc+=1
            xs = [z[1] for z in rocVals[k]]
            ys = [z[0] for z in rocVals[k]]
            auc = np.nansum(ys) / len(ys)
            if print_info:
                print(f'{k}: predict：{ppVal[k][0]}; TP：{ppVal[k][1]}；TPR：{ppVal[k][1] / ppVal[k][0]:.3f}')
            tag_auc[k] = auc
        # axe.plot([0, 1], [0, 1], '-', c='black')
        fig=None
        if plot_fig:
            fig, axe = plt.subplots(figsize=(3.5, 3.5))
            for kc in range(len(tags)):
                k=tags[kc]
                auc=tag_auc[k]
                axe.plot(xs, ys, label=f'{k}, AUC={auc:.3f}',c=colors[kc])
            axe.set_xlabel('FPR')
            axe.set_ylabel('TPR')
            axe.set_title(title)
            axe.spines['top'].set_visible(False)
            axe.spines['right'].set_visible(False)
            plt.legend(loc='lower right')
            # plt.show()
            plt.tight_layout()
        return fig, tag_auc


def compare_assoc_gene_AUC_plot(df, pheno_name):
    pale_cols=COLORS_gene_compare_auc
    data = {'pheno': [], 'auc': [], 'type': []}
    for c in df.columns:
        for i in df.index:
            data['pheno'].append(i)
            data['auc'].append(df.loc[i, c])
            data['type'].append(c)
    pldf = pd.DataFrame(data)
    if pldf.shape[0] == 0:
        return None
    units = df.shape[0]
    h_ratio = 6 / 4
    if units < 8:
        units = 8
        h_ratio = 4 / 2
    w = 6 / 12 * units
    h = w * h_ratio
    fig, ax = plt.subplots(figsize=(15, 3))
    sns.barplot(pldf, x='pheno', y='auc', hue='type', ax=ax,
                palette=pale_cols)  # palette=["#D77988", "#34ABBC"],
    ax.set_xticklabels(ax.get_xticklabels())  # rotation=70,ha='right'
    ax.set_ylim(0.5, np.max(df.values)+0.05)
    ax.set_xlabel('')
    ax.set_ylabel('AUC')
    ax.set_title(pheno_name)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    # plt.show()
    return fig

def compare_assoc_gene_AUC_group_plot(df,plot_box=False,two_side=False,figsize=None):
    test_h0='greater'
    if two_side:
        test_h0='two-sided'
    pales = COLORS_gene_compare_auc
    box_pale = pales[:df.shape[1]]
    data=[df[type].values for type in df.columns]
    types=df.columns.values
    if figsize is None:
        figsize=(1.3*len(data),3.6)
    fig, ax = plt.subplots(figsize=figsize)
    if plot_box:
        seaborn.boxplot(data=data, ax=ax, palette=box_pale)
    else:
        seaborn.violinplot(data=data, ax=ax, palette=box_pale)
    ax.set_xticklabels(types)
    ## p value
    max_y = max([max(d) for d in data])
    min_y = min([min(d) for d in data])
    last_y=max_y*1.15
    y_high=max_y*1.35
    x_positions = ax.get_xticks()
    xi=x_positions[0]
    for i in range(1,len(data)):
        xip1=x_positions[i]
        stat, pv = scipy.stats.ttest_rel(data[i], data[0], alternative=test_h0)
        ax.plot((xi, xip1), (last_y, last_y), '-', c='black')
        ax.plot((xi, xi), (last_y, last_y*0.99), '-', c='black')
        ax.plot((xip1, xip1), (last_y, last_y * 0.99), '-', c='black')
        ax.text((xi + xip1) / 2, last_y*1.01, f'$P$ = {pv:.2g}', ha='center', va='bottom', fontsize=10, color='black')
        last_y+=max_y*0.09
    ax.set_ylim((min_y*0.7,y_high))
    ax.set_ylabel('AUC')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


def compare_assoc_gene_venn(genes, fig_path, show_size=True):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if show_size:
        fmt="{size}\n({percentage:.1f}%)"
    else:
        fmt="{percentage:.1f}%"
    venn(genes, ax=ax,fmt=fmt,cmap=COLORS_gene_compare_auc,legend_loc="lower right")
    plt.tight_layout()
    fig.savefig(fig_path)
    return True


def run_ecs(out_dir,ntasks:int):
    ntasks=int(ntasks)
    gwas_dir=GWAS_case_dir
    exclude_gwas=['MDD2']
    gwas=[f.split('.')[0] for f in os.listdir(gwas_dir)]
    gwas=[g for g in gwas if g not in exclude_gwas]
    gwas_para=[{'abbr':g,'gwas_path':f'{gwas_dir}/{g}.gwas.tsv.gz'} for g in gwas]
    ## test BIP
    # gwas=['BIP']
    # gwas_para=[{'abbr':f'{g}-test-ehe','gwas_path':f'{gwas_dir}/{g}.gwas.tsv.gz'} for g in gwas]
    ## end test
    cmds=[]
    for gwas in gwas_para:
        ks=KGGSEE(f'{out_dir}/{gwas["abbr"]}')
        cmd=ks.ecs_run(gwas['gwas_path'])
        cmds.append(cmd)
    batch_shell_plus(cmds,ntasks)

def run_ehe(out_dir,ntasks:int):
    ntasks=int(ntasks)
    gwas_dir=GWAS_case_dir
    exclude_gwas=UNSELECT_PHENOS
    gwas=[f.split('.')[0] for f in os.listdir(gwas_dir)]
    gwas=[g for g in gwas if g not in exclude_gwas]
    ## test BIP
    gwas=['BIP']
    ## end test
    gwas_para=[{'abbr':g,'gwas_path':f'{gwas_dir}/{g}.gwas.tsv.gz'} for g in gwas]
    in_gwas_dir=f'{GWAS_ROOT}/gold_case/raw'
    meta_path=f'{in_gwas_dir}/meta.xlsx'
    mdf=pd.read_excel(meta_path)
    pheno_type={}
    for x in mdf.index:
        pheno_type[mdf.loc[x,'abbr']]=mdf.loc[x,'model']
    cmds=[]
    for gwas_info in gwas_para:
        ks=KGGSEE(f'{out_dir}/{gwas_info["abbr"]}')
        phenotype_model=pheno_type[gwas_info["abbr"]]
        cmd=ks.ehe_run(gwas_info['gwas_path'],phenotype_model)
        cmds.append(cmd)
    batch_shell_plus(cmds,ntasks)

def search_ncbi(ecs_dir,gwas_conf,db_local_dir):
    top_gene_num = 500
    fdr_q_cutoff=0.05
    make_dir(db_local_dir)
    gdf = pd.read_excel(gwas_conf,dtype=str)
    gwas_alias = {}
    for i in gdf.index:
        gwas_alias[gdf.loc[i, 'abbr']] = f"{gdf.loc[i, 'ncbi_search_alias']}".split(';')
    phenos = sorted(set([x.split('.')[0] for x in os.listdir(ecs_dir) if os.path.isfile(f'{ecs_dir}/{x}')]))
    for pheno in phenos:
        out_path=f'{db_local_dir}/{pheno}.tsv'
        genes = KGGSEE(f'{ecs_dir}/{pheno}').ecs_assoc_genes()
        genes=sorted(genes)
        log(f'start {pheno}: {gwas_alias[pheno]} with {len(genes)} genes')
        if len(genes)==0:
            continue
        pg = PhenoGene('pubmed')
        pg.access_db(gwas_alias[pheno],out_path,genes)

def eval_pheno_gene_ROC(dgn_dir,ecs_dir,db_local_dir):
    top_gene_num = 500
    fdr_q_cutoff=0.05
    min_genes=50
    db_name='pubmed'
    phenos = sorted(set([x.split('.')[0] for x in os.listdir(ecs_dir) if os.path.isfile(f'{ecs_dir}/{x}')]))
    phenos = [p for p in phenos if p not in UNSELECT_PHENOS]
    phe_auc={}

    fig_dir = dgn_dir
    tags=['ECS','DGN'] # DGN
    kggsee_gene_scores=['degree.k_adjusted.txt.gz'] ## must match with tags. 'expr.mean.txt.gz',
    combine_data=[]
    combine_pheno_genes={}
    for pheno in phenos:
        log(f'start {pheno}')
        sig_genes = {}
        sort_genes_dfs = []
        genes_dfs=[]
        kgs=KGGSEE(f'{ecs_dir}/{pheno}')
        df=kgs.ecs_assoc_genes_df()
        sig_p_cutoff=kgs.gene_based_p_cutoff()
        genes_dfs.append([df,'CondiECSP','Gene'])
        if df is None or df.shape[0]<min_genes:
            log(f'ignore {pheno} due to genes < {min_genes}')
            continue
        for gene_score in kggsee_gene_scores:
            df = KGGSEE(f'{dgn_dir}/{pheno}').ecs_assoc_genes_df(gene_score)
            genes_dfs.append([df,'CondiECSP','Gene'])
        for i in range(len(genes_dfs)):
            df, p_col, gene_col=genes_dfs[i]
            df['unique_p']=df[p_col]
            df['unique_gene']=df[gene_col].map(lambda x:f'{pheno}:{x}')
            df.sort_values(by='unique_p', inplace=True, ignore_index=True)
            sort_genes_dfs.append(df)
            sig_genes[tags[i]]=set(df.loc[df['unique_p']<sig_p_cutoff,'unique_gene'].values.tolist())

        ncbi_df=pd.read_table(f'{db_local_dir}/{pheno}.tsv')
        ncbi_df['gene']=ncbi_df['gene'].map(lambda x: f'{pheno}:{x}')

        combine_data.append([sort_genes_dfs,ncbi_df])
        fig, tag_auc = PhenoGene(db_name).eval_ROC_gene_based(sort_genes_dfs, ncbi_df,
                                          tags, 5,title=pheno)
        if fig:
            fig.savefig(f'{fig_dir}/{pheno}.{db_name}.fine.ROC.png')
            phe_auc[pheno] = [tag_auc[k] for k in tags]
        for tag in sig_genes.keys():
            if tag not in combine_pheno_genes:
                combine_pheno_genes[tag]=[]
            combine_pheno_genes[tag]+=sorted([f'{pheno}:{sg}' for sg in sig_genes[tag]])
    combine_pheno_genes={k:[';'.join(combine_pheno_genes[k])] for k in combine_pheno_genes.keys()}
    pd.DataFrame(combine_pheno_genes).to_csv(f'{fig_dir}/compare.combine.pheno-gene.tsv',sep='\t',index=False)

    df = pd.DataFrame(phe_auc, index=tags).T
    fig = compare_assoc_gene_AUC_plot(df, '')
    fig.savefig(f'{fig_dir}/compare.AUC.{db_name}.fine.png')
    fig = compare_assoc_gene_AUC_group_plot(df)
    fig.savefig(f'{fig_dir}/compare.AUC.group.{db_name}.fine.box.png')
    df.to_csv(f'{fig_dir}/compare.AUC.{db_name}.fine.tsv',sep='\t')
    # combine ROC
    sdfss=[[] for i in range(len(tags))]
    ndfs=[]
    for i in range(len(combine_data)):
        sort_dfs,ncbi_df=combine_data[i]
        for k in range(len(sort_dfs)):
            sdfss[k].append(sort_dfs[k])
        ndfs.append(ncbi_df)
    com_sdfs=[pd.concat(dfs,ignore_index=True).sort_values(by=['unique_p']) for dfs in sdfss]
    com_ndf=pd.concat(ndfs,ignore_index=True)
    fig, tag_auc = PhenoGene(db_name).eval_ROC_gene_based(com_sdfs, com_ndf,
                                                          tags, 5, title="",plot_step=5)
    if fig:
        fig.savefig(f'{fig_dir}/combine.{db_name}.fine.ROC.png')
    with open(f'{fig_dir}/combine.{db_name}.fine.ROC.data.pyd','wb') as bw:
        pickle.dump((com_sdfs,com_ndf),bw)


def eval_gene_combine(dgn_dirs,dgn_labels,fig_dir):
    make_dir(fig_dir)
    dfs=[]
    pg_dfs=[]
    com_type_dgn_dfs=[]
    for i in range(len(dgn_dirs)):
        dp=f'{dgn_dirs[i]}/compare.AUC.pubmed.fine.tsv'
        df=pd.read_table(dp,index_col=0)
        df.columns=df.columns.map(lambda x:x if x=='ECS' else dgn_labels[i])
        pg_df = pd.read_table(f'{dgn_dirs[i]}/compare.combine.pheno-gene.tsv')
        pg_df.columns=pg_df.columns.map(lambda x:x if x=='ECS' else dgn_labels[i])
        with open(f'{dgn_dirs[i]}/combine.pubmed.fine.ROC.data.pyd','rb') as br:
            com_dgn_dfs,com_ncbi_df=pickle.load(br)
        if i!=0:
            df=df.loc[:,df.columns[1:]]
            pg_df=pg_df.loc[:,pg_df.columns[1:]]
            com_type_dgn_dfs+=com_dgn_dfs[1:]
        else:
            com_type_dgn_dfs+=com_dgn_dfs
        dfs.append(df)
        pg_dfs.append(pg_df)

    fdf=pd.concat(dfs,axis=1)
    fig = compare_assoc_gene_AUC_group_plot(fdf)
    fig.savefig(f'{fig_dir}/compare.gene.AUC.group.fine.png')
    fig = compare_assoc_gene_AUC_plot(fdf, '')
    fig.savefig(f'{fig_dir}/compare.gene.AUC.fine.png')

    fpgdf=pd.concat(pg_dfs,axis=1)
    sig_g={c:set(str(fpgdf.loc[0,c]).split(';')) for c in fpgdf.columns}
    compare_assoc_gene_venn(sig_g, f'{fig_dir}/combine_pheno_gene.fine.Venn.png',show_size=False)

    fig, tag_auc = PhenoGene('pubmed').eval_ROC_gene_based(com_type_dgn_dfs, com_ncbi_df,
                                                          ['ECS']+dgn_labels, 5, title="",plot_step=5)
    if fig:
        fig.savefig(f'{fig_dir}/combine.pheno_gene.fine.ROC.png')

if __name__ == '__main__':
    gwas_conf = GWAS_meta_xlsx
    ecs_dir=f'{GWAS_ROOT}/gold_case/gene_based/ecs'
    ehe_dir=f'{GWAS_ROOT}/gold_case/gene_based/ehe'
    db_local_dir = f'{GWAS_ROOT}/gold_case/gene_based/pubmed'
    fig_dir = f'{output_DIR}/analysis/combine'
    args=sys.argv[1:]
    # 1. run ecs (same GWAS para with dgn), magma
    run_ecs(ecs_dir,int(args[0]))
    # 2. search in ncbi
    search_ncbi(ecs_dir,gwas_conf,db_local_dir)
    # 3. eval and compare
    for tdb in ['hs_gtex_gse97930_fc', 'tms_global']: #'hs_gtex_gse97930_fc',
        dgn_dir = f'{output_DIR}/{tdb}/result'
        eval_pheno_gene_ROC(dgn_dir,ecs_dir,db_local_dir)
    # 4. combine evaluation
    dgn_dirs=[f'{output_DIR}/{k}/result' for k in ['hs_gtex_gse97930_fc', 'tms_global']]
    eval_gene_combine(dgn_dirs,['DGN.hs','DGN.mm'],fig_dir)
    # 5. run EHE
    run_ehe(ehe_dir, int(args[0]))



