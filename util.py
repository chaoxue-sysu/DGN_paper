# -*- coding: utf-8 -*-
# @Author  : Xue Chao
# @Time    : 2023/07/03 17:22
# @Function:
import concurrent
import concurrent.futures
import gzip
import inspect
import multiprocessing
import re
import subprocess
import sys
import tempfile
import time
import os

import filetype
import gseapy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn
from gprofiler import GProfiler
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import spearmanr,pearsonr
import jpype
import pandas as pd
import matplotlib.colors as mcolors

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR=f'{LOCAL_DIR}/resources'
FONTS_DIR=f'{RESOURCE_DIR}/fonts'

def log(content):
    """
    Log function
    :param content:
    """
    content = time.strftime("%Y-%m-%d %H:%M:%S [INFO] ", time.localtime(time.time())) + "%s" % content
    print(content, flush=True)


def log_exception(e):
    content = time.strftime("%Y-%m-%d %H:%M:%S [Exception] ", time.localtime(time.time())) + f"{type(e)}: {e}"
    print(content, flush=True)


def make_dir(*dirs):
    """
    Create the directory if it does not exist.
    :param dirs:
    """
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def kggseq_ref_genes():
    ref_gene_path = f'{LOCAL_DIR}/resources/kggseqv1.1_hg19_refGene..txt'
    df=pd.read_table(ref_gene_path,header=None)
    genes=set(df[0].unique())
    gene_s=get_gene_symbol(sorted(genes))
    gene_syms=set()
    for g in gene_s:
        g=g.strip()
        if g!='' and g!='NA':
            gene_syms.add(g)
    return gene_syms

def batch_run_function(func,args,nt):
    if len(args)<nt:
        nt=len(args)
    if nt<=1:
        for arg in args:
            func(*arg)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=nt) as executor:
            futures = [executor.submit(func, *item) for item in args]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    log_exception(e)
    # with multiprocessing.Pool(processes=nt) as pool:
    #     futures = [pool.apply_async(func, args=item) for item in args]
    #     pool.close()
    #     pool.join()
    #     for future in futures:
    #         try:
    #             future.get()
    #         except Exception as e:
    #             log_exception(e)

def remove_files(*args):
    for a in args:
        if os.path.isfile(a):
            os.remove(a)

def batch_run_function_linux(func,args,nt):
    if len(args)<nt:
        nt=len(args)
    if nt<=1:
        for arg in args:
            func(*arg)
    else:
        func_name=func.__name__
        module = inspect.getmodule(func)
        module_file = inspect.getfile(module)
        module_name = os.path.basename(module_file)[:-3]
        unique_name=f'{module_name}_{func_name}.{int(time.time())}'
        tmp_py=f'{LOCAL_DIR}/bat.{unique_name}.tmp'
        with open(tmp_py,'w') as bw:
            bw.write(f"import sys \
                \nfrom {module_name} import {func_name} \
                \nif __name__=='__main__':  \
                \n {func_name}(*sys.argv[1:]) ")
        cmds=[]
        for arg in args:
            cmd=f'python {tmp_py} '+' '.join([f'{a}' for a in arg])
            cmds.append(cmd)
        batch_shell_plus(cmds,nt)
        remove_files(tmp_py)

def clear_str_arr(genes:[]):
    ngenes = []
    for g in genes:
        g = str(g).strip()
        if g != '':
            ngenes.append(g)
    return ngenes

class GeneID():
    def __init__(self):
        self.type_id_symbol = None
        hgnc_path = f'{LOCAL_DIR}/resources/HGNC_20210810.txt'
        df = pd.read_table(hgnc_path, header=0, index_col=0, dtype='str')
        ## 过滤舍弃的Gene Symbol记录
        self.df = df.loc[df['Status'] == 'Approved',]
        self.__get_gene_symbol_map()
        self.__get_gene_symbol_alias()

    def __gene_id_type(self,gene_id:str):
        gene_type = 'symbol'
        if len(gene_id) > 14 and gene_id.startswith('ENSG'):
            gene_type = 'ensg_id'
        if re.match('^[\d]+$', gene_id):
            gene_type = 'ncbi_id'
        return gene_type

    def __get_gene_symbol_map(self):
        df=self.df
        gt_cols={
            'symbol':['Approved symbol', 'Previous symbols', 'Alias symbols'],
            'ensg_id':['Ensembl gene ID'],
            'ncbi_id':['NCBI Gene ID']
        }
        type_id_symbol = {gt:{} for gt in gt_cols.keys()}
        for i in df.index:
            if pd.isna(df.loc[i, 'Approved symbol']):
                continue
            v = str(df.loc[i, 'Approved symbol']).strip()
            for gt in gt_cols.keys():
                gene_type = gt_cols[gt]
                for col in gene_type:
                    if df.loc[i, col] and df.loc[i, 'Approved symbol']:
                        for x in str(df.loc[i, col]).split(','):
                            k = x.strip()
                            if k=='' or v=='':
                                continue
                            type_id_symbol[gt][k] = v
        self.type_id_symbol=type_id_symbol

    def __get_gene_symbol_alias(self):
        df=self.df
        symbol_arr={}
        gene_col=['Approved symbol', 'Previous symbols', 'Alias symbols']
        for i in df.index:
            if pd.isna(df.loc[i, 'Approved symbol']):
                continue
            v = str(df.loc[i, 'Approved symbol']).strip()
            if v=='':
                continue
            symbol_arr[v]=set()
            for col in gene_col:
                if pd.isna(df.loc[i, col]):
                    continue
                for x in str(df.loc[i, col]).split(','):
                    k = x.strip()
                    if k=='':
                        continue
                    symbol_arr[v].add(k)
        self.symbol_alias=symbol_arr

    def get_symbol_list(self):
        return sorted(self.symbol_alias.keys())

    def get_gene_symbol(self,genes,na_raw=False):
        genes=clear_str_arr(genes)
        if len(genes)<1:
            return None
        gene_symbols = []
        gt=self.__gene_id_type(genes[0])
        id_symbol=self.type_id_symbol[gt]
        for g in genes:
            if gt=='ensg_id':
                g = g.strip().split('.')[0]
            gs = 'NA'
            if na_raw:
                gs = g
            if g in id_symbol:
                gs = id_symbol[g]
            gene_symbols.append(gs)
        return gene_symbols

    def get_gene_alias(self,genes,include_self=False):
        genes_sym=self.get_gene_symbol(genes,True)
        if not genes_sym:
            return None
        gene_alias=[]
        for g in genes_sym:
            alias=set()
            if include_self:
                alias.add(g)
            if g in self.symbol_alias:
                alias.union(self.symbol_alias[g])
            gene_alias.append(alias)
        return gene_alias

def get_gene_symbol(genes:[]):
    gid=GeneID()
    return gid.get_gene_symbol(genes)

def get_gene_alias(genes:[]):
    gid=GeneID()
    return gid.get_gene_alias(genes,include_self=True)

def need_trans_gene_name(gene_id:str):
    gene_id=str(gene_id).strip()
    need_trans=False
    if len(gene_id) > 14 and gene_id.startswith('ENSG'):
        need_trans=True
    if re.match('^[\d]+$', gene_id):
        need_trans=True
    return need_trans

def cluster_df(cor,method='average', metric= 'euclidean'):
    Z = hierarchy.linkage(cor, method=method, metric=metric)
    x=dendrogram(Z,no_plot=True)
    x_order=x['leaves']
    Z = hierarchy.linkage(cor.T, method=method, metric=metric)
    x=dendrogram(Z,no_plot=True)
    y_order=x['leaves']
    cor=cor.iloc[x_order,y_order]
    return cor


def corr_p(df:pd.DataFrame,method='pearson',sort_by_cluster=True):
    """
    calculate correlation coefficients and p-values.
    :param df:
    :param method:
    :return:
    """
    cols=df.columns.tolist()
    col_num=len(cols)
    cor=np.zeros(np.repeat(col_num,2))
    p=np.zeros(np.repeat(col_num,2))
    for i in range(col_num):
        for j in range(i,col_num):
            if method=='pearson':
                cc,pv=pearsonr(df.iloc[:,i],df.iloc[:,j])
            if method=='spearman':
                cc,pv=spearmanr(df.iloc[:,i],df.iloc[:,j])
            cor[i,j]=cc
            cor[j,i]=cc
            p[i,j]=pv
            p[j,i]=pv
    cor_df=pd.DataFrame(cor,index=cols,columns=cols)
    p_df=pd.DataFrame(p,index=cols,columns=cols)
    if sort_by_cluster:
        cor_df=cluster_df(cor_df)
        p_df=p_df.loc[cor_df.index,cor_df.columns]
    return cor_df,p_df

def read_line(file_path):
    isGzip = True
    try:
        if str(filetype.guess(file_path).extension) == "gz":
            isGzip = True
    except:
        isGzip = False
    if isGzip:
        reader = gzip.open(file_path, "r")
    else:
        reader = open(file_path, "r")
    while True:
        line = reader.readline()
        if not line:
            reader.close()
            break
        if isGzip:
            lineArr = line.decode().strip('\n')
        else:
            lineArr = line.strip('\n')
        yield lineArr

def batch_shell(all_task,limit_task,log_file,time_sleep=0.1):
    make_dir(os.path.dirname(log_file))
    log(f'stdin/stdou/stderr info in: {log_file}')
    log_bw = open(log_file, "w")
    task_pool=[]
    task_remain=len(all_task)
    for task in all_task:
        task_remain+=-1
        break_out = True
        p = subprocess.Popen(task, shell=True, stdin=log_bw, stdout=log_bw, stderr=log_bw)
        task_pool.append(p)
        log(f' ({len(all_task)-task_remain}/{len(all_task)}) '+str(p.pid)+': '+task+' start ...')
        if len(task_pool)==limit_task or task_remain==0:
            while break_out:
                for intask_Popen in task_pool:
                    if intask_Popen.poll()!=None:
                        log(f'{str(intask_Popen.pid)} finish')
                        task_pool.remove(intask_Popen)
                        break_out = False
                        if task_remain==0:
                            break_out=True
                        if len(task_pool)==0:
                            break_out=False
                        break
                time.sleep(time_sleep)
    log_bw.close()

def print_std(l,close=True):
    l.seek(0)
    output = l.read().decode().strip()
    if output!='':
        print(output, flush=True)
    if close:
        l.close()

class FlushLog:
    def __init__(self,io):
        self.last_seek=0
        self.io=io
    def print_std(self):
        self.io.seek(self.last_seek)
        output = self.io.read().decode().strip('\n')
        print(output, flush=True)
        self.io.last_seek=self.io.tell()
        print(self.io.last_seek)

def batch_shell_plus(all_task,limit_task,time_sleep=0.1):
    task_pool=[]
    task_remain=len(all_task)
    for task in all_task:
        task_remain+=-1
        break_out = True
        fileno = tempfile.NamedTemporaryFile(delete=True)
        p = subprocess.Popen(task, shell=True, stdout=fileno, stderr=fileno)
        log(f'({len(all_task)-task_remain}/{len(all_task)}) {str(p.pid)}: {task}')
        task_pool.append([p,fileno])
        if len(task_pool)>=limit_task or task_remain==0:
            while break_out:
                for p,fileno in task_pool:
                    if p.poll()!=None:
                        print_std(fileno)
                        log(f'complete {str(p.pid)}')
                        task_pool.remove([p,fileno])
                        break_out = False
                        if task_remain==0:
                            break_out=True
                        if len(task_pool)==0:
                            break_out=False
                        break
                time.sleep(time_sleep)

def replace_with_gene_symbol_of_df(expr_df):
    expr_df.index=expr_df.index.map(lambda x:str(x).strip())
    gene_syms = get_gene_symbol(expr_df.index)
    gene_idxs = []
    genes = set()
    for i in range(len(gene_syms)):
        gene_syms[i] = gene_syms[i].strip()
        if gene_syms[i] == '' or gene_syms[i] == 'NA' or gene_syms[i] in genes:
            continue
        gene_idxs.append(i)
        genes.add(gene_syms[i])
    expr_df = expr_df.iloc[gene_idxs, :]
    # expr_df = expr_df[sorted(expr_df.columns)]
    expr_df.index=np.array(gene_syms)[gene_idxs]
    return expr_df

def replace_with_gene_symbol_in_refgene_of_df(expr_df):
    expr_df.index=expr_df.index.map(lambda x:str(x).strip())
    gene_syms = get_gene_symbol(expr_df.index)
    kggseq_gene = kggseq_ref_genes()
    gene_idxs = []
    genes = set()
    for i in range(len(gene_syms)):
        gene_syms[i] = gene_syms[i].strip()
        if gene_syms[i] == '' or gene_syms[i] == 'NA' or gene_syms[i] in genes or gene_syms[i] not in kggseq_gene:
            continue
        gene_idxs.append(i)
        genes.add(gene_syms[i])
    expr_df = expr_df.iloc[gene_idxs, :]
    expr_df = expr_df[sorted(expr_df.columns)]
    expr_df.index=np.array(gene_syms)[gene_idxs]
    return expr_df

def index_in_gene_symbol_and_refgene_of_df(expr_df):
    expr_df.index=expr_df.index.map(lambda x:str(x).strip())
    gene_syms = get_gene_symbol(expr_df.index)
    kggseq_gene = kggseq_ref_genes()
    gene_idxs = []
    genes = set()
    for i in range(len(gene_syms)):
        gene_syms[i] = gene_syms[i].strip()
        if gene_syms[i] == '' or gene_syms[i] == 'NA' or gene_syms[i] in genes or gene_syms[i] not in kggseq_gene:
            continue
        gene_idxs.append(i)
        genes.add(gene_syms[i])
    return gene_idxs

def index_in_gene_symbol_and_refgene_of_gene_list(genes:[]):
    gene_syms = get_gene_symbol(genes)
    kggseq_gene = kggseq_ref_genes()
    gene_idxs = []
    genes_arr = set()
    for i in range(len(gene_syms)):
        gene_syms[i] = gene_syms[i].strip()
        if gene_syms[i] == '' or gene_syms[i] == 'NA' or gene_syms[i] in genes_arr or gene_syms[i] not in kggseq_gene:
            continue
        gene_idxs.append(i)
        genes_arr.add(gene_syms[i])
    return gene_idxs,gene_syms

def index_in_gene_symbol_of_gene_list(genes:[]):
    gene_syms = get_gene_symbol(genes)
    gene_idxs = []
    genes_arr = set()
    for i in range(len(gene_syms)):
        gene_syms[i] = gene_syms[i].strip()
        if gene_syms[i] == '' or gene_syms[i] == 'NA' or gene_syms[i] in genes_arr:
            continue
        gene_idxs.append(i)
        genes_arr.add(gene_syms[i])
    return gene_idxs,gene_syms

def pvalue_adjust(pvalue:[],method='FDR'):
    # print(pvalue)
    if method not in ['FDR','Bonf']:
        log('Not support other Method, try to adjust p value by FDR or Bonf!')
    if method=='FDR':
        leng=len(pvalue)
        if leng<3:
            return pvalue
        pvalue_idx=[(i,pvalue[i]) for i in range(leng)]
        sortpvalue=sorted(pvalue_idx,key=lambda x:x[1])
        bh_fdr=[]
        if sortpvalue[-1][1]>1:
            bh_fdr.append((sortpvalue[-1][0],1))
        else:
            bh_fdr.append(sortpvalue[-1])
        for i in range(2,leng+1):
            rank=leng - i+1
            pval_idx= sortpvalue[leng - i]
            fdr=pval_idx[1]* leng /rank
            fdr_front=bh_fdr[-1][1]
            if fdr>fdr_front:
                bh_fdr.append((pval_idx[0],fdr_front))
            else:
                bh_fdr.append((pval_idx[0],fdr))
        return [x[1] for x in sorted(bh_fdr,key=lambda x:x[0])]
    if method=='Bonf':
        return np.array(pvalue)*len(pvalue)

def cpm_df(counts_df,row_is_gene=True):
    """
    Return CPM (counts per million) modified xc
    """
    if not row_is_gene:
        counts_df=counts_df.T
    lib_size = counts_df.sum()
    fdf=counts_df / lib_size * 1e6
    if not row_is_gene:
        fdf=fdf.T
    return fdf

def cpm(counts_mat,row_is_gene=True):
    """
    Return CPM (counts per million) modified xc
    """
    cp_mat = counts_mat
    if not row_is_gene:
        cp_mat=np.copy(counts_mat).T
    lib_size=np.nansum(cp_mat,axis=0)
    fmat=cp_mat/lib_size*1e6
    if not row_is_gene:
        fmat=fmat.T
    return fmat

def dgn_heatmap(df:pd.DataFrame):
    '''
    heatmap by matplotlib
    :return:
    '''

    pass


def test_visual_corr(x1:np.ndarray,x2:np.ndarray,corr_method='pearson',tag=''):
    fig,ax=plt.subplots()
    ax.plot(x1,x2,'.')
    if corr_method=='pearson':
        cc,p=scipy.stats.pearsonr(x1,x2)
    if corr_method=='spearman':
        cc,p=scipy.stats.spearmanr(x1,x2)
    ax.set_title(f'{tag}{corr_method} cc={cc:.3f}; p={p:.3g}')
    plt.tight_layout()
    plt.show()
    return cc,p

def test_visual_diff(x1:np.ndarray,x2:np.ndarray,tag=''):
    fig,ax=plt.subplots(figsize=(3,4))
    stat,p = scipy.stats.ranksums(x1, x2, alternative='greater')
    seaborn.set(style="white")
    seaborn.boxplot(data=[x1,x2], palette=['#62B298','#EF8A66'],ax=ax)
    ax.set_xticklabels(['High','Low'], fontsize=12)
    # ax.boxplot([x1,x2], vert=True, patch_artist=True)
    ax.set_title(f'{tag}\n\n$P$ = {p:.2g}')
    plt.tight_layout()
    plt.show()
    return ax


def get_index(arr_des,arr_src):
    arr_src_idx={arr_src[i]:i for i in range(len(arr_src))}
    des_idx_src={}
    for i in range(len(arr_des)):
        if arr_des[i] in arr_src:
            des_idx_src[i]=arr_des[i]
    idxs=sorted(des_idx_src.keys(),key=lambda x:arr_src_idx[des_idx_src[x]])
    return idxs

def generate_sub_axes():
    pass

def cal_hist_density(arr):
    arr=arr[~np.isnan(arr) & ~np.isinf(arr)]
    return np.histogram(arr, bins=100, density=True)

def kggsee_rez(expr_path, out_prefix, min_tissue:int, kggsee_jar,resource_dir)->str:
    cmd = f'java -Xmx30G -jar {kggsee_jar} --calcu-selectivity-rez-webapp --min-tissue {min_tissue} --gene-expression {expr_path} --out {out_prefix}'
    cmd += f' --resource {resource_dir}'
    return cmd

def kggsee_dese(gwas_file,gene_score,out_prefix,kggsee_jar,resource_dir,multi_correct_method,fwer,top_n_gene,nt,chr_col,bp_col,p_col,
                           ref_genome='hg19',remove_hla=False,java_path='java',jvm_gb='80',
                vcf_ref=None,keep_ref=None,saved_ref=None,run_rez=True):
    '''
    run DESE function in KGGSEE. see docs in https://kggsee.readthedocs.io/en/latest/quick_tutorials.html#dese-driver-tissue-inference.
    :return:
    '''
    para = f'''
        --no-plot
        --resource {resource_dir}
        --sum-file {gwas_file}
        --expression-file {gene_score}
        --out {out_prefix}
        --multiple-testing {multi_correct_method}
        --p-value-cutoff {fwer}
        --buildver {ref_genome}
        --nt {nt}
        --chrom-col {chr_col}
        --pos-col {bp_col}
        --p-col {p_col}
        --filter-maf-le 0.05
        --gene-finemapping
        --dese-permu-num 100
        --db-gene refgene
        --no-gz
        --min-tissue 3
        --top-gene {top_n_gene}
    '''
    if not run_rez:
        para+=f' --calc-selectivity false'
    if saved_ref is None:
        para += f' --vcf-ref {vcf_ref}'
        if keep_ref is not None:
            para += f' --keep-ref {keep_ref}'
    else:
        para += f' --saved-ref {saved_ref}'
    hla_range={'hg19':'27477797-34448354','hg38':'28510120-33480577'}
    if remove_hla:
        para += f' --regions-out chr6:{hla_range[ref_genome]}'
    para=re.sub('\s+',' ',para)
    cmd = f'{java_path} -Xmx{jvm_gb}G -jar {kggsee_jar} {para}'
    return cmd

def run_command(command,log_prefix=''):
    popen = subprocess.Popen(command,shell=True, stdout = subprocess.PIPE,text=True)
    while True:
        line = popen.stdout.readline()
        if not line:
            break
        line=line.strip()
        if line!='':
            print(f'{log_prefix}{line}')
    return popen.returncode

class FunctionEnrichment:
    def __init__(self,db='KEGG'):
        self.db=db
        self.__load_term()

    def __load_term(self):
        db_paths={
            'KEGG':f'{LOCAL_DIR}/resources/KEGG_2021_Human.txt',
            'GO:BP':f'{LOCAL_DIR}/resources/GO_Biological_Process_2023.txt',
            'GO:MF': f'{LOCAL_DIR}/resources/GO_Molecular_Function_2023.txt',
            'GO:CC': f'{LOCAL_DIR}/resources/GO_Cellular_Component_2023.txt'
        }
        db_path=db_paths[self.db]
        bg_genes=[]
        gene_lib = {}
        gene_anno = {}
        gid=GeneID()
        with open(db_path, 'r') as br:
            for line in br:
                arr = line.strip().split('\t')
                term=arr[0].strip()
                gene_syms = gid.get_gene_symbol([x.strip() for x in arr[2:]],True)
                gene_lib[term]=gene_syms
                bg_genes+=gene_syms
                for g in gene_syms:
                    if g not in gene_lib:
                        gene_anno[g]=[]
                    gene_anno[g].append(term)

        self.gene_lib=gene_lib
        self.gene_anno=gene_anno
        self.bg_gene_num=len(set(bg_genes))
    def enrich(self,genes:[],enrich_adj_p_cutoff=0.05,bg_genes_num=None,no_plot=True):
        if bg_genes_num is None:
            bg_genes_num=self.bg_gene_num
        try:
            enr = gseapy.enrichr(genes, self.gene_lib, background=bg_genes_num, no_plot=no_plot)
        except:
            return None
        rdf = enr.results
        rdf = rdf.loc[rdf['Adjusted P-value'] < enrich_adj_p_cutoff,]
        return rdf

    def get_gene_term(self,gene):
        term=[]
        if gene in self.gene_anno:
            term=self.gene_anno[gene]
        return term

    def get_genes_term(self,genes:[],none_rep='None'):
        '''
        返回一组基因名的注释术语。
        :param genes:
        :param none_rep:
        :return:
        '''
        terms=[]
        for g in genes:
            term=self.get_gene_term(g)
            t=none_rep
            if len(term)>0:
                t=term[0]
            terms.append(t)
        return terms

class KGGSEE:
    def __init__(self, prefix):
        self.prefix = prefix

    def gene_based_p_cutoff(self):
        pcut = None
        i = 0
        for line in read_line(f'{self.prefix}.log'):
            if 'Significance level of p value cutoffs for the overall error rate' in line:
                i += 1
                pcut = float(line.strip().split(':')[-1].strip())
        return pcut

    def cond_assoc_genes_df(self,gene_score_file=None,index_col=None):
        if gene_score_file is None:
            gene_path=f'{self.prefix}.gene.assoc.condi.txt'
        else:
            gene_path=f'{self.prefix}{gene_score_file}.gene.assoc.condi.txt'
        df = pd.read_table(gene_path,index_col=index_col)
        return df

    def assoc_cell_df(self,gene_score_file=None):
        if gene_score_file is None:
            cell_path=f'{self.prefix}.celltype.txt'
        else:
            cell_path=f'{self.prefix}{gene_score_file}.celltype.txt'
        df = pd.read_table(cell_path,skipfooter=1, engine='python')
        df = df.sort_values(by=['Adjusted(p)'])
        return df

    def cond_sig_assoc_gene_p(self,gene_score_file=None):
        gene_p={}
        p_cut=self.gene_based_p_cutoff()
        df=self.cond_assoc_genes_df(gene_score_file)
        df=df.loc[df['CondiECSP']<p_cut,:]
        for i in df.index:
            gene_p[df.loc[i, 'Gene']] = df.loc[i, 'CondiECSP']
        return gene_p

    def cond_sig_assoc_gene_herit(self,gene_score_file=None):
        gene_herit={}
        df=self.cond_assoc_genes_df(gene_score_file)
        for i in df.index:
            gene_herit[df.loc[i, 'Gene']] = df.loc[i, 'CondiHeritEHE']
        return gene_herit

    def cond_sig_assoc_gene(self,gene_score_file=None):
        return sorted(self.cond_sig_assoc_gene_p(gene_score_file).keys())

    def assoc_cell_adj_p(self,gene_score_file=None):
        cell_p={}
        df=self.assoc_cell_df(gene_score_file)
        for i in df.index:
            cell_p[df.loc[i,'TissueName']]=float(df.loc[i,'Adjusted(p)'])
        return cell_p

    def assoc_cell_raw_p(self,gene_score_file=None):
        cell_p={}
        df=self.assoc_cell_df(gene_score_file)
        for i in df.index:
            cell_p[df.loc[i,'TissueName']]=float(df.loc[i,'Unadjusted(p)'])
        return cell_p

    def assoc_sig_cell_p(self,adj_p_cut,gene_score_file=None):
        cell_p=self.assoc_cell_adj_p(gene_score_file)
        sig_cell_p={}
        for c,p in cell_p.items():
            if p<adj_p_cut:
                sig_cell_p[c]=p
        return sig_cell_p

    def assoc_sig_cells(self,adj_p_cut,gene_score_file=None,min_top_n=1):
        cell_p = self.assoc_cell_adj_p(gene_score_file)
        cells=sorted(cell_p.keys(), key=lambda x: cell_p[x])
        k=0
        for c in cells:
            if cell_p[c]<adj_p_cut:
                k+=1
        if k<min_top_n:
            k=min_top_n
        return cells[:k]

    def genes_by_module_id(self,module_id):
        df=pd.read_table(f'{self.prefix}.assoc_gene_module.txt',index_col=0)
        genes=df.loc[module_id,'module_gene']
        return [x.strip() for x in str(genes).split(',') if x.strip()!='']



class HomoGene:
    def __init__(self,type='name'):
        self.type=type
        self.load_data()

    def load_data(self):
        type = self.type
        df=pd.read_table(f'{LOCAL_DIR}/resources/mart_hs2mm.txt.gz')
        type_col={'name':'Mouse gene name','id':'Mouse gene stable ID'}
        homo_map={}
        uniq_hs_genes=set()
        col=type_col[type]
        for i in df.index:
            mid=df.loc[i,col]
            hsg = df.loc[i, 'Gene name']
            if (not pd.isna(mid)) and (hsg not in uniq_hs_genes):
                homo_map[mid]=hsg
                uniq_hs_genes.add(hsg)
        self.homo_map=homo_map
    def mm_to_hs_genes(self,mm_gene_names:[]):
        '''
        translate mouse gene to human.
        :return:
        '''
        homo_map = self.homo_map
        homo_genes=[]
        map_idxs=[]
        for i in range(len(mm_gene_names)):
            mg=mm_gene_names[i].strip()
            if mg in homo_map:
                map_idxs.append(i)
                homo_genes.append(homo_map[mg])
        return np.array(map_idxs),np.array(homo_genes)

def remove_last_bracket(input_str):
    pattern = re.compile(r'\([^)]*\)$')
    match = pattern.search(input_str)
    if match:
        result_str = input_str[:match.start()] + input_str[match.end():]
        return result_str
    else:
        return input_str


def get_last_digits(input_str):
    match = re.search(r'\d+$', input_str)
    if match:
        return match.group()
    else:
        return ""

def __plot_bar(cmap,label='Spearman R',vmin=-1, vmax=1,save_path=''):
    fig, ax = plt.subplots(figsize=(2, 1))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, label=label, orientation='horizontal', aspect=5)
    plt.tight_layout()
    if save_path!='':
        plt.savefig(save_path)
    else:
        plt.show()

def heatmap_color_rstyle():
    return __heatmap_color_rstyle()

def heatmap_color_rstyle_single():
    colors = ['#FFFFFF', '#FE0100']
    cmap = LinearSegmentedColormap.from_list("heatmap_rstyle", list(zip([0, 1], colors)))
    return cmap

def __heatmap_color_rstyle():
    colors = ['#1B1AFD', '#FFFFFF', '#FE0100']
    cmap = LinearSegmentedColormap.from_list("heatmap_rstyle", list(zip([0, 0.5, 1], colors)))
    return cmap

def __plot_color_bar(cmap,label='Spearman R',vmin=-1, vmax=1,save_path='',horiz=True):
    fig, ax = plt.subplots(figsize=(1, 2))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    orient='vertical'
    if horiz:
        orient='horizontal'
        fig, ax = plt.subplots(figsize=(2, 1))
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, label=label, orientation=orient, aspect=5)
    plt.tight_layout()
    if save_path!='':
        plt.savefig(save_path)
    else:
        plt.show()

def cell_corr_heatmap(cor: pd.DataFrame, out_fig):
    if cor.shape[0]<2:
        log(f'warning: empty df.')
        return
    cmap = __heatmap_color_rstyle()
    h_size = 0.2 * cor.shape[0]
    w_size = h_size
    fig, ax = plt.subplots(figsize=(w_size, h_size))
    seaborn.heatmap(cor, cmap=cmap, vmin=-1, vmax=1, linewidths=0.1, linecolor='#AEACAD', ax=ax, cbar=False)
    plt.tight_layout()
    plt.savefig(out_fig)
    plt.close()

def cluster_df(cor,method='average', metric= 'euclidean'):
    nrow,ncol=cor.shape
    if nrow<2 or ncol<2:
        return cor
    Z = hierarchy.linkage(cor, method=method, metric=metric)
    x=dendrogram(Z,no_plot=True)
    x_order=x['leaves']
    Z = hierarchy.linkage(cor.T, method=method, metric=metric)
    x=dendrogram(Z,no_plot=True)
    y_order=x['leaves']
    cor=cor.iloc[x_order,y_order]
    return cor

def compare_cell_correlation(expr_path, degree_path):
    """
    compare tissue/cell type correlation according to expression or centrality of network.
    :return:
    """
    corr_method = 'spearman'


def jaccard(li1, li2, print_info=False):
    li1 = set(li1)
    li2 = set(li2)
    x=len(li1.intersection(li2))
    y=len(li1.union(li2))
    if y<1:
        return 0
    cc = x / float(y)
    if print_info:
        print(f'jaccard: {cc}; {len(li1)} vs {len(li2)}')
    return cc

def heatmap_custom1(df:pd.DataFrame,color_df:pd.DataFrame,annot_df:pd.DataFrame,save_fig):
    cmap=__heatmap_color_rstyle()
    base_size=100
    color_df=color_df.loc[df.index,df.columns]
    annot_df = annot_df.loc[df.index, df.columns]
    xs=[]
    ys=[]
    sizes=[]
    colors=[]
    fig, ax = plt.subplots(figsize=(5,4.4))
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            x,y=j,df.shape[0]-i-1
            ys.append(y)
            xs.append(x)
            sizes.append(df.iloc[i,j])
            colors.append(color_df.iloc[i,j])
            ax.text(x,y,annot_df.iloc[i,j], ha='center', va='center',c='black',fontsize=6)
    sizes=np.abs(sizes)
    sizes=sizes*base_size/np.max(sizes)
    max_abs_r=color_df.abs().max().max()
    sc=ax.scatter(xs,ys, c=colors, s=sizes, alpha=0.5, cmap=cmap,marker='s',vmin=-max_abs_r,vmax=max_abs_r) #,vmin=-1,vmax=1
    for i in range(df.shape[0]-1):
        ax.axhline(y=0.5+i,c='gray',linewidth=0.5)
    for i in range(df.shape[1]-1):
        ax.axvline(x=0.5+i,c='gray',linewidth=0.5)

    ax.set_xlim(-0.5,df.shape[1]-0.5)
    ax.set_ylim(-0.5, df.shape[0] - 0.5)
    ax.set_xticks(np.arange(0,df.shape[1]),df.columns,rotation=90,ha='center')
    ax.set_yticks(np.arange(0, df.shape[0]),df.index[::-1])
    ax.tick_params(axis='x', pad=2)
    ax.tick_params(axis='y', pad=2)
    cbar = plt.colorbar(sc)
    # cbar.set_label('R', fontsize=12)
    plt.tight_layout()
    plt.show()
    fig.savefig(save_fig)

def bio_enrich_plot(df:pd.DataFrame,y,x,size,color,size_log=False,ax=None,fig_path=None):
    color_pale=['#E2614D','#FE9927','#4BAA52']
    df=df.copy()
    base_size=120
    size_legend_eg=[0.1,0.5,0.9]
    df['y_idx']=np.arange(df.shape[0],0,-1)
    raw_max_size=df[size].max()
    if size_log:
        df[size]=df[size].map(lambda x:np.log2(x+1))
    max_size=df[size].max()
    df['size_norm']=df[size]*base_size/max_size
    max_char=max([len(c) for c in df[y]])
    if ax is None:
        ratio=1.5
        min_h=5
        h=df.shape[0]/15*4.1
        w_char_base=6/90
        w=w_char_base*(45+max_char)
        if h<min_h:
            h=min_h
        fig, ax = plt.subplots(figsize=(w,h))
    k=0
    for category, group in df.groupby(color):
        ax.scatter(group[x], group['y_idx'], s=group['size_norm'], label=f"{category}", alpha=1, c=color_pale[k])
        k+=1
    ax.set_yticks(df['y_idx'])
    ax.set_yticklabels(df[y])

    le1=ax.legend(loc='upper left',bbox_to_anchor=(1, 1))
    le1.set_title('Database')
    le1._legend_box.align = "left"
    le1.set_frame_on(False)
    for handle in le1.legendHandles:
        handle.set_sizes([base_size*0.7])
    ax.set_xlabel(r'$-log_{10}(adj.P)$')
    ax2 = ax.twinx()
    for sl in size_legend_eg:
        show_eg=int(sl*max_size)
        eg=show_eg
        if size_log:
            show_eg=int(sl*raw_max_size//10*10)
            eg=np.log2(show_eg+1)
        ax2.scatter([], [], s=eg*base_size/max_size, label=f'{show_eg}', alpha=0.8, color='black')
    le2=ax2.legend(loc='upper left',bbox_to_anchor=(1, 0.78))
    le2.set_title('Count')
    le2._legend_box.align = "left"
    le2.set_frame_on(False)
    ax2.set_yticks([])

    # 设置网格线
    ax.grid(axis='y', color='lightgrey', linestyle='--')
    ax.set_axisbelow(True)
    ax.grid(axis='x', color='lightgrey', linestyle='--')
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path,dpi=200)
    plt.show()


def go_enrich_plot(m_genes,out_path,max_term=10,anno_dbs=['GO:BP','GO:CC','GO:MF']):
    gp = GProfiler(return_dataframe=True)
    anno_df = gp.profile(organism='hsapiens', query=m_genes, sources=anno_dbs,
                         user_threshold=0.05)
    # anno_df=anno_df.loc[anno_df.sort_values(by=['p_value']).index[:max_term],:]
    remain_idx=[]
    for cate,sdf in anno_df.groupby('source'):
        cut_term_n=max_term
        if cut_term_n > sdf.shape[0]:
            cut_term_n = sdf.shape[0]
        remain_idx+=sdf.sort_values(by=['p_value']).index[:cut_term_n].tolist()
    anno_df=anno_df.loc[remain_idx,]
    anno_df = anno_df.sort_values(by=['source','p_value'])
    anno_df['p_value']=-np.log10(anno_df['p_value'])
    anno_df.to_excel(f'{out_path}.xlsx')

    try:
        bio_enrich_plot(anno_df,'name','p_value','intersection_size','source',fig_path=out_path)
    except:
        log(f'WARNING: error in plot go enrichment')

def extract_df_col(df,col_suffix):
    xdf = df[[x for x in df.columns if str(x).endswith(col_suffix)]]
    xdf.columns = xdf.columns.map(lambda x: '.'.join(x.split('.')[:-1]))
    return xdf

def hex_to_rgba(hex_color, alpha):
    rgba = mcolors.to_rgba(hex_color)
    return (rgba[0], rgba[1], rgba[2], alpha)

# -*- coding: utf-8 -*-
# @Author  : Xue Chao
# @Time    : 2024/01/11 11:37
# @Function: Main module of DGN (disease-associated gene network module).
#
import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd
import scipy
import seaborn
import umap
from CSCORE import CSCORE_IRLS
from gprofiler import GProfiler
from matplotlib import pyplot as plt, font_manager
from scipy import stats
from scipy.optimize import fsolve
from scipy.stats import zscore, norm
from sklearn.cluster import KMeans, spectral_clustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from construct_gene_network import GeneNetwork, get_no_scale_beta, calculate_similarity_matrix
from util import index_in_gene_symbol_of_gene_list, log, cpm, make_dir, cal_hist_density, FONTS_DIR, kggsee_dese, \
    run_command, KGGSEE, pvalue_adjust, need_trans_gene_name, kggsee_rez, compare_cell_correlation, cell_corr_heatmap, \
    cluster_df

font_files = font_manager.findSystemFonts(fontpaths=FONTS_DIR)
for file in font_files:
    font_manager.fontManager.addfont(file)
plt.rcParams["font.sans-serif"] = "Arial"


class DGN_API:
    '''
    DGN class
    '''
    def __init__(self):
        pass
    def __select_sub_cell(self, cell_ids, max_cell):
        if len(cell_ids)<max_cell:
            max_cell=len(cell_ids)
        np.random.seed(10086)
        sub=np.random.choice(cell_ids,max_cell,replace=False)
        return sub
    def read_expr_dir(self,expr_paths:[],cell_names:[],min_cell=100,max_cell=1000,trans_gene_symbol=False,unified_genes_path=None,
                      same_sample_size=True):
        cp_name=dict(zip(expr_paths,cell_names))
        na_fill_val = 0
        unique_genes = set()
        if unified_genes_path is None or not os.path.isfile(unified_genes_path):
            for f_path in expr_paths:
                try:
                    df = pd.read_table(f_path, header=0, index_col=0)
                except:
                    raise Exception(f'error unzip in {f_path}')
                if df.shape[1] < min_cell:
                    continue
                unique_genes = unique_genes.union(df.index.values)
            unique_genes = sorted(unique_genes)
            if not trans_gene_symbol:
                if need_trans_gene_name(unique_genes[0]):
                    trans_gene_symbol=True
            if trans_gene_symbol:
                gene_idx, gene_syms = index_in_gene_symbol_of_gene_list(unique_genes)
                fine_genes = np.array(unique_genes)[gene_idx]
                fine_gene_sym = np.array(gene_syms)[gene_idx]
                log(f'raw {len(unique_genes)} genes; remain {len(fine_genes)} genes.')
            else:
                fine_genes = np.array(unique_genes)
                fine_gene_sym = np.copy(fine_genes)
            gene_id = {'gene_id': fine_genes, 'gene_symbol': fine_gene_sym}
            if unified_genes_path is not None:
                ug_df=pd.DataFrame(gene_id)
                ug_df.to_csv(unified_genes_path,sep='\t',index=False)
        else:
            ug_df=pd.read_table(unified_genes_path)
            gene_id={c:ug_df[c].values for c in ug_df.columns}
        gene_raw_id=gene_id['gene_id']
        log(f'genes {len(gene_raw_id)}')
        if same_sample_size:
            ss=[]
            for f_path in expr_paths:
                df = pd.read_table(f_path, header=0, index_col=0)
                ss.append(df.shape[1])
            max_cell=np.min(ss)
            min_cell=10

        for f_path in expr_paths:
            f=os.path.basename(f_path)
            last_idx=1
            if f.endswith('.txt.gz'):
                last_idx=2
            arr=f.split('.')
            if len(arr)<2:
                last_idx=0
            cell_name = '.'.join(arr[:-last_idx])
            df = pd.read_table(f_path, header=0, index_col=0)
            cell_ids=df.columns.values
            if len(cell_ids) < min_cell:
                continue
            if len(cell_ids) > max_cell:
                cell_ids=self.__select_sub_cell(cell_ids,max_cell)
            df = df.loc[~df.index.duplicated(), cell_ids]
            add_genes = set(gene_raw_id).union(df.index) - set(df.index)
            for g in add_genes:
                df.loc[g, :] = na_fill_val
            df = df.loc[gene_raw_id, :].fillna(na_fill_val)
            expr=df.values
            yield expr, cp_name[f_path], gene_id

    def normalize_expr(self,expr,method):
        method_map={'no':lambda x:x,'cpm':cpm}
        normalize_expr=method
        return method_map[normalize_expr](expr)

    def corr_matrix(self,expr,seq_depth,method='cs-core',min_expr_val=0):
        def cs_core(expr,depth):
            counts=expr.T
            net = CSCORE_IRLS(counts, depth)
            return net[0]
        zero_idx=np.where(np.nanmean(expr,axis=1)<min_expr_val)[0]
        if min_expr_val>0:
            log(f'{len(zero_idx)} (of {expr.shape[0]}) genes with low expression values (mean<{min_expr_val}).')
        if method=='cs-core':
            corr=cs_core(expr,seq_depth)
        if method=='pearson':
            corr=np.corrcoef(expr)
        if len(zero_idx)>0:
            corr[zero_idx,:]=0
            corr[:,zero_idx]=0
        np.fill_diagonal(corr, 1)
        return corr

    def k_factor_math(self,resample_corr,norm_range=[0, 1]):
        def k_equation(k, X, v):
            return np.nanmean(np.power(X, k)) - v

        datas = np.abs(resample_corr)
        st_idx, ed_idx = (np.array(norm_range) * datas.shape[0]).astype(int)
        sort_datas = np.sort(datas, axis=0)
        norm_datas = sort_datas[st_idx:ed_idx, :]
        v = np.nanmean(np.nanmean(norm_datas, axis=1))
        ks = []
        for i in range(norm_datas.shape[1]):
            k = fsolve(k_equation, 0.01, (norm_datas[:, i], v))
            ks.append(k[0])
        ks=np.array(ks)
        return ks
    def __plot_distribution(self,datas,cells,fig_path,info=None):
        sort_datas = np.sort(datas, axis=0)
        all_bg=sort_datas.flatten()
        nrow, ncol = datas.shape
        need_axes_num=ncol
        sub_axes_unit=3
        n = int(np.sqrt(need_axes_num) - 0.0001) + 1
        plot_ncol = n
        plot_nrow = int(need_axes_num/plot_ncol-0.0001)+1
        fig, axes = plt.subplots(plot_nrow, plot_ncol, figsize=(plot_ncol * sub_axes_unit, plot_nrow * sub_axes_unit))
        k=0
        ref_ghist, ref_gbins = cal_hist_density(all_bg)
        ref_x = (ref_gbins[1:] + ref_gbins[:-1]) / 2
        for i in range(plot_nrow):
            for j in range(plot_ncol):
                ax = axes[i, j]
                if k<need_axes_num:
                    cell_v=sort_datas[:,k]
                    cell=cells[k]
                    if info is not None:
                        ax.set_title(f'{info[k]}')
                    ax.plot(ref_x, ref_ghist, '-', label='All')
                    ghist, gbins = cal_hist_density(cell_v)
                    ax.plot((gbins[1:] + gbins[:-1]) / 2, ghist, '-', label=cell)
                    ax.legend(loc='upper right')
                else:
                    plt.delaxes(ax)
                k+=1
        plt.tight_layout()
        make_dir(os.path.dirname(fig_path))
        plt.savefig(fig_path)

    def cal_k_factor(self,expr_paths,cell_names,intermediate_dir,method='cs-core',min_expr_val=0,resample_size=1000000,
                            trans_gene_symbol=False):
        cell_k={}
        k_factor_path=f'{intermediate_dir}/k_factor.txt'
        if os.path.isfile(k_factor_path):
            k_df=pd.read_table(k_factor_path,index_col=0,header=0)
            for c in k_df.columns:
                cell_k[c]=k_df.loc['k_factor',c]
            log('The k factor has already been calculated.')
        else:
            sample_size = []
            cells = []
            resample_corr=[]
            expr_datas=self.read_expr_dir(expr_paths,cell_names,trans_gene_symbol=trans_gene_symbol)
            for expr, cell_name, gene_id in expr_datas:
                seq_depth=np.nansum(expr,axis=0)
                log(f'{cell_name}: {expr.shape[1]} cells, {expr.shape[0]} genes.')
                sample_size.append(expr.shape[1])
                cor_mat=np.abs(self.corr_matrix(expr,seq_depth,method,min_expr_val))
                triu_mat = cor_mat[np.triu_indices(cor_mat.shape[0], k=1)]
                if len(triu_mat)<resample_size:
                    resample_size=len(triu_mat)
                np.random.seed(10086)
                random_resample = np.random.choice(triu_mat[~np.isnan(triu_mat)], resample_size, replace=False)
                resample_corr.append(random_resample)
                cells.append(cell_name)
                log(f'resampled from {cell_name}')
            resample_corr=np.array(resample_corr).T
            k_factors = self.k_factor_math(resample_corr)
            cell_k=dict(zip(cells,k_factors))
            log(f'calculated k factors')
            rc_df=pd.DataFrame(resample_corr,columns=cells)
            rc_df.to_csv(f'{intermediate_dir}/resample_corr.txt.gz',sep='\t',float_format='%.4f',index=None)
            ck_df=pd.DataFrame({'k_factor':k_factors,'sample_size':sample_size},index=cells).T
            ck_df.index.name='index'
            ck_df.to_csv(k_factor_path,sep='\t')
            log(f'saved k factors')
        return cell_k


    def cal_degree_adjusted(self,expr_paths,cell_names,intermediate_dir,degree_dir,method='cs-core',min_expr_val=0,resample_size=1000000,
                            trans_gene_symbol=False):
        cell_k=self.cal_k_factor(expr_paths,cell_names,intermediate_dir,method,min_expr_val,resample_size,trans_gene_symbol)
        net_cate = ['raw', 'k_adjusted']
        adj_degree=f'{degree_dir}/degree.k_adjusted.txt.gz'
        if os.path.isfile(adj_degree):
            log('The degree of gene co-expression network has already been calculated.')
        else:
            cells=[]
            sample_size = []
            cate_cell_degree={cate:[] for cate in net_cate}
            expr_datas=self.read_expr_dir(expr_paths,cell_names,trans_gene_symbol=trans_gene_symbol)
            cell_depth_gene=[]
            for expr, cell_name, gene_id in expr_datas:
                depth=np.sum(expr,axis=0)
                n_gene=np.count_nonzero(expr,axis=0)
                for s in range(len(depth)):
                    cell_depth_gene.append([depth[s],n_gene[s]])
                cells.append(cell_name)
                sample_size.append(expr.shape[1])
                cor_mat=np.abs(self.corr_matrix(expr,method,min_expr_val))
                cor_mat[np.diag_indices(cor_mat.shape[0])] = 0
                for cate in net_cate:
                    if cate=='k_adjusted':
                        cor_mat=np.power(cor_mat,cell_k[cell_name])
                    cate_cell_degree[cate].append(np.nansum(cor_mat,axis=1))
                log(f'calculated degree of {cell_name}')
            genes = gene_id['gene_symbol']
            for cate in cate_cell_degree.keys():
                deg_vals=np.array(cate_cell_degree[cate]).T
                deg_df=pd.DataFrame(deg_vals,columns=cells,index=genes)
                deg_df.index.name='Gene'
                deg_df.to_csv(f'{degree_dir}/degree.{cate}.txt.gz',sep='\t',float_format='%.4f')
            log(f'saved and plot degree')
        df=pd.read_table(adj_degree,index_col=0)
        return df
    def get_module_corr(self,module_genes,expr_paths,cell_names,intermediate_dir,method='cs-core',min_expr_val=0,
                            trans_gene_symbol=False,resample_size=1000000):
        cell_k=self.cal_k_factor(expr_paths,cell_names,intermediate_dir,method,min_expr_val,resample_size,trans_gene_symbol)

        cells=[]
        sample_size = []
        expr_datas=self.read_expr_dir(expr_paths,cell_names,trans_gene_symbol=trans_gene_symbol)
        vals={}
        common_genes=set()
        for expr, cell_name, gene_id in expr_datas:
            seq_depth=np.nansum(expr,axis=0)
            genes = gene_id['gene_symbol']
            cells.append(cell_name)
            raw_cor_mat=self.corr_matrix(expr,seq_depth,method,min_expr_val)
            # test_gene_a='PIP'
            # test_gene_b='IL3'
            # print(f'{cell_name}:{test_gene_a}:{test_gene_b}={raw_cor_mat[np.where(genes == test_gene_a)[0],np.where(genes == test_gene_a)[0]]}')
            # return
            cor_mat=np.abs(raw_cor_mat)
            sign_cor_mat=raw_cor_mat/cor_mat
            cor_mat = np.power(cor_mat, cell_k[cell_name])
            cor_mat[np.diag_indices(cor_mat.shape[0])] = 1
            cor_mat=cor_mat*sign_cor_mat

            if module_genes is not None:
                midxs=np.array([i for i in range(len(genes)) if genes[i] in module_genes])
                cor_mat=cor_mat[midxs,:][:,midxs]
                genes=genes[midxs]
            cor_mat[np.isnan(cor_mat)] = 0
            cor_mat[np.isposinf(cor_mat)] = 1
            cor_mat[np.isneginf(cor_mat)] = -1
            df=pd.DataFrame(cor_mat,index=genes,columns=genes)
            vals[cell_name]=df
            if len(common_genes)==0:
                common_genes=set(genes)
            else:
                common_genes=common_genes.intersection(genes)
        sgenes=sorted(common_genes)
        for c in vals.keys():
            vals[c]=vals[c].loc[sgenes,sgenes]
        return vals

if __name__ == '__main__':
    # x=np.array([[1,2,3],[3,4,5]])
    # print(cpm(x,row_is_gene=False))
    print(remove_last_bracket('Tuft cell_Sh2d6 (d)high (Adult-Intestine)'))