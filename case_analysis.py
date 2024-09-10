# -*- coding: utf-8 -*-
# @Author: Xue Chao
# @Time: 2024/07/08 11:22
# @Function: Case analysis
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.font_manager import FontProperties
from scipy import stats
import seaborn
from scipy.stats import norm

from coexpression_characteristics import __extract_z_col
from associated_cell_types import get_cell_type_abbr_cate
from util import DGN_API
from para import PsychENCODE_fine_data_dir, output_DIR, \
    case_control_box_pale, up_down_no_pale
from util import make_dir, log, cpm, kggsee_rez, LOCAL_DIR, heatmap_color_rstyle, KGGSEE, \
    pvalue_adjust

def cal_expr_degree(normal_method='cpm',log_trans=True):
    case_analysis_data_dir=f'{output_DIR}/case_analysis/data'
    make_dir(case_analysis_data_dir)
    dirs={}
    name_map={'control':'control','MDD':'MDD','PTSD':'PTSD','BipolarDisorder':'BIP','Schizophrenia':'SCZ'}
    for f in os.listdir(PsychENCODE_fine_data_dir):
        if f in name_map:
            dirs[name_map[f]]=f'{PsychENCODE_fine_data_dir}/{f}'
    dgn_api=DGN_API()
    for k in dirs.keys():
        log(f'start deal {k}')
        dir=dirs[k]
        dfs=[]
        degree_dfs=[]
        for f in os.listdir(dir):
            cell_name=f.replace('.txt.gz','')
            df=pd.read_table(f'{dir}/{f}',index_col=0,header=0)
            df=df.fillna(0)
            log(f'load {k}:{cell_name} with shape={df.shape}')
            expr=df.values
            # cal degree
            seq_depth=np.nansum(expr,axis=0)
            cor_mat=dgn_api.corr_matrix(expr,seq_depth,'cs-core',0)
            degree=np.nansum(np.abs(cor_mat),axis=0)-1
            degree_dfs.append(pd.DataFrame({cell_name:degree},index=df.index))
            # cal expr
            if normal_method=='cpm':
                expr=cpm(expr)
            if log_trans:
                expr=np.log2(expr+1)
            mean_expr=np.nanmean(expr,axis=1)
            dfs.append(pd.DataFrame({cell_name:mean_expr},index=df.index))
        fdf=pd.concat(dfs,axis=1)
        fdf.index.name='Gene'
        fdf = fdf.fillna(0)
        expr_path=f'{case_analysis_data_dir}/{k}.expr.txt.gz'
        fdf.to_csv(expr_path,sep='\t')

        deg_df=pd.concat(degree_dfs,axis=1)
        deg_df.index.name='Gene'
        deg_df = deg_df.fillna(0)
        deg_path=f'{case_analysis_data_dir}/{k}.degree.txt.gz'
        deg_df.to_csv(deg_path,sep='\t')

def run_rez():
    kggsee_dir= f'{LOCAL_DIR}/lib/kggsee'
    kggsee_jar = f'{kggsee_dir}/kggsee.jar'
    kggsee_resources = f'{kggsee_dir}/resources'
    dir=f'{output_DIR}/case_analysis/data'
    for f in os.listdir(dir):
        if f.endswith('expr.txt.gz'):
            cmd = kggsee_rez(f'{dir}/{f}',f'{dir}/{f.replace(".txt.gz","")}',5,kggsee_jar,kggsee_resources)
            log(cmd)
            os.system(cmd)


def cell_similarity_cross_datasets():
    corr_dir = f'{output_DIR}/case_analysis/corr'
    make_dir(corr_dir)
    ctl_path=f'{output_DIR}/case_analysis/data/control.expr.REZ.webapp.genes.txt.gz'
    ref_tags=['hs_gtex_gse97930_fc']
    ref_exprs=[f'{output_DIR}/{rt}/intermediate/expr.mean.REZ.webapp.genes.txt.gz' for rt in ref_tags]
    tag_expr=dict(zip(ref_tags,ref_exprs))
    name_abbr,abbr_cate=get_cell_type_abbr_cate()
    cmap=heatmap_color_rstyle()
    for tag in tag_expr.keys():
        ref_df= pd.read_table(tag_expr[tag],index_col=0,header=0)
        df=pd.read_table(ctl_path,index_col=0,header=0)
        ref_df=__extract_z_col(ref_df)
        df=__extract_z_col(df)
        common_index=ref_df.index.intersection(df.index)
        sele_cols=[c for c in ref_df.columns if abbr_cate[name_abbr[c]]=='Brain']
        ref_df=ref_df.loc[common_index,sele_cols]
        df=df.loc[common_index,:]
        cor_df=pd.DataFrame(np.zeros((ref_df.shape[1],df.shape[1])),index=ref_df.columns,columns=df.columns)
        for c1 in ref_df.columns:
            for c2 in df.columns:
                r,p=scipy.stats.pearsonr(ref_df.loc[:,c1],df.loc[:,c2])
                cor_df.loc[c1, c2] =r
        cor_df.to_csv(f'{corr_dir}/{tag}.corr_matrix.txt',sep='\t')
        annot = np.empty(cor_df.shape, dtype=str)
        for i in range(cor_df.shape[0]):
            max_idx = np.argmax(cor_df.values[i])
            annot[i, max_idx] = '*'
            for j in range(cor_df.shape[1]):
                if annot[i, j] != '*':
                    annot[i, j] = ''
        seaborn.clustermap(cor_df,cmap=cmap,figsize=(7,7),vmin=-1,vmax=1,annot=annot,fmt='',
                           annot_kws={'fontsize': 15,'color':'black'})
        plt.savefig(f'{corr_dir}/{tag}.corr.heatmap.png')
        src_tag=cor_df.idxmax(axis=1)
        src_tag.to_csv(f'{corr_dir}/{tag}.corr.txt',sep='\t',header=False)

def __load_expr_degree(prefix,label=''):
    expr_path=f'{prefix}.expr.{label}txt.gz'
    degree_path = f'{prefix}.degree.{label}txt.gz'
    tag_paths={'expr':expr_path,'degree':degree_path} #
    tag_dfs={}
    genes=[]
    for tag in tag_paths.keys():
        df=pd.read_table(tag_paths[tag],index_col=0)
        tag_dfs[tag]=df
        genes.append(df.index)
    common_genes=np.intersect1d(genes[0],genes[1])
    for tag in tag_dfs.keys():
        tag_dfs[tag]=tag_dfs[tag].loc[common_genes,:]
    return tag_dfs

def __load_values(df:pd.DataFrame,genes,cell):
    if genes is None:
        common_genes=df.index
    else:
        common_genes=df.index.intersection(genes)
    return df.loc[common_genes,cell]

def test_case_control_dif(ctl_val:np.ndarray,case_val:np.ndarray,permutation_times=100):
    dif_val=case_val-ctl_val
    np.random.seed(10086)
    all_arr=[]
    for i in range(permutation_times):
        ctl_copy=ctl_val.copy()
        case_copy=case_val.copy()
        np.random.shuffle(ctl_copy)
        np.random.shuffle(case_copy)
        shu_dif=case_copy-ctl_copy
        all_arr.append(shu_dif)
    null_arr=np.concatenate(all_arr)
    sorted_null_arr=np.sort(null_arr)
    indices = np.searchsorted(sorted_null_arr, dif_val, side='right')
    pv = (len(sorted_null_arr) - indices)/len(sorted_null_arr)
    z_values = norm.ppf(1 - pv)
    # for data in [pv,z_values,dif_val,sorted_null_arr]:
    #     plt.figure(figsize=(6, 4))
    #     seaborn.kdeplot(data, shade=True)
    #     plt.title('Density Plot of Concatenated Array')
    #     plt.xlabel('Value')
    #     plt.ylabel('Density')
    #     plt.show()
    return pv,z_values



def gene_dif_test():
    ctl_dfs=__load_expr_degree(f'{output_DIR}/case_analysis/data/control')
    out_d=f'{output_DIR}/case_analysis/dif'
    make_dir(out_d)
    phenos=['BIP','MDD','SCZ','PTSD'] #,'MDD','SCZ','PTSD'
    for pheno in phenos:
        case_dfs=__load_expr_degree(f'{output_DIR}/case_analysis/data/{pheno}')
        for tag in ctl_dfs.keys():
            ctl_df=ctl_dfs[tag]
            case_df=case_dfs[tag]
            common_cells=ctl_df.columns.intersection(case_df.columns)
            common_genes=ctl_df.index.intersection(case_df.index)
            df_data={}
            for c in common_cells:
                ctl_val=ctl_df.loc[common_genes,c].values
                case_val=case_df.loc[common_genes,c].values
                # pv,z_values=test_case_control_dif(ctl_val,case_val)
                # z_values=stats.zscore(case_val-ctl_val)
                z_values=case_val-ctl_val
                # qv=np.array(pvalue_adjust(pv))
                # sig_gene=np.sum(qv<0.1)
                # log(f'{pheno}:{tag}:{c}:{sig_gene}')
                df_data[c]=z_values
            df=pd.DataFrame(df_data,index=common_genes)
            df.index.name='Gene'
            df.to_csv(f'{out_d}/{pheno}.{tag}.dif_raw.txt.gz',sep='\t')


def assoc_gene_degree_dif():
    dif_dir=f'{output_DIR}/case_analysis/dif'
    ref_tags=['hs_gtex_gse97930_fc']
    phenos=['BIP','MDD','SCZ']
    gene_score_file='degree.k_adjusted.txt.gz'
    # gene_score_file = 'expr.mean.txt.gz'
    ctl_dfs=__load_expr_degree(f'{output_DIR}/case_analysis/data/control')
    stat_dir=f'{output_DIR}/case_analysis/stat'
    name_abbr,abbr_cate=get_cell_type_abbr_cate()
    make_dir(stat_dir)
    # read map
    cell_dict={}
    for tag in ref_tags:
        corr_path = f'{output_DIR}/case_analysis/corr/{tag}.corr.txt'
        mdf = pd.read_table(corr_path, header=None)
        for i in mdf.index:
            cell_dict[mdf.loc[i, 0]]=mdf.loc[i, 1]
    for tag in ref_tags:
        with pd.ExcelWriter(f'{stat_dir}/case_control_test.{tag}.xlsx') as bw:
            pheno_cells={}
            pheno_dfs={}
            pheno_genep={}
            for pheno in phenos:
                pheno_dfs[pheno]={}
                case_dfs=__load_expr_degree(f'{output_DIR}/case_analysis/data/{pheno}')
                kprefix=f'{output_DIR}/{tag}/result/{pheno}'
                ks=KGGSEE(kprefix)
                gene_p=ks.cond_sig_assoc_gene_p(gene_score_file)
                pheno_genep[pheno]=gene_p
                sig_cells=ks.assoc_sig_cells(0.05,gene_score_file,1)
                cell_p=ks.assoc_cell_adj_p(gene_score_file)
                assoc_genes=ks.cond_sig_assoc_gene(gene_score_file)
                mapped_cells=sorted(set([cell_dict[c] for c in sig_cells]))
                data=[]
                for c in sig_cells:
                    # mapped_cell=c
                    mapped_cell=cell_dict[c]
                    if mapped_cell not in case_dfs['degree'].columns:
                        continue
                    com_genes=ctl_dfs['degree'].index.intersection(case_dfs['degree'].index).intersection(assoc_genes)
                    xs=ctl_dfs['degree'].loc[com_genes,mapped_cell]
                    ys=case_dfs['degree'].loc[com_genes,mapped_cell]
                    stat, pv = scipy.stats.ranksums(xs, ys, alternative='two-sided')
                    # stat, pv = scipy.stats.ttest_rel(xs, ys, alternative='two-sided')
                    pheno_dfs[pheno][mapped_cell]=(xs,ys,stat,pv)
                    cp=cell_p[c]
                    # print(f'{pheno}:{c}={stat},{pv}')
                    data.append([c,cp,mapped_cell,stat,pv])
                test_df=pd.DataFrame(data,columns=['Cell type (DGN)','Adjusted P (DGN)','Mapped cell type (Case/Control)',
                                           'Stat (Case/Control)','P (Case/Control)'])
                test_df['Adjusted P (Case/Control)']=pvalue_adjust(test_df.loc[:,'P (Case/Control)'],'Bonf')
                test_df=test_df.sort_values(by=['P (Case/Control)'],ignore_index=True)
                test_df['Cell type (DGN)']=test_df['Cell type (DGN)'].map(lambda x:name_abbr[x])
                test_df.to_excel(bw,sheet_name=pheno,index=False)
                pheno_cells[pheno]=test_df.loc[0,'Mapped cell type (Case/Control)']
        with pd.ExcelWriter(f'{stat_dir}/case_degree_expr_compare.{tag}.xlsx') as bw:
            ## plot most sig cell.
            for pheno in phenos:
                print(f'{pheno}:{pheno_cells[pheno]}')
                ## compare degree
                c=pheno_cells[pheno]
                gene_p=pheno_genep[pheno]
                assoc_genes=sorted(gene_p.keys())
                xs,ys,stat,pv=pheno_dfs[pheno][c]
                fig,axs=plt.subplots(2,1,figsize=(4.5,8))
                ax=axs[0]
                # seaborn.violinplot(data=[xs,ys],ax=ax,palette=case_control_box_pale)
                seaborn.boxplot(data=[xs, ys], ax=ax,palette=case_control_box_pale)
                y_unit=np.max(np.concatenate([xs,ys]))
                last_y=y_unit * 1.3
                ax.plot((0, 1), (last_y, last_y), '-', c='black')
                ax.plot((0, 0), (last_y, last_y-y_unit*0.03), '-', c='black')
                ax.plot((1, 1), (last_y, last_y - y_unit * 0.03), '-', c='black')
                ax.set_ylim(ax.get_ylim()[0],last_y+y_unit*0.2)
                ax.set_xticklabels(['Control',pheno])
                ax.set_ylabel('Degree')
                ax.text(0.5, last_y*1.02, f'$Stat$ = {stat:.2f}, $P$ = {pv:.2g}', ha='center', va='bottom', fontsize=10, color='black')
                ax.set_title(f'{pheno} in {c} cells',fontproperties=FontProperties(size=15, weight='bold'))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                # plt.tight_layout()
                # plt.show()
                # fig.savefig(f'{stat_dir}/{pheno}_{c}.violin.png')
                # fig.savefig(f'{stat_dir}/{pheno}_{c}.box.png')
                # plt.close()
                ## compare genes in table and fig.
                # fig, ax = plt.subplots(figsize=(4, 4))
                ax = axs[1]
                dif_dfs=__load_expr_degree(f'{output_DIR}/case_analysis/dif/{pheno}','dif_z_norm.')
                dif_ex_df=dif_dfs['expr']
                dif_de_df=dif_dfs['degree']
                cgs=dif_ex_df.index.intersection(dif_de_df.index).intersection(assoc_genes)
                dxs=dif_ex_df.loc[cgs,c]
                dys=dif_de_df.loc[cgs,c]
                zcut= stats.norm.ppf(1-0.05/2)
                nosig_idx=[]
                desig_idx=[]
                for i in range(len(dxs)):
                    # if np.abs(dxs[i])<zcut and np.abs(dys[i])<zcut:
                    #     nosig_idx.append(i)
                    if np.abs(dxs[i])<zcut and dys[i]<=-zcut:
                        desig_idx.append(i)
                    else:
                        nosig_idx.append(i)

                ax.plot(dxs[nosig_idx],dys[nosig_idx],'.',c=up_down_no_pale[2])
                ax.plot(dxs[desig_idx], dys[desig_idx], '.', c=up_down_no_pale[0])
                # ax.set_title(f'{pheno}:{c}')
                for y in [zcut,-zcut]:
                    ax.axhline(y,linestyle='--',c=up_down_no_pale[2])
                    ax.axvline(y, linestyle='--',c=up_down_no_pale[2])
                ymin, ymax = np.array(ax.get_ylim())*1.1
                y_bottom = ymin*1.1
                ax.fill_between([-zcut, zcut], y_bottom, -zcut, color='red', alpha=0.1)
                ax.set_ylim(y_bottom, ymax)
                xmin, xmax = np.array(ax.get_xlim())*1.1
                ax.set_xlim(xmin, xmax)
                pr,pp=stats.pearsonr(dxs,dys)
                text_x=(xmax-xmin)*0.1+xmin
                text_y=(ymax-y_bottom)*0.9+y_bottom
                ax.text(text_x, text_y,f"Pearson's R={pr:.2f}", ha = 'left', va = 'bottom')
                annot_idxs = sorted(desig_idx, key=lambda x: dys[x])[:3]
                pheno_annot_type={'BIP':['l3', 'l2', 'l21'],'SCZ':['l3', 'l2', 'l1'],'MDD':['r3', 'r2', 'r1']}
                pos = {'l1': ((-25, 15), 'right'),'l2': ((-25, 0), 'right'),'l21': ((-25, 5), 'right'),'l3': ((-25, -15), 'right'),
                       'r1': ((25, 15), 'left'),'r2': ((25, 0), 'left'),'r3': ((25, -15), 'left')}
                for ii in range(len(annot_idxs)):
                    i = annot_idxs[ii]
                    xyt, ha = pos[pheno_annot_type[pheno][ii]]
                    label = cgs[i]
                    ax.annotate(label, (dxs[i], dys[i]), xytext=xyt,
                                arrowprops=dict(arrowstyle="->", facecolor='gray', edgecolor='gray'),
                                fontsize=9, textcoords="offset points", ha=ha, bbox=dict(boxstyle='round', alpha=0.1))

                ax.set_xlabel(r"Expression's $Z$ (Case - Control)")
                ax.set_ylabel(r"Degree's $Z$ (Case - Control)")
                # ax.set_aspect('equal', adjustable='box')
                # fig.savefig(f'{stat_dir}/{pheno}_{c}.scatter_annot.png')
                plt.tight_layout()
                fig.savefig(f'{stat_dir}/{pheno}_{c}.case_analysis.png')
                plt.close()

                sig_data=[]
                for i in desig_idx:
                    sig_data.append([cgs[i],pheno_genep[pheno][cgs[i]],dxs[i],dys[i],stats.norm.sf(abs(dxs[i]))*2,
                                    stats.norm.sf(abs(dys[i]))*2])
                sig_df=pd.DataFrame(sig_data,columns=['Gene','P(DGN)','Z(Expr)','Z(Degree)','P(Expr)','P(Degree)'])
                sig_df.to_excel(bw,sheet_name=f'{pheno}_{c}',index=False)

def main():
    cal_expr_degree()
    run_rez()
    cell_similarity_cross_datasets()
    gene_dif_test()
    assoc_gene_degree_dif()

if __name__ == '__main__':
    main()
