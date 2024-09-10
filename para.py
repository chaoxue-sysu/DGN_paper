# -*- coding: utf-8 -*-
# @Author  : Xue Chao
# @Time    : 2024/01/15 18:05
# @Function: parameters for run DGN in paper.
import platform

import seaborn
from seaborn import cm

# Common
DATA_DIR = '/home/xc/local/data'
if platform.system() == 'Windows':
    DATA_DIR = r'E:\WorkData\syncHPC\home\data'

## scRNA-seq expression matrix
TRANS_DIR = f'{DATA_DIR}/resources/Transcriptome'
simulate_dir=f'{TRANS_DIR}/simulate/analysis'
TMS_global_dir=f'{TRANS_DIR}/TMS/analysis/facs/global'
hs_GTEx_GSE97930_fc_dir=f'{TRANS_DIR}/hs_GTEx_GSE97930_fc/analysis/global'
PsychENCODE_fine_data_dir=f'{TRANS_DIR}/Brain/PsychENCODE2/Prashant_Science_2024/analysis/by_category_resample1000'

# GWAS
GWAS_ROOT=f'{DATA_DIR}/resources/GWAS'
GWAS_case_dir=f'{GWAS_ROOT}/gold_case/formatted'
GWAS_raw_meta_path=f'{GWAS_ROOT}/gold_case/raw/meta.xlsx'

# Output
output_main_version='dgn_0516'
output_son_version='a'
output_relative_dir=f'projects/{output_main_version}{output_son_version}'
output_DIR=f'{DATA_DIR}/{output_relative_dir}' #

# Gene annotation
## from https://github.com/broadinstitute/gtex-pipeline/blob/master/TOPMed_RNAseq_pipeline.md
gtexv8_pip_gene_annot_gtf=f'{DATA_DIR}/resources/GeneAnnotation/gencode.v34.annotation.gtf.gz'

# Parameters for analysis in paper.
META_DIR=f'{DATA_DIR}/resources/meta'
transcriptome_meta_xlsx=f'{META_DIR}/transcriptome.xlsx'
GWAS_meta_xlsx=f'{META_DIR}/GWAS.xlsx'
meta_transcriptome_dir=f'{META_DIR}/transcriptome'

## Common results
Common_results_dir=f'{DATA_DIR}/projects/dgn_common'

## COLOR para
COLORS_gene_compare_auc=['#F8766D','#00bf7d','#00BFC4']
CELL_CATE_colors= ['#DB5F57','#56DB5F','#5F56DB']
case_compare_box_pale= ['#F29802','#F8766D','#00BFC4','#C5C6C3']
up_down_no_pale = ['#F87572','#87CEEB','#C5C6C3']
case_control_box_pale=COLORS_gene_compare_auc
UNSELECT_PHENOS=[]
