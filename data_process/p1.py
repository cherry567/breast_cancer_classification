# -*- coding:utf-8 -*-

import convert, cross_valiation
from enum import Enum
import rpy2.robjects as robjects
import os

# os.environ['R_HOME'] = 'C:\Program Files\R\R-3.6.1'
# os.environ['R_USER'] = r'C:\Users\JQ\Anaconda3\Lib\site-packages\rpy2'  # 生成的rpy2目录C:\Python27\Lib\site-packages\rpy2


# 注意

# 准备工作：1、将miRNAmatrix.txt复制到当前目录下
# 2、将FC_FDR_RNA.R复制到当前目录下,并设置对照组的样本数
# 3、fenzu.py设置对照组的样本数


class Enum_rna(Enum):
    lncrna = "lncrna"
    mirna = "mirna"
    mrna = "mrna"
    other = "other"


#DATA_DIR0 = r"C:\Users\JQ\Py_R_data_process_new\dataset\TCGA-BREA"
#DATA_DIR0 = r"D:\CODE\Py_R_data_process_new\dataset\TCGA-KIRC"
DATA_DIR0 = r"C:\Users\JQ\experiment\data\BREA"
# gene_file = DATA_DIR0 + r"\gene_symbol.txt"
# miRNA_file = DATA_DIR0 + r"\miRNAmatrix.txt"
#
# co_gene = DATA_DIR0 + r"\co_gene.txt"
# co_gene_T = DATA_DIR0 + r"\co_gene_T.txt"
# FDR = DATA_DIR0 + "\\" + r"FC_FDR_RNA.R"
gene = DATA_DIR0 + r"\gene_symbol.txt"   #标准化后，从5万7千多维降至3万5千维
gene_T = DATA_DIR0 + r"\gene_symbol_T.txt"
# 第一步加入高质量miRNA
# print("第一步:加入高质量miRNA")
# RongHe.ronghe(DATA_DIR0, gene_file, miRNA_file)  # 融合高质量的miRNA得到co_gene.txt
# #
#
# # 第二步分为十折数据,对每一折数据划分训练集和验证集，并进行转置，放入共10个文件夹
print("第二步:分为十折数据,对每一折数据划分训练集和验证集，并进行转置，放入共10个文件夹")
# print("共表达谱数据co_gene.txt转置")
#convert.T_name(co_gene_T, co_gene)  # 转置
#convert.T_name(gene_T, gene)
DATA_Y = []
Data_X = []
print("划分十折数据")
# cross_valiation.cross_count(DATA_DIR0, co_gene_T, 10, Data_X, DATA_Y)
cross_valiation.cross_count(DATA_DIR0, gene_T, 5, Data_X, DATA_Y)


#
#
# # 第三步，对每一折的学习集的转置分类别提取，并放入各自的目录下（tiqumirna.py）
# print("第三步：对每一折的学习集的转置分类别提取，并放入各自的目录下，并提取对照组")
#
# for i in range(0, 10):
#     DATA_DIR2 = DATA_DIR0 + "\\" + str(i) + "\\"
#     filename2 = DATA_DIR2 + "/" + str(i) + "_train_T.txt"
#
#     tqmirna.tiqu(DATA_DIR2, filename2)  # 分四类提取基因数据
#
#     for enum_rna in Enum_rna:
#         DATA_DIR3 = DATA_DIR2 + enum_rna.value + "\\"
#         filename3 = DATA_DIR3 + "/" + enum_rna.value + ".txt"
#         filename3_T = DATA_DIR3 + "/" + enum_rna.value + "_T.txt"
#         FDR_each = DATA_DIR3 + r"FC_FDR_RNA.R"
#         convert.T_name(filename3_T, filename3)  # 转置函数
#         fenzu.devide_count(DATA_DIR3, filename3_T, 4)  # FDR所用的训练集分组，每组20个正例、20个反例
#         copy_test.copyfile(FDR, FDR_each)
#         robjects.r.setwd(DATA_DIR3)
#         print(robjects.r.getwd())
#         robjects.r.source('FC_FDR_RNA.R')
#


#
#
# # 第三步，对每一折的学习集的转置分类别提取，并放入各自的目录下（tiqumirna.py）
# print("第三步：对每一折的学习集的转置分类别提取，并放入各自的目录下，并提取对照组")
#
#
#
# for i in range(0, 10):
#     DATA_DIR2 = DATA_DIR0 + "\\" + str(i) + "\\"
#     filename2 = DATA_DIR2 + "/" + str(i) + "_out_T.txt"
#
#     tqmirna.tiqu(DATA_DIR2, filename2)  # 分四类提取基因数据
#
#     for enum_rna in Enum_rna:
#         DATA_DIR3 = DATA_DIR2 + enum_rna.value + "\\"
#         filename3 = DATA_DIR3 + "/" + enum_rna.value + ".txt"
#         filename3_T = DATA_DIR3 + "/" + enum_rna.value + "_T.txt"
#         FDR_each = DATA_DIR3 + r"FC_FDR_RNA.R"
#         convert.T_name(filename3_T, filename3)  # 转置函数
#         fenzu.devide_count(DATA_DIR3, filename3_T, 4)  # FDR所用的训练集分组，每组20个正例、20个反例
#         copy_test.copyfile(FDR, FDR_each)
#         robjects.r.setwd(DATA_DIR3)
#         print(robjects.r.getwd())
#         robjects.r.source('FC_FDR_RNA.R')
