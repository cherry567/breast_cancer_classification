import numpy as np
import convert



from enum import Enum

class Enum_rna(Enum):
    lncrna = "lncrna"
    mirna = "mirna"
    mrna = "mrna"
    other = "other"


DATA_DIR0 = r"C:\Users\JQ\experiment\data\KIRC"

for i in range(0, 10):
    DATA_DIR2 = DATA_DIR0 + "\\" + str(i) + "\\"

    for enum_rna in Enum_rna:
        DATA_DIR3 = DATA_DIR2 + enum_rna.value + "\\"
        for i in range(0, 4):
            filename = DATA_DIR3 + "/" + "hnsc_mrna_train" + str(i) + "T.txt"
            filename_T = DATA_DIR3 + "/" + "hnsc_mrna_train" + str(i) + "_T.txt"
            # filename3 = DATA_DIR3 + "/" + enum_rna.value + ".txt"
            # filename3_T = DATA_DIR3 + "/" + enum_rna.value + "_T.txt"
            filename_in = DATA_DIR3 + "/" + "kirc_" + str(i) + "_inputs.csv"
            filename_out = DATA_DIR3 + "/" + "kirc_" + str(i) + "_outputs.csv"


            f = open(filename)
            convert.T_name(filename_T,filename)
            f = open(filename_T)
            f1 = open(filename_in, 'w')  # 没有就创建文件
            f2 = open(filename_out, 'w')



            # data = np.genfromtxt(filename)
            # transpose_data = np.transpose(data)
            # print(transpose_data)
            # transpose_data.to_csv(filename1,sep = ',')

            lines = f.readlines()

            # 将原始数据写入inputs
            datamat = []
            for i in range(0, len(lines)):
                data = lines[i].strip().split('\t')
                datamat.append(data)  # 存放整个矩阵

            N_i = len(datamat)  # 行数
            N_j = len(datamat[0])  # 列数
            # print(N_i)
            # print(N_j)
            # print(datamat[1][2])
            tumor = 0
            normal = 0

            for i in range(1, N_i):
                Y = datamat[i][0].strip().split('-')
                if (Y[3][0] == '1' and Y[3][1] == '1'):
                    f2.write("2")
                    normal += 1
                else:
                    f2.write("1")
                    tumor += 1
                f2.write('\n')

            # print(normal)
            # print(tumor)
            # #
            for i in range(1, N_i):
                for j in range(1, N_j):
                    if (j < N_j - 1):
                        f1.write(datamat[i][j])
                        f1.write(',')
                    else:
                        f1.write(datamat[i][j])
                if i < N_i - 1:
                    f1.write("\n")





#
#
#
# DATA_DIR0 = r"C:\Users\JQ\TCGA_TOOL\TCGA-KIRC\0\lncrna"
# filename = DATA_DIR0+ r"\hnsc_mrna_train0_T.txt"   #行为样本，列为基因
#
# filename1 = DATA_DIR0+r"\kirc_0_inputs.csv"    #除去行名，除去列名
# filename2 = DATA_DIR0+r"\kirc_0_outputs.csv"   #添加标签
#
# f = open(filename)
# f1 = open(filename1,'w')    #没有就创建文件
# f2 = open(filename2,'w')
#
# # data = np.genfromtxt(filename)
# # transpose_data = np.transpose(data)
# # print(transpose_data)
# # transpose_data.to_csv(filename1,sep = ',')
#
# lines = f.readlines()
#
# #将原始数据写入inputs
# datamat = []
# for i in range(0, len(lines)):
#
#     data = lines[i].strip().split('\t')
#     datamat.append(data)   #存放整个矩阵
#
# N_i = len(datamat)    #行数
# N_j = len(datamat[0])  #列数
# print(N_i)
# print(N_j)
# # print(datamat[1][2])
# tumor = 0
# normal = 0
#
# for i in range(1, N_i):
#     Y = datamat[i][0].strip().split('-')
#     if (Y[3][0] == '1' and Y[3][1] == '1'):
#         f2.write("2")
#         normal += 1
#     else:
#         f2.write("1")
#         tumor += 1
#     f2.write('\n')
#
# print(normal)
# print(tumor)
# # #
# for i in range(1, N_i):
#     for j in range(1, N_j):
#         if (j < N_j- 1):
#             f1.write(datamat[i][j])
#             f1.write(',')
#         else:
#             f1.write(datamat[i][j])
#     if i < N_i - 1:
#         f1.write("\n")





