from enum import Enum

# class Enum_rna(Enum):
#     lncrna = "lncrna"
#     mirna = "mirna"
#     mrna = "mrna"
#     other = "other"
#


#为每个样本添加标签,去掉第一列，样本名
def add(fileDir, filename, filename1):

    f = open(filename)
    f1 = open(filename1, "w")

    lines = f.readlines()
    datamat = []

    for i in range(0, len(lines)):
        data = lines[i].strip().split('\t')
        datamat.append(data)  # 存放整个矩阵

    N_i = len(datamat)  # 行数
    N_j = len(datamat[0])  # 列数
    print(datamat[1][0])  # TCGA-91-A4BC-01
    # Y = datamat[1][0].strip().split('-')
    # print(Y[3][1])
    print(N_j)
    normal = 0
    tumor =0
    for i in range(0, N_i):
        print(i + 1)
        # f1.write(datamat[i][0])  #保留样本名
        # mark.append(datamat[i][0])
        # f1.write(',')
        for j in range(1, N_j):  # 从第二列开始
            f1.write(datamat[i][j])
            f1.write(',')
        if (i == 0):
            f1.write("label")
        else:
            Y = datamat[i][0].strip().split('-')
            # print(Y)
            if (Y[3][0] == '1' and Y[3][1] == '1'):
                f1.write("2")
                normal+=1
            else:
                f1.write("1")
                normal+=1
        if (i < N_i):
            f1.write('\n')
    print(tumor)
    print(normal)
#

# DATA_DIR0 = r"C:\Users\JQ\experiment\data\BREA"
# filename = DATA_DIR0 + "\\diff_T.txt"
# filename1 = DATA_DIR0 + "\\diff_new.csv"
# add(DATA_DIR0, filename, filename1)




    # for enum_rna in Enum_rna:


#
# DATA_DIR0 = r"C:\Users\JQ\experiment\data\LAUD"
# for enum_rna in Enum_rna:
#     for i in range(0,9):
#         DATA_DIR1 = DATA_DIR0 + "\\" + "0" + "\\"
#         DATA_DIR2 = DATA_DIR1 + enum_rna.value + "\\"
#         filename = DATA_DIR2 + "hnsc_mrna_train" + str(i) + ".txt"
#         filename1 = DATA_DIR2 + "hnsc_mrna_train" + str(i) + ".csv"
#         add(DATA_DIR1, filename, filename1)