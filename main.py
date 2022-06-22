import pandas as pd
import matplotlib.pyplot as plt
import copy
import math
import numpy as np
from itertools import groupby

f1 = pd.read_csv('./Rawdata/ChirpPre.csv')
f2 = pd.read_csv('./Rawdata/FlarePre.csv')
f3 = pd.read_csv('./Rawdata/StabPre.csv')
f4 = pd.read_csv('./Rawdata/BabyPre.csv')


def f(row):
    val = (row['Group_number'] - 1) * 10 + row['Trial_number']
    return val


f1['No.'] = f1.apply(f, axis=1)
f2['No.'] = f2.apply(f, axis=1)
f3['No.'] = f3.apply(f, axis=1)
f4['No.'] = f4.apply(f, axis=1)

f1.to_csv('f1.csv')
f2.to_csv('f2.csv')
f3.to_csv('f3.csv')
f4.to_csv('f4.csv')




def shanchu(x):
    del x["Trial_number"]
    del x["Group_number"]
    del x["Position_number"]
    return x




def get_data(raw_data, component_num):
    result = []
    for idx, row in raw_data.iterrows():
        if row['Component_number'] == component_num:
            result.append(row)
    result2 = []
    if len(result) > 0:
        result2.append(result[0])
    for i in range(1, int(len(result))):
        if result[i]['Time_stamp'] == result2[-1]['Time_stamp']:
            pass
        else:
            result2.append(result[i])
    return result2


def data_pro(data):
    # 1. 得到组件数据
    data_for_182 = get_data(data, 182)
    data_for_176 = get_data(data, 176)

    # 2. 计算position 并 取一行
    data_uniq_for_182 = []

    for i in range(int(len(data_for_182))-1):
        x = data_for_182[i]['Velocity']
        y = data_for_182[i + 1]['Velocity']
        pos = (x + y / 128) / 100
        data_for_182[i]['Position'] = pos
        data_for_182[i + 1]['Position'] = pos
        data_uniq_for_182.append(data_for_182[i])

    # for i in range(100)):
    #     print(list(data_uniq_for_182[i]))


    data_uniq_for_176 = []
    sum_data = 0
    no = 1
    for i in range(1, len(data_for_176) - 1, 1):
        if i == 1:
            data_for_176[0]['Position'] = 0
            data_uniq_for_176.append(data_for_176[0])
        if data_for_176[i]['No.'] == no:
            vel = data_for_176[i]['Velocity']
            pos = (int(vel) - 64) * (float(data_for_176[i]['Time_stamp']) - float(data_for_176[i - 1]['Time_stamp']))
            data_for_176[i]['Position'] = sum_data + pos
            sum_data += pos
            data_uniq_for_176.append(data_for_176[i])
        else:
            sum_data = 0
            vel = data_for_176[i]['Velocity']
            pos = (int(vel) - 64) * (float(data_for_176[i]['Time_stamp']) - float(data_for_176[i - 1]['Time_stamp']))
            data_for_176[i]['Position'] = sum_data + pos
            sum_data += pos
            data_uniq_for_176.append(data_for_176[i])
            no += 1

    #     for i in range(100)):
    #         print(list(data_uniq_for_176[i]))

    # 融合每组数据
    list_group = []
    if len(data_uniq_for_182) > 0:
        for i in range(50):
            no = i + 1
            group = []
            group_182 = []
            group_176 = []
            for j in range(len(data_uniq_for_182)):
                if data_uniq_for_182[j]['No.'] == no:
                    group.append(data_uniq_for_182[j])
                    group_182.append(data_uniq_for_182[j])
            for j in range(len(data_uniq_for_176)):
                if data_uniq_for_176[j]['No.'] == no:
                    group.append(data_uniq_for_176[j])
                    group_176.append(data_uniq_for_176[j])

            group_182.sort(key=lambda x: x['Time_stamp'])
            group_176.sort(key=lambda x: x['Time_stamp'])
            group.sort(key=lambda x: x['Time_stamp'])

            first = group[0]
            start_time = min(group_182[0]['Time_stamp'], group_176[0]['Time_stamp'])
            step = 0.001
            end_182 = len(group_182) - 1
            end_176 = len(group_176) - 1
            end_time = max(group_182[end_182]['Time_stamp'], group_176[end_176]['Time_stamp'])

            result = []
            m = 0
            n = 0
            error = 0.0001
            size = int((end_time - start_time) / step)
            for v in range(size):
                current_time = start_time + step * v
                if math.isclose(group_176[m]['Time_stamp'], current_time) and math.isclose(group_182[n]['Time_stamp'],
                                                                                           current_time):
                    result.append(group_176[m])
                    result.append(group_182[n])
                    if m + 1 < len(group_176):
                        m += 1
                    if n + 1 < len(group_182):
                        n += 1

                elif math.isclose(group_176[m]['Time_stamp'], current_time) and math.isclose(group_182[n]['Time_stamp'],
                                                                                             current_time) == False:
                    result.append(group_176[m])
                    if m + 1 < len(group_176):
                        m += 1
                    temp = copy.copy(group_182[n])
                    temp['Time_stamp'] = current_time
                    result.append(temp)
                elif math.isclose(group_176[m]['Time_stamp'], current_time) == False and math.isclose(
                        group_182[n]['Time_stamp'], current_time):
                    result.append(group_182[n])
                    if n + 1 < len(group_182):
                        n += 1
                    temp = copy.copy(group_176[m])
                    temp['Time_stamp'] = current_time
                    result.append(temp)
                else:
                    temp = copy.copy(group_176[m])
                    temp['Time_stamp'] = current_time
                    result.append(temp)

                    temp = copy.copy(group_182[n])
                    temp['Time_stamp'] = current_time
                    result.append(temp)

            result.sort(key=lambda x: x['Time_stamp'])
            for i in range(len(result)):
                list_group.append(result[i])
            return pd.DataFrame(list_group)
    else:
        # 50 组数据
        only_176_data = []
        for i in range(50):
            no = i + 1
            group = []
            group_176 = []
            for j in range(len(data_uniq_for_176)):
                if data_uniq_for_176[j]['No.'] == no:
                    group.append(data_uniq_for_176[j])
                    group_176.append(data_uniq_for_176[j])

            group_176.sort(key=lambda x: x['Time_stamp'])
           
            # 补齐时间戳
            start_time = group_176[0]['Time_stamp']
            end_time = group_176[len(group_176)-1]['Time_stamp']
            step = 0.001
            index = 0
            
            size = int((end_time - start_time) / step)
            for v in range(size):
                current_time = start_time + step * v
                if math.isclose(current_time,group_176[index]['Time_stamp']) == True:
                    only_176_data.append(group_176[index])
                    t = copy.copy(group_176[index])
                    t['Component_number'] = 182
                    t['Position'] = 1
                    only_176_data.append(t)
                    index += 1
                else:
                    temp = copy.copy(group_176[index-1])
                    temp['Time_stamp'] = current_time
                    only_176_data.append(temp)
                    t = copy.copy(group_176[index-1])
                    t['Component_number'] = 182
                    t['Position'] = 1.0
                    only_176_data.append(t)
        return pd.DataFrame(only_176_data)

f1 = pd.read_csv('f1.csv')
f2 = pd.read_csv('f2.csv')
f3 = pd.read_csv('f3.csv')
f4 = pd.read_csv('f4.csv')
cfChirp = data_pro(f1)
print(cfChirp.head())
cfChirp.to_csv('cfChirp_merge.csv')

cfChirp = data_pro(f2)
print(cfChirp.head())
cfChirp.to_csv('cfFlare_merge.csv')


def gen_data():
    cfChirp = data_pro(f1)
    print(cfChirp.head())
    cfChirp.to_csv('cfChirp_merge.csv')

    cfFlare = data_pro(f2)
    print(cfFlare.head())
    cfFlare.to_csv('cfFlare_merge.csv')

    cfStab = data_pro(f3)
    print(cfStab.head())
    cfStab.to_csv('cfStab_merge.csv')

    cfBaby = data_pro(f4)
    print(cfBaby.head())
    cfBaby.to_csv('cfBaby_merge.csv')

# gen_data()


from sklearn.model_selection import train_test_split
# 划分数据集
def split_data(x,y,test_size):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = test_size)
    return x_train,x_test,y_train,y_test




def gen_test_and_train_data(filename):
    f = pd.read_csv(filename)
    result = []
    label = []
    for idx, row in f.iterrows():
        result.append(row)
        label.append(row['Scratch_type'])
    f_x_train,f_x_test,f_y_train,f_y_test = split_data(result,label,0.25)
    return f_x_train,f_x_test,f_y_train,f_y_test 

def gen_test_and_train_datas():
    f1_x_train,f1_x_test,f1_y_train,f1_y_test  = gen_test_and_train_data('cfChirp_merge.csv')
    print(len(f1_x_train),len(f1_x_test),len(f1_y_train),len(f1_y_test),len(f1_x_test)/len(f1_x_train))
    f2_x_train,f2_x_test,f2_y_train,f2_y_test  = gen_test_and_train_data('cfFlare_merge.csv')
    f3_x_train,f3_x_test,f3_y_train,f3_y_test  = gen_test_and_train_data('cfStab_merge.csv')
    f4_x_train,f4_x_test,f4_y_train,f4_y_test  = gen_test_and_train_data('cfBaby_merge.csv')
    
    x_test = []
    y_test = []
    x_train = []
    y_train = []
    for i in range(len(f1_x_test)):
        x_test.append(f1_x_test[i])
    for i in range(len(f2_x_test)):
        x_test.append(f2_x_test[i])
    for i in range(len(f3_x_test)):
        x_test.append(f3_x_test[i])
    for i in range(len(f4_x_test)):
        x_test.append(f4_x_test[i])

    for i in range(len(f1_y_test)):
        y_test.append(f1_y_test[i])
    for i in range(len(f2_y_test)):
        y_test.append(f2_y_test[i])
    for i in range(len(f3_y_test)):
        y_test.append(f3_y_test[i])
    for i in range(len(f4_y_test)):
        y_test.append(f4_y_test[i])
    
    for i in range(len(f1_x_train)):
        x_train.append(f1_x_train[i])
    for i in range(len(f2_x_train)):
        x_train.append(f2_x_train[i])
    for i in range(len(f3_x_train)):
        x_train.append(f3_x_train[i])
    for i in range(len(f4_x_train)):
        x_train.append(f4_x_train[i])
    

    for i in range(len(f1_y_train)):
        y_train.append(f1_y_train[i])
    for i in range(len(f2_y_train)):
        y_train.append(f2_y_train[i])
    for i in range(len(f3_y_train)):
        y_train.append(f3_y_train[i])
    for i in range(len(f4_y_train)):
        y_train.append(f4_y_train[i])

    return x_test,y_test,x_train,y_train


# x_test,y_test,x_train,y_train = gen_test_and_train_datas()

# print(len(x_test),len(y_test),len(x_train),len(y_train),len(x_test)/len(x_train))
    



