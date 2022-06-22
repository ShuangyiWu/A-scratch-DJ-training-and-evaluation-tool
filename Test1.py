from sre_constants import SUCCESS
import pandas as pd
import matplotlib.pyplot as plt
import copy
import math

from pytest import fail

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


def shanchu(x):
    del x["Trial_number"]
    del x["Group_number"]
    del x["Position_number"]
    return x


# f1=shanchu(f1)
# f2=shanchu(f2)
# f3=shanchu(f3)
import numpy as np
from itertools import groupby


def get_data(raw_data, component_num):
    result = []
    for idx, row in raw_data.iterrows():
        if row['Component_number'] == component_num:
            result.append(row)
    result2 = []
    result2.append(result[0])
    for i in range(1, int(len(result))):
        if result[i]['Time_stamp'] == result2[-1]['Time_stamp']:
            pass
        else:
            result2.append(result[i])
    return result2


def print_data(data_for_182, i):
    print('Component_number', 'Position_number', 'Time_stamp')
    print(data_for_182[i]['Component_number'], data_for_182[i]['Position_number'], data_for_182[i]['Time_stamp'])


# 3. 取类别每组的最大time stamp
def get_each_type_of_group_max_time_stamp(data):
    max_time_stamp_by_group = []
    time_stamp_group = groupby(data, key=lambda x: x['No.'])
    for key, group in time_stamp_group:
        g = list(group)
        max_value = max(g, key=lambda item: item[3])
        max_time_stamp_by_group.append(max_value['Time_stamp'])
    return max_time_stamp_by_group


def data_pro(data):
    # 1. 得到组件数据
    data_for_182 = get_data(data, 182)
    data_for_176 = get_data(data, 176)

    # 2. 计算position 并 取一行
    data_uniq_for_182 = []

    for i in range(int(len(data_for_182)) - 1):
        x = data_for_182[i]['Velocity']
        y = data_for_182[i + 1]['Velocity']
        pos = (x + y / 128) / 100
        data_for_182[i]['Position'] = pos
        data_for_182[i + 1]['Position'] = pos
        data_uniq_for_182.append(data_for_182[i])

    #     for i in range(len(data_uniq_for_182)):
    #         print(list(data_uniq_for_182[i]))
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

    # for i in range(len(data_uniq_for_176)):
    #     print(list(data_uniq_for_176[i]))
    # 融合每组数据
    list_group = []
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
    #         df = pd.DataFrame(result)
    #         print(df.head())

    return pd.DataFrame(list_group)


def data_pro_for_baby(data):
    # 1. 得到组件数据
    data_for_176 = get_data(data, 176)

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
        end_time = group_176[len(group_176) - 1]['Time_stamp']
        step = 0.001
        index = 0

        size = int((end_time - start_time) / step)
        for v in range(size):
            current_time = start_time + step * v
            if math.isclose(current_time, group_176[index]['Time_stamp']) == True:
                only_176_data.append(group_176[index])
                t = copy.copy(group_176[index])
                t['Component_number'] = 182
                t['Position'] = 1
                only_176_data.append(t)
                index += 1
            else:
                temp = copy.copy(group_176[index - 1])
                temp['Time_stamp'] = current_time
                only_176_data.append(temp)
                t = copy.copy(group_176[index - 1])
                t['Component_number'] = 182
                t['Position'] = 1.0
                only_176_data.append(t)
    return pd.DataFrame(only_176_data)


def gen_data():
    cfChirp = data_pro(f1)
    cfChirp.to_csv('cfChirp_merge.csv')
    cfFlare = data_pro(f2)
    cfFlare.to_csv('cfFlare_merge.csv')
    cfStab = data_pro(f3)
    # print(cfStab.head())
    cfStab.to_csv('cfStab_merge.csv')
    cfBaby = data_pro_for_baby(f4)
    # print(cfBaby.head())
    cfBaby.to_csv('cfBaby_merge.csv')


# gen_data()

from sklearn.model_selection import train_test_split


# 划分数据集
def split_data(x, y, test_size):
    # x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = test_size)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(len(x)):  # 1-40
        if x[i]["No."] <= 10:
            x_train.append(x[i])
            y_train.append(y[i])
        elif x[i]["No."] >= 41:
            x_test.append(x[i])
            y_test.append(y[i])

    return x_train, x_test, y_train, y_test


def gen_test_and_train_data(filename):
    f = pd.read_csv(filename)
    result = []
    label = []
    for idx, row in f.iterrows():
        result.append(row)
        label.append(row['Scratch_type'])
    f_x_train, f_x_test, f_y_train, f_y_test = split_data(result, label, 0.25)

    return f_x_train, f_x_test, f_y_train, f_y_test


def gen_test_and_train_datas():
    f1_x_train, f1_x_test, f1_y_train, f1_y_test = gen_test_and_train_data('cfChirp_merge.csv')

    f2_x_train, f2_x_test, f2_y_train, f2_y_test = gen_test_and_train_data('cfFlare_merge.csv')

    f3_x_train, f3_x_test, f3_y_train, f3_y_test = gen_test_and_train_data('cfStab_merge.csv')
    f4_x_train, f4_x_test, f4_y_train, f4_y_test = gen_test_and_train_data('cfBaby_merge.csv')

    pd.DataFrame(f1_x_test).to_csv('f1_x.csv')
    pd.DataFrame(f2_x_test).to_csv('f2_x.csv')
    pd.DataFrame(f3_x_test).to_csv('f3_x.csv')
    pd.DataFrame(f4_x_test).to_csv('f4_x.csv')

    x_test = []
    y_test = []
    x_train = []
    y_train = []

    count = 1
    for j in range(41, 51):
        for i in range(len(f1_x_test)):
            if f1_x_test[i]['No.'] == j:
                temp = copy.copy(f1_x_test[i])
                temp['No.'] = count
                x_test.append(temp)
        count += 1

    pd.DataFrame(x_test).to_csv("1.csv")

    for j in range(41, 51):
        for i in range(len(f2_x_test)):
            if f2_x_test[i]['No.'] == j:
                temp = copy.copy(f2_x_test[i])
                temp['No.'] = count
                x_test.append(temp)
        count += 1

    pd.DataFrame(x_test).to_csv("2.csv")

    for j in range(41, 51):
        for i in range(len(f3_x_test)):
            if f3_x_test[i]['No.'] == j:
                temp = copy.copy(f3_x_test[i])
                temp['No.'] = count
                x_test.append(temp)
        count += 1
    pd.DataFrame(x_test).to_csv("3.csv")

    for j in range(41, 51):
        for i in range(len(f4_x_test)):
            if f4_x_test[i]['No.'] == j:
                temp = copy.copy(f4_x_test[i])
                temp['No.'] = count
                x_test.append(temp)
        count += 1
    pd.DataFrame(x_test).to_csv("4.csv")

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

    count = 11
    for j in range(1, 11):
        for i in range(len(f2_x_train)):
            if f2_x_train[i]['No.'] == j:
                temp = copy.copy(f2_x_train[i])
                temp['No.'] = count
                x_train.append(temp)
        count += 1

    for j in range(1, 11):
        for i in range(len(f3_x_train)):
            if f3_x_train[i]['No.'] == j:
                temp = copy.copy(f3_x_train[i])
                temp['No.'] = count
                x_train.append(temp)
        count += 1

    for j in range(1, 11):
        for i in range(len(f4_x_train)):
            if f4_x_train[i]['No.'] == j:
                temp = copy.copy(f4_x_train[i])
                temp['No.'] = count
                x_train.append(temp)
        count += 1

    for i in range(len(f1_y_train)):
        y_train.append(f1_y_train[i])
    for i in range(len(f2_y_train)):
        y_train.append(f2_y_train[i])
    for i in range(len(f3_y_train)):
        y_train.append(f3_y_train[i])
    for i in range(len(f4_y_train)):
        y_train.append(f4_y_train[i])

    return x_test, y_test, x_train, y_train


x_test, y_test, x_train, y_train = gen_test_and_train_datas()
test_x = pd.DataFrame(x_test)
test_x.to_csv('test_x.csv')

test_y = pd.DataFrame(y_test)
test_y.to_csv('test_y.csv')
train_x = pd.DataFrame(x_train)
train_x.to_csv('train_x.csv')
train_y = pd.DataFrame(y_train)
train_y.to_csv('train_y.csv')
print(len(x_test), len(y_test), len(x_train), len(y_train), len(x_test) / (len(x_train) + len(x_test)))

print("split data done!")


# 计算test数据集中的对应组别的正确label
def test_right_label(file, test_group_no):
    test = pd.read_csv(file)
    test_right = [0, 0, 0, 0]

    for idx, row in test.iterrows():
        if row['No.'] == test_group_no:
            if row['Scratch_type'] == 'Chirp':
                test_right[0] += 1
            elif row['Scratch_type'] == 'Flare':
                test_right[1] += 1
            elif row['Scratch_type'] == 'Stab':
                test_right[2] += 1
            elif row['Scratch_type'] == 'Baby_Scratch':
                test_right[3] += 1
    max_value = 0
    label = None

    for i in range(len(test_right)):
        if test_right[i] > max_value:
            max_value = test_right[i]
            if i == 0:
                label = 'Chirp'
                # print("Chirp:",test_right[i])
            elif i == 1:
                label = 'Flare'
                # print("Flare:",test_right[i])
            elif i == 2:
                label = 'Stab'
                # print("Stab:",test_right[i])
            elif i == 3:
                label = 'Baby'
                # print("Baby:",test_right[i])
    # print('=>',label)
    return label


import pandas as pd
import tslearn.metrics as ts


def align_gen_hh(test_data, no1, train_data, no2):
    # sub_df1 = test_data[['Time_stamp', 'Position', 'No.']][test_data['No.'] == no1][
    #     test_data['Component_number'] == 182]
    sub_df1 = test_data.loc[((test_data["No."] == no1) & (test_data["Component_number"] == 182)), ["Time_stamp", "Position", "No."]]
    # sub_df1 = test_data['Time_stamp', 'Position', 'No.']
    # sub_df1 = sub_df1['No.' == no1]
    # sub_df1 = sub_df1['Component_number' == 182]
    pos1 = sub_df1.iloc[:, 1].values
    pos1 = pos1 - pos1[0]
    # 让pos都从0开始，为了消除之前第二组会从第一组的最后一个数开始算的影响
    # sub_df2 = train_data[['Time_stamp', 'Position', 'No.']][train_data['No.'] == no2][
    #     train_data['Component_number'] == 182]
    sub_df2 = train_data.loc[
        ((train_data["No."] == no2) & (train_data["Component_number"] == 182)), ["Time_stamp", "Position", "No."]]
    # sub_df2 = train_data['Time_stamp', 'Position', 'No.']
    # sub_df2 = sub_df2['No.' == no2]
    # sub_df2 = sub_df2['Component_number' == 182]
    pos2 = sub_df2.iloc[:, 1].values
    pos2 = pos2 - pos2[0]
    align, dist = ts.dtw_path(pos1, pos2)
    # print(dist)
    # 这个是求相似程度的函数，align是怎么把两组数据变形，dist是相似程度的指标
    # https://tslearn.readthedocs.io/en/latest/gen_modules/metrics/tslearn.metrics.dtw_path.html
    return align, dist


def align_project_hh(align, test_data, no1, train_data, no2, comp):
    # sub_df1 = test_data[['Time_stamp', 'Position', 'No.']][test_data['No.'] == no1][
    #     test_data['Component_number'] == comp]
    sub_df1 = test_data.loc[
        ((test_data["No."] == no1) & (test_data["Component_number"] == comp)), ["Time_stamp", "Position", "No."]]
    # sub_df1 = test_data['Time_stamp', 'Position', 'No.']
    # sub_df1 = sub_df1['No.' == no1]
    # sub_df1 = sub_df1['Component_number' == comp]
    pos1 = sub_df1.iloc[:, 1].values
    pos1 = pos1 - pos1[0]

    # sub_df2 = train_data[['Time_stamp', 'Position', 'No.']][train_data['No.'] == no2][
    #     train_data['Component_number'] == comp]
    sub_df2 = train_data.loc[
        ((train_data["No."] == no2) & (train_data["Component_number"] == comp)), ["Time_stamp", "Position", "No."]]
    # sub_df2 = train_data['Time_stamp', 'Position', 'No.']
    # sub_df2 = sub_df2['No.' == no2]
    # sub_df2 = sub_df2['Component_number' == comp]
    pos2 = sub_df2.iloc[:, 1].values
    pos2 = pos2 - pos2[0]
    alignedp1 = []
    alignedp2 = []
    for i in range(0, len(align)):
        alignedp1.append(pos1[align[i][0]])
        alignedp2.append(pos2[align[i][1]])
    return pos1, pos2, alignedp1, alignedp2
    # 把这个变形方法应用到同一个test和train的176的数据上
    # align里面长这样：（0，0），（0，1）就表示第一项和第一项对应，test的第一项和train的第二项也对应，所以就把test的第一项延长
    # 可以看一下下面的图，变形之后两个pos会变成一样长，而且都比原来长


test_data = pd.read_csv('test_x.csv')
train_data = pd.read_csv('train_x.csv')


# for i in range(1,51):
#     label = test_right_label(i)


def test(test_group_no):
    result = []
    labels = []
    for i in range(1, 41):
        print('train:', i,end = '')
        a, dist = align_gen_hh(test_data, test_group_no, train_data, i)
        # print(a)
        pos1, pos2, p1, p2 = align_project_hh(a, test_data, test_group_no, train_data, i, 176)
        dist = ts.dtw(p1, p2)
        result.append(dist)
        label = test_right_label('train_x.csv', i)
        labels.append(label)

    print(labels)
    min_dist = min(result)
    index_min = np.argmin(result)

    pre_label = labels[index_min]
    print("pre --> ", 'dist:', min_dist, 'label:', labels[index_min])

    act_label = test_right_label('test_x.csv', test_group_no)
    print('act:--> ', 'label:', act_label)

    return pre_label, act_label


ok = 0
Failed = 0
acts = []
pres = []
for i in range(1, 41):
    print('test:', i)
    pre, act = test(i)
    pres.append(pre)
    acts.append(act)
    if pre == act:
        ok += 1
    else:
        Failed += 1

print(ok, Failed, ok / (ok + Failed))
print('acts:', acts)
print('pres:', pres)
# precision    recall  f1-score   support
from sklearn.metrics import classification_report

y_true = acts
y_pred = pres
target_names = ['Chirp', 'Flare', 'Stab', 'Baby']
print(classification_report(y_true, y_pred, target_names=target_names))

from sklearn.metrics import confusion_matrix

y_true = acts
y_pred = pres
print(confusion_matrix(y_true, y_pred))

