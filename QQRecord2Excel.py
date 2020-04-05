from pandas import Series, DataFrame, ExcelWriter, read_excel
from json import load as j_load
import os
import sys


def line_reader(df):
    file = open(record_path, encoding='UTF-8')

    while True:
        # 读取日器标签
        label = file.readline()
        if not label:
            break
        # 如果读到空行则读取下一行
        if label == '\n':
            file.readline()
        # 读取文本回复
        data = file.readline()
        # 跳过空行读取
        file.readline()
        # 读取QQ号
        d_tail = label[-10: -1]

        tail_str = ''
        for key_ in key_list:
            if d_tail == tail[key_]:
                tail_str = key_
        if tail_str == '':
            tail_str = tail['else']
        # 创建名称——文本Series
        new_series = Series([tail_str, data[:-1]])
        df = df.append(new_series, ignore_index=True)
        print(df.shape)

    return df


def tab_split(df):
    with open(tab_split_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines:
        input_text, target_text = line.split('\t')
        new_series = Series([input_text, target_text])
        df = df.append(new_series, ignore_index=True)
        print(df.shape)
    return df


def sign_tab_split(df):
    with open(sign_tab_split_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        tab_input = ''
    for line in lines:
        tab_split = line.split('\t')
        # 如果重复了input
        if tab_input == tab_split[1]:
            pass
        else:
            # 更新当前input并添加
            tab_input = tab_split[1]
            new_series = Series([tab_split[1], tab_split[2]])
            df = df.append(new_series, ignore_index=True)
        print(df.shape)
    return df


application_path = ''
# 确定存放目录的相对位置
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)

# base_dir = os.path.abspath(os.path.join(application_path, os.path.pardir))
base_dir = application_path

works_dir = os.path.join(base_dir, 'workspace')
split_path = os.path.join(works_dir, 'split.txt')

# 群聊记录地址
record_path = os.path.join(works_dir, 'record.txt')

# excel输入输出地址
raw_excel = os.path.join(works_dir, 'record.xlsx')
excel_out = os.path.join(works_dir, 'train.xlsx')
excel_sort = os.path.join(works_dir, 'sort_record.xlsx')
tab_split_path = os.path.join(works_dir, 'tab_split.txt')
tab_split_out_path = os.path.join(works_dir, 'tab_split.xlsx')
sign_tab_split_path = os.path.join(works_dir, 'sign_tab_split.txt')
sign_tab_split_out_path = os.path.join(works_dir, 'sign_tab_split.xlsx')

# QQ号码后8位作为区分
with open(split_path, 'r', encoding='UTF-8') as f:
    tail = j_load(f)
print(tail)

# 获取字典键值
key_list = []
for key, val in tail.items():
    key_list.append(key)


mode = '0'
# 模式1：将txt文本读取为excel
# 模式2：将excel中标记的元素提取
# 模式3：将对话转换
print('Version: 0.0.3 Alpha')
print('模式1：将原始QQ导出的数据转换为excel表格')
print('模式2：将标记好的excel表格进行筛选')
print('模式3：将筛选后的表格转换为训练数据')
print('模式4：将TAB分隔的数据转换为excel')
print('模式5：将多轮，带标签的TAB分隔的数据转换为excel')
while True:
    if mode == '0':
        mode = input('输入工作模式：')
    elif mode == '1':
        df = DataFrame()
        df = line_reader(df)
        df['label'] = 0
        with ExcelWriter(raw_excel, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='record', encoding='UTF-8')
        mode = '0'

    elif mode == '2':
        df = read_excel(raw_excel)
        filter = df['label'] > 0
        new_df = df[filter]
        with ExcelWriter(excel_sort, engine='xlsxwriter') as writer:
            new_df.to_excel(writer, index=False, sheet_name='record', encoding='UTF-8')
        mode = '0'

    elif mode == '3':
        df = read_excel(excel_sort)
        input_seq = ''
        target_seq = ''
        new_pair = 0
        train_seq = []
        # 对都进来的excel进行行遍历
        for index, row in df.iterrows():
            if row['label'] == 1:
                input_seq = row[1]
            else:
                target_seq = row[1]
                new_pair = 1
            if new_pair:
                train_seq.append([input_seq, target_seq])
                new_pair = 0
        df = DataFrame(train_seq)
        print(df)
        df.columns = ['input', 'output']
        with ExcelWriter(excel_out, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='record', encoding='UTF-8')
        mode = '0'
    elif mode == '4':
        df = DataFrame()
        df = tab_split(df)
        df.columns = ['input', 'output']
        with ExcelWriter(tab_split_out_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='train_data', encoding='UTF-8')

        mode = '0'
    elif mode == '5':
        df = DataFrame()
        df = sign_tab_split(df)
        df.columns = ['input', 'output']
        with ExcelWriter(sign_tab_split_out_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='train_data', encoding='UTF-8')
        mode = '0'
    else:
        break




