import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

# 修改为适配 eICU 的设置
EICU_DIR = 'eicu/'  # eICU 数据文件目录
RESULT_ROOT_DIR = 'records/'  # 结果输出目录
TUPLE_DIR = RESULT_ROOT_DIR + 'tuple/'
IDX_DIR = RESULT_ROOT_DIR + 'index/'

# 确保目录存在
os.makedirs(RESULT_ROOT_DIR, exist_ok=True)
os.makedirs(TUPLE_DIR, exist_ok=True)
os.makedirs(IDX_DIR, exist_ok=True)

def generate_diagnosis_tuples(tablename='diagnosis'):
    '''
    为eICU诊断表生成元组

    Parameters:
    ----
        tablename: 表名

    Returns:
    ----
        无返回值
    '''
    print('\ngenerating tuples of', tablename)

    # 加载代码字典
    code2idx = _load_code_dict(tablename)

    # 加载诊断表
    path = EICU_DIR + tablename + '.csv'
    cols = ['patientunitstayid', 'diagnosisoffset', 'diagnosisstring']
    setting = {'patientunitstayid': str, 'diagnosisoffset': int, 'diagnosisstring': str}
    table = pd.read_csv(path, usecols=setting.keys(), dtype=setting, index_col=False)

    # 统一列名和顺序
    table.rename({'patientunitstayid': 'subject_id',
                  'diagnosisstring': 'code',
                  'diagnosisoffset': 'time'}, axis=1, inplace=True)
    table = table.loc[:, ['subject_id', 'code', 'time']]

    # 过滤不在字典中的代码
    table = table.loc[table['code'].isin(code2idx), :]
    table.loc[:, 'code'] = table.loc[:, 'code'].apply(code2idx.get)

    # 输出元组
    _table2tuples(table, TUPLE_DIR + tablename, has_value=False)


def generate_lab_tuples(tablename='lab'):
    '''
    为eICU实验室检查表生成元组

    Parameters:
    ----
        tablename: 表名

    Returns:
    ----
        无返回值
    '''
    print('\ngenerating tuples of', tablename)

    # 加载代码字典
    dic = pd.read_csv(IDX_DIR + 'code_dict.csv', dtype={'code': str, 'code_type': str, 'with_value': str})
    dic = dic.loc[dic['source_table'] == tablename, ['index', 'code', 'code_type', 'with_value']]
    code2idx = {row['code']: str(row['index']) for _, row in dic.iterrows()}
    print('code dict size:', len(code2idx))

    code_with_value = set(dic.loc[dic['with_value'] == '1', 'code'])

    # 加载患者字典
    origin_patients = _load_patients()

    # 分块加载lab表
    src_path = EICU_DIR + tablename + '.csv'
    setting = {'patientunitstayid': str, 'labresultoffset': int, 'labname': str, 'labresult': float}

    with pd.read_csv(src_path, usecols=setting.keys(), index_col=False,
                     chunksize=30000000, dtype=setting) as reader:
        for i, chunk in enumerate(reader):
            patients = {i: [] for i in origin_patients}

            chunk = chunk.loc[:, ['patientunitstayid', 'labresultoffset', 'labname', 'labresult']]
            for pid, offset, labname, labresult in tqdm(chunk.itertuples(False), total=chunk.shape[0]):
                # 过滤不在字典中的代码
                if labname not in code2idx:
                    continue

                # 创建元组: [admission_id, time, code, value]
                tuple = ['', str(offset), code2idx[labname], '']

                if labname in code_with_value:  # 带值的代码
                    if not pd.isna(labresult):
                        tuple[3] = str(labresult)
                    else:
                        tuple[3] = '_MISSING'
                else:
                    tuple[3] = 'NaN'

                # 添加到患者记录
                patients[pid].append(tuple)

            # 输出元组
            _value_table2tuples(patients, TUPLE_DIR + tablename + str(i))


def generate_medication_tuples(tablename='medication'):
    '''
    为eICU药物表生成元组

    Parameters:
    ----
        tablename: 表名

    Returns:
    ----
        无返回值
    '''
    print('\ngenerating tuples of', tablename)

    # 加载代码字典
    code2idx = _load_code_dict(tablename)

    # 加载药物表
    path = EICU_DIR + tablename + '.csv'
    cols = ['patientunitstayid', 'drugstartoffset', 'drugname']
    setting = {'patientunitstayid': str, 'drugstartoffset': int, 'drugname': str}
    table = pd.read_csv(path, usecols=setting.keys(), dtype=setting, index_col=False)

    # 统一列名和顺序
    table.rename({'patientunitstayid': 'subject_id',
                  'drugname': 'code',
                  'drugstartoffset': 'time'}, axis=1, inplace=True)
    table = table.loc[:, ['subject_id', 'code', 'time']]

    # 过滤不在字典中的代码
    table = table.loc[table['code'].isin(code2idx), :]
    table.loc[:, 'code'] = table.loc[:, 'code'].apply(code2idx.get)

    # 输出元组
    _table2tuples(table, TUPLE_DIR + tablename, has_value=False)


def generate_infusiondrug_tuples(tablename='infusiondrug'):
    '''
    为eICU输液药物表生成元组

    Parameters:
    ----
        tablename: 表名

    Returns:
    ----
        无返回值
    '''
    print('\ngenerating tuples of', tablename)

    # 加载代码字典
    dic = pd.read_csv(IDX_DIR + 'code_dict.csv', dtype={'code': str, 'code_type': str, 'with_value': str})
    dic = dic.loc[dic['source_table'] == tablename, ['index', 'code', 'code_type', 'with_value']]
    code2idx = {row['code']: str(row['index']) for _, row in dic.iterrows()}
    print('code dict size:', len(code2idx))

    code_with_value = set(dic.loc[dic['with_value'] == '1', 'code'])

    # 加载患者字典
    origin_patients = _load_patients()

    # 分块加载infusiondrug表
    src_path = EICU_DIR + tablename + '.csv'
    setting = {'patientunitstayid': str, 'infusionoffset': int, 'drugname': str, 'infusionrate': str}

    with pd.read_csv(src_path, usecols=setting.keys(), index_col=False,
                     chunksize=30000000, dtype=setting) as reader:
        for i, chunk in enumerate(reader):
            patients = {i: [] for i in origin_patients}

            chunk = chunk.loc[:, ['patientunitstayid', 'infusionoffset', 'drugname', 'infusionrate']]
            for pid, offset, drugname, infusionrate in tqdm(chunk.itertuples(False), total=chunk.shape[0]):
                # 过滤不在字典中的代码
                if drugname not in code2idx:
                    continue

                # 创建元组: [admission_id, time, code, value]
                tuple = ['', str(offset), code2idx[drugname], '']

                if drugname in code_with_value and not pd.isna(infusionrate) and infusionrate.strip() != '':
                    tuple[3] = infusionrate.replace(',', '/')
                else:
                    tuple[3] = 'NaN'

                # 添加到患者记录
                patients[pid].append(tuple)

            # 输出元组
            _value_table2tuples(patients, TUPLE_DIR + tablename + str(i))


def _load_code_dict(tablename):
    '''
    加载代码字典，使用 index 列作为 code 的值
    '''
    dic = pd.read_csv(IDX_DIR + 'code_dict.csv', dtype={'code': str, 'code_type': str, 'with_value': str})
    dic = dic.loc[dic['source_table'] == tablename, ['index', 'code', 'code_type', 'with_value']]
    code2idx = {row['code']: str(row['index']) for _, row in dic.iterrows()}
    print('code dict size:', len(code2idx))
    return code2idx


def _table2tuples(table, oFile, has_value=False):
    '''
    将pandas.DataFrame表转换为一批元组并输出

    Parameters:
    ----
        table: 要输出为元组的表
        oFile: 输出文件的路径
        has_value: 是否包含 value 列（对于 diagnosis 和 medication 表，设为 False）

    Returns:
    ----
        无返回值
    '''
    patients = _load_patients()

    for p, c, t in tqdm(table.itertuples(False), total=table.shape[0]):
        if p in patients:
            if has_value:
                patients[p].append(('', str(t), c, ''))  # value 会在调用时处理
            else:
                patients[p].append(('', str(t), c, 'NaN'))  # 非 lab 表，value 为 NaN

    with open(oFile + '.tri', 'w', encoding='utf8') as f:
        for id, info in patients.items():
            if len(info) > 0:  # 只写入有记录的患者
                f.write(str(id) + '\n')
                for l in info:
                    f.write(','.join(l) + '\n')
                f.write('\n')


def _value_table2tuples(patients, oFile):
    '''
    输出带值的表的元组

    Parameters:
    ----
        patients: 要输出的元组
        oFile: 输出文件的路径

    Returns:
    ----
        无返回值
    '''
    with open(oFile + ".tri", 'w', encoding='utf8') as f:
        for id, info in patients.items():
            if len(info) > 0:  # 只写入有记录的患者
                f.write(str(id) + '\n')
                for l in info:
                    l[3] = str(l[3]).replace(',', '/')
                    f.write(','.join(l) + '\n')
                f.write('\n')


def _load_patients():
    '''
    加载所有患者ID
    '''
    patients = pd.read_csv(EICU_DIR + 'patient.csv', usecols=['patientunitstayid'], dtype='str')
    patients = {i: [] for i in patients['patientunitstayid']}
    return patients


def merge_tuples_simple(src_dir, cols, out_path):
    '''
    使用更简单的方法合并所有表的元组

    Parameters:
    ----
        src_dir: 元组的源目录
        cols: 输出文件的列名
        out_path: 输出合并元组的文件路径

    Returns:
    ----
        无返回值
    '''
    print("\nMerging tuples in {}".format(src_dir))

    # 检查目录中是否有.tri文件
    tri_files = [i for i in os.listdir(src_dir) if '.tri' in i]
    if not tri_files:
        print(f"No .tri files found in {src_dir}")
        return

    # 创建患者字典，用于存储所有患者的所有元组
    all_patients = {}

    # 从每个.tri文件中读取元组
    for tri_file in tri_files:
        print(f"Processing {tri_file}...")

        with open(src_dir + tri_file, 'r', encoding='utf8') as f:
            current_patient = None
            for line in f:
                line = line.strip()

                if not line:  # 空行表示一个患者的记录结束
                    current_patient = None
                    continue

                if current_patient is None:  # 当前行是患者ID
                    current_patient = line
                    if current_patient not in all_patients:
                        all_patients[current_patient] = []
                else:  # 当前行是一条元组记录
                    tuple_data = line.split(',')
                    all_patients[current_patient].append(tuple_data)

    # 按患者ID顺序写入输出文件
    with open(out_path, 'w', encoding='utf8') as f:
        # 写入表头
        f.write(','.join(cols) + '\n')

        # 对患者ID进行排序，确保顺序一致
        for patient_id in sorted(all_patients.keys()):
            tuples = all_patients[patient_id]

            # 如果患者没有元组，则跳过
            if not tuples:
                continue

            # 按时间排序
            tuples.sort(key=lambda x: int(x[1]))  # 确保时间是数值型排序

            # 写入每条元组记录
            for tuple_data in tuples:
                f.write(patient_id + ',' + ','.join(tuple_data) + '\n')

    print(f"Merged {len(all_patients)} patients' data to {out_path}")


def main():
    # 为每个eICU表生成元组
    generate_diagnosis_tuples('diagnosis')
    generate_lab_tuples('lab')
    generate_medication_tuples('medication')
    generate_infusiondrug_tuples('infusiondrug')

    # 使用简化的合并函数
    cols = ['patient_id', 'admission_id', 'time', 'code', 'value']
    merge_tuples_simple(TUPLE_DIR, cols, RESULT_ROOT_DIR + 'tuples.csv')


if __name__ == '__main__':
    main()