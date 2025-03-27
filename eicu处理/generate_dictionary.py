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
STRING_TUPLE_DIR = RESULT_ROOT_DIR + 'string_tuple/'
IDX_DIR = RESULT_ROOT_DIR + 'index/'

# 确保目录存在
os.makedirs(RESULT_ROOT_DIR, exist_ok=True)
os.makedirs(TUPLE_DIR, exist_ok=True)
os.makedirs(IDX_DIR, exist_ok=True)
os.makedirs(STRING_TUPLE_DIR, exist_ok=True)

V_FREQ = 'value_frequency'
FREQ = 'total_frequency'

idx_cols = ['code', 'code_type', V_FREQ, FREQ, 'source_table', 'unit_of_measurement', 'with_value']


def _output_dict(table: pd.DataFrame, tablename: str):
    '''
    Parameters:
    ----
        table:
            The dictionary to output
        tablename:
            tablename of the input/output file

    Returns:
    ----
        No return
    '''

    table = table.groupby(['code', 'code_type']).count()
    print('unknown freq', int(table.loc['<unk>', FREQ]) if '<unk>' in table.index else 0)
    if '<unk>' in table.index:
        table.drop(['<unk>'], inplace=True)

    table.sort_values(FREQ, inplace=True)
    table[V_FREQ] = 0
    table['source_table'] = tablename
    table['unit_of_measurement'] = ''
    table['with_value'] = 0
    table.to_csv(IDX_DIR + tablename + '_dict.dict', columns=idx_cols[2:], index_label=['code', 'code_type'])

    table.reset_index(inplace=True)
    all_type = table['code_type'].unique()
    for ctype in all_type:
        temp_table = table.loc[table['code_type'] == ctype]
        print('type', 'code_num', 'final', 'mean', 'median', 'max', 'min', sep='\t')
        print(ctype, temp_table.shape[0], temp_table[FREQ].sum(), int(temp_table[FREQ].mean()),
              temp_table[FREQ].median(), temp_table[FREQ].max(), temp_table[FREQ].min(), sep='\t')
        print('')


def generate_diagnosis_dict(tablename='diagnosis'):
    '''
    Generate dictionary for diagnosis.csv from eICU

    Parameters:
    ----
        tablename:
            tablename of the file

    Returns:
    ----
        No return
    '''

    print('\ngenerating dict of', tablename)

    path = EICU_DIR + tablename + '.csv'
    setting = {'diagnosisstring': str}
    table = pd.read_csv(path, usecols=setting.keys(), dtype=setting, index_col=False)

    # 重命名列并添加代码类型
    table.rename({'diagnosisstring': 'code'}, axis=1, inplace=True)
    print('number of unique diagnoses:',
          table['code'].drop_duplicates(keep='first', inplace=False).shape[0])

    table.loc[:, 'total_frequency'] = 1
    table.loc[:, 'code_type'] = 'eicu_diagnosis'

    _output_dict(table, tablename)


def generate_lab_dict(tablename='lab'):
    '''
    Generate dictionary for lab.csv from eICU

    Parameters:
    ----
        tablename:
            tablename of the file

    Returns:
    ----
        No return
    '''

    print('\ngenerating dict of', tablename)

    # 记录频率的字典
    freq_record = {}

    # 记录值是否变化的字典
    value_record = {}

    # 加载源表
    src_path = EICU_DIR + tablename + '.csv'
    setting = {'labname': str, 'labresult': float, 'labmeasurenamesystem': str}

    # 使用 chunking 处理大文件
    with pd.read_csv(src_path, usecols=setting.keys(), index_col=False,
                     dtype=setting, chunksize=30000000) as reader:
        for i, chunk in enumerate(reader):
            for labname, labresult, unit in tqdm(chunk.itertuples(False), total=chunk.shape[0]):
                # 0:labname, 1:labresult, 2:unit

                if labname not in freq_record:
                    freq_record[labname] = {}
                    freq_record[labname]['value'] = 0
                    freq_record[labname]['total'] = 0

                # 计数
                freq_record[labname]['total'] += 1

                # 标准化单位
                unit = _normalize_unit(unit)

                if not pd.isna(labresult):
                    freq_record[labname]['value'] += 1

                    # 检查代码是否总是出现相同的值
                    if labname in value_record:
                        if value_record[labname] != None and value_record[labname] != labresult:
                            value_record[labname] = None
                    else:
                        value_record[labname] = labresult

    table = []
    for k, v in freq_record.items():
        if v['total'] >= 1:
            if v['value'] >= 1 and k in value_record and value_record[k] == None:
                table.append([k, v['value'], v['total'], 1])
            else:
                table.append([k, 0, v['total'], 0])

    table = pd.DataFrame(table,
                         columns=['code', V_FREQ, FREQ, 'with_value']).sort_values('code')

    # 填充其他列
    table['unit_of_measurement'] = ''  # 可以根据需要从数据中提取
    table['source_table'] = tablename
    table['code_type'] = 'eicu_lab'

    table = table.loc[:, idx_cols]
    table.sort_values(['with_value', FREQ], inplace=True)
    table.to_csv(IDX_DIR + tablename + '_dict.dict', index=False)


def generate_medication_dict(tablename='medication'):
    '''
    Generate dictionary for medication.csv from eICU

    Parameters:
    ----
        tablename:
            tablename of the file

    Returns:
    ----
        No return
    '''

    print('\ngenerating dict of', tablename)

    path = EICU_DIR + tablename + '.csv'
    setting = {'drugname': str}
    table = pd.read_csv(path, usecols=setting.keys(), dtype=setting, index_col=False)

    # 重命名列并添加代码类型
    table.rename({'drugname': 'code'}, axis=1, inplace=True)
    print('number of unique medications:',
          table['code'].drop_duplicates(keep='first', inplace=False).shape[0])

    table.loc[:, 'total_frequency'] = 1
    table.loc[:, 'code_type'] = 'eicu_medication'

    _output_dict(table, tablename)


def generate_infusiondrug_dict(tablename='infusiondrug'):
    '''
    Generate dictionary for infusiondrug.csv from eICU

    Parameters:
    ----
        tablename:
            tablename of the file

    Returns:
    ----
        No return
    '''

    print('\ngenerating dict of', tablename)

    # 记录频率的字典
    freq_record = {}

    # 加载源表
    src_path = EICU_DIR + tablename + '.csv'
    setting = {'drugname': str, 'infusionrate': str}

    # 使用 chunking 处理大文件
    with pd.read_csv(src_path, usecols=setting.keys(), index_col=False,
                     dtype=setting, chunksize=30000000) as reader:
        for i, chunk in enumerate(reader):
            for drugname, infusionrate in tqdm(chunk.itertuples(False), total=chunk.shape[0]):
                # 0:drugname, 1:infusionrate

                if drugname not in freq_record:
                    freq_record[drugname] = {}
                    freq_record[drugname]['value'] = 0
                    freq_record[drugname]['total'] = 0

                # 计数
                freq_record[drugname]['total'] += 1

                if not pd.isna(infusionrate) and infusionrate.strip() != '':
                    freq_record[drugname]['value'] += 1

    table = []
    for k, v in freq_record.items():
        if v['total'] >= 1:
            if v['value'] >= 1:
                table.append([k, v['value'], v['total'], 1])
            else:
                table.append([k, 0, v['total'], 0])

    table = pd.DataFrame(table,
                         columns=['code', V_FREQ, FREQ, 'with_value']).sort_values('code')

    # 填充其他列
    table['unit_of_measurement'] = ''
    table.loc[table['with_value'] == 1, 'unit_of_measurement'] = 'rate'
    table['source_table'] = tablename
    table['code_type'] = 'eicu_infusiondrug'

    table = table.loc[:, idx_cols]
    table.sort_values(['with_value', FREQ], inplace=True)
    table.to_csv(IDX_DIR + tablename + '_dict.dict', index=False)


def _normalize_unit(unit):
    '''
    normalize unit of measurement
    '''

    if pd.isna(unit):
        return 'nan'
    else:
        unit = unit.lower().strip()
        if unit == '' or unit == 'none' or unit == 'nan':
            return 'nan'
        else:
            return unit


def merge_dict(out_path):
    '''
    Merge dictionaries of all tables together.

    Parameters:
    ----
        out_path: filepath to output the merged dictionary

    Returns:
    ----
        No return
    '''

    print('Merging all dictionaries together...')

    table = pd.DataFrame(columns=idx_cols)

    for tablename in os.listdir(IDX_DIR):
        if '.dict' in tablename:
            path = IDX_DIR + tablename
            temp = pd.read_csv(path, dtype={'code': 'str'}, index_col=False)
            table = pd.concat((table, temp), ignore_index=True)

    # 排序所有条目
    table.sort_values(['code_type', 'with_value', 'total_frequency'], inplace=True, ignore_index=True)
    table.index += 1

    # 统计
    value_table = table.loc[table['with_value'] == 1]
    if not value_table.empty:
        print('ratio:', value_table[V_FREQ].divide(value_table[FREQ]).mean())

    print('total value:', table[V_FREQ].sum(), 'total freq', table[FREQ].sum())

    print('all:')
    print('code_num', 'final', 'mean', 'median', 'max', 'min', sep='\t')
    print(table.shape[0], table[FREQ].sum(), int(table[FREQ].mean()),
          table[FREQ].median(), table[FREQ].max(), table[FREQ].min(), sep='\t')
    print('-' * 20)

    if not value_table.empty:
        print('value:')
        print('code_num', 'final', 'mean', 'median', 'max', 'min', sep='\t')
        print(value_table.shape[0], value_table[V_FREQ].sum(), int(value_table[V_FREQ].mean()),
              value_table[V_FREQ].median(), value_table[V_FREQ].max(), value_table[V_FREQ].min(), sep='\t')
        print('-' * 20)

    # 输出字典
    table.to_csv(out_path, index_label='index')


def main():
    # 为每个 eICU 表生成字典
    generate_diagnosis_dict('diagnosis')
    generate_lab_dict('lab')
    generate_medication_dict('medication')
    generate_infusiondrug_dict('infusiondrug')

    # 合并所有字典
    merge_dict(IDX_DIR + 'code_dict.csv')


if __name__ == '__main__':
    main()