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


def generate_patient_dict(tuple_path, out_path):
    '''
    生成包含患者个人信息的患者字典

    Parameters:
    ----
        tuple_path: 元组文件路径
        out_path: 输出路径

    Returns:
    ----
        None
    '''

    print('===================================')
    print('开始加载并处理患者数据')

    # 先获取有医疗记录的患者ID列表
    recorded_patients = _get_patients_with_records(tuple_path)
    print(f'有医疗记录的患者总数: {len(recorded_patients)}')

    # 将有记录的患者ID转换为DataFrame
    recorded_df = pd.DataFrame({'patientunitstayid': list(recorded_patients)})

    # 逐块读取患者表，避免一次加载全部数据
    chunks = []
    for chunk in pd.read_csv(EICU_DIR + 'patient.csv',
                             dtype={'patientunitstayid': 'str'},
                             chunksize=50000):
        # 检查重复的patientunitstayid并去重
        if chunk.duplicated(subset=['patientunitstayid']).any():
            print(f"发现重复的patientunitstayid，保留第一条记录")
            chunk = chunk.drop_duplicates(subset=['patientunitstayid'], keep='first')

        # 只保留有记录的患者
        chunk = pd.merge(recorded_df, chunk, on='patientunitstayid', how='inner')
        chunks.append(chunk)

    # 如果没有数据，报错并退出
    if not chunks:
        raise ValueError("患者表为空或没有匹配的患者")

    # 合并所有数据块
    patients = pd.concat(chunks, ignore_index=True)

    # 确保没有重复ID
    if patients.duplicated(subset=['patientunitstayid']).any():
        print("警告: 合并后仍有重复ID，进行最终去重")
        patients = patients.drop_duplicates(subset=['patientunitstayid'], keep='first')

    print(f'最终患者数量: {patients.shape[0]}')

    # 选择需要的列，基于表结构
    available_columns = list(patients.columns)
    print(f"可用列: {available_columns}")

    selected_columns = [
        'patientunitstayid', 'gender', 'age', 'ethnicity',
        'hospitaladmittime24', 'hospitaladmitoffset', 'hospitaladmitsource',
        'hospitaldischargestatus', 'unittype', 'unitadmittime24', 'unitadmitsource',
        'unitstaytype', 'admissionweight', 'unitdischargetime24', 'unitdischargeoffset',
        'unitdischargelocation', 'unitdischargestatus'
    ]

    final_columns = ['patientunitstayid']  # 始终保留患者ID
    for col in selected_columns:
        if col in available_columns:
            final_columns.append(col)

    patients = patients[final_columns]

    # 输出患者字典
    patients.to_csv(out_path, index=False)
    print(f'生成的患者字典已保存到 {out_path}')
    print(f'字典大小: {patients.shape[0]} 行, {patients.shape[1]} 列')
    print(f'包含的列: {list(patients.columns)}')
    print('===================================')

    return patients


def _get_patients_with_records(tuple_path):
    '''
    找出在tuples.csv中有记录的患者，并返回这些患者的ID

    Parameters:
    ----
        tuple_path: 元组文件路径

    Returns:
    ----
        有记录的患者ID集合
    '''

    print('===================================')
    print('寻找有医疗记录的患者')

    try:
        # 尝试直接读取文件的第一列
        patients = set()
        with pd.read_csv(tuple_path, index_col=False, usecols=[0],
                         chunksize=30000000, dtype='str') as reader:
            for i, chunk in enumerate(reader):
                for pid in tqdm(chunk.itertuples(False), total=chunk.shape[0]):
                    patients.add(pid[0])

        print('有记录的患者总数:', len(patients))
        print('===================================')
        return patients

    except Exception as e:
        print(f"读取元组文件时出错: {e}")
        print("返回所有患者ID作为备选")
        # 读取患者表时确保去重
        all_patients = []
        for chunk in pd.read_csv(EICU_DIR + 'patient.csv',
                                 usecols=['patientunitstayid'],
                                 dtype='str',
                                 chunksize=50000):
            chunk = chunk.drop_duplicates()
            all_patients.append(chunk)

        all_patients_df = pd.concat(all_patients, ignore_index=True)
        all_patients_df = all_patients_df.drop_duplicates()
        return set(all_patients_df['patientunitstayid'])


def revise_code_dict(input_dict_path, tuple_path, output_dict_path):
    '''
    根据元组修订代码字典中的频率

    Parameters:
    ----
        input_dict_path: 原始字典路径
        tuple_path: 元组文件路径
        output_dict_path: 输出字典路径

    Returns:
    ----
        None
    '''

    print("===================================")
    print("修订代码字典中的频率")

    try:
        # 加载原始字典
        original_dict = pd.read_csv(input_dict_path,
                                    dtype={'code': 'str', 'code_type': 'str',
                                           'value_frequency': 'int', 'total_frequency': 'int'},
                                    index_col=False)

        # 检查字典中的列
        dict_columns = list(original_dict.columns)
        print(f"原始字典中的列: {dict_columns}")

        # 确保有index列，如果没有，尝试使用行索引
        if 'index' not in dict_columns:
            print("字典中没有index列，使用行索引代替")
            original_dict['index'] = original_dict.index + 1

        # 创建映射
        original_dict_map = {str(row['index']): (row['value_frequency'], row['total_frequency'])
                             for _, row in original_dict.iterrows()}

        value_freq_dict = {k: 0 for k in original_dict_map}
        total_freq_dict = {k: 0 for k in original_dict_map}

        # 根据元组计算代码频率
        with pd.read_csv(tuple_path, index_col=False,
                         chunksize=30000000, dtype='str') as reader:
            for i, chunk in enumerate(reader):
                for pid, hadm, time, code, value in tqdm(chunk.itertuples(False), total=chunk.shape[0]):

                    if code in total_freq_dict:
                        total_freq_dict[code] += 1

                        # 检查值是否为数值
                        if not pd.isna(value) and value != 'NaN' and value != '_MISSING':
                            try:
                                float(value.replace('/', '.'))  # 尝试转换为浮点数
                                value_freq_dict[code] += 1
                            except ValueError:
                                pass  # 不是数值，不增加值频率

        # 输出变化的代码频率
        print('检查频率变化...')
        changes_found = False
        for k, v in original_dict_map.items():
            if v[0] != value_freq_dict[k]:
                print(f'[Value freq changed] code: {k}, freq: {v[0]} -> {value_freq_dict[k]}')
                changes_found = True
            if v[1] != total_freq_dict[k]:
                print(f'[Total freq changed] code: {k}, freq: {v[1]} -> {total_freq_dict[k]}')
                changes_found = True

        if not changes_found:
            print("没有发现频率变化")

        # 更新字典
        new_dict = original_dict.copy()
        new_dict['value_frequency'] = new_dict['index'].apply(lambda x: value_freq_dict.get(str(x), 0))
        new_dict['total_frequency'] = new_dict['index'].apply(lambda x: total_freq_dict.get(str(x), 0))

        # 输出更新后的字典
        new_dict.to_csv(output_dict_path, index=False)
        print(f"已保存修订后的代码字典到 {output_dict_path}")
        print("===================================")

    except Exception as e:
        print(f"修订代码字典时出错: {e}")
        print("将原始字典复制到输出路径")
        try:
            import shutil
            shutil.copy(input_dict_path, output_dict_path)
            print(f"已复制原始字典到 {output_dict_path}")
            print("===================================")
        except Exception as copy_err:
            print(f"复制文件时出错: {copy_err}")
            print("===================================")


def main():
    try:
        # 生成患者字典
        generate_patient_dict(RESULT_ROOT_DIR + 'tuples.csv', RESULT_ROOT_DIR + 'patients_dict.csv')

        # 修订代码字典
        revise_code_dict(IDX_DIR + 'code_dict.csv', RESULT_ROOT_DIR + 'tuples.csv',
                         RESULT_ROOT_DIR + 'code_dict_revised.csv')

        print("后处理完成!")
    except Exception as e:
        print(f"后处理过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()