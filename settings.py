'''
Settings of cleaning
'''

MIMIC_DIR = 'mimic/'    # original files of MIMIC-IV v1.0
ROLL_UP_SRC = 'rollup_tables/'   # files of roll-up tables
#UOM_SRC  = 'records/tools/'   # files of roll-up tables
UOM_SRC = "uom_dependency/"
RESULT_ROOT_DIR = 'records/'    # the directory to output the result

# the following files are under RESULT_ROOT_DIR
TUPLE_DIR = RESULT_ROOT_DIR + 'tuple/'
STRING_TUPLE_DIR = RESULT_ROOT_DIR + 'string_tuple/'
IDX_DIR = RESULT_ROOT_DIR + 'index/'
