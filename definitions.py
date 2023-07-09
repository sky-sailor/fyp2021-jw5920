import os

# Root
ROOT_DIR = 'D:/IC Final Year Project'  # os.path.dirname(os.path.abspath(__file__))

# Data
DATA_DIR = os.path.join(ROOT_DIR, 'OUCRU')
DATASET_GENERATION = 'adults'
DATASET_PATIENT = '01NVa-003-2001'
DENGUE_PATH = os.path.join(DATA_DIR, DATASET_GENERATION, DATASET_PATIENT, 'PPG', DATASET_PATIENT+' Smartcare.csv')
PATHOLOGY_PATH = os.path.join(DATA_DIR, 'daily-profile.csv')

# Logs
LOG_PATH = '/vol/bitbucket/oss1017/logs'
TEMPLATES_PATH = os.path.join(ROOT_DIR, 'fyp2021-jw5920', 'main', 'pkgname', 'utils', 'templates')
