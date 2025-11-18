import os

F_START_MHZ = 11489.7
F_STOP_MHZ = 11590.7
# F_START_MHZ = 12695
# F_STOP_MHZ = 12755

BATCH_SIZE = 32
BASE_DATASET_SIZE = 1_000
# FILTER_NAME = "EAMU4T1-BPFC2"
# FILTER_NAME = "ERV-KuIMUXT1-BPFC1"
# FILTER_NAME = "EAMU4-KuIMUXT3-BPFC1"
FILTER_NAME = "EAMU4-KuIMUXT2-BPFC2"
# FILTER_NAME = "EAMU4-KuIMUXT2-BPFC4"
# FILTER_NAME = "SCYA501-KuIMUXT5-BPFC3"
ENV_ORIGIN_DATA_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "origins_data")
ENV_DATASET_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "datasets_data")
ENV_TUNE_DATASET_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "tune_data")
