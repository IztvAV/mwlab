import os


BATCH_SIZE = 32
BASE_DATASET_SIZE = 100_000
FILTER_NAME = "ERV-KuIMUXT1-BPFC1"
# FILTER_NAME = "SCYA501-KuIMUXT5-BPFC3"
ENV_ORIGIN_DATA_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "origins_data")
ENV_DATASET_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "datasets_data")
ENV_TUNE_DATASET_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "tune_data")
