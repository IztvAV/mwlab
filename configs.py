import os


BATCH_SIZE = 64
BASE_DATASET_SIZE = 34250
FILTER_NAME = "EAMU4-KuIMUXT3-BPFC1"
# FILTER_NAME = "SCYA501-KuIMUXT5-BPFC3"
ENV_ORIGIN_DATA_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "origins_data")
ENV_DATASET_PATH = os.path.join(os.getcwd(), "filters", "FilterData", FILTER_NAME, "datasets_data")
