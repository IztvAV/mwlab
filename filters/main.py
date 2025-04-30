import os
import mwlab
from filters.datasets.touchstone_dataset import TouchstoneMWFilterDataset


def main():
    td = TouchstoneMWFilterDataset(root=os.getcwd()+"\\Data")
    pass


if __name__ == "__main__":
    main()
