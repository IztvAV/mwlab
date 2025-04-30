import os
import mwlab
from filters import TouchstoneMWFilterDataset


def main():
    td = TouchstoneMWFilterDataset(root=os.getcwd()+"\\Data")
    pass


if __name__ == "__main__":
    main()
