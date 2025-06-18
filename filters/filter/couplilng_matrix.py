import torch
import numpy as np
import matplotlib.pyplot as plt


class CouplingMatrix:
    def __init__(self, matrix: torch.Tensor):
        self._matrix = matrix
        self._links = self.get_links_from_matrix(matrix)
        self._links_for_analysis = self.get_links_for_analysis_from_matrix(matrix)
        self._matrix_order = matrix.shape[0]

    @classmethod
    def from_file(cls, file_path, device='cpu'):
        """
        Reads matrix elements from file and makes it symmetric.

        Parameters:
        file_path (str): Path to the coupling matrix file.
        device (str): Device to store the tensor on ('cpu' or 'cuda')

        Returns:
        CouplingMatrix: Instance with symmetric square coupling matrix.
        """
        matrix = []
        with open(file_path, mode='r') as file:
            lines = file.readlines()
            for line in lines:
                row = list(map(float, line.strip().split(',')))
                matrix.append(row)

        matrix = torch.tensor(matrix, dtype=torch.float32, device=device)

        # Check if matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square")

        # Make matrix symmetric
        sym_matrix = (matrix + matrix.T) / 2
        return cls(sym_matrix)

    @staticmethod
    def from_factors(factors, links, matrix_order, device='cpu'):
        """
        Creates a coupling matrix from factors and links.

        Parameters:
        factors (torch.Tensor): Tensor of coupling factors
        links (list): List of tuples representing matrix links
        matrix_order (int): Size of the square matrix
        device (str): Device to store the tensor on

        Returns:
        torch.Tensor: Symmetric coupling matrix
        """
        indices = torch.tensor(links, device=device).T  # (2, num_links)
        values = factors.to(device)

        M = torch.zeros((matrix_order, matrix_order),
                       dtype=torch.float32,
                       device=device)
        M = M.index_put((indices[0], indices[1]), values)
        M = M.index_put((indices[1], indices[0]), values)  # symmetric part
        return M

    @staticmethod
    def get_links_from_matrix(matrix):
        """Get upper triangular non-zero links from matrix"""
        upper_tri = torch.triu(matrix)
        nonzero_indices = torch.nonzero(upper_tri)
        return [(i.item(), j.item()) for i, j in nonzero_indices]

    @classmethod
    def error_matrix(cls, orig_matrix, pred_matrix):
        links = torch.tensor(orig_matrix.links).T
        orig_elements = orig_matrix.matrix[links[0], links[1]]
        pred_elements = pred_matrix.matrix[links[0], links[1]]
        error_elements = torch.abs((orig_elements - pred_elements) / orig_elements) * 100
        error_matrix = CouplingMatrix.from_factors(error_elements, links.T, orig_matrix.matrix_order)
        return cls(error_matrix)

    def _get_main_diagonals_indices(self, matrix):
        arr = np.array(matrix)
        n = arr.shape[0]

        # Главная диагональ (i = j)
        main_diag = []
        for i in range(n):
            if matrix[i,i] != 0:
                main_diag.append((i, i))

        # Смещенная главная диагональ (i = j + 1)
        main_shifted = []
        for i in range(0, n-1):
            main_shifted.append((i, i + 1))

        indices = main_diag + main_shifted

        return indices

    def _get_anti_diagonals_indices(self, matrix):
        n = len(matrix)
        indices = []

        for i in range(n):
            for j in range(n):
                if matrix[i, j] != 0:
                    # Побочная диагональ (i + j = n - 1)
                    if i + j == n - 1:
                        indices.append((i, j))
                    # Смещенная побочная диагональ (i + j = n - 2)
                    elif i + j == n - 2:
                        indices.append((i, j))

        return indices

    def get_links_for_analysis_from_matrix(self, matrix):
        main_diagonal_indices = self._get_main_diagonals_indices(matrix)
        anti_diagonal_indices = self._get_anti_diagonals_indices(np.triu(matrix))
        indices = list(set(main_diagonal_indices + anti_diagonal_indices))
        return indices


    @property
    def matrix(self):
        return self._matrix.clone()

    @property
    def links(self):
        return self._links

    @property
    def links_for_analysis(self):
        return self._links_for_analysis

    @property
    def matrix_order(self):
        return self._matrix_order

    @property
    def factors(self):
        rows, cols = zip(*self.links)
        return self.matrix[rows, cols]

    @property
    def factors_for_analysis(self):
        rows, cols = zip(*self.links_for_analysis)
        return self.matrix[rows, cols]

    def to(self, device):
        """Move the matrix to specified device"""
        self._matrix = self._matrix.to(device)
        return self

    def plot_matrix(self, decimals=4, title="Coupling matrix", cmap="viridis"):
        """
        Displays the matrix as a table on a plot

        Parameters:
            decimals : int, optional
                Number of decimal places (default 4)
            title : str, optional
                Plot title
            cmap : str, optional
                Color map (from matplotlib)
        """
        matrix_np = self._matrix.cpu().numpy() if self._matrix.is_cuda else self._matrix.numpy()

        fig, ax = plt.subplots(figsize=(8, 6))

        # Hide axes
        ax.axis('off')
        ax.axis('tight')

        # Create table
        table = ax.table(
            cellText=np.round(matrix_np, decimals=decimals),
            cellLoc='center',
            loc='center',
            colLabels=None,
            rowLabels=None
        )

        # Style adjustments
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)  # Scaling

        # Add color map and handle zeros
        if cmap is not None:
            norm_matrix = matrix_np.copy()
            non_zero_mask = matrix_np != 0
            if np.any(non_zero_mask):
                norm_matrix[non_zero_mask] = (matrix_np[non_zero_mask] - np.min(matrix_np[non_zero_mask])) / \
                                           (np.max(matrix_np[non_zero_mask]) - np.min(matrix_np[non_zero_mask]))

            colors = plt.cm.get_cmap(cmap)(norm_matrix)

            for (i, j), val in np.ndenumerate(matrix_np):
                if val == 0:
                    table[(i, j)].set_facecolor("white")
                else:
                    table[(i, j)].set_facecolor(colors[i, j])

        plt.title(title)