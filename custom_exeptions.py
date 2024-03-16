# Exption subclssese for data_processing
class CSVReadErr(Exception):
    def __init__(self, message='Error reading CSV file'):
        """
        Error raised for issues reading a CSV file.

        Parameters
        ----------
        message : str, optional
            Custom error message. Default is 'Error reading CSV file'.
        """
        self.message = message
        super().__init__(self.message)


class LenMismatchErr(Exception):
    def __init__(self, train_dim, ideal_dim, message='Mismatch in length between ideal and training data:'):
        """
        Error raised for length mismatches between ideal and training data.

        Parameters
        ----------
        train_dim : int
            Number of points in the training data.
        ideal_dim : int
            Number of points in the ideal data.
        message : str, optional
            Custom error message prefix. Default is 'Mismatch in length between ideal and training data:'.
        """
        self.message = '\n'.join([
            message,
            f'Amount of train points {train_dim}',
            f'Amount of ideal points {ideal_dim}'])
        super().__init__(self.message)


class MissPointsErr(Exception):
    def __init__(self, message='Some points missing, features (x) are not equal'):
        """
        Error raised when some points are missing, leading to unequal features.

        Parameters
        ----------
        message : str, optional
            Custom error message. Default is 'Some points missing, features (x) are not equal'.
        """
        self.message = message
        super().__init__(self.message)