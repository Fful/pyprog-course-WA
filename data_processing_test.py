import unittest
import warnings
from data_processing_funcs import Colors, CustomDF

from collections import defaultdict
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestColors(unittest.TestCase):
    def test_index_setter(self):
        """
        Test the setter method for the current_index property of the Colors class.

        This test initializes a Colors object, sets the current_index property to 2,
        and asserts that the value is correctly set to 2.
        """

        colors = Colors()
        colors.current_index = 2
        self.assertEqual(colors.current_index, 2)

    def test_reset(self):
        """
        Test the reset method of the Colors class.

        This test initializes a Colors object, sets the current_index property to 2,
        calls the reset method, and asserts that the current_index property is reset to 0.
        """

        colors = Colors()
        colors.current_index = 2
        colors.reset()
        self.assertEqual(colors.current_index, 0)


class TestCustomDF(unittest.TestCase):
    def setUp(self):
        """
        Set up sample data for testing.

        This method initializes sample dataframes and other variables needed for testing CustomDF methods.
        """

        # Sample data for testing
        self.ideal_df = CustomDF({'X': [1, 2, 3], 'Y1(i)': [2, 4, 9], 'Y2(i)': [1, 2, 4]})
        self.train_df = CustomDF({'X': [1, 2, 3], 'Y1(t)': [2, 4, 8], 'Y2(t)': [1, 3, 5]})
        # Train CustomDF same as ideal
        self.train_df_same = CustomDF({'X': [1, 2, 3], 'Y1(t)': [2, 4, 9], 'Y2(t)': [1, 2, 4]})
        # Sample of mse
        self.mse_df = CustomDF({'Y1(t)': [3, 0], 'Y2(t)': [1, 2]})  # Sample mean square error dataframe
        # Sample of test and chosen ideal funcs labels
        self.test_df = CustomDF({'X': [1, 2], 'Y(test)': [2, 4]})
        self.chosen_ideal_labels = ['Y1(i)', 'Y2(i)']
        # Sample merged_df of testing fit
        self.merged_df = CustomDF({'X': [1, 2], 'Y(test func)': [2, 4], 'Y1(i)': [2, 4], 'Y2(i)': [1, 2]})
        self.margin = np.sqrt(2)

    def test_constructor(self):
        """
        Test the constructor method of the CustomDF class.

        This test checks whether the constructor method returns an instance of CustomDF.
        """

        self.assertIsInstance(self.ideal_df._constructor(), CustomDF)

    def test_strip_brackets(self):
        """
        Test the _strip_brackets method of the CustomDF class.

        This test calls the _strip_brackets method with sample input and checks if it returns the expected output.
        """

        # Call the method to be tested
        result_list = CustomDF()._strip_brackets(['Y1(data_in_brackets)', 'Y2(data_in_brackets)'])
        # Assert that the returned object is a list
        self.assertIsInstance(result_list, list)
        # Assert that the brackets are stripped correctly
        self.assertEqual(result_list, ['Y1', 'Y2'])

    def test_sq_err_matrix1(self):
        """
        Test the _sq_err_matrix method of the CustomDF class with different labels between ideal and train data.

        This test calls the _sq_err_matrix method with sample input where the labels between ideal and train data are different,
        and checks if the method returns a CustomDF instance with the correct index, columns, and values.
        """

        # Call the method to be tested
        result_df = CustomDF._sq_err_matrix(self.ideal_df[['Y1(i)', 'Y2(i)']], self.train_df['Y1(t)'])
        # Assert that the returned object is a CustomDF instance
        self.assertIsInstance(result_df, CustomDF)
        # Assert that the index and columns are correctly named
        self.assertIsNone(result_df.index.name)
        self.assertEqual(result_df.columns.tolist(), ['sq_dif_Y1(i)', 'sq_dif_Y2(i)'])
        # Assert that the values are calculated correctly
        expected_values = {'sq_dif_Y1(i)': [0.0, 0.0, 1.0], 'sq_dif_Y2(i)': [1.0, 4.0, 16.0]}
        for column, expected in expected_values.items():
            np.testing.assert_allclose(result_df[column], expected)

    def test_sq_err_matrix2(self):
        """
        Test the _sq_err_matrix method of the CustomDF class with the same labels between ideal and train data.

        This test calls the _sq_err_matrix method with sample input where the labels between ideal and train data are the same,
        and checks if the method returns a CustomDF instance with the correct index, columns, and values.
        """

        # Call the method to be tested
        result_df = CustomDF._sq_err_matrix(self.ideal_df[['Y1(i)', 'Y2(i)']], self.train_df_same['Y1(t)'])
        # Assert that the returned object is a CustomDF instance
        self.assertIsInstance(result_df, CustomDF)
        # Assert that the index and columns are correctly named
        self.assertIsNone(result_df.index.name)
        self.assertEqual(result_df.columns.tolist(), ['sq_dif_Y1(i)', 'sq_dif_Y2(i)'])
        # Assert that the values are calculated correctly
        expected_values = {'sq_dif_Y1(i)': [0.0, 0.0, 0.0], 'sq_dif_Y2(i)': [1.0, 4.0, 25.0]}
        for column, expected in expected_values.items():
            np.testing.assert_allclose(result_df[column], expected)

    def test_mean_square_error1(self):
        """
        Test the mean_square_error method of the CustomDF class with different labels between ideal and train data.

        This test calls the mean_square_error method with sample input where the labels between ideal and train data are different,
        and checks if the method returns a CustomDF instance with the correct index, columns, and values.
        """

        # Call the method to be tested
        result_df = CustomDF.mean_square_error(self.ideal_df, self.train_df)
        # Assert that the returned object is a CustomDF instance
        self.assertIsInstance(result_df, CustomDF)
        # Assert that the index and columns are correctly named
        self.assertEqual(result_df.index.name, 'No ideal func')
        self.assertEqual(result_df.columns.tolist(), ['Y1(t)', 'Y2(t)'])
        # Assert that the values are calculated correctly
        expected_values = {'Y1(t)': [1/3, 7.0], 'Y2(t)': [6.0, 2/3]}
        for column, expected in expected_values.items():
            np.testing.assert_allclose(result_df[column], expected)

    def test_mean_square_error2(self):
        """
        Test the mean_square_error method of the CustomDF class with the same labels between ideal and train data.

        This test calls the mean_square_error method with sample input where the labels between ideal and train data are the same,
        and checks if the method returns a CustomDF instance with the correct index, columns, and values.
        """

        # Call the method to be tested
        result_df = CustomDF.mean_square_error(self.ideal_df, self.train_df_same)
        # Assert that the returned object is a CustomDF instance
        self.assertIsInstance(result_df, CustomDF)
        # Assert that the index and columns are correctly named
        self.assertEqual(result_df.index.name, 'No ideal func')
        self.assertEqual(result_df.columns.tolist(), ['Y1(t)', 'Y2(t)'])
        # Assert that the values are calculated correctly
        expected_values = {'Y1(t)': [0.0, 10.0], 'Y2(t)': [10.0, 0.0]}
        for column, expected in expected_values.items():
            np.testing.assert_allclose(result_df[column], expected)


    def test_find_best_fit(self):
        """
        Test the find_best_fit method of the CustomDF class.

        This test calls the find_best_fit method with sample mean square error data and checks if the method returns a CustomDF instance
        with the correct index, columns, and values.
        """

        # Call the method to be tested
        result_df = CustomDF.find_best_fit(self.mse_df)
        # Assert that the returned object is a CustomDF instance
        self.assertIsInstance(result_df, CustomDF)
        # Assert that the index and columns are correctly named
        self.assertEqual(result_df.index.tolist(), ['No. of ideal func', 'MSE'])
        self.assertEqual(result_df.columns.tolist(), ['Y1(t)', 'Y2(t)'])
        # Assert that the values are calculated correctly
        self.assertEqual(result_df.iloc[0]['Y1(t)'], 1)  # Index of ideal
        self.assertEqual(result_df.iloc[1]['Y1(t)'], 0.0)
        self.assertEqual(result_df.iloc[0]['Y2(t)'], 0)  # Index of ideal
        self.assertEqual(result_df.iloc[1]['Y2(t)'], 1.0)

    def test_lmerge(self):
        """
        Test the lmerge method of the CustomDF class.

        This test calls the lmerge method with sample test and ideal data and checks if the method returns a CustomDF instance
        with the correct columns and values.
        """

        # Call the method to be tested
        merged_df = CustomDF.lmerge(self.test_df, self.ideal_df, self.chosen_ideal_labels)
        # Assert that the returned object is a CustomDF instance
        self.assertIsInstance(merged_df, CustomDF)

        # Assert that the merged has the correct columns
        expected_columns = ['X', 'Y(test)', 'Y1(i)', 'Y2(i)']
        self.assertEqual(list(merged_df.columns), expected_columns)

        # Assert that the merged has the correct values
        expected_values = [(1, 2, 2, 1), (2, 4, 4, 2)]
        for row, expected_row in zip(merged_df.itertuples(index=False), expected_values):
            self.assertEqual(row, expected_row)

    def test_fit_unnested(self):
        """
        Test the fit method of the CustomDF class with unnested output.

        This test calls the fit method with sample data and checks if the method returns a CustomDF instance
        with the correct columns, values, and ideal function fitting.
        """

        # Call the method to be tested
        fitted_data, counter = self.merged_df.fit(self.chosen_ideal_labels, self.margin)

        # Assert that the returned objects are of the expected types
        self.assertIsInstance(fitted_data, CustomDF)
        self.assertIsInstance(counter, defaultdict)

        # Assert that the fitted data DataFrame has the correct columns
        expected_columns = ['X (test func)', 'Y (test func)', 'Delta Y (test func)', 'No. of ideal func']
        self.assertEqual(list(fitted_data.columns), expected_columns)

        # Assert that the values are calculated correctly
        expected_values = {'X (test func)': [1.0, 1.0, 2.0],
                           'Y (test func)': [2.0, 2.0, 4.0],
                           'Delta Y (test func)': [0.0, 1.0, 0.0]}
        for column, expected in expected_values.items():
            np.testing.assert_allclose(fitted_data[column], expected)

        # Assert ideal functions are fitted correctly
        self.assertEqual(list(fitted_data['No. of ideal func']), ['Y1', 'Y2', 'Y1'])

        # Assert that the counter dictionary contains the correct counts
        expected_counts = {'Y1': 2, 'Y2': 1}  # Assuming both ideal functions fit the test data
        self.assertDictEqual(dict(counter), expected_counts)

    def test_fit_nested(self):
        """
        Test the fit method of the CustomDF class with nested output.

        This test calls the fit method with sample data and checks if the method returns a CustomDF instance
        with the correct columns, values, and ideal function fitting.
        """

        # Call the method to be tested
        fitted_data, counter = self.merged_df.fit(self.chosen_ideal_labels, self.margin, nested_list=True)

        # Assert that the returned objects are of the expected types
        self.assertIsInstance(fitted_data, CustomDF)
        self.assertIsInstance(counter, defaultdict)

        # Assert that the fitted data DataFrame has the correct columns
        expected_columns = ['X (test func)', 'Y (test func)', 'Delta Y (test func)', 'No. of ideal func']
        self.assertEqual(list(fitted_data.columns), expected_columns)

        # Assert that the values are calculated correctly
        expected_values = {'X (test func)': [1.0, 2.0],
                           'Y (test func)': [2.0, 4.0]}
        for column, expected in expected_values.items():
            np.testing.assert_allclose(fitted_data[column], expected)

        # Assert ideal functions are fitted correctly
        self.assertEqual(list(fitted_data['No. of ideal func']), [['Y1', 'Y2'], 'Y1'])
        self.assertEqual(list(fitted_data['Delta Y (test func)']), [[0.0, 1.0], 0.0])

        # Assert that the counter dictionary contains the correct counts
        expected_counts = {'Y1': 2, 'Y2': 1}  # Assuming both ideal functions fit the test data
        self.assertDictEqual(dict(counter), expected_counts)


if __name__ == '__main__':
    unittest.main()

