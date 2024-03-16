# Main libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Establish sql connection
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# Additional imports from standart lib
import os
import re
from collections import defaultdict

# Custom
from custom_exeptions import CSVReadErr, LenMismatchErr, MissPointsErr

class Colors:
    def __init__(self):
        """
        Initialize Colors object with a default color cycle.
        """
        self.__color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.current_index = 0

    @property
    def next_color(self):
        """
        Get the next color from the color cycle.

        Returns
        -------
        str
            The next color in the color cycle.
        """
        color = self.__color_cycle[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.__color_cycle)
        return color

    def reset(self):
        """
        Reset the color cycle index to the initial state.
        """
        self.current_index = 0


class CSVToDataBase:
    def __init__(self, csv_name, csv_source, db_name, db_url):
        """
        Initialize CSVToDataBase object.

        Parameters
        ----------
        csv_name : str
            Name of the CSV file.
        csv_source : str
            Path to the directory containing the CSV file.
        db_name : str
            Name of the database table.
        db_url : str
            URL of the database.

        Attributes
        ----------
        csv_name : str
            Name of the CSV file.
        db_name : str
            Name of the database table.
        url : str
            URL of the database.
        csv_source : str
            Path to the directory containing the CSV file.
        col_names : list
            List of column names.
        """
        self.csv_name = csv_name
        self.db_name = db_name
        self.url = db_url
        self.csv_source = csv_source
        self.col_names = list()

    def rename_columns(self, col_names):
        """
        Rename columns of the CSV file.

        Parameters
        ----------
        col_names : list
            List of new column names.
        """
        self.col_names = col_names

    def _create_table(self, engine):
        """
        Create a database table from a CSV file.

        Parameters
        ----------
        engine : sqlalchemy.engine.base.Engine
            Database engine.
        """
        df = pd.read_csv(os.path.join(self.csv_source, self.csv_name))
        if self.col_names:
            df.columns = self.col_names
        df.to_sql(self.db_name, engine, if_exists='replace', index=False)

    def create(self):
        """
        Create a database table from the CSV file.
        """
        engine = create_engine(self.url, pool_pre_ping=True)
        self._create_table(engine)
        print(f'Database "{self.db_name}" was created!')


class CustomDF(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        """
        Initialize CustomDF object.

        Parameters
        ----------
        *args, **kwargs :
            Positional and keyword arguments passed to the pandas DataFrame constructor.
        """
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        """
        Constructor for subclassing operations.
        """
        return CustomDF

    @classmethod
    def read_sql_query(cls, query, url, *args, **kwargs):
        """
        Read SQL query into a CustomDF object.
        Returns a table of data corresponding to the result of the query string.

        Parameters
        ----------
        query : str
            SQL query to be executed.
        url : sqlalchemy.engine.URL object
            URL object representing the database connection.
        *args, **kwargs :
            Additional arguments to be passed to pd.read_sql_query.

        Returns
        -------
        CustomDF
            A custom DataFrame object containing the result set of the executed SQL query,
            in relation to the specified database connection.
        """
        engine = create_engine(url, pool_pre_ping=True)
        with engine.connect() as conn:
            result_df = pd.read_sql_query(text(query), con=conn, *args, **kwargs)
            return cls(result_df)

    @classmethod
    def read_by_lines(cls, file_name, data_source, new_col_names=[]):
        """
        Read CSV file line by line into a CustomDF object.
        CSV file must contain columns name in fi

        Parameters
        ----------
        file_name: str
            Name of the CSV file.
        data_source: str | os.path object
            Corresponding to the path to the directory with CSV files.
        new_col_names: list; optional
            Insert new columns names

        Returns
        -------
        CustomDF
            A custom DataFrame object containing the result set of the data from the
            CSV file.
        """
        try:
            file_path = os.path.join(data_source, file_name)
            rows = []
            # read CSV file line by line
            with open(file_path, 'r') as file:
                # init CustomDF object
                columns = file.readline().strip().split(',')
                if new_col_names:
                    df = cls(columns=new_col_names)
                else:
                    df = cls(columns=columns)

                # iter trought lines
                for line in file:
                    values = line.strip().split(',')
                    rows.append(values)

                # add file data to CustomDF object
                df = pd.concat([df, CustomDF(rows, columns=df.columns)], ignore_index=True)
                # change dtypes to numeric
                df = df.astype(float)
            return df

        except Exception as e:
            raise CSVReadErr(f'Error reading CSV file: {str(e)}')

    @classmethod
    def mean_square_error(cls, ideal, train):
        """
        Calculate the mean square error between ideal and train datasets.

        Parameters
        ----------
        ideal : CustomDF
            DataFrame containing the ideal dataset.
        train : CustomDF
            DataFrame containing the train dataset.

        Returns
        -------
        CustomDF
            DataFrame containing the mean square error for each label.

        Raises
        ------
        IndexError
            If the 'X' feature is not found in either the ideal or train datasets.
        LenMismatchErr
            If the length of the train dataset does not match the length of the ideal dataset.
        MissPointsErr
            If some points are missing between the ideal and train datasets.
        """
        # check features
        if not (ideal.columns == 'X').any() or not (train.columns == 'X').any():
            raise IndexError('Feature "X" has not been found')
        # check matching columns len
        if not train.shape[0] == ideal.shape[0]:
            raise LenMismatchErr(train.shape[0], ideal.shape[0])
        # check missing points
        if not (ideal['X'] == train['X']).all():
            raise MissPointsErr()

        # split data
        features = train['X']
        train_labels = train.drop(columns=['X'])
        ideal_labels = ideal.drop(columns=['X'])

        # init mse datatable
        MSE = cls()

        for (train_name, train_label) in train_labels.items():
            frame = cls._sq_err_matrix(ideal_labels, train_label)
            frame.loc['sum'] = frame.sum()
            MSE[train_name] = frame.loc['sum'] / features.shape[0]

        MSE.index = [idx.split('_')[-1] for idx in MSE.index]
        MSE.index.name = 'No ideal func'

        return MSE

    @classmethod
    def _sq_err_matrix(cls, ideal_labels, train_col):
        """
        Calculate the square error matrix.

        Parameters
        ----------
        ideal_labels : CustomDF
            DataFrame containing the ideal labels.
        train_col : pd.Series
            Series containing the train column.

        Returns
        -------
        CustomDF
            DataFrame containing the square error matrix.
        """
        sq_err_matrix = cls().from_dict({'reference': train_col})
        for (ideal_name, ideal_label) in ideal_labels.items():
            deviation = sq_err_matrix['reference'] - ideal_label
            sq_err_matrix[f'sq_dif_{ideal_name}'] = np.power(deviation, 2)
        return sq_err_matrix.drop(columns=['reference'])

    @classmethod
    def find_best_fit(cls, mse):
        """
        Find the best fit from the mean square error DataFrame.

        Parameters
        ----------
        mse : CustomDF
            DataFrame containing the mean square error.

        Returns
        -------
        CustomDF
            DataFrame containing the best fit.
        """
        best_fit = cls({
            'No. of ideal func': mse.iloc[mse.agg(np.argmin)].index,
            'MSE': mse.agg('min')
        }).transpose()
        return best_fit

    def _strip_brackets(self, input_list):
        """
        Remove brackets from strings in a list.

        Parameters
        ----------
        input_list : list
            List of strings.

        Returns
        -------
        list
            List of strings with brackets removed.
        """
        # delistification
        if len(input_list) == 1:
            return re.sub(r'\([^)]*\)', '', input_list[0])
        # return list
        return [re.sub(r'\([^)]*\)', '', string) for string in input_list]

    def ideal_to_list(self, strip_brackets=False):
        """
        Convert DataFrame to a list of ideal functions.

        Parameters
        ----------
        strip_brackets : bool, optional
            Whether to strip brackets from the output list. Default is False.

        Returns
        -------
        list
            List of ideal functions.
        """
        chosen_ideal = self.loc['No. of ideal func'].to_list()
        if strip_brackets:
            chosen_ideal = self._strip_brackets(input_list=chosen_ideal)
        return chosen_ideal

    def bar_plot(self, strip_brackets=False):
        """
        Generate and display a bar plot of the mean square error (MSE) for each train function.

        Parameters
        ----------
        strip_brackets : bool, optional
            Whether to strip brackets from the index labels. Default is False.
        """
        if strip_brackets:
            index = self.index.to_list()
            self.index = self._strip_brackets(input_list=index)

        ax = self.plot(
            figsize=(8.27, 11.69),
            kind='barh',
            logx=True,
            fontsize=10,
        )
        ax.set_title('MSE for each train func in log scale', fontsize=12)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    def scatter_on_4funcs(self, train):
        """
        Generate and display scatter plots of the train data along with the corresponding ideal functions.

        Parameters
        ----------
        train : CustomDF
            DataFrame containing the train data.

        Notes
        -----
        This method assumes that the 'X' feature is present in the train DataFrame.
        """
        fig, _ = plt.subplots(2, 2, figsize=(8.27, 11.69), sharex=True)
        axis = fig.get_axes()  # get list of axis
        color = Colors()  # set color obj
        X = train['X']  # get features
        train = train.drop(columns=['X'])  # get labels

        plt.subplots_adjust(hspace=0.05, top=0.95)
        fig.suptitle('Selected Ideal_functions with corresponding Test_data')

        # set labels
        for idx, ax in enumerate(axis):
            if idx > 1:
                ax.set(xlabel='x')
            if idx % 2 == 0:
                ax.set(ylabel='y')

        # zip axis, train, and ideal data
        ziped_obj = zip(axis, train.items(), self.items())
        # plot graphs
        for ax, (train_name, train_data), (ideal_name, ideal_data) in ziped_obj:
            ax.scatter(x=X, y=train_data, alpha=0.3, c='C5')
            ax.plot(X, ideal_data, c=color.next_color)
            ax.legend([train_name, ideal_name], loc='upper right')

        plt.tight_layout()
        plt.show()

    @classmethod
    def lmerge(cls, test, ideal, chosen_ideal_labels, *args, **kwargs):
        """
        Merge the test and ideal dataframes based on the 'X' column.

        Parameters
        ----------
        test : CustomDF
            DataFrame containing the test data.
        ideal : CustomDF
            DataFrame containing the ideal data.
        chosen_ideal_labels : list
            List of ideal labels to merge with the test data.
        *args, **kwargs :
            Additional arguments to be passed to pd.merge.

        Returns
        -------
        CustomDF
            Merged DataFrame containing the test and ideal data.

        Raises
        ------
        MissPointsErr
            If some points are missing between the test and ideal datasets.
        """
        # Check missing points
        if not (test['X'].isin(ideal['X'])).all():
            raise MissPointsErr()
        # Merge CustomDF's
        merged_df = pd.merge(
            left=test,
            right=ideal[chosen_ideal_labels + ['X']],
            how='left',
            on='X',
            *args,
            **kwargs
        )
        return cls(merged_df)

    def plot_ideal_to_test(self, chosen_ideal_labels, test_label='Y(test func)'):
        """
        Plot the selected ideal functions along with the test data.

        Parameters
        ----------
        chosen_ideal_labels : list
            List of labels corresponding to the selected ideal functions.
        test_label : str, optional
            Label for the test function. Default is 'Y(test func)'.
        """
        ax2 = self.plot(
            x='X',
            y=chosen_ideal_labels,
            ylabel='Y',
            figsize=(8.27, 5.84)
        )
        ax2.set_title('Selected Ideal_functions and Test_data', fontsize=12)
        ax2.plot(self['X'], self[test_label], 'C5o:', mec='1.0')

        labels = chosen_ideal_labels + [test_label]
        ax2.legend(labels=labels, loc='upper left', bbox_to_anchor=(1, 1))
        plt.subplots_adjust(right=0.8)
        plt.tight_layout()
        plt.show()

    def fit(self, chosen_ideal_labels, margin=np.sqrt(2), nested_list=False):
        """
        Fit the test data to the chosen ideal functions.

        Parameters
        ----------
        chosen_ideal_labels : list
            List of labels corresponding to the chosen ideal functions.
        margin : float, optional
            Maximum allowable margin between the test data and the ideal functions. Default is sqrt(2).
        nested_list : bool, optional
            Whether to output nested lists for multiple fits. Default is False.

        Returns
        -------
        CustomDF
            DataFrame containing the fitted test data.
        defaultdict
            Dictionary containing the counts of how many times each ideal function fits the test data.
        """
        test_mapped = CustomDF(columns=['X (test func)', 'Y (test func)', 'Delta Y (test func)', 'No. of ideal func'])
        idx = 0
        counter = defaultdict(int)
        for i, row in self.iterrows():
            # Init row that will be implicted in df
            row_test_data = [row['X'].round(2), row['Y(test func)']]  # 1st and 2nd columns
            delta, No = list(), list()  # 3rd and 4th columns

            for ideal_func_No, div in zip(chosen_ideal_labels, row[chosen_ideal_labels]):
                if abs(div - row['Y(test func)']) < margin:
                    fitted_func = self._strip_brackets([ideal_func_No])
                    counter[fitted_func] += 1
                    delta.append(round(abs(div - row['Y(test func)']), 6))
                    No.append(fitted_func)

            # Missing values case
            if not delta:
                delta = [pd.NA]
                No = [pd.NA]

            if nested_list:
                # Covering into list multiple values
                if len(delta) > 1:
                    delta = [delta]
                    No = [No]
                test_mapped.loc[i] = row_test_data + delta + No
            else:
                # Duplicate row with multiple fitting ideal_func datapoints
                for d, N in zip(delta, No):
                    test_mapped.loc[idx] = row_test_data + [d, N]
                    idx += 1

        return test_mapped, counter

    def to_db(self, db_name, url):
        """
        Export the DataFrame to a database table.

        Parameters
        ----------
        db_name : str
            Name of the database table.
        url : str
            URL of the database.
        """
        engine = create_engine(url, pool_pre_ping=True)
        self.to_sql(db_name, engine, if_exists='replace', index=False)
        print(f'Database "{db_name}" was created!')