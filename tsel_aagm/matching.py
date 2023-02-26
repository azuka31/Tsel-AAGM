import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression, HuberRegressor, RANSACRegressor
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import time
pd.set_option('mode.chained_assignment',None)
plt.style.use('seaborn')

class PropensityScoreMatch:
    '''
    PropensityScoreMatch is a class for matching propensity score and treatment effect.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data frame.
    features : list
        A list of features to be used for matching.
    treatment : str
        The treatment column name.
    outcome : str
        The outcome column name.

    Attributes
    ----------
    df : pandas.DataFrame
        The input data frame.
    features : list
        A list of features to be used for matching.
    treatment : str
        The treatment column name.
    outcome : str
        The outcome column name.
    df_matched : pandas.DataFrame
        The matched data frame.
    df_smd : pandas.DataFrame
        The standardized mean difference data frame.
    df_TE : pandas.DataFrame
        The treatment effect data frame.

    Methods
    -------
    check_data_types()
        Check if the data types of features, treatment and outcome are correct.
    logistic_regression()
        Calculate propensity score using logistic regression.
    matching_score()
        Match the treated and control groups using nearest neighbors.
    plot_smd()
        Plot standardized mean difference for each feature.
    plot_individual_treatment()
        Plot the treatment effect for each individual.
    linear_regression()
        Calculate the treatment effect using linear regression.
    calculate_treatment_effect()
        Calculate the treatment effect for each feature.
    calculate_smd()
        Calculate the standardized mean difference for each feature.

    Example
    -------
    df = pd.read_csv('data.csv')
    psm = PropensityScoreMatch(df, ['feature_1', 'feature_2'], 'treatment', 'outcome')
    psm.plot_smd()
    psm.plot_individual_treatment()
    '''
    
    def __init__(self, df, features, treatment, outcome):
        # Initiating Variable
        self.df = df
        self.features = features 
        self.treatment = treatment
        self.outcome = outcome

        # Checking features, treatment and outcoume data type
        PropensityScoreMatch.check_data_types(self)

        # Logistic Regression and Matching
        PropensityScoreMatch.logistic_regression(self)
        PropensityScoreMatch.matching_score(self)
        PropensityScoreMatch.linear_regression(self)
        PropensityScoreMatch.calculate_treatment_effect(self)

        # Evaluating Propensity Score
        df_smd_bef = PropensityScoreMatch.calculate_smd(self, self.df)
        df_smd_aft = PropensityScoreMatch.calculate_smd(self, self.df_matched)
        df_smd = df_smd_bef.merge(df_smd_aft, on='features', how='left')
        df_smd.columns = ['features','smd_bef', 'smd_aft']
        self.df_smd = df_smd
        
    def check_data_types(self):
        if type(self.features) != list:
            raise Exception('The features dtype should be in list')
        if type(self.outcome) != str:
            raise Exception('The outcome dtype should be in string')
        if type(self.treatment) != str:
            raise Exception('The treatment dtype should be in string')

    def logistic_regression(self):
        df_features = self.df[self.features]

        # Initiating features and target 
        X = StandardScaler().fit_transform(df_features)
        y = self.df[self.treatment]

        # Fitting model 
        logit = LogisticRegression()
        logit.fit(X, y)
        ps = logit.predict_proba(X)[:,1]
        p_nt = logit.predict_proba(X)[:,0]
        pred = logit.predict(X)

        # Generating output (Probability and Prediction)
        self.df['proba'] = ps
        self.df['prediction'] = pred

        return self.df

    def matching_score(self):
        
        # Splitting treatement
        mask = self.df[self.treatment]==1
        df_treatment_1 = self.df[mask]
        df_treatment_0 = self.df[~mask].reset_index().drop(['index'], axis=1)

        # Adding Dummy ID
        df_treatment_1['dummy_id'] = df_treatment_1.index
        df_treatment_0['dummy_id'] = df_treatment_0.index

        # Getting Probability value and convert to Numpy Array
        proba_treatment_1 = df_treatment_1['proba'].values
        proba_treatment_0 = df_treatment_0['proba'].values

        # Reshaping Array to 2D
        proba_treatment_1 = proba_treatment_1.reshape(-1,1)
        proba_treatment_0 = proba_treatment_0.reshape(-1,1)

        # Finding Nearest Neighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(proba_treatment_0)
        distances, indices = nbrs.kneighbors(proba_treatment_1)

        # Creating LCG and Matching ID
        df_lcg = df_treatment_0.loc[indices[:,0]].copy() 
        df_lcg = df_lcg.reset_index().drop(['index'], axis=1)
        df_treatment_1['matched_id'] = df_lcg['dummy_id'].values
        df_lcg['matched_id'] = df_treatment_1['dummy_id'].values

        # Presribing Treatment and LCG 
        df_matched = pd.concat([df_treatment_1, df_lcg], ignore_index=True)
        self.df_matched = df_matched

        return self.df_matched 

    def plot_individual_treatment(self):

        return

    def linear_regression(self):

        # Defining Variable
        df_temp = self.df_matched.copy()
        target = self.outcome
        features = self.features
        treatment = self.treatment
        reversed_treatment = 'reversed_'.format(treatment)
        
        # Preparation Features and Target
        df_temp[reversed_treatment] = df_temp[treatment].apply(lambda b: abs(b-1))
        ols_features = features + [treatment,reversed_treatment]
        X = df_temp[ols_features].values
        y = df_temp[target].values

        # Fitting Model
        ols = LinearRegression()
        ols.fit(X, y)
        
        # Reversing (0 1 to 1 0) to Predict Counter Factuals
        ols_features_reversed = features + [reversed_treatment,treatment]
        X_reversed = df_temp[ols_features_reversed].values
        y_counter = ols.predict(X_reversed)

        # Editing Variable Names
        counter = 'counter_{}'.format(target)
        y1 = 'Y1_{}'.format(target)
        y0 = 'Y0_{}'.format(target)
        y1_y0 = 'Y1_Y0_{}'.format(target)

        # Deterimining Counter, y0 and y1
        df_temp[counter] = y_counter

        # Predicting Y0, Y1, and Y0 - Y1
        df_temp[y1] = df_temp[target] * df_temp[treatment] + df_temp[counter] * df_temp[reversed_treatment]
        df_temp[y0] = df_temp[target] * df_temp[reversed_treatment] + df_temp[counter] * df_temp[treatment]
        df_temp[y1_y0] = df_temp[y1] - df_temp[y0]

        # Assigning Attribute
        self.df_TE = df_temp.copy()
        return self.df_TE

    def calculate_treatment_effect(self):

        # Defining Variables
        target = self.outcome
        treatment = self.treatment
        y1_y0 = 'Y1_Y0_{}'.format(target)

        # Calculating ATT
        self.ATT = self.df_TE[self.df_TE[treatment]==1][[y1_y0]].mean().values[0]
        self.ATE = self.df_TE[[y1_y0]].mean().values[0]
        self.ATC = self.df_TE[self.df_TE[treatment]==0][[y1_y0]].mean().values[0]

        # Printing ATT, ATE, ATC
        print('ATT: {}'.format(self.ATT))
        print('ATE: {}'.format(self.ATE))
        print('ATC: {}'.format(self.ATC))
        return

    def calculate_smd(self, df):

        # Defining Variables
        features = self.features
        treatment = self.treatment

        # Agg Pre SME
        agg_operations = {treatment: 'count'}
        agg_operations.update({
            feature: ['mean', 'std'] for feature in features
        })
        table_one = df.groupby(treatment).agg(agg_operations)

        # Calculate SMD
        def compute_table_one_smd(table_one: pd.DataFrame, round_digits: int=4) -> pd.DataFrame:
            feature_smds = []
            for feature in features:
                feature_table_one = table_one[feature].values
                neg_mean = feature_table_one[0, 0]
                neg_std = feature_table_one[0, 1]
                pos_mean = feature_table_one[1, 0]
                pos_std = feature_table_one[1, 1]
                smd = (pos_mean - neg_mean) / np.sqrt((pos_std ** 2 + neg_std ** 2) / 2)
                smd = round(abs(smd), round_digits)
                feature_smds.append(smd)
            return pd.DataFrame({'features': features, 'smd': feature_smds})

        table_one_smd = compute_table_one_smd(table_one)
        return table_one_smd

    def plot_smd(self):
        df_vis = self.df_smd
        x = df_vis.index
        y1 = df_vis['smd_bef']
        y2 = df_vis['smd_aft']


        plt.style.use('default')
        plt.plot(y1, x, 'ko', label='Bef. Match')
        plt.plot(y2, x, 'kx', label='Aft. Match')
        plt.axvline(0.1, linestyle='--', color='k')
        plt.axvline(0.2, linestyle='-.', color='k')
        plt.xlim(0,0.5)
        plt.legend()
        plt.yticks(ticks=x, labels=df_vis['features'])
        plt.title('Standardized Mean Difference of PS Model')
        plt.show()

if __name__ == "__main__":

    # Data Preparation
    takers = pd.read_csv('sample_takers_mendadak_hepi.csv', delimiter='|').fillna(0)
    nontakers = pd.read_csv('sample_nontakers_mendadak_hepi.csv', delimiter='|').fillna(0)
    takers['label'] = 1
    nontakers['label'] = 0
    df_main = pd.concat([takers, nontakers], ignore_index=True)

    # Defining Variable
    features = ['rev_data_package_bef', 'last_poin_24', 'payload_bef', 'last_balance_24', 'los']
    treatment = 'label'
    outcome = 'rev_data_package_aft'

    # Propensity Score Model
    PS_model = PropensityScoreMatch(df_main, features, treatment, outcome)
