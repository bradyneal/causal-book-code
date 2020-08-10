"""
Estimating the causal effect of sodium on blood pressure in a simulated example
adapted from Luque-Fernandez et al. (2018):
    https://academic.oup.com/ije/article/48/2/640/5248195
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def generate_data(n=1000, seed=0, beta1=1.05, alpha1=0.4, alpha2=0.3, binary_treatment=True, binary_cutoff=3.5):
    np.random.seed(seed)
    age = np.random.normal(65, 5, n)
    sodium = age / 18 + np.random.normal(size=n)
    if binary_treatment:
        if binary_cutoff is None:
            binary_cutoff = sodium.mean()
        sodium = (sodium > binary_cutoff).astype(int)
    blood_pressure = beta1 * sodium + 2 * age + np.random.normal(size=n)
    proteinuria = alpha1 * sodium + alpha2 * blood_pressure + np.random.normal(size=n)
    hypertension = (blood_pressure >= 140).astype(int)  # not used, but could be used for binary outcomes
    return pd.DataFrame({'blood_pressure': blood_pressure, 'sodium': sodium,
                         'age': age, 'proteinuria': proteinuria})


def estimate_causal_effect(Xt, y, model=LinearRegression(), treatment_idx=0, regression_coef=False):
    model.fit(Xt, y)
    if regression_coef:
        return model.coef_[treatment_idx]
    else:
        Xt1 = pd.DataFrame.copy(Xt)
        Xt1[Xt.columns[treatment_idx]] = 1
        Xt0 = pd.DataFrame.copy(Xt)
        Xt0[Xt.columns[treatment_idx]] = 0
        return (model.predict(Xt1) - model.predict(Xt0)).mean()


if __name__ == '__main__':
    binary_t_df = generate_data(beta1=1.05, alpha1=.4, alpha2=.3, binary_treatment=True, n=10000000)
    continuous_t_df = generate_data(beta1=1.05, alpha1=.4, alpha2=.3, binary_treatment=False, n=10000000)

    ate_est_naive = None
    ate_est_adjust_all = None
    ate_est_adjust_age = None

    for df, name in zip([binary_t_df, continuous_t_df],
                        ['Binary Treatment Data', 'Continuous Treatment Data']):
        print()
        print('### {} ###'.format(name))
        print()

        # Adjustment formula estimates
        ate_est_naive = estimate_causal_effect(df[['sodium']], df['blood_pressure'], treatment_idx=0)
        ate_est_adjust_all = estimate_causal_effect(df[['sodium', 'age', 'proteinuria']],
                                                    df['blood_pressure'], treatment_idx=0)
        ate_est_adjust_age = estimate_causal_effect(df[['sodium', 'age']], df['blood_pressure'])
        print('# Adjustment Formula Estimates #')
        print('Naive ATE estimate:\t\t\t\t\t\t\t', ate_est_naive)
        print('ATE estimate adjusting for all covariates:\t', ate_est_adjust_all)
        print('ATE estimate adjusting for age:\t\t\t\t', ate_est_adjust_age)
        print()

        # Linear regression coefficient estimates
        ate_est_naive = estimate_causal_effect(df[['sodium']], df['blood_pressure'], treatment_idx=0,
                                               regression_coef=True)
        ate_est_adjust_all = estimate_causal_effect(df[['sodium', 'age', 'proteinuria']],
                                                    df['blood_pressure'], treatment_idx=0,
                                                    regression_coef=True)
        ate_est_adjust_age = estimate_causal_effect(df[['sodium', 'age']], df['blood_pressure'],
                                                    regression_coef=True)
        print('# Regression Coefficient Estimates #')
        print('Naive ATE estimate:\t\t\t\t\t\t\t', ate_est_naive)
        print('ATE estimate adjusting for all covariates:\t', ate_est_adjust_all)
        print('ATE estimate adjusting for age:\t\t\t\t', ate_est_adjust_age)
        print()
