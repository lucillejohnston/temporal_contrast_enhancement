"""Statistical analysis utilities"""
from statsmodels.stats.multitest import multipletests
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

def apply_fdr_correction(p_values, alpha=0.05, method='fdr_bh'):
    """
    Apply FDR correction to multiple comparisons.
    """
    rejected, p_corrected, _, _ = multipletests(p_values, method=method, alpha=alpha)
    return rejected, p_corrected


def run_mixed_effects_model(formula, data, groups_col, reml=False):
    """
    Run linear mixed effects model.
    """
    model = mixedlm(formula, data=data, groups=data[groups_col])
    result = model.fit(reml=reml)
    return result


def compare_models_lr_test(model_main, model_interaction):
    """
    Likelihood ratio test comparing nested models.
    """
    lr_stat = 2 * (model_interaction.llf - model_main.llf)
    df_diff = len(model_interaction.params) - len(model_main.params)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=df_diff)
    
    return lr_stat, df_diff, p_value