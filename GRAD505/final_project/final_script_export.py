
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.multivariate.manova import MANOVA
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.covariance import EllipticEnvelope
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan


fleet = pd.read_excel('10_1_Class 8 Fleet Data.xlsx', header=0)
efficiency = pd.read_excel('10_2_Class 8 Fleet Fuel Economy.xlsx')
fleet.drop(columns=['Unnamed: 0'], inplace=True)
fleet.drop(index=fleet.index[0], inplace=True)
fleet.columns = fleet.columns.str.strip()


fleet.head()

efficiency.tail()

fleet.describe()

# convert non-numeric values to NaN and compute correlations
numeric = fleet.apply(pd.to_numeric, errors='coerce')

# correlation matrix (drop all-empty columns first)
corr = numeric.dropna(axis=1, how='all').corr()

# heatmap of correlation matrix
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Pearson correlation')
cols = corr.columns
plt.xticks(range(len(cols)), cols, rotation=90)
plt.yticks(range(len(cols)), cols)
plt.title('Correlation matrix heatmap')
plt.tight_layout()
plt.savefig(fname='Heatmap.png')
plt.show()

# pairwise scatter matrix for numeric columns
pd.plotting.scatter_matrix(numeric.dropna(axis=1, how='all'), figsize=(12, 12), diagonal='kde')
plt.suptitle('Scatter matrix (pairwise plots)', y=1.02)
plt.tight_layout()
plt.savefig(fname='pairplot.png')
plt.show()

print('Observations in fleet data: ' + str(len(fleet.index)))
print('Observations in efficiency data: ' + str(len(efficiency.index)))

wear_metals = ['Iron', 'Lead', 'Copper', 'Chro', 'Aluminum']
wear_metals_corr = corr.loc[wear_metals, wear_metals]
with pd.option_context('display.float_format', '{:.3f}'.format,
                       'display.width', None):
    print(wear_metals_corr)

df_long = fleet.melt(
    value_vars=wear_metals,
    var_name="Wear Metal",
    value_name="Concentration"
)
sns.set_theme(style="whitegrid")

g = sns.catplot(
    data=df_long,
    x="Wear Metal",
    y="Concentration",
    col="Wear Metal",
    kind="box",
    col_wrap=3,
    sharey=False,
    height=4,
    aspect=0.9
)

g.set_titles("{col_name}")
g.set_axis_labels("", "Concentration (ppm)")
g.fig.suptitle("Distribution of Wear Metals Across Fleets", y=1.05)
g.savefig(fname="boxplot.png")
plt.show()

# %%


# Filter to soot < 2%
fleet_filtered = fleet[fleet["Soot"] < 2]

# Wear metal columns
wear_metals = [
    "Iron", "Lead", "Copper", "Chro", "Aluminum", "Silicon"
]

plt.figure(figsize=(8, 6))

for metal in wear_metals:
    plt.scatter(fleet_filtered["Soot"], fleet_filtered[metal], label=metal)

plt.xlabel("Soot (%)")
plt.ylabel("Wear Metal Concentration")
plt.title("Soot (<2%) vs Wear Metals")
plt.legend()
plt.grid(True)
plt.savefig(fname="soot.png")
plt.show()

formula = 'Iron + Lead + Copper + Chro + Aluminum + Silicon ~ Oxidation + Nitration'
predictors = ['Oxidation','Nitration']

manova_df = fleet[predictors + wear_metals].copy().apply(pd.to_numeric, errors='coerce').dropna()
manova_df[wear_metals] = np.log1p(manova_df[wear_metals])
detector = EllipticEnvelope(contamination=0.06, random_state=42) # Removes significant outliers


manova_df['is_outlier'] = detector.fit_predict(manova_df[wear_metals])

manova_df = manova_df[manova_df['is_outlier'] == 1]

maov = MANOVA.from_formula(formula, data=manova_df)
print(maov.mv_test())


mahal_distances = detector.mahalanobis(manova_df[wear_metals])
stats.probplot(mahal_distances, dist="chi2", sparams=(len(wear_metals),), plot=plt)
plt.title("Q-Q Plot")
plt.savefig("Q1_qq.png")
plt.show()


X = sm.add_constant(manova_df[predictors])

results = {}
for metal in wear_metals:
    model = sm.OLS(manova_df[metal], X).fit()
    results[metal] = model

for metal in wear_metals:
    print(f"\n{metal}:")
    print(f"  Oxidation p = {results[metal].pvalues['Oxidation']:.4f}")
    print(f"  Nitration p = {results[metal].pvalues['Nitration']:.4f}")

one_way_df = fleet[['Soot'] + wear_metals].copy()
one_way_df[wear_metals] = one_way_df[wear_metals].astype(float)
one_way_df['Soot'] = pd.to_numeric(one_way_df['Soot'], errors='coerce')
one_way_df[wear_metals] = one_way_df[wear_metals].apply(pd.to_numeric, errors='coerce')
one_way_df[wear_metals] = np.log1p(one_way_df[wear_metals])
soot_threshold = one_way_df['Soot'].quantile(0.99)
one_way_df = one_way_df[one_way_df['Soot'] <= soot_threshold]

detector = EllipticEnvelope(contamination=0.05, random_state=42) # Removes significant outliers
one_way_df['is_outlier'] = detector.fit_predict(one_way_df[wear_metals])

one_way_df = one_way_df[one_way_df['is_outlier'] == 1]


formula = 'Iron + Lead + Copper + Chro + Aluminum + Silicon ~ Soot'
maov = MANOVA.from_formula(formula, data=one_way_df)
print(maov.mv_test())

sns.regplot(data=one_way_df, x='Soot', y='Iron', scatter_kws={'alpha':0.5})
plt.title("Checking Linearity: Log(Iron) vs Soot")
plt.savefig("Q2-linearity")
plt.show()

mahal_distances = detector.mahalanobis(one_way_df[wear_metals])
stats.probplot(mahal_distances, dist="chi2", sparams=(len(wear_metals),), plot=plt)
plt.title("Q-Q Plot")
plt.savefig("Q2-qq.png")
plt.show()


def test_wear_trends(df):
    results = []
    working_df = df.copy()
    
    for metal in wear_metals:
        working_df[metal] = pd.to_numeric(working_df[metal], errors='coerce')
        temp_df = working_df.dropna(subset=[metal, 'Samp. #']).copy()
        temp_df['log_metal'] = np.log1p(temp_df[metal])
        
        temp_df['time_step'] = temp_df['Samp. #']
        temp_df['time_step_sq'] = temp_df['time_step']**2
        
        try:

            formula = 'log_metal ~ time_step + time_step_sq'
            model = smf.ols(formula=formula, data=temp_df).fit()
            
            p_val_linear = model.pvalues['time_step']
            p_val_quad = model.pvalues['time_step_sq']
            coeff_quad = model.params['time_step_sq']
            
            if p_val_quad < 0.05:
                trend = "Quadratic" if coeff_quad > 0 else "Curved"
            else:
                trend = "Linear" if p_val_linear < 0.05 else "No Significant Trend"
                
            results.append({
                'Metal': metal,
                'Result': trend,
                'P_Linear': round(p_val_linear, 4),
                'P_Quadratic': round(p_val_quad, 4),
                'R_squared': round(model.rsquared, 4)
            })
        except Exception as e:
            results.append({'Metal': metal, 'Result': f"Error: {str(e)}"})
            
    return pd.DataFrame(results)

test_wear_trends(fleet)

residuals = model.resid
fitted = model.fittedvalues

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(x=fitted, y=residuals, ax=ax[0])
ax[0].axhline(0, color='red', linestyle='--')
ax[0].set_title('Residuals vs. Fitted')
ax[0].set_xlabel('Fitted log(Metal)')
ax[0].set_ylabel('Residuals')

sm.qqplot(residuals, line='45', fit=True, ax=ax[1])
ax[1].set_title('Normal Q-Q Plot')

plt.tight_layout()
plt.savefig('Q3-diagnostics.png')
plt.show()
