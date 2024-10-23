# Cookie-cats
I recently completed a project about analysing results of A/B testing using Python and Pandas. The following code illustrates the key steps I took to clean and analze the dataset.
## Dataset Overview

This dataset comprises A/B test outcomes for Cookie Cats, investigating the impact of relocating the initial gate in the game from level 30 to level 40. Upon installing the game, players were randomly assigned to either gate_30 or gate_40.

The data we have is from 90,189 players that installed the game while the AB-test was running. The variables are:

userid: A unique number that identifies each player.

version: Whether the player was put in the control group (gate_30 - a gate at level 30) or the group with the moved gate (gate_40 - a gate at level 40).

sum_gamerounds: the number of game rounds played by the player during the first 14 days after install.

retention_1: Did the player come back and play 1 day after installing?

retention_7: Did the player come back and play 7 days after installing?

## Code Overview
Import Libraries and Configure Settings
```python
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
import statsmodels.api as sm
```
Then I imported dataset and checked it for missing values.
```python
data = pd.read_csv('cookie_cats.txt', sep=',')
df = pd.DataFrame(data)
print(data.head())
print(data.isnull().sum())
```
To make results more reliable it was cleand with IQR method from outliers.
```python
Q1 = df['sum_gamerounds'].quantile(0.25)
Q3 = df['sum_gamerounds'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_clean = df[(df['sum_gamerounds'] >= lower_bound) & (df['sum_gamerounds'] <= upper_bound)]
print(f"Num of deleted rows: {len(df) - len(df_clean)}")
```
After I sorted data into two groups: gate_30 and gate_40 and calculated the mean for each group.
```python
group_30 = data[data['version'] == 'gate_30']['sum_gamerounds']
group_40 = data[data['version'] == 'gate_40']['sum_gamerounds']

print(f'Mean for gate_30: {group_30.mean()}')
print(f'Mean for gate_40: {group_40.mean()}')
```
The next step was to make different math test for better understanding this A/B testing results.
The Mann-Whitney U Test was performed to compare the distribution of game rounds between the two groups.
```python
u_stat, p_value = stats.mannwhitneyu(group_30, group_40)
print(f'u_stat: {u_stat}')
print(f'p-value: {p_value}')
```
This section counted the number of players who returned to the game on the 1st day and 7th day after installation, grouped by game version.
```python
returned_1 = df[df['retention_1'] == True].groupby('version').size()
returned_7 = df[df['retention_7'] == True].groupby('version').size()
print(f'returned_1 true: {returned_1}')
print(f'returned_7 true: {returned_7}')
```
The Z-Test checked whether the proportion of returning players differs significantly between the two groups.
```python
z_stat, p_value = sm.stats.proportions_ztest([20034, 20119], [44700, 45489])
print(f'Z-st: {z_stat}')
print(f'p-value: {p_value}')
```
This logistic regression model assessed the influence of game version and game rounds on player retention on day 1.
```python
X = df[['version', 'sum_gamerounds']]
y = df['retention_1']
X = sm.add_constant(X)
logit_model = sm.Logit(y, X)
result = logit_model.fit()
print(result.summary())
```
A similar regression was performed to analyze the effect of the game version and rounds played on 7-day retention.
```python
X = df[['version', 'sum_gamerounds']]
y = df['retention_7']
X = sm.add_constant(X)
lin_model = sm.OLS(y, X)
lin_result = lin_model.fit()
print(lin_result.summary())
```

## Analyzing of results

