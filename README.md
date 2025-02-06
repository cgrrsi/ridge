# ridge

## Linear least squares with L2 regularization

We start with reimplementing Ridge regression from scratch that is OLS with L2 penalization.

The goal is to minimize the objective function given by $||y - X\beta||^2_2 + \lambda * ||\beta||^2_2$

This classically leads to the closed-form solution $\hat{\beta} = (X^TX+\lambda I)^{-1}X^Ty$.

## Non-zero priors

Actually, we can generalize the previous formula and incorporate non-zero priors.

$$\hat{\beta}(\lambda) = \arg \min_{\beta \in \mathbb{R}^p} \left( (Y - X\beta)^TW(Y - X\beta) + (\beta - \beta_0)^T\Delta(\beta - \beta_0) \right)$$

- $W$: Weighting matrix, where it is diagonal. For simple Ridge, $W = I_p$, which means all data points are equally weighted.
- $\beta_0$: Beta prior. For simple Ridge, $\beta_0 = 0$.
- $\Delta$: Positive semi-definite matrix. For simple Ridge, $\Delta = \lambda I_p$.

## Shrinking a subset of features only

Next, we investigate how to shkrink a subset of features only, in the case we do not want to shrink all of the features, but only some of them.

$$\hat{\gamma}, \hat{\beta} = \arg \min_{\gamma \in \mathbb{R}^m, \beta \in \mathbb{R}^p} \|Y - U\gamma - X\beta\|^2 + (\beta - \beta_0)^T \Lambda (\beta - \beta_0)$$

- $X$: Penalised features
- $U$: Unpenalised features

This is particularly useful when incorporating interaction terms — which we might want to penalize more than other features — or when we have features from various fundamental types.
