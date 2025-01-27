# Ridge regression from scratch

## Linear least squares with L2 regularization

The goal is to minimize the objective function given by $||y - X\beta||^2_2 + \lambda * ||\beta||^2_2$

## Generalized Ridge Regression

$$\hat{\beta}(\lambda) = \arg \min_{\beta \in \mathbb{R}^p} \left( (Y - X\beta)^TW(Y - X\beta) + (\beta - \beta_0)^T\Delta(\beta - \beta_0) \right)$$

- $W$: Weighting matrix, where it is diagonal. For simple Ridge, $W = I_p$, which means all data points are equally weighted.
- $\beta_0$: Beta prior. For simple Ridge, $\beta_0 = 0$.
- $\Delta$: Positive semi-definite matrix. For simple Ridge, $\Delta = \lambda I_p$.
