# AI Challenge 2026 — User Retention Prediction

## Overview

This project was completed for the qualification stage of the **AI Challenge 2026** international artificial intelligence competition for children and youth.

The task was to predict whether a user would **return to a mobile application**. The solution was implemented in **R** and uses statistical feature engineering, logistic Bayesian Additive Regression Trees, and threshold optimization based on ROC analysis.

The final model achieved an **ROC-AUC of 0.65** on the leaderboard.

---

## Problem Statement

Given historical user activity data, predict the binary target variable:

- `retention = 1` — the user returned to the app
- `retention = 0` — the user did not return

This is a binary classification problem.

---

## Dataset

The data was provided in two CSV files:

- `train.csv` — training data with the target variable `retention`
- `test.csv` — test data without the target variable

Key features included:

- `id`
- `is_weekend_user`
- `sessions_count`
- `avg_session_time`
- `purchases_count`
- `avg_purchase_value`
- `active_days`
- `retention` — target variable, present only in the training set

The categorical variable `is_weekend_user` was converted to a factor.

Missing values were checked using:

```r
colSums(is.na(training))
```

---

## Approach

### 1. Data Loading and Preprocessing

The training and test datasets were loaded separately. The categorical feature `is_weekend_user` was converted to a factor in both datasets.

```r
testing$is_weekend_user = as.factor(testing$is_weekend_user)
training$is_weekend_user = as.factor(training$is_weekend_user)
```

Missing values were inspected before further processing.

---

### 2. Feature Engineering

Several new informative features were created to better capture user engagement and monetization behavior:

```r
dta$time_spent = dta$sessions_count * dta$avg_session_time
dta$money_spent = dta$purchases_count * dta$avg_purchase_value
```

These features combine frequency and intensity:

- `time_spent` estimates total user engagement time.
- `money_spent` estimates total user spending.

To reduce the influence of skewed distributions and heavy tails, log transformations were applied to several numerical features:

```r
dta$sessions_count = log(dta$sessions_count)
dta$avg_session_time = log(dta$avg_session_time)
dta$purchases_count = log(dta$purchases_count)
dta$avg_purchase_value = log(dta$avg_purchase_value)
dta$active_days = log(dta$active_days)
```

This helped smooth extreme values and improve model stability.

---

### 3. Train / Validation Split

The data was split into training and validation subsets using the `rsample` package.

```r
require(rsample)

set.seed(42)
split_CV = initial_split(training)
train_CV = training(split_CV)
test_CV = testing(split_CV)
```

A fixed random seed was used to ensure reproducibility.

---

### 4. Modeling with Bayesian Additive Regression Trees

The main model was a **Bayesian Additive Regression Tree** model for binary outcomes, implemented via `BART::lbart`.

```r
require(BART)

x_train = subset(train_CV, select = -c(id, retention))
y_train = train_CV$retention
x_test = subset(test_CV, select = -c(id, retention))

res = lbart(x_train, y_train, x.test = x_test)
test_CV$prob = res$prob.test.mean
```

The model outputs posterior mean probabilities for the positive class. These probabilities were used as retention scores.

---

### 5. Threshold Optimization

Because the model returns probabilities, a decision threshold is needed to convert them into binary predictions.

The `ROCR` package was used to evaluate different cutoffs:

```r
require(ROCR)

pred_fit = prediction(test_CV$prob, test_CV$retention)
perf_fit = performance(pred_fit, 'tpr', 'fpr')

perf1 = performance(pred_fit, x.measure = 'cutoff', measure = 'spec')
perf2 = performance(pred_fit, x.measure = 'cutoff', measure = 'sens')
perf3 = performance(pred_fit, x.measure = 'cutoff', measure = 'acc')

plot(perf1, col = 'red', lwd = 2)
plot(add = T, perf2, col = 'green', lwd = 2)
plot(add = T, perf3, col = 'blue', lwd = 2)
abline(v = 0.77, lwd = 2)
```

Sensitivity, specificity, and accuracy were plotted against the cutoff value to choose a reasonable decision boundary.

The final threshold was set to approximately:

```r
0.757
```

---

### 6. Final Model and Submission

After selecting the threshold, the model was retrained on the full training dataset and used to predict the test set.

```r
x_train = subset(training, select = -c(id, retention))
y_train = training$retention
x_test = subset(testing, select = -c(id))

res = lbart(x_train, y_train, x.test = x_test)

testing$retention = ifelse(res$prob.test.mean >= 0.757, 1, 0)

to_save = testing[c('id', 'retention')]
write.csv(to_save, 'answer.csv', row.names = FALSE)
```

The final submission file contained two columns:

- `id`
- `retention`

---

## Results

| Metric | Value |
|---|---|
| Model | Logistic Bayesian Additive Regression Trees (`lbart`) |
| Validation | Train/validation split |
| Threshold | 0.757 |
| Leaderboard ROC-AUC | 0.65 |

---

## Technologies Used

- **R**
- **BART** — Bayesian Additive Regression Trees
- **ROCR** — ROC curve and threshold analysis
- **rsample** — data splitting
- Base R for data preprocessing and feature engineering

---

## Key Takeaways

- Log transformations helped reduce skewness and heavy tails in numerical features.
- Combining frequency and intensity features produced more informative user behavior signals.
- Bayesian Additive Regression Trees provided a flexible nonlinear model for binary retention prediction.
- Threshold tuning was important for converting predicted probabilities into final binary labels.
- The final solution achieved **ROC-AUC 0.65** on the leaderboard.