# loading data and making factor out of obviously categorical variable
testing = read.csv('test.csv')
testing$is_weekend_user = as.factor(testing$is_weekend_user)
training = read.csv('train.csv')
training$is_weekend_user = as.factor(training$is_weekend_user)

# checking for missing values
colSums(is.na(training))

# performing some feature-engineering
qual = function(dta) {
  dta$time_spent = dta$sessions_count * dta$avg_session_time
  dta$money_spent = dta$purchases_count * dta$avg_purchase_value
  
  dta$sessions_count = log(dta$sessions_count)
  dta$avg_session_time = log(dta$avg_session_time)
  dta$purchases_count = log(dta$purchases_count)
  dta$avg_purchase_value = log(dta$avg_purchase_value)
  dta$active_days = log(dta$active_days)
  dta$avg_session_time = log(dta$avg_session_time)
  
  return(dta)
}

training = qual(training)
testing = qual(testing)

# splitting data into train and test
require(rsample)

set.seed(42)
split_CV = initial_split(training)
train_CV = training(split_CV)
test_CV = testing(split_CV)

# predicting retention via Bayesian sum of trees
require(BART)

x_train = subset(train_CV, select = -c(id, retention))
y_train = train_CV$retention
x_test = subset(test_CV, select = -c(id, retention))
res = lbart(x_train, y_train, x.test = x_test)
test_CV$prob = res$prob.test.mean

#choosing the proper cutoff
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

# training model on all the data and forming predictions
x_train = subset(training, select = -c(id, retention))
y_train = training$retention
x_test = subset(testing, select = -c(id))
res = lbart(x_train, y_train, x.test = x_test)

testing$retention = ifelse(res$prob.test.mean >= 0.757, 1, 0)
to_save = testing[c('id', 'retention')]
write.csv(to_save, 'answer.csv', row.names = FALSE)