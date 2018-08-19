# Predictive Analysis HW2
#p171 - 11
#(a) 
library(ISLR)

Auto <- Auto
attach(Auto)
median_mpg <- median(Auto$mpg)
mpg01 <- rep(0, length(mpg))
i <- 1
while(i <= length(Auto$mpg)){
    if(mpg[i] > median_mpg){
        mpg01[i] <- 1
    }
    else{
        mpg01[i] <- 0
    }
    i <- i + 1
}
Auto_bi <- cbind(Auto, mpg01)

attach(Auto_bi)
par(mfrow = c(2, 2))
boxplot(mpg ~ mpg01, xlab = "MPG01", ylab = "MPG")
boxplot(cylinders ~ mpg01, xlab = "MPG01", ylab = "Cylinders")
boxplot(displacement ~ mpg01, xlab = "MPG01", ylab = "Displacement")
boxplot(horsepower ~ mpg01, xlab = "MPG01", ylab = "Horsepower")
boxplot(weight ~ mpg01, xlab = "MPG01", ylab = "Weight")
boxplot(acceleration ~ mpg01, xlab = "MPG01", ylab = "Acceleration")
boxplot(year ~ mpg01, xlab = "MPG01", ylab = "Year")
boxplot(origin ~ mpg01, xlab = "MPG01", ylab = "Origin")

library(dplyr)
    Auto_bi.sub <- dplyr::select(Auto_bi, one_of(c("cylinders", "acceleration", "year", "origin", "mpg01")))
i <- 1
Auto_bi.chisq <- as.data.frame(rep(0, 4))
while(i < length(Auto_bi.sub[1 ,])){
    data <- cbind(Auto_bi.sub[, i], Auto_bi.sub$mpg01)
    data <- table(data)
    Auto_bi.chisq[i ,] <- chisq.test(data)$p.value
    i <- i + 1
}
Auto_bi.chisq <- as.data.frame(Auto_bi.chisq, row.names = c("cylinders", "acceleration", "year", "origin"))
names(Auto_bi.chisq) <- "Chisq with mpg01"

Auto_model <- dplyr::select(Auto_bi, one_of(c("horsepower",
                                       "weight",
                                       "displacement",
                                       "cylinders",
                                       "origin",
                                       "mpg01")))
cros <- sample(c(0, 1), length(Auto_model[, 1]), replace = TRUE, prob = c(0.3, 0.7))
Auto_model_train <- Auto_model[cros == 1,]
Auto_model_test <- Auto_model[cros == 0,]

library(MASS)
attach(Auto_model)
#perform lda on the data
lda.fit <- lda(mpg01 ~., data = Auto_model_train)
lda.fit
lda.pred = predict(lda.fit, Auto_model_test)

lda.class = lda.pred$class
table(lda.class, Auto_model_test$mpg01)
test_lda_err <- 1 - mean(lda.class == Auto_model_test$mpg01)
test_lda_err

# perform qda on the data
qda.fit <- qda(mpg01 ~., data = Auto_model_train)
qda.fit
qda.pred = predict(qda.fit, Auto_model_test)

qda.class = qda.pred$class
table(qda.class, Auto_model_test$mpg01)
test_qda_err <- 1 - mean(qda.class == Auto_model_test$mpg01)
test_qda_err

#perform logistic on the data
logistic <- glm(mpg01 ~., family = binomial, data = Auto_model_train)
logistic
logistic.prod <- predict(logistic, Auto_model_test, type = "response")
head(logistic.prod)

logistic.pred <- rep(0, length(Auto_model_test[, 1]))
logistic.pred[logistic.prod > 0.5] <- 1
table(logistic.pred, Auto_model_test$mpg01)
test_log_err <- 1 - mean(logistic.pred == Auto_model_test$mpg01)
test_log_err

#perform KNN on the data
library(class)
i <- 1
test_knn_err <- rep(0, length(Auto_model_train[, 1]))
set.seed(2017)
while (i < length(Auto_model_train[, 1])){
    knn.pred <- knn(Auto_model_train, Auto_model_test, Auto_model_train$mpg01, k = i)
    test_knn_err[i] <- 1 - mean(knn.pred == Auto_model_test$mpg01)
    i <- i + 1
}
plot(1:length(Auto_model_train[, 1]), test_knn_err, xlab = "k")

## Exercise 6 from section 5.4 of the textbook (ISLR p. 199).
#(a).
attach(Default)
Default <- Default
logistic.fit <- glm(default ~ balance + income, family = binomial, data = Default)
summary(logistic.fit)

#(b).
boot.fn <- function(data, index){
    logistic_reg <- glm(default ~ balance + income, family = binomial, data = Default[index, ])
    return(logistic_reg$coef)
}

#c.
library(boot)
boot(Default, boot.fn, R = 1000)

## Question 4
#(a). 
set.seed(1)
X <- rnorm(100, mean = 24, sd = 11)
epsilon <- rnorm(100)

#(b).
Y <- 1 + 1*X +2*X^2 + 4*X^3 + epsilon

#(c).
newdata <- data.frame(X, Y)
i <- 2
while(i <= 10){
    newdata <- cbind(newdata, X^i)
    i <- i + 1
}
colnames(newdata) <- c("X", "Y", "X^2", "X^3", "X^4", "X^5", "X^6", "X^7", "X^8", "X^9", "X^10")
library(leaps)
regfit = regsubsets(Y ~., data = newdata, nvmax = 10)
reg.summary <- summary(regfit)
adjr2 <- reg.summary$adjr2
bic <- reg.summary$bic
cp <- reg.summary$cp

par(mfrow = c(2, 2))
plot(reg.summary$rss, xlab = "Number of Predictors", ylab = "RSS", type = "l")

plot(reg.summary$adjr2, xlab="Number of Predictors", ylab = "Adjusted RSq", type = "l")

points(which.max(reg.summary$adjr2),
       reg.summary$adjr2[which.max(reg.summary$adjr2)],
       col = "red",
       cex = 2,
       pch = 20)

plot(reg.summary$cp, xlab="Number of Predictors", ylab = "Cp", type = "l")

points(which.min(reg.summary$cp),
       reg.summary$cp[which.min(reg.summary$cp)],
       col = "red",
       cex = 2,
       pch = 20)

plot(reg.summary$bic,xlab="Number of Predictors", ylab = "BIC", type = "l")

points(which.min(reg.summary$bic),
       reg.summary$bic[which.min(reg.summary$bic)],
       col = "red",
       cex = 2,
       pch = 20)

#d.
regfit.forward = regsubsets(Y ~., data = newdata, nvmax = 10, method = "forward")
regfit.backward = regsubsets(Y ~., data = newdata, nvmax = 10, method = "backward")

## best subset info
reg.summary <- summary(regfit)
adjr2.best <- reg.summary$adjr2
adjr2.best.model <- which.max(reg.summary$adjr2)

bic.best <- reg.summary$bic
bic.best.model <- which.min(reg.summary$bic)

cp.best <- reg.summary$cp
cp.best.model <- which.min(reg.summary$cp)

## forward info
regfor.summary <- summary(regfit.forward)
adjr2.forward <- regfor.summary$adjr2
adjr2.for.model <- which.max(regfor.summary$adjr2)

bic.forward <- regfor.summary$bic
bic.for.model <- which.min(regfor.summary$bic)

cp.forward <- regfor.summary$cp
cp.for.model <- which.min(regfor.summary$cp)

## backward info
regback.summary <- summary(regfit.backward)
adjr2.backward <- regback.summary$adjr2
adjr2.back.model <- which.max(regback.summary$adjr2)

bic.backward <- regback.summary$bic
bic.back.model <- which.min(regback.summary$bic)

cp.backward <- regback.summary$cp
cp.back.model <- which.min(regback.summary$cp)

## combine the data into one data.frame
adjr2 <- rbind(adjr2.best, adjr2.forward, adjr2.backward)
bic <- rbind(bic.best, bic.forward, bic.backward)
cp <- rbind(cp.best, cp.forward, cp.backward)

model.adjr2 <- rbind(adjr2.best.model, adjr2.for.model, adjr2.back.model)
model.bic <- rbind(bic.best.model, bic.for.model, cp.back.model)
model.cp <- rbind(cp.best.model, cp.for.model, cp.back.model)

adjr2.row <- cbind(adjr2, model.adjr2)
bic.row <- cbind(bic, model.bic)
cp.row <- cbind(cp, model.cp)
compare <- rbind(adjr2.row, bic.row, cp.row)
compare

#e.lasso
set.seed(1)
library(glmnet)
x <- model.matrix(Y ~ ., newdata)[, -1]
y <- newdata$Y
#generate a grid to test different 100 lamda value
grid <- 10^seq(10, -2, length = 100)
out <- glmnet(x, y, alpha = 1, lambda = grid)
#use the cross validation on data set.
cv.out <- cv.glmnet(x, y, alpha = 1)
plot(cv.out)
bestlam <- cv.out$lambda.min
lasso.coef <- predict(out, type = "coefficients", exact = TRUE, x = x, y = y, s = bestlam)
lasso.coef

## question 9
# (a)
attach(College)
College <- College
train <- sample(1:length(College[, 1]), length(College[, 1]) * 0.7, replace = FALSE)
College_train <- College[train, ]
College_test <- College[-train, ]

# (b) linear regression
lm.fit <- lm(Apps ~., data = College_train)
lm.predict <- predict(lm.fit, newdata = College_test)
test_mse <- sqrt(sum((lm.predict - College_test$Apps)^2)/length(College_test$Apps))
test_mse

# (c) ridge regression
x.r <- model.matrix(Apps ~., College_train)[, -c(1, 3)]
y.r <- College_train$Apps
set.seed(1124)
cv.ridge.out <- cv.glmnet(x.r, y.r, alpha = 0)
bestlam.ridge <- cv.ridge.out$lambda.min
ridge <- glmnet(x.r, y.r, alpha = 0)
ridge.pred <- predict.glmnet(ridge, 
                             newx = model.matrix(Apps ~., College_test)[, -c(1, 3)],
                             exact = TRUE,
                             x = x.r,
                             y = y.r,
                             s = bestlam.ridge)
ridge.testMSE <- sqrt(mean((ridge.pred - College_test$Apps)^2))
ridge.testMSE

# (d) lasso regression
x.r <- model.matrix(Apps ~., College)[, -c(1, 3)]
y.r <- College$Apps
set.seed(1124)
cv.lasso.out <- cv.glmnet(x.r, y.r, alpha = 1)
bestlam.lasso <- cv.lasso.out$lambda.min
lasso <- glmnet(x.r, y.r, alpha = 0)
lasso.pred <- predict(lasso, newx = model.matrix(Apps ~., College_test)[, -c(1, 3)],
                      exact = TRUE,
                      x = x.r,
                      y = y.r,
                      s = bestlam.lasso)
lasso.testMSE <- sqrt(mean((lasso.pred - College_test$Apps)^2))
lasso.testMSE

# Number of non-zero coefficient estimates
lasso.coef = predict(lasso, type = "coefficients", s = bestlam.lasso)
lasso.coef
length(lasso.coef)
