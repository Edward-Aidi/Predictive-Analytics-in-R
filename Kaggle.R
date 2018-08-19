# For 500s Kaggle Competition
setwd("/Users/ai/Desktop/MKT.500S - Predictive Analytics for Business Decisions-Making/Kaggle")
data_tr <- read.csv("Train.csv")
data_te <- read.csv("Test.csv")

library(dplyr)
# 1. Unit Price (P), 2. Promotion (PR), 3. Display (D), 4. Feature (F) 5. Volume of Sales (Y)
# split the train data into 24 products
i <- 1 
while(i <= 24){
    f <- paste("F", i, sep = "")
    d <- paste("D", i, sep = "")
    pr <- paste("PR", i, sep = "")
    p <- paste("P", i, sep = "")
    y <- paste("Y", i, sep = "")
    
    product <- dplyr::select(data_tr, one_of(c(f, d, pr, p, y)))
    fname <- paste("./Train_by pro/Train_P", i, ".csv", sep = "")
    write.csv(product, file = fname)
    i <- i + 1
}

# split the train data into 24 products, test did not contain y
i <- 1 
while(i <= 24){
    f <- paste("F", i, sep = "")
    d <- paste("D", i, sep = "")
    pr <- paste("PR", i, sep = "")
    p <- paste("P", i, sep = "")
    
    product <- dplyr::select(data_te, one_of(c(f, d, pr, p)))
    fname <- paste("./Test_by pro/Test_P", i, ".csv", sep = "")
    write.csv(product, file = fname)
    i <- i + 1
}

# Product 1 - Carbonated Beverages
set.seed(1124)
p1 <- read.csv("./Train_by pro/Train_P1.csv")
p1 <- p1[-1]
cor(p1[,1:4])

# split the data into training and testing
train <- sample(1:length(p1[,1]), length(p1[, 1])*0.7)
p1_train <- p1[train, ]
p1_test <- p1[-train, ]

# MARS
library(earth)
MARS.model <- earth(p1_train$Y1 ~., data = p1_train, degree = 1)
summary(MARS.model)
predict.p1 = predict(MARS.model, newdata = p1_test)
mse <- mean((predict.p1-p1_test$Y1)^2)

# Scan Pro
p1_log <- data.frame(log(p1_train$Y1), log(p1_train$P1), p1_train[,-4:-5])
colnames(p1_log) <- c("log_y", "log_p","F1", "D1", "PR1")
p1_log_s <- data.frame(log(p1_test$P1), p1_test[-4])
colnames(p1_log_s) <- c("log_p","F1", "D1", "PR1", "Y1")
lm.fit <- lm(log_y ~., data = p1_log)
lm.predict <- predict(lm.fit, newdata = p1_log_s)
mse <- mean((exp(lm.predict) - p1_test$Y1)^2)

## XGboost
library(xgboost)
df_train <- read.csv("./Train_by pro/Train_P1.csv")
df_train <- df_train[, -1]
df_test <- read.csv(".//Test_by pro/Test_P1.csv")
df_test <- df_test[, -1]
set.seed(1124)

library(data.table)
data <- p1_log[2:5]
label <- p1_log$log_y
train <- list(data=data, label=labels)
dim(train$data)

xgb <- xgboost(data = data.matrix(train$data), label = train$labels,
               max_depth = 2, eta = 1, nthread = 2, nrounds = 2, 
               objective = "reg:linear")

xgb.grid <- expand.grid(nrounds = 500,
                        max_depth = seq(6,10),
                        eta = c(0.01,0.3, 1),
                        gamma = c(0.0, 0.2, 1),
                        colsample_bytree = c(0.5,0.8, 1)
)

xgb_tune <-train(SalePrice ~.,
                 data=train,
                 method="xgbTree",
                 metric = "RMSE",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid
)

## Boosted Tree
library(gbm)
set.seed(1124)

boost.p1g <- gbm(log_y~.,p1_log, distribution = "gaussian",
                   n.trees = 5000, cv.folds = 10, interaction.depth = 3)
summary(boost.p1g)
yhat.boost <- predict(boost.p1g, newdata = p1_log_s, n.trees = 5000)
mean((exp(yhat.boost) - p1_log_s$Y1)^2)

## Student t for product 1
set.seed(1124)
p1 <- read.csv("./Train_by pro/Train_P1.csv")
p1 <- p1[-1]
cor(p1[,1:4])

# split the data into training and testing
set.seed(1124)
train <- sample(1:length(p1[,1]), length(p1[, 1])*0.7)
p1_train <- p1[train, ]
p1_test <- p1[-train, ]

set.seed(1124)
boost.p1t <- gbm(log_y ~.-PR1, p1_log, distribution = "tdist",
                n.trees = 5000, interaction.depth = 4)
summary(boost.p1t)
yhat.boost <- predict(boost.p1t, newdata = p1_log_s, n.trees = 5000)
mean((exp(yhat.boost) - p1_log_s$Y1)^2)

## Prediction
set.seed(1124)
p1 <- read.csv("./Train_by pro/Train_P1.csv")
p1 <- p1[-1]
p1 <- data.frame(log(p1$Y1), log(p1$P1), p1[,-4:-5])
colnames(p1) <- c("log_y", "log_p","F1", "D1", "PR1")

pte1 <- read.csv("./Test_by pro/Test_P1.csv")
pte1 <- pte1[-1]
pte1 <- data.frame(log(pte1$P1), pte1[-4])
colnames(pte1) <- c("log_p","F1", "D1", "PR1")

boost.p1tf <- gbm(log_y ~.-PR1, p1, distribution = "tdist",
                 n.trees = 5000, interaction.depth = 4)

yhat.boost <- predict(boost.p1tf, newdata = pte1, n.trees = 5000)
write.csv(yhat.boost, file = "./predict/p1.csv")



## Product 2 - Cigarettes
########
set.seed(1124)
p2 <- read.csv("./Train_by pro/Train_P2.csv")
p2 <- p2[-1]
cor(p2[,1:4])

# split the data into training and testing
set.seed(1124)
train <- sample(1:length(p2[,1]), length(p2[, 1])*0.7)
p2_train <- p2[train, ]
p2_test <- p2[-train, ]
set.seed(1124)

p2_log <- data.frame(log(p2_train$Y2), log(p2_train$P2), p2_train[,-4:-5])
colnames(p2_log) <- c("log_y", "log_p","F2", "D2", "PR2")
p2_log_s <- data.frame(log(p2_test$P2), p2_test[-4])
colnames(p2_log_s) <- c("log_p","F2", "D2", "PR2", "Y2")

boost.p2t <- gbm(log_y ~.-F2-D2, p2_log, distribution = "tdist",
                 n.trees = 5000, interaction.depth = 2)
summary(boost.p2t)
yhat.boost <- predict(boost.p2t, newdata = p2_log_s, n.trees = 5000)
mean((exp(yhat.boost) - p2_log_s$Y2)^2)

## Product 3 - Coffee
set.seed(1124)
p3 <- read.csv("./Train_by pro/Train_P3.csv")
p3 <- p3[-1]
cor(p3[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p3[,1]), length(p3[, 1])*0.7)
p3_train <- p3[train, ]
p3_test <- p3[-train, ]

set.seed(1124)
p3_log <- data.frame(log(p3_train$Y3), log(p3_train$P3), p3_train[,-4:-5])
colnames(p3_log) <- c("log_y", "log_p","F3", "D3", "PR3")
p3_log_s <- data.frame(log(p3_test$P3), p3_test[-4])
colnames(p3_log_s) <- c("log_p","F3", "D3", "PR3", "Y3")

boost.p3t <- gbm(log_y ~., p3_log, distribution = "tdist",
                 n.trees = 100000, interaction.depth = 2)
summary(boost.p3t)
yhat.boost <- predict(boost.p3t, newdata = p3_log_s, n.trees = 100000)
mean((exp(yhat.boost) - p3_log_s$Y3)^2)

## Product 4 - Cold Cereal
set.seed(1124)
p4 <- read.csv("./Train_by pro/Train_P4.csv")
p4 <- p4[-1]
cor(p4[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p4[,1]), length(p4[, 1])*0.7)
p4_train <- p4[train, ]
p4_test <- p4[-train, ]

set.seed(1124)
p4_log <- data.frame(log(p4_train$Y4), log(p4_train$P4), p4_train[,-4:-5])
colnames(p4_log) <- c("log_y", "log_p","F4", "D4", "PR4")
p4_log_s <- data.frame(log(p4_test$P4), p4_test[-4])
colnames(p4_log_s) <- c("log_p","F4", "D4", "PR4", "Y4")

boost.p4t <- gbm(log_y ~., p4_log, distribution = "tdist",
                 n.trees = 5000, interaction.depth = 2)
summary(boost.p4t)
yhat.boost <- predict(boost.p4t, newdata = p4_log_s, n.trees = 5000)
mean((exp(yhat.boost) - p4_log_s$Y4)^2)

## Product 5 -Deodorant
set.seed(1124)
p5 <- read.csv("./Train_by pro/Train_P5.csv")
p5 <- p5[-1]
cor(p5[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p5[,1]), length(p5[, 1])*0.7)
p5_train <- p5[train, ]
p5_test <- p5[-train, ]

set.seed(1124)
p5_log <- data.frame(log(p5_train$Y5), log(p5_train$P5), p5_train[,-4:-5])
colnames(p5_log) <- c("log_y", "log_p","F5", "D5", "PR5")
p5_log_s <- data.frame(log(p5_test$P5), p5_test[-4])
colnames(p5_log_s) <- c("log_p","F5", "D5", "PR5", "Y5")

boost.p5t <- gbm(log_y ~., p5_log, distribution = "tdist",
                 n.trees = 5000, interaction.depth = 7)
summary(boost.p5t)
yhat.boost <- predict(boost.p5t, newdata = p5_log_s, n.trees = 5000)
mean((exp(yhat.boost) - p5_log_s$Y5)^2)

## Product 6 - Diapers
set.seed(1124)
p6 <- read.csv("./Train_by pro/Train_P6.csv")
p6 <- p6[-1]
cor(p6[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p6[,1]), length(p6[, 1])*0.7)
p6_train <- p6[train, ]
p6_test <- p6[-train, ]

set.seed(1124)
p6_log <- data.frame(log(p6_train$Y6), log(p6_train$P6), p6_train[,-4:-5])
colnames(p6_log) <- c("log_y", "log_p","F6", "D6", "PR6")
p6_log_s <- data.frame(log(p6_test$P6), p6_test[-4])
colnames(p6_log_s) <- c("log_p","F6", "D6", "PR6", "Y6")

boost.p6t <- gbm(log_y ~., p6_log, distribution = "tdist",
                 n.trees = 7000, interaction.depth = 7)
summary(boost.p6t)
yhat.boost <- predict(boost.p6t, newdata = p6_log_s, n.trees = 7000)
mean((exp(yhat.boost) - p6_log_s$Y6)^2)

## Product 7 - Face Tissue
set.seed(1124)
p7 <- read.csv("./Train_by pro/Train_P7.csv")
p7 <- p7[-1]
cor(p7[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p7[,1]), length(p7[, 1])*0.7)
p7_train <- p7[train, ]
p7_test <- p7[-train, ]

set.seed(1124)
p7_log <- data.frame(log(p7_train$Y7), log(p7_train$P7), p7_train[,-4:-5])
colnames(p7_log) <- c("log_y", "log_p","F7", "D7", "PR7")
p7_log_s <- data.frame(log(p7_test$P7), p7_test[-4])
colnames(p7_log_s) <- c("log_p","F7", "D7", "PR7", "Y7")

boost.p7t <- gbm(log_y ~., p7_log, distribution = "tdist",
                 n.trees = 5000, interaction.depth = 3)
summary(boost.p7t)
yhat.boost <- predict(boost.p7t, newdata = p7_log_s, n.trees = 5000)
mean((exp(yhat.boost) - p7_log_s$Y7)^2)

## 8. Frozen Dinner Entre
set.seed(1124)
p8 <- read.csv("./Train_by pro/Train_P8.csv")
p8 <- p8[-1]
cor(p8[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p8[,1]), length(p8[, 1])*0.7)
p8_train <- p8[train, ]
p8_test <- p8[-train, ]

set.seed(1124)
p8_log <- data.frame(log(p8_train$Y8), log(p8_train$P8), p8_train[,-4:-5])
colnames(p8_log) <- c("log_y", "log_p","F8", "D8", "PR8")
p8_log_s <- data.frame(log(p8_test$P8), p8_test[-4])
colnames(p8_log_s) <- c("log_p","F8", "D8", "PR8", "Y8")

set.seed(1124)
boost.p8t <- gbm(log_y ~., p8_log, distribution = "tdist",
                 n.trees = 4000, interaction.depth = 2)
summary(boost.p8t)
yhat.boost <- predict(boost.p8t, newdata = p8_log_s, n.trees = 4000)
mean((exp(yhat.boost) - p8_log_s$Y8)^2)

## 9. Frozen Pizza

set.seed(1124)
p9 <- read.csv("./Train_by pro/Train_P9.csv")
p9 <- p9[-1]
cor(p9[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p9[,1]), length(p9[, 1])*0.7)
p9_train <- p9[train, ]
p9_test <- p9[-train, ]

set.seed(1124)
p9_log <- data.frame(log(p9_train$Y9), log(p9_train$P9), p9_train[,-4:-5])
colnames(p9_log) <- c("log_y", "log_p","F9", "D9", "PR9")
p9_log_s <- data.frame(log(p9_test$P9), p9_test[-4])
colnames(p9_log_s) <- c("log_p","F9", "D9", "PR9", "Y9")

set.seed(1124)
boost.p9t <- gbm(log_y ~., p9_log, distribution = "tdist",
                 n.trees = 5000, interaction.depth = 3)
summary(boost.p9t)
yhat.boost <- predict(boost.p9t, newdata = p9_log_s, n.trees = 5000)
mean((exp(yhat.boost) - p9_log_s$Y9)^2)

## 10. Hot Dog

set.seed(1124)
p10 <- read.csv("./Train_by pro/Train_P10.csv")
p10 <- p10[-1]
cor(p10[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p10[,1]), length(p10[, 1])*0.7)
p10_train <- p10[train, ]
p10_test <- p10[-train, ]

set.seed(1124)
p10_log <- data.frame(log(p10_train$Y10), log(p10_train$P10), p10_train[,-4:-5])
colnames(p10_log) <- c("log_y", "log_p","F10", "D10", "PR10")
p10_log_s <- data.frame(log(p10_test$P10), p10_test[-4])
colnames(p10_log_s) <- c("log_p","F10", "D10", "PR10", "Y10")

set.seed(1124)
boost.p10t <- gbm(log_y ~., p10_log, distribution = "tdist",
                  n.trees = 5000, interaction.depth = 3)
summary(boost.p10t)
yhat.boost <- predict(boost.p10t, newdata = p10_log_s, n.trees = 5000)
mean((exp(yhat.boost) - p10_log_s$Y10)^2)

## 11. Laundry Detergent

set.seed(1124)
p11 <- read.csv("./Train_by pro/Train_P11.csv")
p11 <- p11[-1]
cor(p11[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p11[,1]), length(p11[, 1])*0.7)
p11_train <- p11[train, ]
p11_test <- p11[-train, ]

set.seed(1124)
p11_log <- data.frame(log(p11_train$Y11), log(p11_train$P11), p11_train[,-4:-5])
colnames(p11_log) <- c("log_y", "log_p","F11", "D11", "PR11")
p11_log_s <- data.frame(log(p11_test$P11), p11_test[-4])
colnames(p11_log_s) <- c("log_p","F11", "D11", "PR11", "Y11")

set.seed(1124)
boost.p11t <- gbm(log_y ~., p11_log, distribution = "tdist",
                  n.trees = 5000, interaction.depth = 3)
summary(boost.p11t)
yhat.boost <- predict(boost.p11t, newdata = p11_log_s, n.trees = 5000)
mean((exp(yhat.boost) - p11_log_s$Y11)^2)


## 12. Margarine & Butter
set.seed(1124)
p12 <- read.csv("./Train_by pro/Train_P12.csv")
p12 <- p12[-1]
cor(p12[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p12[,1]), length(p12[, 1])*0.7)
p12_train <- p12[train, ]
p12_test <- p12[-train, ]

set.seed(1124)
p12_log <- data.frame(log(p12_train$Y12), log(p12_train$P12), p12_train[,-4:-5])
colnames(p12_log) <- c("log_y", "log_p","F12", "D12", "PR12")
p12_log_s <- data.frame(log(p12_test$P12), p12_test[-4])
colnames(p12_log_s) <- c("log_p","F12", "D12", "PR12", "Y12")

set.seed(1124)
boost.p12t <- gbm(log_y ~., p12_log, distribution = "tdist",
                  n.trees = 7000, interaction.depth = 2)
summary(boost.p12t)
yhat.boost <- predict(boost.p12t, newdata = p12_log_s, n.trees = 7000)
mean((exp(yhat.boost) - p12_log_s$Y12)^2)

## 13. Mayonnaise

set.seed(1124)
p13 <- read.csv("./Train_by pro/Train_P13.csv")
p13 <- p13[-1]
cor(p13[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p13[,1]), length(p13[, 1])*0.7)
p13_train <- p13[train, ]
p13_test <- p13[-train, ]

set.seed(1124)
p13_log <- data.frame(log(p13_train$Y13), log(p13_train$P13), p13_train[,-4:-5])
colnames(p13_log) <- c("log_y", "log_p","F13", "D13", "PR13")
p13_log_s <- data.frame(log(p13_test$P13), p13_test[-4])
colnames(p13_log_s) <- c("log_p","F13", "D13", "PR13", "Y13")

set.seed(1124)
boost.p13t <- gbm(log_y ~., p13_log, distribution = "tdist",
                  n.trees = 5000, interaction.depth = 3)
summary(boost.p13t)
yhat.boost <- predict(boost.p13t, newdata = p13_log_s, n.trees = 5000)
mean((exp(yhat.boost) - p13_log_s$Y13)^2)

## 14. Mustard & Ketchup

set.seed(1124)
p14 <- read.csv("./Train_by pro/Train_P14.csv")
p14 <- p14[-1]
cor(p14[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p14[,1]), length(p14[, 1])*0.7)
p14_train <- p14[train, ]
p14_test <- p14[-train, ]

set.seed(1124)
p14_log <- data.frame(log(p14_train$Y14), log(p14_train$P14), p14_train[,-4:-5])
colnames(p14_log) <- c("log_y", "log_p","F14", "D14", "PR14")
p14_log_s <- data.frame(log(p14_test$P14), p14_test[-4])
colnames(p14_log_s) <- c("log_p","F14", "D14", "PR14", "Y14")

set.seed(1124)
boost.p14t <- gbm(log_y ~., p14_log, distribution = "tdist",
                  n.trees = 5000, interaction.depth = 2)
summary(boost.p14t)
yhat.boost <- predict(boost.p14t, newdata = p14_log_s, n.trees = 5000)
mean((exp(yhat.boost) - p14_log_s$Y14)^2)

## 15. Paper Towel

set.seed(1124)
p15 <- read.csv("./Train_by pro/Train_P15.csv")
p15 <- p15[-1]
cor(p15[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p15[,1]), length(p15[, 1])*0.7)
p15_train <- p15[train, ]
p15_test <- p15[-train, ]

set.seed(1124)
p15_log <- data.frame(log(p15_train$Y15), log(p15_train$P15), p15_train[,-4:-5])
colnames(p15_log) <- c("log_y", "log_p","F15", "D15", "PR15")
p15_log_s <- data.frame(log(p15_test$P15), p15_test[-4])
colnames(p15_log_s) <- c("log_p","F15", "D15", "PR15", "Y15")

set.seed(1124)
boost.p15t <- gbm(log_y ~., p15_log, distribution = "tdist",
                  n.trees = 5000, interaction.depth = 4)
summary(boost.p15t)
yhat.boost <- predict(boost.p15t, newdata = p15_log_s, n.trees = 5000)
mean((exp(yhat.boost) - p15_log_s$Y15)^2)

## 16. Peanut Butter

set.seed(1124)
p16 <- read.csv("./Train_by pro/Train_P16.csv")
p16 <- p16[-1]
cor(p16[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p16[,1]), length(p16[, 1])*0.7)
p16_train <- p16[train, ]
p16_test <- p16[-train, ]

set.seed(1124)
p16_log <- data.frame(log(p16_train$Y16), log(p16_train$P16), p16_train[,-4:-5])
colnames(p16_log) <- c("log_y", "log_p","F16", "D16", "PR16")
p16_log_s <- data.frame(log(p16_test$P16), p16_test[-4])
colnames(p16_log_s) <- c("log_p","F16", "D16", "PR16", "Y16")

set.seed(1124)
boost.p16t <- gbm(log_y ~., p16_log, distribution = "tdist",
                  n.trees = 2000, interaction.depth = 4)
summary(boost.p16t)
yhat.boost <- predict(boost.p16t, newdata = p16_log_s, n.trees = 2000)
mean((exp(yhat.boost) - p16_log_s$Y16)^2)

## 17. Shampoo

set.seed(1124)
p17 <- read.csv("./Train_by pro/Train_P17.csv")
p17 <- p17[-1]
cor(p17[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p17[,1]), length(p17[, 1])*0.7)
p17_train <- p17[train, ]
p17_test <- p17[-train, ]

set.seed(1124)
p17_log <- data.frame(log(p17_train$Y17), log(p17_train$P17), p17_train[,-4:-5])
colnames(p17_log) <- c("log_y", "log_p","F17", "D17", "PR17")
p17_log_s <- data.frame(log(p17_test$P17), p17_test[-4])
colnames(p17_log_s) <- c("log_p","F17", "D17", "PR17", "Y17")

set.seed(1124)
boost.p17t <- gbm(log_y ~.-D17, p17_log, distribution = "tdist",
                  n.trees = 1000, interaction.depth = 3)
summary(boost.p17t)
yhat.boost <- predict(boost.p17t, newdata = p17_log_s, n.trees = 1000)
mean((exp(yhat.boost) - p17_log_s$Y17)^2)

## 18. Soup

set.seed(1124)
p18 <- read.csv("./Train_by pro/Train_P18.csv")
p18 <- p18[-1]
cor(p18[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p18[,1]), length(p18[, 1])*0.7)
p18_train <- p18[train, ]
p18_test <- p18[-train, ]

set.seed(1124)
p18_log <- data.frame(log(p18_train$Y18), log(p18_train$P18), p18_train[,-4:-5])
colnames(p18_log) <- c("log_y", "log_p","F18", "D18", "PR18")
p18_log_s <- data.frame(log(p18_test$P18), p18_test[-4])
colnames(p18_log_s) <- c("log_p","F18", "D18", "PR18", "Y18")

set.seed(1124)
boost.p18t <- gbm(log_y ~., p18_log, distribution = "tdist",
                  n.trees = 5000, interaction.depth = 2)
summary(boost.p18t)
yhat.boost <- predict(boost.p18t, newdata = p18_log_s, n.trees = 5000)
mean((exp(yhat.boost) - p18_log_s$Y18)^2)

## 19. Spaghetti Sauce
set.seed(1124)
p19 <- read.csv("./Train_by pro/Train_P19.csv")
p19 <- p19[-1]
cor(p19[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p19[,1]), length(p19[, 1])*0.7)
p19_train <- p19[train, ]
p19_test <- p19[-train, ]

set.seed(1124)
p19_log <- data.frame(log(p19_train$Y19), log(p19_train$P19), p19_train[,-4:-5])
colnames(p19_log) <- c("log_y", "log_p","F19", "D19", "PR19")
p19_log_s <- data.frame(log(p19_test$P19), p19_test[-4])
colnames(p19_log_s) <- c("log_p","F19", "D19", "PR19", "Y19")

set.seed(1124)
boost.p19t <- gbm(log_y ~., p19_log, distribution = "tdist",
                  n.trees = 5000, interaction.depth = 3)
summary(boost.p19t)
yhat.boost <- predict(boost.p19t, newdata = p19_log_s, n.trees = 5000)
mean((exp(yhat.boost) - p19_log_s$Y19)^2)

## 20. Sugar Substitute
set.seed(1124)
p20 <- read.csv("./Train_by pro/Train_P20.csv")
p20 <- p20[-1]
cor(p20[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p20[,1]), length(p20[, 1])*0.7)
p20_train <- p20[train, ]
p20_test <- p20[-train, ]

set.seed(1124)
p20_log <- data.frame(log(p20_train$Y20), log(p20_train$P20), p20_train[,-4:-5])
colnames(p20_log) <- c("log_y", "log_p","F20", "D20", "PR20")
p20_log_s <- data.frame(log(p20_test$P20), p20_test[-4])
colnames(p20_log_s) <- c("log_p","F20", "D20", "PR20", "Y20")

set.seed(1124)
boost.p20t <- gbm(log_y ~., p20_log, distribution = "tdist",
                  n.trees = 5000, interaction.depth = 2)
summary(boost.p20t)
yhat.boost <- predict(boost.p20t, newdata = p20_log_s, n.trees = 5000)
mean((exp(yhat.boost) - p20_log_s$Y20)^2)

## 21. Toilet Tissues 
set.seed(1124)
p21 <- read.csv("./Train_by pro/Train_P21.csv")
p21 <- p21[-1]
cor(p21[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p21[,1]), length(p21[, 1])*0.7)
p21_train <- p21[train, ]
p21_test <- p21[-train, ]

set.seed(1124)
p21_log <- data.frame(log(p21_train$Y21), log(p21_train$P21), p21_train[,-4:-5])
colnames(p21_log) <- c("log_y", "log_p","F21", "D21", "PR21")
p21_log_s <- data.frame(log(p21_test$P21), p21_test[-4])
colnames(p21_log_s) <- c("log_p","F21", "D21", "PR21", "Y21")

set.seed(1124)
boost.p21t <- gbm(log_y ~., p21_log, distribution = "tdist",
                  n.trees = 5000, interaction.depth = 2)
summary(boost.p21t)
yhat.boost <- predict(boost.p21t, newdata = p21_log_s, n.trees = 5000)
mean((exp(yhat.boost) - p21_log_s$Y21)^2)

## 22. Tooth Paste

set.seed(1124)
p22 <- read.csv("./Train_by pro/Train_P22.csv")
p22 <- p22[-1]
cor(p22[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p22[,1]), length(p22[, 1])*0.7)
p22_train <- p22[train, ]
p22_test <- p22[-train, ]

set.seed(1124)
p22_log <- data.frame(log(p22_train$Y22), log(p22_train$P22), p22_train[,-4:-5])
colnames(p22_log) <- c("log_y", "log_p","F22", "D22", "PR22")
p22_log_s <- data.frame(log(p22_test$P22), p22_test[-4])
colnames(p22_log_s) <- c("log_p","F22", "D22", "PR22", "Y22")

set.seed(1124)
boost.p22t <- gbm(log_y ~., p22_log, distribution = "tdist",
                  n.trees = 20000, interaction.depth = 3)
summary(boost.p22t)
yhat.boost <- predict(boost.p22t, newdata = p22_log_s, n.trees = 20000)
mean((exp(yhat.boost) - p22_log_s$Y22)^2)

## 23. Yogurt
set.seed(1124)
p23 <- read.csv("./Train_by pro/Train_P23.csv")
p23 <- p23[-1]
cor(p23[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p23[,1]), length(p23[, 1])*0.7)
p23_train <- p23[train, ]
p23_test <- p23[-train, ]

set.seed(1124)
p23_log <- data.frame(log(p23_train$Y23), log(p23_train$P23), p23_train[,-4:-5])
colnames(p23_log) <- c("log_y", "log_p","F23", "D23", "PR23")
p23_log_s <- data.frame(log(p23_test$P23), p23_test[-4])
colnames(p23_log_s) <- c("log_p","F23", "D23", "PR23", "Y23")

set.seed(1124)
boost.p23t <- gbm(log_y ~., p23_log, distribution = "tdist",
                  n.trees = 2000, interaction.depth = 3)
summary(boost.p23t)
yhat.boost <- predict(boost.p23t, newdata = p23_log_s, n.trees = 2000)
mean((exp(yhat.boost) - p23_log_s$Y23)^2)

## 24. Beer

set.seed(1124)
p24 <- read.csv("./Train_by pro/Train_P24.csv")
p24 <- p24[-1]
cor(p24[,1:4])

# split the data into training and testing

set.seed(1124)
train <- sample(1:length(p24[,1]), length(p24[, 1])*0.7)
p24_train <- p24[train, ]
p24_test <- p24[-train, ]

set.seed(1124)
p24_log <- data.frame(log(p24_train$Y24), log(p24_train$P24), p24_train[,-4:-5])
colnames(p24_log) <- c("log_y", "log_p","F24", "D24", "PR24")
p24_log_s <- data.frame(log(p24_test$P24), p24_test[-4])
colnames(p24_log_s) <- c("log_p","F24", "D2$4", "PR24", "Y24")

set.seed(1124)
boost.p24t <- gbm(log_y ~.-log_p, p24_log, distribution = "tdist",
                  n.trees = 1000, interaction.depth = 3)
summary(boost.p24t)
yhat.boost <- predict(boost.p24t, newdata = p24_log_s, n.trees = 1000)
mean((exp(yhat.boost) - p24_log_s$Y24)^2)

######
###### Directly MARS using 96 predictors
# For 500s Kaggle Competition
setwd("/Users/ai/Desktop/MKT.500S - Predictive Analytics for Business Decisions-Making/Kaggle")
data_tr <- read.csv("Train.csv")
data_te <- read.csv("Test.csv")

library(dplyr)
set.seed(1124)
Y <- dplyr::select(data_tr, starts_with("Y"))
data_tr_p <- dplyr::select(data_tr, -starts_with("Y"))
library(earth)
set.seed(1124)

#####
train <- sample(1:length(data_tr_p[,1]), length(data_tr_p[, 1])*0.7)
data_train <- data_tr_p[train, ]
data_test <- data_tr_p[-train, ]
Y_train <- Y[train, ]
Y_test <- Y[-train, ]
#####

p1 <- cbind(data_tr_p, Y[1])
p1 <- p1[-1]
model1 = earth(p1$Y1 ~., data = p1, degree = 1)
pred.y1 = predict(model1, newdata = data_te)
colnames(pred.y1) <- c("y_hat_1")

p2 <- cbind(data_tr_p, Y[2])
p2 <- p2[-1]
model2 = earth(p2$Y2 ~., data = p2, degree = 1)
pred.y2 = predict(model2, newdata = data_te)
colnames(pred.y2) <- c("y_hat_2")

# nornal MARS, degree = 1
setwd("/Users/ai/Desktop/MKT.500S - Predictive Analytics for Business Decisions-Making/Kaggle")
data_tr <- read.csv("Train.csv")
data_te <- read.csv("Test.csv")

library(dplyr)
set.seed(1124)
Y <- dplyr::select(data_tr, starts_with("Y"))
data_tr_p <- dplyr::select(data_tr, -starts_with("Y"))
library(earth)
set.seed(1124)

p <- matrix(nrow = 250, ncol = 97)
pred_all <- matrix(nrow = 62, ncol = 24)
i <- 1
while(i <= 24){
    p <- cbind(data_tr_p, Y[i])
    p <- p[-1]
    # c <- colnames(Y[i])
    model <- earth(x = data_tr_p, y = Y[i], degree = 1)
    pred <- predict(model, newdata = data_te)
    pred_all[, i] <- pred 
    i <- i + 1
}
write.csv(pred_all, file = "./noral_MARS.csv")

# nornal MARS, degree = 2 not good 456
p <- matrix(nrow = 250, ncol = 97)
pred_all <- matrix(nrow = 62, ncol = 24)
i <- 1
while(i <= 24){
    p <- cbind(data_tr_p, Y[i])
    p <- p[-1]
    c <- colnames(Y[i])
    model <- earth(x = data_tr_p, y = Y[i], degree = 2)
    pred <- predict(model, newdata = data_te)
    pred_all[, i] <- pred 
    i <- i + 1
}
write.csv(pred_all, file = "./noral_MARS_d2.csv")


## scan_pro MARS 414
p <- matrix(nrow = 250, ncol = 97)
pred_all <- matrix(nrow = 62, ncol = 24)
i <- 1
data_tr_p <- data_tr_p[-1]
log.data_tr <- data_tr_p

data_te <- data_te[-1]
log.data_te <- data_te

j <- 4
while(j <= 96){
    log.data_tr[, j] <- log(log.data_tr[, j])
    log.data_te[, j] <- log(log.data_te[, j])
    j <- j+4
}

while(i <= 24){
    p <- cbind(log.data_tr, log(Y[i]))
    c <- colnames(Y[i])
    model <- earth(x = log.data_tr, y = log(Y[i]), degree = 1)
    pred <- predict(model, newdata = log.data_te)
    pred_all[, i] <- exp(pred) 
    i <- i + 1
}
write.csv(pred_all, file = "./scanpro_MARS.csv")

# Lasso prediction 347
library(glmnet)
grid <- 10^seq(10, -2, length=100)

setwd("/Users/ai/Desktop/MKT.500S - Predictive Analytics for Business Decisions-Making/Kaggle")
data_tr <- read.csv("Train.csv")
data_te <- read.csv("Test.csv")

library(dplyr)
set.seed(1124)
Y <- dplyr::select(data_tr, starts_with("Y"))
data_tr_p <- dplyr::select(data_tr, -starts_with("Y"))
library(earth)
set.seed(1124)
data_tr_p <- data_tr_p[-1]
data_te <- data_te[-1]

p <- matrix(nrow = 250, ncol = 97)
pred_all <- matrix(nrow = 62, ncol = 24)
i <- 1

while (i <= 24){
    i
    p <- cbind(data_tr_p, Y[i])
    x <- model.matrix(p$Y1 ~., p)[, -1]
    y <- Y[,i]
    lasso.mod <- glmnet(x, y, alpha = 1, lambda = grid)
    cv.out <- cv.glmnet(x, y, alpha = 1)
    bestlam <- cv.out$lambda.min
    lasso.pred <- predict(lasso.mod, x = x, y = y, s = bestlam, newx = as.matrix(data_te), exact = TRUE)
    pred_all[,i] <- lasso.pred
    i <- i + 1
    i
}
write.csv(pred_all, file = "./normal_Lasso.csv")

## log lasso
library(glmnet)
grid <- 10^seq(10, -2, length=100)

setwd("/Users/ai/Desktop/MKT.500S - Predictive Analytics for Business Decisions-Making/Kaggle")
data_tr <- read.csv("Train.csv")
data_te <- read.csv("Test.csv")

library(dplyr)
set.seed(1124)
Y <- dplyr::select(data_tr, starts_with("Y"))
data_tr_p <- dplyr::select(data_tr, -starts_with("Y"))
set.seed(1124)
data_tr_p <- data_tr_p[-1]
data_te <- data_te[-1]

log.data_tr <- data_tr_p
log.data_te <- data_te

j <- 4
while(j <= 96){
    log.data_tr[, j] <- log(log.data_tr[, j])
    log.data_te[, j] <- log(log.data_te[, j])
    j <- j+4
}

p <- matrix(nrow = 250, ncol = 97)
pred_all <- matrix(nrow = 62, ncol = 24)
i <- 1

while (i <= 24){
    p <- cbind(log.data_tr, log(Y[i]))
    c <- colnames(p)
    c <- c[-length(c)]
    c <- c(c, "log.Y")
    colnames(p) <- c
    x <- model.matrix(log.Y ~., p)[, -1]
    y <- log(Y[,i])
    lasso.mod <- glmnet(x, y, alpha = 1, lambda = grid)
    cv.out <- cv.glmnet(x, y, alpha = 1)
    bestlam <- cv.out$lambda.min
    lasso.pred <- predict(lasso.mod, x = x, y = y, s = bestlam, newx = as.matrix(log.data_te), exact = TRUE)
    pred_all[,i] <- exp(lasso.pred)
    i <- i + 1
    i
}
write.csv(pred_all, file = "./log_Lasso.csv")


## Lasso subset selection
out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef = predict(out,type="coefficients",s=bestlam)[1:20,]

## normal boosted 580
setwd("/Users/ai/Desktop/MKT.500S - Predictive Analytics for Business Decisions-Making/Kaggle")
data_tr <- read.csv("Train.csv")
data_te <- read.csv("Test.csv")

library(dplyr)
set.seed(1124)
Y <- dplyr::select(data_tr, starts_with("Y"))
data_tr_p <- dplyr::select(data_tr, -starts_with("Y"))
data_tr_p <- data_tr_p[-1]
library(gbm)
set.seed(1124)

p <- matrix(nrow = 250, ncol = 97)
pred_all <- matrix(nrow = 62, ncol = 24)

i <- 1
while(i <= 24){
    p <- cbind(data_tr_p, Y[i])
    c <- colnames(p)
    c <- c[-length(c)]
    c <- c(c, "Y")
    colnames(p) <- c
    set.seed(1124)
    boost <- gbm(Y ~., p, distribution = "tdist",
                     n.trees = 5000, interaction.depth = 2)
    pred <- predict(boost, newdata = data_te, n.trees = 5000)
    pred_all[, i] <- pred 
    i <- i + 1
}
write.csv(pred_all, file = "./normal_boost_2trees.csv")

## randonforest 368
setwd("/Users/ai/Desktop/MKT.500S - Predictive Analytics for Business Decisions-Making/Kaggle")
data_tr <- read.csv("Train.csv")
data_te <- read.csv("Test.csv")

library(dplyr)
set.seed(1124)
Y <- dplyr::select(data_tr, starts_with("Y"))
data_tr_p <- dplyr::select(data_tr, -starts_with("Y"))
data_tr_p <- data_tr_p[-1]
library(randomForest)
set.seed(1124)

p <- matrix(nrow = 250, ncol = 97)
pred_all <- matrix(nrow = 62, ncol = 24)

i <- 1
while(i <= 24){
    p <- cbind(data_tr_p, Y[i])
    c <- colnames(p)
    c <- c[-length(c)]
    c <- c(c, "Y")
    colnames(p) <- c
    set.seed(1124)
    rf <- randomForest(Y ~., p)
    pred <- predict(rf, newdata = data_te)
    pred_all[, i] <- pred 
    i <- i + 1
}
write.csv(pred_all, file = "./normal_randomForest.csv")

## scan pro random forest
setwd("/Users/ai/Desktop/MKT.500S - Predictive Analytics for Business Decisions-Making/Kaggle")
data_tr <- read.csv("Train.csv")
data_te <- read.csv("Test.csv")

library(dplyr)
set.seed(1124)
Y <- dplyr::select(data_tr, starts_with("Y"))
data_tr_p <- dplyr::select(data_tr, -starts_with("Y"))
data_tr_p <- data_tr_p[-1]
library(randomForest)
set.seed(1124)

log.data_tr <- data_tr_p

data_te <- data_te[-1]
log.data_te <- data_te

j <- 4
while(j <= 96){
    log.data_tr[, j] <- log(log.data_tr[, j])
    log.data_te[, j] <- log(log.data_te[, j])
    j <- j+4
}

p <- matrix(nrow = 250, ncol = 97)
pred_all <- matrix(nrow = 62, ncol = 24)

i <- 1
while(i <= 24){
    p <- cbind(log.data_tr, log(Y[i]))
    c <- colnames(p)
    c <- c[-length(c)]
    c <- c(c, "log.Y")
    colnames(p) <- c
    set.seed(1124)
    rf <- randomForest(log.Y ~., p, mtry = 1)
    pred <- predict(rf, newdata = log.data_te)
    pred_all[, i] <- exp(pred) 
    i <- i + 1
}
write.csv(pred_all, file = "./scanpro_randomForest,tr1.csv")

## pca standardized selection and then mars
setwd("/Users/ai/Desktop/MKT.500S - Predictive Analytics for Business Decisions-Making/Kaggle")
data_tr <- read.csv("Train.csv")
data_te <- read.csv("Test.csv")

library(dplyr)
set.seed(1124)
Y <- dplyr::select(data_tr, starts_with("Y"))
data_tr_p <- dplyr::select(data_tr, -starts_with("Y"))

## the result of pca is not that good
pr.out <- prcomp(data_tr_p, scale = TRUE)
pr.var <- pr.out$sdev^2
pve <- pr.var/sum(pr.var)

## using the time to make clusters
km.out <- kmeans(data_tr_p, 4)
plot(data_tr_p, 
     col=(km.out$cluster+1), 
     main = "K-Means Clustering Results with K = 4", pch=20,cex=2)


library(earth)
set.seed(1124)

p <- matrix(nrow = 250, ncol = 97)
pred_all <- matrix(nrow = 62, ncol = 24)

d <- matrix(nrow = 62, ncol = 24)
i <- 1
lalog <- lalog[-1]
rilog <- rilog[-1]
rf <- rf[-1]
ma <- ma[-1]
while(i <= 24){
    d[,i] <- (lalog[,i]+rilog[,i]+rf[,i]+ma[, i])/4
    i <- i+1
}
write.csv(d, file = "./Combine_loglasso+logridge+rf+mars.csv")

#### script for plot
library(forecast)
library(openxlsx)
rfilename <- paste("Store", 2, ".xlsx", sep = "")
data <- read.xlsx(rfilename)
plot(data$Y1, 
     main = "Store2 Product1 Sales Time Series Plot",
     xlab = "id/week", 
     ylab = "Product1 sales", 
     type = "l")
fit <- stl(data$Y1, s.window="period")
plot(fit)

## normal arima
setwd("/Users/ai/Desktop/MKT.500S - Predictive Analytics for Business Decisions-Making/Kaggle")
data_tr <- read.csv("Train.csv")
data_te <- read.csv("Test.csv")

library(dplyr)
set.seed(1124)
Y <- dplyr::select(data_tr, starts_with("Y"))
data_tr_p <- dplyr::select(data_tr, -starts_with("Y"))
data_tr_p <- data_tr_p[-1]
data_te <- data_te[-1]
library(forecast)
set.seed(1124)

i <- 1
while(i <= 24){
    Y[,i] <- ts(Y[,i], frequency = 52)
    i<- i+1
}

pred_all <- matrix(nrow = 62, ncol = 24)

i <- 1
while(i <= 24){
    a <- auto.arima(Y[, i])
    pp <- length(a$model$phi)
    q <- length(a$model$theta)
    d <- length(a$model$Delta)
    fit <- arima(Y[, i], order = c(pp, d, q), method = "ML")
    pred_all[,i] <- as.matrix(predict(fit)$pred)
    i <- i+1
}

