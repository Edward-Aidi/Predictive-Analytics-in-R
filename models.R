```{r S2P1, echo = TRUE}
library(openxlsx)
rfilename <- paste("Store", 2, ".xlsx", sep = "")
data <- read.xlsx(rfilename)
plot(data$Y1, 
     main = "Store2 Product1 Sales Time Series Plot",
     xlab = "id/week", 
     ylab = "Product1 sales", 
     type = "l")
```


setwd("/Users/ai/Desktop/MKT.500S - Predictive Analytics for Business Decisions-Making/Project")
library(openxlsx)
h <- 2
while(h <= 9){
    rfilename <- paste("Store", h, ".xlsx", sep = "")
    data <- read.xlsx(rfilename)
    
    library(dplyr)
    data_com <- dplyr::select(data, -one_of("Random", "Store", "Week"))
    data_all.predictors <- dplyr::select(data_com, -contains("Y"))
    data_all.Y <- dplyr::select(data_com, contains("Y"))
    
    data_tr <- data_com[data$Random == "Train", ]
    data_te <- data_com[data$Random != "Train", ]
    
    ## Linear Regression, using all the predictors
    i <- 1
    rmse.lm <- rep(0, 24)
    while(i <= 24){
        # combine all predictors with one y
        c <- colnames(data_all.predictors)
        y <- paste("Y", i, sep = "")
        c <- c(c,y)
        dat <- dplyr::select(data_tr, one_of(c))
        
        # change the name of columns
        c2 <- colnames(data_all.predictors)
        y2 <- "Y"
        c2 <- c(c2,y2)
        colnames(dat) <- c2
        
        # fit the linear regression using all the predictors
        lm.fit <- lm(Y ~., data = dat)
        lm.predict <- predict(lm.fit, newdata = data_te)
        ytrue <- dplyr::select(data_te, one_of(y))
        rmse.lm[i] <- sqrt(mean((lm.predict - ytrue)^2))
        i <- i+1
    }
    
    ## KNN
    
    ## Lasso Regression
    library(glmnet)
    set.seed(1)
    grid <- 10^seq(10, -2, length = 100)
    
    i <- 1
    rmse.la <- rep(0, 24)
    while (i <= 24){
        set.seed(1)
        # combine all predictors with one y
        c <- colnames(data_all.predictors)
        y <- paste("Y", i, sep = "")
        c <- c(c,y)
        dat <- dplyr::select(data_tr, one_of(c))
        dat.te <- data_all.predictors[data$Random != "Train", ]
        
        # change the name of columns
        c2 <- colnames(data_all.predictors)
        y2 <- "Y"
        c2 <- c(c2,y2)
        colnames(dat) <- c2
        
        # select reponses from train data
        x <- model.matrix(Y ~., dat)[, -1]
        yy <- data_all.Y[data$Random == "Train", ]
        
        lasso.mod <- glmnet(x = x, y = yy[,i], alpha = 1, lambda = grid)
        cv.out <- cv.glmnet(x = x, y = yy[,i], alpha = 1)
        bestlam <- cv.out$lambda.min
        lasso.pred <- predict(lasso.mod, 
                              x = x, y = yy[,i], 
                              s = bestlam, 
                              newx = as.matrix(dat.te), 
                              exact = TRUE)
        ytrue <- dplyr::select(data_te, one_of(y))
        rmse.la[i] <- sqrt(mean((lasso.pred - ytrue)[,1]^2))
        i <- i + 1
    }
    
    ## Lasso log
    library(glmnet)
    set.seed(1)
    grid <- 10^seq(10, -2, length = 100)
    
    # log transformation
    log.data <- data_com
    j <- 4
    while(j <= 96){
        log.data[, j] <- log(data_com[, j])
        j <- j+4
    }
    
    # lasso regression
    i <- 1
    rmse.la.log <- rep(0, 24)
    while (i <= 24){
        set.seed(1)
        # combine all predictors with one y
        c <- colnames(data_all.predictors)
        y <- paste("Y", i, sep = "")
        c <- c(c,y)
        dat <- dplyr::select(log.data, one_of(c))
        dat.tr <- dat[data$Random == "Train", ]
        dat.te <- log.data[data$Random != "Train", ]
        log.dat.te <- select(dat.te, -contains("Y"))
        
        # change the name of columns
        c2 <- colnames(data_all.predictors)
        y2 <- "Y"
        c2 <- c(c2,y2)
        colnames(dat.tr) <- c2
        
        # select reponses from train data
        x <- model.matrix(Y ~., dat.tr)[, -1]
        yy <- data_all.Y[data$Random == "Train", ]
        
        lasso.mod <- glmnet(x = x, y = log(yy[,i]), alpha = 1, lambda = grid)
        cv.out <- cv.glmnet(x = x, y = log(yy[,i]), alpha = 1)
        bestlam <- cv.out$lambda.min
        lasso.pred <- predict(lasso.mod, 
                              x = x, y = log(yy[,i]), 
                              s = bestlam, 
                              newx = as.matrix(log.dat.te), 
                              exact = TRUE)
        ytrue <- dplyr::select(data_te, one_of(y))
        rmse.la.log[i] <- sqrt(mean((exp(lasso.pred) - ytrue)[,1]^2))
        i <- i + 1
    }
    
    ## Ridge Regression
    library(glmnet)
    set.seed(1)
    grid <- 10^seq(10, -2, length = 100)
    
    i <- 1
    rmse.ri <- rep(0, 24)
    while (i <= 24){
        set.seed(1)
        # combine all predictors with one y
        c <- colnames(data_all.predictors)
        y <- paste("Y", i, sep = "")
        c <- c(c,y)
        dat <- dplyr::select(data_tr, one_of(c))
        dat.te <- data_all.predictors[data$Random != "Train", ]
        
        # change the name of columns
        c2 <- colnames(data_all.predictors)
        y2 <- "Y"
        c2 <- c(c2,y2)
        colnames(dat) <- c2
        
        # select reponses from train data
        x <- model.matrix(Y ~., dat)[, -1]
        yy <- data_all.Y[data$Random == "Train", ]
        
        ridge.mod <- glmnet(x = x, y = yy[,i], alpha = 0, lambda = grid)
        cv.out <- cv.glmnet(x = x, y = yy[,i], alpha = 0)
        bestlam <- cv.out$lambda.min
        ridge.pred <- predict(ridge.mod, 
                              x = x, y = yy[,i], 
                              s = bestlam, 
                              newx = as.matrix(dat.te), 
                              exact = TRUE)
        ytrue <- dplyr::select(data_te, one_of(y))
        rmse.ri[i] <- sqrt(mean((ridge.pred - ytrue)[, 1]^2))
        i <- i + 1
    }
    
    ## Ridge log
    library(glmnet)
    set.seed(1)
    grid <- 10^seq(10, -2, length = 100)
    
    # log transformation
    log.data <- data_com
    j <- 4
    while(j <= 96){
        log.data[, j] <- log(data_com[, j])
        j <- j+4
    }
    
    # lasso regression
    i <- 1
    rmse.ri.log <- rep(0, 24)
    while (i <= 24){
        set.seed(1)
        # combine all predictors with one y
        c <- colnames(data_all.predictors)
        y <- paste("Y", i, sep = "")
        c <- c(c,y)
        dat <- dplyr::select(log.data, one_of(c))
        dat.tr <- dat[data$Random == "Train", ]
        dat.te <- log.data[data$Random != "Train", ]
        log.dat.te <- select(dat.te, -contains("Y"))
        
        # change the name of columns
        c2 <- colnames(data_all.predictors)
        y2 <- "Y"
        c2 <- c(c2,y2)
        colnames(dat.tr) <- c2
        
        # select reponses from train data
        x <- model.matrix(Y ~., dat.tr)[, -1]
        yy <- data_all.Y[data$Random == "Train", ]
        
        ridge.mod <- glmnet(x = x, y = log(yy[,i]), alpha = 0, lambda = grid)
        cv.out <- cv.glmnet(x = x, y = log(yy[,i]), alpha = 0)
        bestlam <- cv.out$lambda.min
        ridge.pred <- predict(ridge.mod, 
                              x = x, y = log(yy[,i]), 
                              s = bestlam, 
                              newx = as.matrix(log.dat.te), 
                              exact = TRUE)
        ytrue <- dplyr::select(data_te, one_of(y))
        rmse.ri.log[i] <- sqrt(mean((exp(ridge.pred) - ytrue)[,1]^2))
        i <- i + 1
    }
    
    ## Bagging
    library(randomForest)
    set.seed(1)
    i <- 1
    rmse.bagging <- rep(0, 24)
    while(i <= 24){
        # combine all predictors with one y
        c <- colnames(data_all.predictors)
        y <- paste("Y", i, sep = "")
        c <- c(c,y)
        dat <- dplyr::select(data_tr, one_of(c))
        
        # change the name of columns
        c2 <- colnames(data_all.predictors)
        y2 <- "Y"
        c2 <- c(c2,y2)
        colnames(dat) <- c2
        
        set.seed(1)
        rf <- randomForest(Y ~., dat, mtry = length(data_all.predictors[1,]))
        pred <- predict(rf, newdata = data_te)
        ytrue <- dplyr::select(data_te, one_of(y))
        rmse.bagging[i] <- sqrt(mean((pred - ytrue)^2))
        i <- i + 1
    }
    
    ## Random Forest
    library(randomForest)
    set.seed(1)
    i <- 1
    rmse.rf <- rep(0, 24)
    while(i <= 24){
        # combine all predictors with one y
        c <- colnames(data_all.predictors)
        y <- paste("Y", i, sep = "")
        c <- c(c,y)
        dat <- dplyr::select(data_tr, one_of(c))
        
        # change the name of columns
        c2 <- colnames(data_all.predictors)
        y2 <- "Y"
        c2 <- c(c2,y2)
        colnames(dat) <- c2
        
        set.seed(1)
        rf <- randomForest(Y ~., dat)
        pred <- predict(rf, newdata = data_te)
        ytrue <- dplyr::select(data_te, one_of(y))
        rmse.rf[i] <- sqrt(mean((pred - ytrue)^2))
        i <- i + 1
    }
    
    ## Boosted
    library(gbm)
    set.seed(1)
    i <- 1
    rmse.bo <- rep(0, 24)
    while(i <= 24){
        # combine all predictors with one y
        c <- colnames(data_all.predictors)
        y <- paste("Y", i, sep = "")
        c <- c(c,y)
        dat <- dplyr::select(data_tr, one_of(c))
        
        # change the name of columns
        c2 <- colnames(data_all.predictors)
        y2 <- "Y"
        c2 <- c(c2,y2)
        colnames(dat) <- c2
        
        set.seed(1)
        boosted <- gbm(Y ~., dat,distribution = "tdist",
                       n.trees = 5000, interaction.depth = 3)
        pred <- predict(boosted, newdata = data_te, n.trees = 5000)
        ytrue <- dplyr::select(data_te, one_of(y))
        rmse.bo[i] <- sqrt(mean((pred - ytrue)^2))
        i <- i + 1
    }
    
    ## MARS log
    set.seed(1)
    library(earth)
    i <- 1
    rmse.mars.log <- rep(0, 24)
    
    # select predictors from train data
    dat <- dplyr::select(data_tr, -contains("Y"))
    log.data <- dat
    log.te <- data_te
    j <- 4
    while(j <= 96){
        log.data[, j] <- log(dat[, j])
        log.te[, j] <- log(data_te[, j])
        j <- j+4
    }
    yy <- data_all.Y[data$Random == "Train", ]
    
    while(i <= 24){
        model <- earth(x = log.data, y = log(yy[i]), degree = 1)
        pred <- predict(model, newdata = log.te)
        #select true y from test data
        ytrue <- dplyr::select(data_te, one_of(y))
        rmse.mars.log[i] <- sqrt(mean((exp(pred) - ytrue)^2))
        i <- i + 1
    }
    
    ## MARS
    set.seed(1)
    library(earth)
    i <- 1
    rmse.mars <- rep(0, 24)
    # select predictors from train data
    dat <- dplyr::select(data_tr, -contains("Y"))
    yy <- data_all.Y[data$Random == "Train", ]
    datte <- dplyr::select(data_te, -contains("Y"))
    while(i <= 24){
        model <- earth(x = dat, y = yy[,i], degree = 1)
        pred <- predict(model, newdata = dattte)
        #select true y from test data
        ytrue <- dplyr::select(data_te, one_of(y))
        rmse.mars[i] <- sqrt(mean((pred - ytrue)^2))
        i <- i + 1
    }
    
    ## MARS 2
    set.seed(1)
    library(earth)
    i <- 1
    rmse.mars2 <- rep(0, 24)
    # select predictors from train data
    dat <- dplyr::select(data_tr, -contains("Y"))
    yy <- data_all.Y[data$Random == "Train", ]
    
    while(i <= 24){
        model <- earth(x = dat, y = yy[i], degree = 2)
        pred <- predict(model, newdata = data_te)
        #select true y from test data
        ytrue <- dplyr::select(data_te, one_of(y))
        rmse.mars2[i] <- sqrt(mean((pred - ytrue)^2))
        i <- i + 1
    }
    
    ## MARS2 log
    set.seed(1)
    library(earth)
    i <- 1
    rmse.mars2.log <- rep(0, 24)
    
    # select predictors from train data
    dat <- dplyr::select(data_tr, -contains("Y"))
    log.data <- dat
    log.te <- data_te
    j <- 4
    while(j <= 96){
        log.data[, j] <- log(dat[, j])
        log.te[, j] <- log(data_te[, j])
        j <- j+4
    }
    yy <- data_all.Y[data$Random == "Train", ]
    
    while(i <= 24){
        model <- earth(x = log.data, y = log(yy[i]), degree = 2)
        pred <- predict(model, newdata = log.te)
        #select true y from test data
        ytrue <- dplyr::select(data_te, one_of(y))
        rmse.mars2.log[i] <- sqrt(mean((exp(pred) - ytrue)^2))
        i <- i + 1
    }
    
    result <- matrix(nrow = 24, ncol = 10)
    rmse.store <- cbind(rmse.la,
                        rmse.la.log,
                        rmse.ri,
                        rmse.ri.log,
                        rmse.mars,
                        rmse.mars2,
                        rmse.mars.log,
                        rmse.mars2.log,
                        rmse.bagging,
                        rmse.rf,
                        rmse.bo)
    
    fname <- paste("./rmse_Store", h, ".csv", sep = "")
    write.csv(rmse.store, file = fname)
    h <- h + 1
}
