# hw1 Predictive Analysis
##Problem 3: Exercise 8 from section 2.4 of the textbook (ISLR p. 54)
setwd("/Users/ai/Desktop/MKT.500S - Predictive Analytics for Business Decisions-Making/hw1")
college <- read.csv("College.csv")
rownames(college) <- college[,1]
college <- college[,-1]
s <- summary(college)
pairs(college[, 1:10])

boxplot(college$Outstate~college$Private,
        col = "royalblue", 
        xlab = "Private Schools?",
        ylab = "Out-of-State Tuition")

Elite = rep("No", nrow(college))
Elite[college$Top10perc > 50] <- "Yes"
Elite <- as.factor(Elite)
college <- data.frame(college, Elite)

summary(college$Elite)

boxplot(college$Outstate~college$Elite,
        col = "yellow", 
        xlab = "Elite Schools?",
        ylab = "Out-of-State Tuition")

par(mfrow=c(2,2))
hist(college$Apps)
hist(college$Accept)
hist(college$Enroll)
hist(college$Top10perc)

#Correlation
numer <- c('Room.Board', 'S.F.Ratio', 'perc.alumni', 'Expend')
data <- select(college, one_of(numer))
corr <- round(cor(data, use = "complete.obs"), 2)

fit <- lm(Outstate ~ Private + Elite + Room.Board + 
            S.F.Ratio + perc.alumni + Expend, data = college)
summary(fit)

##Problem 5:Exercise 9, parts (a)-(c) only, from section 3.7 of the textbook (ISLR p. 122)
pairs(Auto[-1, ])
cate <- c("cylinders", "origin", "name")
auto_quan <- select(Auto, -one_of(cate))
corr <- cor(auto_quan)

lm.fit <- lm(mpg ~ . - name, data = Auto)
summary(lm.fit)