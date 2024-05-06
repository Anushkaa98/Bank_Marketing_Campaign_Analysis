#### Installing and loading packages ####

#rm(list = ls())

#install.packages("pacman")
#install.packages("BiocManager")
#install.packages("caret")
#install.packages("InformationValue")
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("randomForest")

pacman::p_load(readxl, rio, tidyverse, ggplot2, dplyr, caret, rpart, rpart.plot, randomForest)

#### Importing Data ####

actual_bank_data <- read_excel("/Users/anushkamukherjee/Downloads/Bank Marketing Data.xlsx")

bank_data <- read_excel("/Users/anushkamukherjee/Downloads/Bank Marketing Data.xlsx") %>% as_tibble()

str(bank_data)

#### Data Cleaning ####

# check if there are any null values

sum(is.na(bank_data))

# Renaming "y" column as "term_deposit for comprehensibility

columns <- colnames(bank_data)

columns[17] <- "term_deposit"

colnames(bank_data) <- columns

colnames(bank_data)

# Attaching the dataset so the variables could be used directly in place of variable

attach(bank_data)

# Checking data type pf each column and making amendments if required

print(sapply(bank_data, class))

# Converting term_deposit datatype to factor after replacing "yes" with 1 and "no" to 0

c <- term_deposit

c <- vapply(c, gsub, pattern = "yes", replacement = 1, character(1), USE.NAMES = FALSE)

c <- vapply(c, gsub, pattern = "no", replacement = 0, character(1), USE.NAMES = FALSE)

bank_data$term_deposit <- as.numeric(c)

str(bank_data)

# Converting poutcome, job and month to factors

#bank_data$poutcome <- as.factor(bank_data$poutcome)

#job <- as.factor(job)

#month <- as.factor(month)

#### Data Visualization ####

# Deep dive into variables' distribution

dist <- function(df, data = bank_data){
  if(is.numeric(df)){
    return(list(Mean = mean(df), Maximum = max(df), Minimum = min(df), Median = median(df)))
  }
  else{
    print(paste("The provided argument is of type:", class(df)))
  }
}

#### Proportion function ####

proportion <- function(df,a,b,c){

  df_gb <- df %>% group_by({{a}},{{b}}) %>% summarize(total_count = n(), .groups = 'drop')
  for(i in 1:nrow(df_gb)){
    cat = as.character(df_gb[i,1])
    sum_of <- sum(subset(df_gb,df_gb[[c]]==cat)$total_count)
    df_gb[i,4] <- as.double(df_gb[i,3]/sum_of)
  }
  colnames(df_gb)[4] <- "proportion"
  print(df_gb)
  return(df_gb)
}

marital_term <- proportion(bank_data,marital, term_deposit,"marital")

ggplot(marital_term, aes(x = marital, y = proportion, fill=term_deposit)) + geom_col()

ggplot(marital_term%>%filter(term_deposit==1), aes(x = marital, y = proportion)) + geom_col()

marital_loan <- proportion(bank_data, marital, loan, "marital")

ggplot(marital_loan, aes(x = marital, y = proportion, fill=loan)) + geom_col()

ggplot(marital_loan%>%filter(loan=="yes"), aes(x = marital, y = proportion)) + geom_col()

job_loan <- proportion(bank_data,job,loan,"job")

education_loan <- proportion(bank_data, education, loan, "education")


# Age distribution across various marital class

ggplot(bank_data, aes(x = age)) + geom_histogram(bins = 10) + facet_wrap(~marital)

# What proportion of marital groups are going for a term deposit

marital_1 <- bank_data %>% group_by(marital, term_deposit) %>%
  summarize(total_count = n(),.groups = 'drop')

marital_1

for(i in 1:nrow(marital_1)){
  if(i == 1){
    temp = c()
  }
  cat <- as.character(marital_1[i,1])
  print(cat)
  marital_1[i,4]<-as.double(marital_1[i,3]/sum(subset(marital_1,marital==cat)$total_count))
}

colnames(marital_1)[4] <- "proportion"

ggplot(marital_1, aes(x = marital, y = proportion, fill = term_deposit)) + geom_col()

# More proportion of divorced and single are going for term deposits

# Age group for most term deposits

ggplot(bank_data%>%filter(term_deposit == 1), aes(x = age)) + geom_histogram(bins = 10)

# People in age range of 30-40 are going for term_deposits the most

# Which months showed highest positive results (term_deposit == 1)

result_month_1 <- bank_data %>% filter(term_deposit == 1) %>% group_by(month) %>% summarize(total_count = n(),.groups = 'drop') %>%
  arrange(desc(total_count)) %>% mutate(proportion = total_count/sum(total_count)*100)

result_month_1

sum(result_month_1$proportion)

ggplot(result_month_1, aes(x = reorder(month,total_count, decreasing = TRUE), y = total_count)) + geom_col() +
  labs(x = "Months", y = "Subscribers")

# which months showed the highest negative results

result_month_0 <- bank_data %>% filter(term_deposit == 0) %>% group_by(month) %>% summarize(total_count = n(),.groups = 'drop') %>%
  arrange(desc(total_count)) %>% mutate(proportion = total_count/sum(total_count)*100)

result_month_0

sum(result_month_0$proportion)

# visualize the month distribution

ggplot(result_month_0, aes(x = reorder(month,total_count, decreasing = TRUE), y = total_count)) + geom_col()+
  labs(x = "Months", y = "Non-Subscribers")

# Data distribution across different months

ggplot(dplyr::count(bank_data,month), aes(x = reorder(month,n, decreasing = TRUE), y = n)) + geom_col() + labs(x = "Months", y = "Observations")


ggplot(dplyr::count(bank_data%>%filter(term_deposit==1),education), aes(x = reorder(education,n), y = n)) + geom_col()

ggplot(dplyr::count(bank_data%>%filter(loan=="yes"),job), aes(x = reorder(job,n), y = n)) + geom_col()

ggplot(dplyr::count(bank_data%>%filter(term_deposit==1),job), aes(x = reorder(job,n), y = n)) + geom_col()

dplyr::count(bank_data, poutcome)

colnames(bank_data)

dist(loan)
unique(loan)
dplyr::count(bank_data, loan)
dplyr::count(bank_data%>%group_by(loan,term_deposit),term_deposit)

bank_data%>%group_by(loan,term_deposit)%>%summarize(total_count = n(),.groups = 'drop')

unique(job)

# Job distribution across dataset

dplyr::count(bank_data, job) %>% arrange(desc(n))

ggplot()


## Modeling for term_deposit ####

# Replacing -1 with 0 in pdays

temp <- bank_data$pdays
temp<-vapply(temp,gsub,pattern=-1,replacement=0,USE.NAMES = FALSE,FUN.VALUE = character(1))
bank_data$pdays <- as.numeric(temp)
bank_data$pdays
# Model 1

colnames(bank_data)

# Columns to be considered for predicting if an individual would opt for a term_deposit
# Continuous: age, balance, duration, campaign, pdays, previous
# categorical: job,marital,education,default,housing,loan,month

# creating dummy variables for categorical variables

cate <- bank_data%>%select(job,marital,education,default,housing,loan,month,age, balance, duration, campaign, pdays, previous)
cate_1 <- model.matrix(~.-1,data = cate) # Suppressing intercept

# Combining continuos and dummy variables for glm model

glm_data<-cbind(cate_1,bank_data%>%select(term_deposit))
colnames(glm_data)
glm_data <- glm_data[,-12] # Removing job uknown because of singularity
glm_data
# Partitioning data into training and validation

set.seed(11)

train_rows <- sample(row.names(glm_data),0.6*dim(glm_data)[1])
valid_rows <- setdiff(row.names(glm_data),train_rows)
train_df <- glm_data[train_rows,]
valid_df <- glm_data[valid_rows,]
dim(train_df)
dim(valid_df)
dim(glm_data)
print(2712+1809)
train_df

term_deposit_glm <- glm(train_df$term_deposit~., data = train_df, family="binomial")

summary(term_deposit_glm)

term_deposit_glm_pred <- predict(term_deposit_glm, valid_df[,-37], type = 'response')
ggplot(as.data.frame(term_deposit_glm_pred),aes(x = term_deposit_glm_pred)) + geom_histogram()
cut_off <- 0.5
predicted_classes <- ifelse(term_deposit_glm_pred > cut_off,1,0)
predicted_classes
conf_matrix <- confusionMatrix(factor(predicted_classes), factor(valid_df$term_deposit))
print(conf_matrix)

# Model 2

colnames(bank_data)

# categorical: month, housing, loan
# continuous: duration, previous

glm_data_2 <- bank_data %>% select(duration, previous, month, housing, loan)
glm_data_2 <- model.matrix(~.-1, data = glm_data_2)
glm_data_2 <- cbind(glm_data_2, bank_data %>% select(term_deposit))
glm_data_2<- glm_data_2[,-14] # Supressing because of singularity
head(glm_data_2)


train_df_2 <- glm_data_2[train_rows,]
valid_df_2 <- glm_data_2[valid_rows,]


term_deposit_glm_2 <- glm(train_df_2$term_deposit~., data = train_df_2, family="binomial")

summary(term_deposit_glm_2)

term_deposit_glm_pred_2 <- predict(term_deposit_glm_2, valid_df_2[,-16], type = 'response')
ggplot(as.data.frame(term_deposit_glm_pred_2),aes(x = term_deposit_glm_pred_2)) + geom_histogram()
cut_off_2 <- 0.2
predicted_classes_2 <- ifelse(term_deposit_glm_pred_2 > cut_off_2,1,0)
conf_matrix_2 <- confusionMatrix(factor(predicted_classes_2), factor(valid_df_2$term_deposit))
conf_matrix_2
conf_matrix_2$byClass["Specificity"]

#### Function for finding optimal cut_off ####

cut_spec <- function(predicted_class, actual,counter = 1){
  cut_mat <- data.frame(cutoff = seq(0,1,0.1), spec = rep(1,11), sen = rep(1,11))
  for(i in seq(0,1,0.1)){
    predicted <- ifelse(predicted_class > i, 1, 0)
    temp <- confusionMatrix(factor(predicted),factor(actual))
    cut_mat[counter,2] <- temp$byClass["Specificity"]
    cut_mat[counter, 3] <- temp$byClass["Sensitivity"]
    counter <- counter+1
  }
  melted_data <- cut_mat %>%
    pivot_longer(cols = c("spec","sen"), names_to = "variable", values_to = "value")
  return(melted_data)
}

ggplot(cut_spec(term_deposit_glm_pred_2,valid_df_2$term_deposit),aes(x=cutoff,y=value, color=variable))+geom_line()

ggplot(cut_spec(term_deposit_glm_pred,valid_df$term_deposit),aes(x=cutoff,y=value, color=variable))+geom_line()

print(conf_matrix_2)
print(conf_matrix)

# Model 3 (Decision Tree)

# categorical: month, housing, loan
# continuous: duration, previous

dt_data <- bank_data %>% select(month, housing, loan, term_deposit)

# Replacing yes and no with 1 and zero for loan and housing

temp <- ifelse(dt_data$housing=='yes',1,0)
dt_data$housing <- temp
temp <- ifelse(dt_data$loan=='yes',1,0)
dt_data$loan<-temp

# converting categorical variables' data type to factor

columns <- c("month","housing","loan","term_deposit")
dt_data[,columns] <- lapply(dt_data[,columns],factor)
# Including the continuous variables into the data

dt_data <- cbind(bank_data%>%select(duration,previous), dt_data)

# Formula for decision tree

formula <- term_deposit ~ month + housing + loan + duration + previous

# Splitting into training and testing

train_df_3 <- dt_data[train_rows,]
valid_df_3 <- dt_data[valid_rows,]

# Training model on training data

dt_model <- rpart(formula, data = train_df_3, method = "class")

# Plotting the decision tree model

rpart.plot(dt_model, yesno = 2, type = 0, extra = 101)

# Creating predictions

dt_model_pred <- predict(dt_model, newdata = valid_df_3[,-6], type="class")

conf_matrix_dt <- caret::confusionMatrix(factor(dt_model_pred),factor(valid_df_3$term_deposit))

conf_matrix_dt

# We see a specificity of ~36% which is much better than the one we got from the 2 glm models with a cutoff of 0.5 (~25)

# Building a random forest model with same training and validation sets used for decision tree

rf_model <- randomForest(term_deposit ~ ., data = train_df_3, ntree = 500, importance = TRUE)

summary(rf_model)

varImpPlot(rf_model)

rf_model_pred <- predict(rf_model, newdata = valid_df_3[,-6])
rf_model_pred

conf_matrix_rf <- caret::confusionMatrix(factor(rf_model_pred),factor(valid_df_3$term_deposit))
conf_matrix_rf

# We see a slightly lower specificity value for random forest model compared to decision tree

#Modeling to check factors affecting balance
#Model1
head(bank_data)

train_df_3 <- bank_data[train_rows,]
valid_df_3 <- bank_data[valid_rows, ]

Bal_model <- lm(balance ~ ., data = train_df_3)
Bal_model_step <- step(lm(balance ~ ., data = train_df_3))
summary(Bal_model)
summary(Bal_model_step)

Bal_model_pred <- predict(Bal_model , newdata = valid_df_3)
Bal_model_pred
summary(Bal_model_pred)
plot(residuals(Bal_model))

ggplot(valid_df_3, aes(x = Bal_model_pred, y = balance)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(title = "Actual vs. Predicted Values", x = "Actual Values", y = "Predicted Values")


rmse <- sqrt(mean((valid_df_3$balance - Bal_model_pred)^2))
print(paste("Root Mean Squared Error (RMSE):", rmse))


#model 2
Bal_data_2 <- bank_data %>% select(age, marital, default, loan, month,balance)

train_df_4 <- Bal_data_2[train_rows,]
valid_df_4 <- Bal_data_2[valid_rows,]
Bal_model_2 <- lm(balance ~ ., data = train_df_4)
summary(Bal_model_2)

Bal_model_pred_2 <- predict(Bal_model_2 , newdata = valid_df_4)
Bal_model_pred_2
summary(Bal_model_pred_2)
plot(residuals(Bal_model_2))

ggplot(valid_df_4, aes(x = Bal_model_pred_2, y = balance)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(title = "Actual vs. Predicted Values", x = "Actual Values", y = "Predicted Values")


rmse <- sqrt(mean((valid_df_4$balance - Bal_model_pred_2)^2))
print(paste("Root Mean Squared Error (RMSE):", rmse))
