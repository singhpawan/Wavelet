  # Wavelet Classification Challenge - Pawandeep Singh
  library(dplyr)
  library(lattice)
  library(ggplot2)
  library(caret)
  library(knitr)
  library(memisc)
  library(gridExtra)
  library(forecast)
  library(GGally)
  library(nnet)
  library(bagRboostR)
  library(e1071)
  library(rpart)
  library(ROCR)
  library(mice)
  
  
  
  # Load the csv file
  wavelet = read.csv('~/Desktop/wavelet/wavelet.csv')
  
  
  # Dimension of data
  head(wavelet)
  str(wavelet)

  positions <- sample(nrow(wavelet), size = floor(nrow(wavelet)/4) * 3 )
  training_data <- wavelet[positions, ]
  testing_data <- wavelet[-positions,]

  ###########################  Prepare graphs of all the features factored with user_id to see visually the feature boundaries of for each user_ids  ####################################
  ggplot(data = wavelet, aes(x = wavelet$X_cap_seq ,y = wavelet$feat2, color = factor(wavelet$X_user_id))) + geom_point(alpha = 0.1, position = position_jitter(h = 0)) + scale_y_log10()
  ggplot(data = wavelet, aes(x = wavelet$X_cap_seq ,y = wavelet$feat3, color = factor(wavelet$X_user_id))) + geom_point(alpha = 0.1, position = position_jitter(h = 0)) + scale_y_log10()
  ggplot(data = wavelet, aes(x = wavelet$X_cap_seq ,y = wavelet$feat5, color = factor(wavelet$X_user_id))) + geom_point(alpha = 0.1, position = position_jitter(h = 0)) + scale_y_log10()
  ggplot(data = wavelet, aes(x = wavelet$X_cap_seq ,y = wavelet$feat6, color = factor(wavelet$X_user_id))) + geom_point(alpha = 0.1, position = position_jitter(h = 0)) + scale_y_log10()
  ggplot(data = wavelet, aes(x = wavelet$X_cap_seq ,y = wavelet$feat7, color = factor(wavelet$X_user_id))) + geom_point(alpha = 0.1, position = position_jitter(h = 0)) + scale_y_log10()
  ggplot(data = wavelet, aes(x = wavelet$X_cap_seq ,y = wavelet$feat9, color = factor(wavelet$X_user_id))) + geom_point(alpha = 0.1, position = position_jitter(h = 0)) + scale_y_log10()
  ggplot(data = wavelet, aes(x = wavelet$X_cap_seq ,y = wavelet$feat10, color = factor(wavelet$X_user_id))) + geom_point(alpha = 0.1, position = position_jitter(h = 0)) + scale_y_log10()
  ggplot(data = wavelet, aes(x = wavelet$X_cap_seq ,y = wavelet$feat11, color = factor(wavelet$X_user_id))) + geom_point(alpha = 0.1, position = position_jitter(h = 0)) + scale_y_log10()
  ggplot(data = wavelet, aes(x = wavelet$X_cap_seq ,y = wavelet$feat12, color = factor(wavelet$X_user_id))) + geom_point(alpha = 0.1, position = position_jitter(h = 0)) + scale_y_log10()
  ggplot(data = wavelet, aes(x = wavelet$X_cap_seq ,y = wavelet$feat15, color = factor(wavelet$X_user_id))) + geom_point(alpha = 0.1, position = position_jitter(h = 0)) + scale_y_log10()
  ggplot(data = wavelet, aes(x = wavelet$X_cap_seq ,y = wavelet$feat17, color = factor(wavelet$X_user_id))) + geom_point(alpha = 0.1, position = position_jitter(h = 0)) + scale_y_log10()
  ggplot(data = wavelet, aes(x = wavelet$X_cap_seq ,y = wavelet$feat18, color = factor(wavelet$X_user_id))) + geom_point(alpha = 0.1, position = position_jitter(h = 0)) + scale_y_log10()
  ggplot(data = wavelet, aes(x = wavelet$X_cap_seq ,y = wavelet$feat20, color = factor(wavelet$X_user_id))) + geom_point(alpha = 0.1, position = position_jitter(h = 0)) + scale_y_log10()
  ggplot(data = wavelet, aes(x = wavelet$X_cap_seq ,y = wavelet$feat21, color = factor(wavelet$X_user_id))) + geom_point(alpha = 0.1, position = position_jitter(h = 0)) + scale_y_log10()
  
  
  #####################  correlations among the variables ################
  with(wavelet,cor.test(wavelet$feat1, wavelet$feat2), method = 'pearson')
  with(wavelet,cor.test(wavelet$feat3, wavelet$feat4), method = 'pearson')
  with(wavelet,cor.test(wavelet$feat3, wavelet$feat7), method = 'pearson')
  with(wavelet,cor.test(wavelet$feat3, wavelet$feat8), method = 'pearson')
  with(wavelet,cor.test(wavelet$feat3, wavelet$feat16), method = 'pearson')
  with(wavelet,cor.test(wavelet$feat4, wavelet$feat5), method = 'pearson')
  with(wavelet,cor.test(wavelet$feat4, wavelet$feat7), method = 'pearson')
  with(wavelet,cor.test(wavelet$feat4, wavelet$feat17), method = 'pearson')
  with(wavelet,cor.test(wavelet$feat4, wavelet$feat18), method = 'pearson')
  with(wavelet,cor.test(wavelet$feat4, wavelet$feat21), method = 'pearson')
  with(wavelet,cor.test(wavelet$feat4, wavelet$feat3), method = 'pearson')
  with(wavelet,cor.test(wavelet$feat6, wavelet$feat4), method = 'pearson')
  with(wavelet,cor.test(wavelet$feat7, wavelet$feat21), method = 'pearson')
  with(wavelet,cor.test(wavelet$feat11, wavelet$feat13), method = 'pearson')
  with(wavelet,cor.test(wavelet$feat14, wavelet$feat15), method = 'pearson')
  with(wavelet,cor.test(wavelet$feat17, wavelet$feat21), method = 'pearson')
  with(wavelet,cor.test(wavelet$feat18, wavelet$feat7), method = 'pearson')
  
  
  # Small function to calculate the percentage of data missing in each feature
  pMiss <- function(x){
    (sum (is.na(x)) / length(x)) * 100
  }
  
  # features chosen which have low missing data %age and low correlation among them
  features <- wavelet[,c(6,7,9,10,11,13,14,15,16,19,21,22,24,25)]
  
  # features with the label data.
  tfeatures <- wavelet[,c(1,6,7,9,10,11,13,14,15,16,19,21,22,24,25)]
  
  # percentage of missing features in data
  apply(wavelet,2,pMiss)  

  
  ## Replacing the na values with columanar medians
  for(i in 1:ncol(tfeatures)){
    tfeatures[is.na(tfeatures[,i]), i] <- median(tfeatures[,i], na.rm = TRUE)
  }
  
  # prepare a matrix of the number of instances missing varying number of features and rank them accordingly
  md.pattern(features)
  
# Write the preprocessed features to the file
  write.csv(tfeatures, "~/pawan/wavelet/test.csv", row.names = FALSE, col.names = TRUE)
  
  
  # Preparign data for a multinomial logistic regression model
  positions <- sample(nrow(tfeatures), size = floor(nrow(tfeatures)/4) * 3 )
  training_data <- tfeatures[positions, ]
  testing_data <- tfeatures[-positions,]
  
  
  # Logistic regression model
  mn <- multinom(training_data$X_user_id ~ feat2 + feat3 +  feat5 +  feat7 +  
                   feat9 +  feat10 +  feat11 +  feat12 +  feat15 +  feat17 +  
                   feat20 + feat6 + feat21, data = training_data)
  
  
  probs <- predict(mn, newdata = testing_data,"probs")
  cum.probs <- t(apply(probs,1,cumsum))
  
  predictMNL <- function(model, newdata) {
    
    # Only works for neural network models
    if (is.element("nnet",class(model))) {
      # Calculate the individual and cumulative probabilities
      probs <- predict(model,newdata,"probs")
      cum.probs <- t(apply(probs,1,cumsum))
      
      # Draw random values
      vals <- runif(nrow(newdata))
      
      # Join cumulative probabilities and random draws
      tmp <- cbind(cum.probs,vals)
      
      # For each row, get choice index.
      k <- ncol(probs)
      ids <- 1 + apply(tmp,1,function(x) length(which(x[1:k] < x[k+1])))
      
      # Return the values
      return(ids)
    }
  }
  
  testing_data$predict <- predictMNL(mn, testing_data)
  
  acc <- accuracy(testing_data$X_user_id, testing_data$predict)
  acc
  xtab <- table(testing_data$X_user_id, testing_data$predict)
  colnames(xtab) <- 0:9
  confusionMatrix(xtab)

" 
I tried svm but the training time was way too much and hence decied to use sklearn for training classifiers.
  svm.model <- svm( training_data$X_user_id ~ feat2 + feat3, data = training_data, cost =100, gamma = 1)
  svm.pred <- predict(svm.model, testing_data)
  
  table(pred = svm.pred, true = testing_data[,1])
  
 " 

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
