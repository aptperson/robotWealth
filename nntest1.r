###############################################
#### ROLLING FORECAST ORIGIN FRAMEWORK ########
###############################################

setwd("C:/Users/athompson2/repos/personalRepos/robotWealth/")

library(caret)
library(nnet)
library(neuralnet)
library(deepnet)
library(foreach)
library(doParallel)

eu <- read.csv("EUvarsD1.csv", header = F, stringsAsFactors = F)
colnames(eu) <- c("date", "deltabWdith3", "velocity10", "mom3", "atrRatFast", "atrRatSlow", "ATR7", "objective")

# summary function for training caret models on maximum profit
absretSummary <- function (data, lev = NULL, model = NULL) { # for training on a next-period return
  positions <- ifelse(data[ , "pred"] > 0.0, 1, ifelse(data[, "pred"] < -0.0, -1, 0))
  trades <- positions*data[, "obs"] 
  profit <- sum(trades)
  names(profit) <- 'profit'
  return(profit)
}

############################################################################
# nnet models --------------------------------------------------------------
############################################################################


lbWin <- c(50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000)
iterations <- length(lbWin)
modellist.nnet <- vector(mode = "list",
                         length = iterations)

nnetGrid <- expand.grid(.layer1 = c(2:3),
                        .layer2 = c(0:2),
                        .layer3=0,
                        .hidden_dropout=c(0, 0.1, 0.2),
                        .visible_dropout=c(0, 0.1, 0.2))

nnetGrid <- expand.grid(.layer1 = c(2:3),
                        .layer2 = 1,
                        .layer3 = 0,
                        .hidden_dropout = 0.1,
                        .visible_dropout = 0.1)

cl <- makeCluster(4)
registerDoParallel(cl)
set.seed(503)
i = 1

for (i in 1:iterations) {
  
  cat("traning Window: ", lbWin[i])
  
  
  timecontrol <- trainControl(method = 'timeslice',
                              initialWindow = lbWin[i],
                              horizon = 1,
                              summaryFunction = absretSummary,
                              selectionFunction = "best", 
                              returnResamp = 'final',
                              fixedWindow = TRUE,
                              savePredictions = 'final') 

  modellist.nnet[[i]] <- train(x = eu[, 2:4],
                               y = eu[, 8],
                               method = "dnn", 
                               trControl = timecontrol,
                               tuneGrid = nnetGrid,
                               preProcess = c('center', 'scale'))

}

stopCluster(cl)

equity.nnet <- data.frame()
sharpe.nnet <- data.frame()
acc.nnet <- data.frame()
for (j in c(1:15)) {
  k <- 1
  
  for (t in c(0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)) {
    threshold <- t
    trades <- ifelse(modellist.nnet[[j]]$pred$pred > threshold, modellist.nnet[[j]]$pred$obs, 
                     ifelse(modellist.nnet[[j]]$pred$pred < -threshold, -modellist.nnet[[j]]$pred$obs, 0))
    cumulative <- cumsum(trades)
    plot(cumulative, type = 'l', col = 'blue', main = paste0('Model: ', j, 'Thresh: ', threshold))
    equity.nnet[j, k] <- cumulative[length(cumulative)]
    
    x <- trades[trades!=0] #total trades
    y <- trades[trades > 0] #winning trades
    acc.nnet[j, k] <- (length(y)/length(x))*100
    
    
    if (length(x) > 100) { # exclude any sharpes with less than 100 trades
      a <- sum(trades)/length(x) #average trade
      b <- sd(x) #std dev of trades
      sharpe.nnet[j, k] <- sqrt(252) * a/b
    }
    else sharpe.nnet[j, k] <- 0
    
    k <- k+1
  }
}

rownames(sharpe.nnet) <- lengths
colnames(sharpe.nnet) <- c('0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', '0.55', '0.6', '0.65', 
                           '0.7', '0.75', '0.8')
# sharpe ratio
sharpe.nnet[ "WINDOW" ] <- rownames(sharpe.nnet)
s.molten <- melt( sharpe.nnet, id.vars="WINDOW", value.name="SHARPE", variable.name="THRESHOLD" )
s.molten <- na.omit(s.molten)

#Factorize 'WINDOW' for plotting
s.molten$WINDOW <- as.character(s.molten$WINDOW)
s.molten$WINDOW <- factor(s.molten$WINDOW, levels=unique(s.molten$WINDOW))

s <- ggplot(s.molten, aes(x=WINDOW, y=THRESHOLD, fill = SHARPE)) + 
  geom_tile(colour = "white")  + 
  scale_fill_gradient(low = "white", high = "blue4") +
  ggtitle("Sharpe Ratio by Window Length and Prediction Threshold")