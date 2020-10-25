#installing packages
install.packages("corrplot")
install.packages("tidyverse")
install.packages("gridExtra")
install.packages("rstatix")

#importing libraries
library(tidyverse)
library(grid)
library(gridExtra)
library(corrplot)
library(rstatix)

getwd()
df <- list.files(path = "C:/Users/george/Desktop", pattern = "*.csv", full.names = TRUE) %>%
      lapply(read_csv) %>%
      bind_rows()
df <- df[-10]

print(summary(df))

#creating vectors for each variable of the dataset

medians <- vector("double", length = 0)  
ranges <- vector("double", length = 0)  
standarddeviation <- vector("double", ncol(df))  
variableslargeQuantiles <- vector("double", ncol(df))  
variablesmallQuantiles <- vector("double", ncol(df)) 
variableMeans <- vector("double", ncol(df))  


#iterating throught the dataset and finding the statistics

for (i in seq_along(df)) {            
  variableMeans[[i]] <- mean(df[[i]])   
  variablesmallQuantiles[[i]] <- quantile(df[[i]], probs = c(0.01)) 
  variableslargeQuantiles[[i]] <- quantile(df[[i]], probs = c(0.99)) 
  standarddeviation[[i]] <- sd(df[[i]])  
  medians[[i]] <- median(df[[i]])  
  ranges[[i]] <- max(df[[i]]) - min(df[[i]])
}


#just printing the statistics



variableMeans <- format(variableMeans, digits = 3, nsmall = 2)
variablesmallQuantiles<- format(variablesmallQuantiles, digits = 3, nsmall = 2)
variableslargeQuantiles<- format(variableslargeQuantiles, digits = 3, nsmall = 2)
standarddeviation<- format(standarddeviation, digits = 3, nsmall = 2)
ranges<- format(ranges, digits = 3, nsmall = 2)
medians<- format(medians, digits = 3, nsmall = 2)


#creating a final dataframe

data <- data.frame(variableMeans, medians, variablesmallQuantiles, variableslargeQuantiles, standarddeviation, ranges)

#plotting spearman's correlation matrix

corrMatrix <- cor(df, method = "spearman")
print(corrMatrix)
corrplot(corrMatrix, type = "full", order = "hclust", 
         tl.col = "black", tl.srt = 45)


#another plot of cor
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(corrMatrix, method = "color", col = col(200),  
         type = "full", 
         addCoef.col = "black", 
         tl.col = "darkblue", tl.srt = 45, 
         diag = TRUE 
)

#another plotof cor matrix
cor.mat <- df %>% cor_mat(method = "spearman")
cor.mat %>%
  cor_plot(label = TRUE)


#plotting the table
rownames(data) <- c("AT", "AP", "AH", "AFDP", "GTEP", "TIT","TAT", "TEY", "CDP", "NOX")
colnames(data) <- c("Mean","median","Percentile 1%", "Percentile 99%", "standard deviation", "Range")
tt3 <- ttheme_default(
  core=list(
    fg_params=list(fontface=3)),
  colhead=list(fg_params=list(col="navyblue", fontface=4L)),
  rowhead=list(fg_params=list(col="black", fontface=3L),bg_params = list(fill = "light blue")))

grid.table(data, theme = tt3)
