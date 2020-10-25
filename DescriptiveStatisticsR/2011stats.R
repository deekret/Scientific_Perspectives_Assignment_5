#installing packages
install.packages("corrplot")
install.packages("tidyverse")
install.packages("gridExtra")

#importing libraries
ibrary(tidyverse)
library(grid)
library(gridExtra)
library(corrplot)




data2011 <- read.csv("gt_2015.csv")
data2011 <- data2011[-10]
head(data2011,5)

print(summary(data2011))


#creating vectors for each variable of the dataset

medians <- vector("double", length = 0)  
ranges <- vector("double", length = 0)  
standarddeviation <- vector("double", ncol(data2011))  
variableslargeQuantiles <- vector("double", ncol(data2011))  
variablesmallQuantiles <- vector("double", ncol(data2011)) 
variableMeans <- vector("double", ncol(data2011))  


#iterating throught the dataset and finding the statistics

for (i in seq_along(data2011)) {            
  variableMeans[[i]] <- mean(data2011[[i]])   
  variablesmallQuantiles[[i]] <- quantile(data2011[[i]], probs = c(0.01)) 
  variableslargeQuantiles[[i]] <- quantile(data2011[[i]], probs = c(0.99)) 
  standarddeviation[[i]] <- sd(data2011[[i]])  
  medians[[i]] <- median(data2011[[i]])  
  ranges[[i]] <- max(data2011[[i]]) - min(data2011[[i]])
}


#just printing the statistics



variableMeans <- format(variableMeans, digits = 2)
variablesmallQuantiles<- format(variablesmallQuantiles, digits = 2)
variableslargeQuantiles<- format(variableslargeQuantiles, digits = 2)
standarddeviation<- format(standarddeviation, digits = 2)
ranges<- format(ranges, digits = 2)
medians<- format(medians, digits = 2)

#creating a final dataframe

data <- data.frame(variableMeans, medians, variablesmallQuantiles, variableslargeQuantiles, standarddeviation, ranges)

#plotting spearman's correlation matrix

corrMatrix <- cor(data2011, method = "spearman")
print(corrMatrix)

#another plot of cor
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(corrMatrix, method = "color", col = col(200),  
         type = "full", 
         addCoef.col = "black", 
         tl.col = "darkblue", tl.srt = 45, 
         diag = TRUE 
)

corrplot(corrMatrix, type = "full", order = "hclust", 
         tl.col = "black", tl.srt = 45)

#plotting the table
rownames(data) <- c("AT", "AP", "AH", "AFDP", "GTEP", "TIT","TAT", "TEY", "CDP", "NOX")
colnames(data) <- c("Mean","median","Percentile 1%", "Percentile 99%", "standard deviation", "Range")
tt3 <- ttheme_default(
  core=list(
            fg_params=list(fontface=3)),
  colhead=list(fg_params=list(col="navyblue", fontface=4L)),
  rowhead=list(fg_params=list(col="black", fontface=3L),bg_params = list(fill = "light blue")))

grid.table(data, theme = tt3)

#colhead=list(fg_params=list(col="navyblue", fontface=4L)),
