# Load libraries
library(tidyverse)
library(caret)
library(e1071)
library(tm)  # For text mining
library(SnowballC)  # For text stemming

# Load dataset (assuming the dataset is in CSV format)
data <- read.csv("Areview.csv")

# Data preprocessing
# Assuming your dataset has two columns: "text" and "emotion"
# Convert emotions to factors
data$emotion <- as.factor(data$emotion)

# Split data into training and testing sets (80% training, 20% testing)
set.seed(123)
train_index <- createDataPartition(data$emotion, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Create a Corpus
corpus <- Corpus(VectorSource(train_data$text))

# Text preprocessing
corpus <- tm_map(corpus, content_transformer(tolower))  # Convert to lowercase
corpus <- tm_map(corpus, removePunctuation)  # Remove punctuation
corpus <- tm_map(corpus, removeNumbers)  # Remove numbers
corpus <- tm_map(corpus, removeWords, stopwords("en"))  # Remove stopwords
corpus <- tm_map(corpus, stemDocument)  # Stemming

# Create a document-term matrix
dtm <- DocumentTermMatrix(corpus)

# Model training
svm_model <- train(as.factor(emotion) ~ ., data = as.data.frame(as.matrix(dtm)), method = "svmLinear")

# Predictions on the test set
test_corpus <- Corpus(VectorSource(test_data$text))
test_corpus <- tm_map(test_corpus, content_transformer(tolower))
test_corpus <- tm_map(test_corpus, removePunctuation)
test_corpus <- tm_map(test_corpus, removeNumbers)
test_corpus <- tm_map(test_corpus, removeWords, stopwords("en"))
test_corpus <- tm_map(test_corpus, stemDocument)
test_dtm <- DocumentTermMatrix(test_corpus)
predictions <- predict(svm_model, newdata = as.data.frame(as.matrix(test_dtm)))

# Evaluate the model
confusionMatrix(predictions, test_data$emotion)

# Plotting emotions
emotion_counts <- table(predictions)
emotion_counts <- as.data.frame(emotion_counts)
colnames(emotion_counts) <- c("emotion", "count")

# Plot
ggplot(emotion_counts, aes(x = emotion, y = count)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Emotion Distribution", x = "Emotion", y = "Count")
