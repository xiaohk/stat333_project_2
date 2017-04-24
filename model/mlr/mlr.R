# Use "../subset/Yelp_mini.csv" to save time
df = read.csv("../../subset/Yelp_train.csv", header = TRUE)
out = lm(with(df, stars ~ useful + funny + cool + sentiment_score))
summary(out)