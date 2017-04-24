df = read.csv("Yelp_train.csv", header = TRUE)
newdata <- na.omit(df)
newdata$stars
model.full = lm(stars ~ useful+funny+cool+city+longitude+latitude+categories+nchar+nword+sentiment_score+gem+incredible+perfection+phenomenal+divine+die+highly+superb+heaven+amazing+favorites+sourced+perfect+knowledgeable+gross+poorly+response+flavorless+waste+terrible+tasteless+rude+awful+inedible+horrible+apology+disgusting+worst
,data=newdata)

model.null = lm(stars ~ 1,
                 data=newdata
)

step(model.null, scope=list(lower=model.null, upper=model.full), direction="both")
model =lm(formula = stars ~ sentiment_score + cool + funny + useful + 
            nchar + nword + die + incredible + perfection + phenomenal + 
            divine, data = newdata)
predict(model)
prediction=predict(model)
write.csv(prediction,file="prediction.csv")
