# Description
The second project of Spring 2017 Stat 333 is a [Kaggle competition](https://www.kaggle.com/c/uw-madison-sp17-stat333), where we are asked to predict Yelp ratings based on the text comments in Madison WI area. Our group got rank one on both public and private leaderboard 🎉.

<p align="center">
	<img src="https://i.imgur.com/RMz0uDn.png" height="250">
</p>


# Models
| Model                      | Directory Name | Description                                                                  |
|----------------------------|----------------|------------------------------------------------------------------------------|
| Deep Learning              | `./dl`         | Use Stanford's GloVe to vectorize text, and a simple CP-CP-CP neural network |
| Linear Regression          | `./lr`         | Use TFIDF text encoding, and lasso, ridge regression and elastic net         |
| Multiple Linear Regression | `./mrl`        | Naive simple multiple linear regression with silly variables                 |
| Neural Network             | `./nn`         | Use tf-idf text encoding, and a simple one hidden layer neural network       |

## Results
Our best model is using Ridge regression with tf-idf text encoding. You can check out the self-explained Jupyter notebook [here](https://github.com/xiaohk/stat333_project_2/blob/master/model/lr/final_model.ipynb).

## Comments
1. Feature engineering is much more important in NLP. We have tried many different text encoding methods here. GLoVe should have worked the best, but it was beaten by tf-idf in this very project.
2. We extracted the stem of words and removed stopping words. It turns out the stopping word level really worths tuning.


You can see our [presentation](https://docs.google.com/presentation/d/e/2PACX-1vQSIkfflqbx-Qlg6wRq-qwXJ2lbG3jrTiu-kara1DTUCQXRSxV_brV_TsjwlCsQJZuUCWuieN94Cqps/pub?start=false&loop=false&delayms=3000) to get more info.
