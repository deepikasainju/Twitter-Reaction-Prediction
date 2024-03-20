import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, confusion_matrix 

st.title("Text Classification for twitter training")
st.title("Model used:-")
st.write("""This profit prediction model utilizes logistic regression to predict reactions of the commenters
         on the twitter based on the review about movies they wrote. By leveraging 
         historical data that includes known reactions and corresponding reviews, the model learns 
         the underlying patterns and correlations. Through this learning process, the model develops a 
         logistic equation that best fits the relationship between the input features and the target variable 
         (reaction).""")

st.title("Dataset used:-")
odf=pd.read_csv("twitter_training.csv")
odf

st.title("Cleaned dataset")
df=pd.read_csv("cleaned_twittertrain_data.csv")
df

from sklearn.linear_model import LogisticRegression
log_regression = LogisticRegression()

vectorizer = TfidfVectorizer() 
X = df['Review'].apply(lambda X: np.str_(X)) #indepedent 
Y = df['Reaction'] #dependent

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15) 

# #Creating Pipeline
pipeline = Pipeline([('vect', vectorizer), 
                     ('chi',  SelectKBest(chi2, k=2000)), 
                     ('clf', LogisticRegression(random_state=1,max_iter=1000))]) 


model = pipeline.fit(X_train, Y_train)


st.title("Reaction Prediction")
react=st.text_area("Text to translate:")
if st.button("Submit"):
    react_data = {'predict_react':[react]}
    react_data_df = pd.DataFrame(react_data)
    predict_react_cat = model.predict(react_data_df['predict_react'])
    st.text("Predicted reaction of review = ",predict_react_cat[0])
else:
    st.text("please enter the news")


st.title("Accuracy of the Model:")
from sklearn.metrics import accuracy_score
predict_review_cat = model.predict(X_test) 
acc = accuracy_score(Y_test,predict_review_cat) *100
st.write(f'Accuracy: {acc:.2f}%')


st.title("Classification Report")
ytest = np.array(Y_test)
report = classification_report(ytest,model.predict(X_test))
st.text("\n{}".format(report)) 


st.title('Confusion Matrix for the Text Classification Model of twitter training')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(model.predict(X_test),Y_test)
fig, ax = plt.subplots(figsize=(8,8), dpi=100)
class_names = ['Positive','Negative','Neutral','Irrelevant']
display = ConfusionMatrixDisplay(cm, display_labels=class_names)
display.plot(ax=ax)
st.pyplot(fig)

