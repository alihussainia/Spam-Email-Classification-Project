import joblib
import re
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
import pandas as pd
import nltk

st.write("# Virtual University of Pakistan")
st.write("# Spam Detection App")

message_text = st.text_input("Enter a message for spam evaluation")

def preprocessor(text):
#     text = re.sub('<[^>]*>', '', text) # Effectively removes HTML markup tags
#     emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
#     text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    
    df = pd.DataFrame([f"{phrase}"],columns=["Text"])

    
    
    
    #REPLACING NUMBERS FROM Digits to Words
    df['Text']=df['Text'].str.replace(r'\d+(\.\d+)?', 'numbers')
    #CONVRTING EVERYTHING TO LOWERCASE
    df['Text']=df['Text'].str.lower()
    #REPLACING NEXT LINES BY 'WHITE SPACE'
    df['Text']=df['Text'].str.replace(r'\n'," ") 
    # REPLACING EMAIL IDs BY 'MAILID'
    df['Text']=df['Text'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','MailID')
    # REPLACING URLs  BY 'Links'
    df['Text']=df['Text'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','Links')
    # REPLACING CURRENCY SIGNS BY 'MONEY'
    df['Text']=df['Text'].str.replace(r'£|\$', 'Money')
    # REPLACING LARGE WHITE SPACE BY SINGLE WHITE SPACE
    df['Text']=df['Text'].str.replace(r'\s+', ' ')

    # REPLACING LEADING AND TRAILING WHITE SPACE BY SINGLE WHITE SPACE
    df['Text']=df['Text'].str.replace(r'^\s+|\s+?$', '')
    #REPLACING CONTACT NUMBERS
    df['Text']=df['Text'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','contact number')
    #REPLACING SPECIAL CHARACTERS  BY WHITE SPACE 
    df['Text']=df['Text'].str.replace(r"[^a-zA-Z0-9]+", " ")
    
    # removing stopwords 
    nltk.download('stopwords')
    stop = stopwords.words('english')
    df['Cleaned_Text'] = df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))	
	
	
	
	
	
    text = df['Cleaned_Text'][0]
    return text

model = joblib.load('spam_classifier.joblib')

def classify_message(model, message):

	if model.predict([message])[0] == 0:
		label = 'Spam'
	else:
		label = 'Non-Spam'
	spam_prob = model.predict_proba([message])

	return {'label': label, 'spam_probability': spam_prob[0][0]}

if message_text != '':

	result = classify_message(model, message_text)

	st.write(result)

	
	explain_pred = st.button('Explain Predictions')

	if explain_pred:
		with st.spinner('Generating explanations'):
			class_names = ['Spam', 'Non-Spam']
			explainer = LimeTextExplainer(class_names=class_names)
			exp = explainer.explain_instance(message_text, 
				model.predict_proba, num_features=10)
			components.html(exp.as_html(), height=800)
	




