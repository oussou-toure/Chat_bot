#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st


# In[3]:


nltk.download('stopwords')


# In[5]:


nltk.download('wordnet')


# In[7]:


nltk.download('omw-1.4')


# In[12]:


#1. Choisissez un sujet
''' Un text de Taylor Swift
    lyrics of "daylight"'''


# In[8]:


#2. Prétraiter les données 
with open('C:/Users/Ousmane/Documents/GOMYCODE/learn.gomycode.co/DeepLearning/ChatBot/Checkpoints/daylight.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')
# Tokenize the text into sentences
sentences = sent_tokenize(data)
# Define a function to preprocess each sentence
def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    # Remove stopwords and punctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words('english') and word not in string.punctuation]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]


# In[13]:


#3. Définissez la fonction de similarité
# Define a function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    # Preprocess the query
    query = preprocess(query)
    # Compute the similarity between the query and each sentence in the text
    max_similarity = 0
    most_relevant_sentence = ""
    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)
    return most_relevant_sentence


# In[14]:


#4. Définir la fonction chatbot
def chatbot(question):
    # Find the most relevant sentence
    most_relevant_sentence = get_most_relevant_sentence(question)
    # Return the answer
    return most_relevant_sentence


# In[15]:


# 5. Créer une application Streamlit
def main():
    st.title("Chatbot")
    st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")
    # Get the user's question
    question = st.text_input("You:")
    # Create a button to submit the question
    if st.button("Submit"):
        # Call the chatbot function with the question and display the response
        response = chatbot(question)
        st.write("Chatbot: " + response)
if __name__ == "__main__":
    main()


# In[ ]:




