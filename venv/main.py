import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Dataset", "EDA", "Model", "Result", "Evaluation"],
    )

if selected == "Dataset":
    st.title("Dataset")
elif selected == "EDA":
    st.title("EDA")
elif selected == "Model":
    st.title("Model")
elif selected == "Result":
    st.title("Result")
elif selected == "Evaluation":
    st.title("Evaluation")
