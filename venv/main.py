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

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

train_file_path = './dataset/train.csv'
test_file_path = './dataset/test.csv'
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

train_data.dropna(subset=['tweets', 'class'], inplace=True)
test_data.dropna(subset=['tweets'], inplace=True)
combined_data = pd.concat([train_data.assign(dataset='train'), test_data.assign(dataset='test')])

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    return

train_data['processed_tweets'] = train_data['tweets'].apply(preprocess_text)
test_data['processed_tweets'] = test_data['tweets'].apply(preprocess_text)

X = train_data['tweets']
y = train_data['class']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Dataset", "EDA", "Model", "Result", "Evaluation"],
    )

if selected == "Dataset":
    st.title("📂 Dataset")
    st.divider()
    st.header("Combined Data")
    st.dataframe(combined_data)
    st.header("Train Data")
    st.dataframe(train_data)
    st.header("Test Data")
    st.dataframe(test_data)

elif selected == "EDA":
    st.title("🔍 Exploratory Data Analysis (EDA)")
    st.divider()
    st.write("Jumlah data Train:", len(train_data))
    st.write("Jumlah data Test:", len(test_data))
    st.header("Distribusi kelas pada data training:")
    st.write(combined_data['class'].value_counts())
    st.divider()

    st.header("Distribusi Jumlah Data: Train vs Test")
    st.bar_chart(combined_data['dataset'].value_counts())
    st.divider()
    st.header("Jumlah data per kelas per dataset")
    st.bar_chart(combined_data.groupby(['dataset', 'class']).size().unstack())
    st.divider()
    st.header("Distribusi Kelas pada Data Training")
    st.bar_chart(train_data['class'].value_counts())
    st.divider()
    
    train_data['text_length'] = train_data['tweets'].apply(lambda x: len(str(x).split()))
    
    st.header("Distribusi Panjang Teks (dalam kata)")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(train_data['text_length'], bins=30, kde=True, ax=ax)
    ax.set_title("Distribusi Panjang Teks (dalam kata)")
    ax.set_xlabel("Jumlah Kata")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)
    st.divider()

    st.header("Wordclouds per Kelas")
    
    classes = train_data['class'].unique()
    
    for cls in classes:
        st.subheader(f"Wordcloud untuk Kelas: {cls}")

        class_data = train_data[train_data['class'] == cls]
        
        text = " ".join(class_data['tweets'].astype(str))
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        st.divider()

elif selected == "Model":
    st.title("Model")
    st.divider()

    st.header("Data Training setelah preprocessing")
    st.dataframe(train_data[['processed_tweets', 'class']])

    st.header("Data Testing setelah preprocessing")
    st.dataframe(test_data[['processed_tweets']])

    user_input = st.text_area("Masukkan teks untuk diprediksi", placeholder="Contoh: I hate waiting in traffic...")

    if st.button("Prediksi"):
        if user_input.strip() == "":
            st.warning("Teks tidak boleh kosong.")
        else:
            # processed_input = preprocess_text(user_input)
            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]

            prediction_map = {
                'figurative': "🧠 *Teks ini mengandung ekspresi figuratif.*",
                'irony': "🎭 *Teks ini mengandung ironi, ada kontras antara yang dikatakan dan kenyataan.*",
                'regular': "📘 *Teks ini biasa saja, tidak mengandung ekspresi khusus.*",
                'sarcasm': "😏 *Teks ini mengandung sarkasme, yaitu sindiran tajam.*"
            }

            st.success(f"**Prediksi Kategori: `{prediction}`**")
            st.info(prediction_map.get(prediction, "Tidak diketahui."))

elif selected == "Result":
    st.title("📄 Hasil Prediksi Data Uji")
    st.divider()

    test_data['predicted_class'] = model.predict(vectorizer.transform(test_data['processed_tweets']))
    st.dataframe(test_data[['tweets', 'predicted_class']].head(20))

    st.subheader("Distribusi Hasil Prediksi pada Data Uji")
    fig_pred, ax_pred = plt.subplots()
    sns.countplot(x='predicted_class', data=test_data, order=test_data['predicted_class'].value_counts().index, ax=ax_pred)
    st.pyplot(fig_pred)

elif selected == "Evaluation":
    st.title("Evaluation")
    st.divider()

    y_pred = model.predict(X_val_tfidf)

    st.header("📌 Classification Report")
    st.dataframe(pd.DataFrame(classification_report(y_val, y_pred, output_dict=True)).transpose())

    acc = accuracy_score(y_val, y_pred)
    st.header("🎯 Akurasi")
    st.metric(label=" ", value=f"{acc:.2%}")

    st.header("🧮 Confusion Matrix")
    conf_matrix = confusion_matrix(y_val, y_pred, labels=model.classes_)
    fig_conf, ax_conf = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues', ax=ax_conf)
    ax_conf.set_xlabel("Predicted")
    ax_conf.set_ylabel("Actual")
    st.pyplot(fig_conf)