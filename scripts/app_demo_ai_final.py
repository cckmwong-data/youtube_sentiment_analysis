# ---------------------- Import Libraries ----------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
from tensorflow.keras.models import load_model
import tensorflow as tf

import streamlit as st
import os
import string
import re
import html
from wordcloud import WordCloud
from collections import Counter

from googleapiclient.discovery import build

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---------------------- Load Transformer Model ----------------------

# Transformer Model (RoBERTa)
TRANSFORMER_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"  

@st.cache_resource

# Load the RoBERTa model from Hugging Face 
def load_transformer_model():
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL)
    return tokenizer, model

hf_tokenizer, hf_model = load_transformer_model()

def predict_with_transformer_3class(text: str):
    # Tokenize
    inputs = hf_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
        return_token_type_ids=False
    )
    # Remove token_type_ids for RoBERTa
    inputs.pop("token_type_ids", None)

    hf_model.eval() # Set the model to evaluation mode

    # Disable gradient calculations for faster processing
    with torch.no_grad():
        outputs = hf_model(**inputs)
        logits = outputs.logits # Extract the raw prediction scores
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0] # Convert logits to probabilities

    # Cardiff mapping: 0=negative, 1=neutral, 2=positive
    index_to_label = {0: "Negative", 1: "Neutral", 2: "Positive"}

    # Predicted label (for the 'sentiment' column)
    pred_id = int(probs.argmax()) # index of the highest-probability class
    sentiment = index_to_label[pred_id]

    # Extract P_Positive and P_Negative based on the assumed order   
    p_negative = float(probs[0])
    p_positive = float(probs[2])

    # Calculate the Continuous Sentiment Score (CSS): CSS = P_Pos - P_Neg
    # CSS ranges from -1.0 (max negative) to +1.0 (max positive)
    css = p_positive - p_negative 
    
    # Rescale the CSS from [-1, +1] to [0, 1]: (CSS + 1) / 2
    score = (css + 1) / 2

    return sentiment, score

# ---------------------- Load the model and cahche resources ----------------------

# The model is stored at the current directory 
MODEL_PATH = "sentiment_lstm_model2.keras"

# Define the global colour mapping for each sentiment type
SENTIMENT_COLOURS = {
    'Positive': '#009E73',  # bluish green
    'Neutral':  '#949494',  # grey
    'Negative': '#D55E00',  # vermillion (orange-red)
}

# Load the model just once and cache it when rerunning the app
@st.cache_resource
def load_sentiment_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model
 
# Read max_length from the file with caching
@st.cache_data
def load_max_length():
    with open('max_length_sentiment2.txt', 'r') as f:
        return int(f.read())

# Load the tokenizer with caching
@st.cache_resource
def load_tokenizer():
    with open('tokenizer_sentiment2.json', 'r') as f:
        data = f.read()  # Read the JSON string from the file
    return tokenizer_from_json(data)  # Reconstruct the tokenizer from the JSON string and return it

# Use the cached functions
model = load_sentiment_model() # load the model
max_length = load_max_length() # load the max_length
tokenizer = load_tokenizer() # load the tokenizer

# ---------------------- Functions for Preprocessing and Analysis ----------------------

# Function to remove mentions, URLs, and emails
def remove_mention_url_email(text):
    # remove mentions, URLs, and emails by replacing these patterns by space
    # and then change to lower case
    text = re.sub(r"@\S+|https?:\S+|http?:\S+|\S+@\S+", ' ', str(text).lower())

    # remove extra spaces and return the result
    return text.strip()

# Function to remove HTML tags
def remove_html_tags(text):
    # search for any forms of HTML tags which start with < and end with >
    clean = re.compile('<.*?>')
    # replace the HTML tags with an empty string
    text = re.sub(clean, '', text)

    # Decode HTML entities
    return html.unescape(text)

# Function to remove punctuation and other non-alphabetic characters
def remove_punc(text):
    # replace anything that is NOT a lowercase letter or space to space
    text = re.sub(r"[^a-z\s]", ' ', text)
    return text

# Function to convert nltk POS (part of speech) tag to WordNet POS tag
def get_wordnet_pos(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Function to find the root form of a word
def lemmatize_text(words):
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(words) # get NLTK’s pos_tag
    # change to lemmatized text with the consideration of the POS (Part of Speech)
    lemmatized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag))
        for word, pos_tag in pos_tags
    ]
    return lemmatized_words

# Function to remove stopwords
def remove_stopwords(words):
    tokens = []

    # add extra stopwords    
    extra_stopwords = {
    'video', 'youtube', 'channel', 'people', 'thing', 'things',
    'get', 'got', 'know', 'see', 'one', 'really', 'something', 'someone',
    'even', 'much', 'make', 'makes', 'made', 'go', 'going', 'take', 'still',
    'look', 'looks', 'way','want', 'never', 'always', 'ever', 'thought', 'thanks', 'thank',
    'lot', 'lotta', 'bit', 'guy', 'guys','say', 'says', 'said', 'stuff', 'anyone', 'everyone',
    'sure', 'okay', 'ok', 'yall', 'yeah', 'yes', 'nope', 'nah', 'hi', 'hello', 'hey',
    'wow', 'hmm', 'huh', 'uh', 'ah', 'im', 'youre', 'hes', 'shes', 'theyre',
    'cant', 'dont', 'didnt', 'doesnt', 'isnt', 'wasnt', 'wont', 'wouldnt', 'also', 'just', 'gotta', 'gonna',
    'would'}

    # Combine standard and extra stopwords
    combined_stopwords = set(stopwords.words('english')).union(extra_stopwords)

    # only keep those words which are not stop words with length greater than 2
    for word in words:
      if len(word) >= 2 and word not in combined_stopwords:
        tokens.append(word)

    return tokens

# Function for full preprocessing and prediction of the comment inputs
def preprocess_and_predict(text):

    text = remove_html_tags(text) # remove HTML tags
    text = remove_mention_url_email(text) # remove mentions, URLs, and emails
    text = remove_punc(text) # remove punctuation and other non-alphabetic characters

    tokenizer_nltk = TweetTokenizer() 
    tokens = tokenizer_nltk.tokenize(text) # tokenize tweet comments
    tokens = lemmatize_text(tokens) # convert to lemmatized text for each token
    tokens = remove_stopwords(tokens) # remove stopwords

    # Convert tokens to sequences
    seq = tokenizer.texts_to_sequences([tokens])
    # Pad or trauncate the sequences to max_length
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')

    # Make predictions using the keras model
    prob = model.predict(padded)[0][0]

    # Convert the probability to sentiment: positive, negative, neutral
    if prob >= 0.60: 
        sentiment = 'Positive'
    elif prob <= 0.40:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
        
    return sentiment, float(prob), tokens

# Function to fetch comments of the selected youtube video
def fetch_comments(index):

    # Read the meta CSV
    df_meta = pd.read_csv("metadata_comments.csv", encoding="latin1")

    video_id = df_meta.iloc[index, 0] # extract video id
    video_title = df_meta.iloc[index, 1] # extract title of the video
    video_author = df_meta.iloc[index, 2] # extract author of the video
    csv_file = df_meta.iloc[index, 3] # extract the corresponding csv file name

    # Extract the comments from the designated csv file
    comments = pd.read_csv(csv_file, encoding="utf-8")

    return comments, video_title, video_author, video_id

# Function for finding the smallest and largest values among 3 numbers
def compare_num(a, b, c):
    # Return the label and value of the smallest and largest numbers
    values = {'Positive': a, 'Neutral': b, 'Negative': c}
    smallest = min(values, key=values.get)
    largest = max(values, key=values.get)
    return (smallest, values[smallest]), (largest, values[largest]) # return the labels and values of the smallest and biggest numbers

# Function to show the summary of the whole sentimental analysis
def show_summary(num_total, num_pos, num_neut, num_neg, mean_score):
    df = st.session_state['df']

    # Determine major tone
    if mean_score >= 0.60:
        major_tone, tone_emoji = "positive", "🟢"
    elif mean_score <= 0.40:
        major_tone, tone_emoji = "negative", "🔴"
    else:
        major_tone, tone_emoji = "neutral", "🟡"

    pos_pct = 0 
    neut_pct = 0 
    neg_pct = 0 

    pos_pct = num_pos/ num_total*100 # the percentage of positive comments
    neut_pct = num_neut/ num_total*100 # the percentage of neutral comments
    neg_pct = num_neg/ num_total*100 # the percentage of negative comments

    # Score distribution stats
    scores = df["score"].astype(float)
    sigma = float(scores.std(ddof=0)) # standard deviation of the scores 

    # Interpret distribution shape and assign the corresponding text
    if sigma < 0.10:
        spread_text = " Sentiment scores are tightly clustered, showing strong agreement among viewers. "
    elif sigma < 0.25:
        spread_text = " Scores show moderate variation, suggesting some differing comments. "
    else:
        spread_text = " Scores are widely dispersed, indicating that comments are polarized. "

    # Initialize words_text
    words_text = ""

    # Finding the top 10 words among all comments
    words = df['tokens']

    # If 'Message' is a list of tokens, join them into strings
    if isinstance(words.iloc[0], list):
        words = words.apply(lambda x: ' '.join(x))

    # Combine all spam messages into one string
    text = ' '.join(words)

    # Generate the word cloud
    wc = WordCloud(width=800, height=400, colormap="Accent").generate(text)

    # Only process if there are any words at all
    if len(wc.words_) > 0:
        # Take the top 10 words or fewer if less available
        top = list(wc.words_.items())[:min(10, len(wc.words_))]
        words_text = f" Common words include {', '.join([f'**{w}**' for w, _ in top])}."

    # Build the text of the summary section
    summary = (
        f"{tone_emoji} Overall, the comments express a **{major_tone}** tone "
        f"with an average sentiment score of **{mean_score:.2f}**. Score has a value between 0 and 1 and a higher score indicates a more positive sentiment. "
        f"**{pos_pct:.1f}%** of the messages are positive, **{neut_pct:.1f}%** are neutral, "
        f"and **{neg_pct:.1f}%** are negative. "
        f"{spread_text}"
        f"{words_text}"
    )

    st.markdown(f"**Average Sentiment Score: {mean_score:.2f}/ 1.00**")
    st.markdown(summary)

    # Show comments statistics if there are comments fetched
    show_table(
        len(st.session_state["df"]),
        len(st.session_state["df_positive"]),
        len(st.session_state["df_neutral"]),
        len(st.session_state["df_negative"]),
    )

# Function to show a table of comment counts
def show_table(total_comments, positive_comments, neutral_comments, negative_comments):
    # Define the columns and table content
    data = {
    "": ["Number of Positive comments 😊", 
               "Number of Negative comments 😞", "Number of Neutral comments 😐", "Total Comments"],
    "Count": [positive_comments, negative_comments, neutral_comments, total_comments]
    }

    df_data = pd.DataFrame(data)

    # Display the table
    st.dataframe(df_data, hide_index=True)  

# Function to generate a word cloud of youtube comments
def generate_wordcloud(df, sentiment, container, colour):
    words = df['tokens']

    # If 'Message' is a list of tokens, join them into strings
    if isinstance(words.iloc[0], list):
        words = words.apply(lambda x: ' '.join(x))

    # Combine all spam messages into one string
    text = ' '.join(words)

    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, colormap=colour).generate(text)

    # Plot the word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f"Most Frequent Words of {sentiment} Messages", fontsize=16, fontweight='bold')
    container.pyplot(fig, use_container_width=True)
    plt.close(fig)

# Function to show a bar chart of sentiment distribution
def show_bar_chart(df, container):

    # Count the number of each sentiment
    sentiment_counts = df['sentiment'].value_counts()

    # Match colour order with sentiment order
    colours = [SENTIMENT_COLOURS[sent] for sent in sentiment_counts.index]

    # Create a figure for Streamlit
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#F5F5F5')
    
    # Horizontal bar chart
    bars = ax.barh(sentiment_counts.index, sentiment_counts.values, color=colours)

    # Add labels directly to bars
    ax.bar_label(bars, fmt='%d', label_type='edge', padding=3)

    # Display the bar chart in Streamlit
    container.pyplot(fig, use_container_width=True)

    plt.close(fig)

# Function to show a pie chart of sentiment distribution
def show_pie_chart(df, container):

    # Count the number of each sentiment
    sentiment_counts = df['sentiment'].value_counts()

    # Match color order with sentiment order
    colours = [SENTIMENT_COLOURS[sent] for sent in sentiment_counts.index]

    # Create a figure for Streamlit
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#F5F5F5')
    
    # Pie chart
    ax.pie(sentiment_counts, 
           labels=sentiment_counts.index, 
           autopct='%1.1f%%', 
           startangle=140, 
           explode=[0.05] * len(sentiment_counts),
           colors=colours)

    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is a circle.

    # Display the pie chart in Streamlit
    container.pyplot(fig, use_container_width=True)
    plt.close(fig)

# Function to show a strip chart of score distribution
def show_strip_chart(df, container):

    fig, ax = plt.subplots(figsize=(8, 5), facecolor='#F5F5F5')

    # Strip plot
    sns.stripplot(x='sentiment', y='score', data=df, jitter=True, palette=SENTIMENT_COLOURS, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Display the strip chart
    container.pyplot(fig, use_container_width=True)

    plt.close(fig)  

# ---------------------- Streamlit App ----------------------
# UI
st.markdown("<h1 style='font-size: 54px;'>Sentiment Analysis of YouTube Comments</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 24px; color: #CCCCCC; font-style: italic; margin-top: -15px;'>Demo Application</p>", unsafe_allow_html=True)

with st.expander("ℹ️ **How to Use This Demo**"):
    st.markdown("1. Select a YouTube video from the dropdown below.")
    st.markdown("2. Select the model for sentimental analysis on the sidebar.") 
    st.markdown("3. Choose charts and tables.")
    st.markdown("4. Click the **Analyze Comments** button.")

with st.expander("ℹ️ **About This Demo Application**"):
    st.markdown("""
    In this demonstration, we have preloaded YouTube comments extracted from several videos. Users can select one of these videos for sentiment analysis based on the preferred model (LSTM/ Transformer).
    The available categories of the videos include:
    1) **new product teaser**,  
    2) **game trailer**, and  
    3) **social topic**.

    Analyzing market sentiment in these areas can offer significant value to companies and organizations. For instance, sentiment analysis of **new product teasers** helps businesses gauge customer excitement, identify potential concerns, refine marketing strategies, and even optimize product features before launch. For **game trailers**, it enables developers and publishers to understand player expectations, enhance engagement, and anticipate audience reception prior to release. In the case of **social topics**, sentiment insights allow organizations to monitor public opinion, assess reputational impact, and inform communication or policy decisions more effectively.
    
    In a full implementation of this application, users would be able to input any YouTube video URL, allowing the system to extract comments in real time using the YouTube Data API. However, due to the daily quota limitations of the API, this demonstration uses preloaded comments to ensure a smooth and consistent experience while effectively showcasing the system’s sentiment analysis capabilities.
    
    Please check out the following link for the full implementation of the application.
    """)

    url = "https://raw.githubusercontent.com/cckmwong-data/youtube_sentiment_analysis/c769c96450a8c3dfc62d8ab62ea8cd76948dc8ea/demo_video/yt_sentiment_analysis.mp4"
    st.video(url)

with st.expander("ℹ️ **About The Models**"):
    st.markdown(
        """
        User can switch between two different sentiment analysis models: **custom-trained LSTM model** and **RoBERTa-based AI model**.

        ### Custom-Trained LSTM Model
        - Built using a Long Short-Term Memory (LSTM) recurrent neural network.  
        - Trained on the Sentiment140 dataset from Stanford, consisting of 1.6M tweets labeled as positive or negative.

        **Strengths**  
        - Lightweight and fast to run.  
        - Performs well on shorter, informal text (e.g., tweets).

        **Limitations**  
        - Relatively lower accuracy on negations, sarcasm, irony, and nuanced wording.   
        - More sensitive to spelling variations, unseen phrases and complex sentences.  
        - Limited training vocabulary compared to modern transformer models. 

        ### RoBERTa-Based AI Model
        - A transformer-based model pretrained on a massive corpus of English text developed by Meta.  
        - Fine-tuned for sentiment classification with modern NLP techniques.

        **Strengths**  
        - High accuracy due to deep contextual understanding.  
        - Fast inference, even on longer or complex sentences.  
        - Handles negations and subtle emotional cues better.  
        - More robust across different writing styles and domains (tweets, reviews, formal text).

        **Limitations**  
        - Longer processing time due to larger model and heavier memory usage compared to the LSTM.  
        - Not sensitive on sarcasm.
        - Pretrained data may introduce subtle biases.

        """
    )

options = [
    "New Product Teaser - Everything New with GoPro HERO13 Black",
    "Game Trailer - Fallout 4: Anniversary Edition",
    "Social Topic - What is London’s controversial Ulez expansion?"
]
# Get the saved index (if any), default to 0 (first option)
selected_index = st.session_state.get("selected_index", 0)

# Create selectbox using the index
selected_option = st.selectbox(
    "",
    options,
    index=selected_index
)

# Find current integer index
current_index = options.index(selected_option)
# Store the index (integer) instead of the text
st.session_state["selected_index"] = current_index

df = pd.DataFrame()  # Initialize an empty DataFrame
df_positive = pd.DataFrame()  # Initialize an empty DataFrame for positive comments
df_negative = pd.DataFrame()  # Initialize an empty DataFrame for negative comments     
df_neutral = pd.DataFrame()  # Initialize an empty DataFrame for neutral comments

# Declare placeholders once
button_container = st.empty()
info_container = st.empty()
summary_container = st.empty()
bar_container = st.empty()
pie_container = st.empty()
wc_container = st.empty()
wc_all_container = st.empty()
wc_pos_container = st.empty()
wc_neg_container = st.empty()
extremes_container = st.empty()
strip_container = st.empty()
no_comment_container = st.empty()

with st.sidebar:
    st.subheader("Sentiment Model")
    model_choice = st.radio(
        "Please select:",
        ["LSTM (Custom-trained)", "Transformer (AI-based)"],
        index=0
    )
    st.markdown("---")

with st.sidebar:
    st.subheader("Visualizations")
    st.write("Please select:")

    summary_check = st.checkbox("Insights Overview", value=True)
    st.caption("Summary of the analysis")

    bar_check = st.checkbox("Comments by Sentiment", value=True)
    st.caption("Number of comments for each sentiment")

    pie_check = st.checkbox("Sentiment Distribution", value=True)
    st.caption("Proportion of sentiment categories")

    wc_check = st.checkbox("Word Cloud", value=True)
    st.caption("The most common words in the comments")

    extremes_check = st.checkbox("Most Polarizing Comments", value=True)
    st.caption("The most extreme positive and negative comments")

    strip_check = st.checkbox("Score Distribution", value=True)
    st.caption("Distribution of sentiment scores")

# Button for analyzing comments
with button_container.container():
    analyze_btn = st.button("🔍 Analyze Comments", use_container_width=True)

# Start anazling the comments after the user has pressed the Analyze Comments button   
if analyze_btn:  
    try:
        comments, title, author, video_id = fetch_comments(st.session_state["selected_index"])
        df = pd.DataFrame(comments, columns=["comment"])

        # Progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()

        sentiment = []
        score = []
        tokens = []

        total_comments = len(df) # total no. of comments fetched

        # make the sentimental analysis for each comment
        for i, row in df.iterrows():
            comment = row['comment']
            if model_choice == "Transformer (AI-based)":
                s, sc = predict_with_transformer_3class(comment)
                
                # finding the tokens for the string
                text_clean = remove_html_tags(comment)
                text_clean = remove_mention_url_email(text_clean)
                text_clean = remove_punc(text_clean)

                tokenizer_nltk = TweetTokenizer()
                tok = tokenizer_nltk.tokenize(text_clean)
                tok = lemmatize_text(tok)
                tok = remove_stopwords(tok)

                t = tok  
            else:
                s, sc, t = preprocess_and_predict(comment)  # LSTM path
            sentiment.append(s)
            score.append(sc)
            tokens.append(t)

            # Update progress of the progress bar
            progress = (i + 1) / total_comments
            progress_bar.progress(progress)
            progress_text.text(f"Processing comment {i + 1} of {total_comments}...")

        # Store analysis results into the comments dataframe
        df['tokens'] = tokens
        df['sentiment'] = sentiment
        df['score'] = score

        # Update session state with the results
        st.session_state["df"] = df
        st.session_state["df_positive"] = df[df.sentiment == "Positive"]
        st.session_state["df_negative"] = df[df.sentiment == "Negative"]
        st.session_state["df_neutral"] = df[df.sentiment == "Neutral"]
        st.session_state["title"] = title
        st.session_state["author"] = author
        st.session_state["video_id"] = video_id
        st.session_state["analysis_done"] = True # Analysis completed successfully

        # reset the progress bar and text upon completion of analysis
        progress_bar.empty()
        progress_text.empty()

    # catch the error when the link is invalid or the video does not support comments function
    except Exception as e:
       st.warning("⚠️ Please provide a valid YouTube link with comments enabled.")
       st.session_state["analysis_done"] = False

# if the analysis completed successfully
if st.session_state.get("analysis_done", False):

    # printing the basic info of the video
    with info_container.container():
        url = f"https://www.youtube.com/watch?v={st.session_state['video_id']}"
        st.video(url) # Incoporate the YouTube Video with the extracted URL
        st.markdown(f"Video Title: {st.session_state['title']}")
        st.markdown(f"Video Author: {st.session_state['author']}")

    # Show comments statistics if there are comments fetched
    if len(st.session_state["df"]) > 0:
        num_total = len(st.session_state["df"])
        num_pos = len(st.session_state["df_positive"])
        num_neg = len(st.session_state["df_negative"])
        num_neut = len(st.session_state["df_neutral"])
        mean_score = st.session_state['df']['score'].mean()

        # Show the summary of the analysis if selected
        if summary_check:
            with summary_container.container():
                st.divider() 
                st.header("Insights Overview")
                show_summary(num_total, num_pos, num_neut, num_neg, mean_score)
        else:
                summary_container.empty()

        # Show bar chart if selected
        if bar_check:
            smallest, largest = compare_num(num_pos, num_neut, num_neg)
            with bar_container.container():
                section = st.container()  # one section container
                section.divider()
                section.header("No. of Comments by Sentiment")
                section.markdown(f"A majority of comments expressed {largest[0]} sentiment with a total count of {largest[1]}. {smallest[0]} sentiment received fewest comments, totaling {smallest[1]}.")
                show_bar_chart(st.session_state["df"], section)  
        else:
            bar_container.empty()

        # Show pie chart if selected
        if pie_check:
            pos_pct = num_pos/ num_total*100
            neut_pct = num_neut/ num_total*100
            neg_pct = num_neg/ num_total*100
            smallest, largest = compare_num(pos_pct, neut_pct, neg_pct)
            with pie_container.container():
                section = st.container()  # one section container
                section.divider()
                section.header("Sentiment Distribution")
                section.markdown(f"Most of the comments showed {largest[0]} sentiment, representing {largest[1]:.2f}% of the total number of comments.")
                show_pie_chart(st.session_state["df"], section)
        else:
            pie_container.empty()

        # Show word cloud if selected
        if wc_check:
            with wc_container.container():
                st.divider() 
                st.header("Word Cloud")
                if not st.session_state["df"].empty:
                    with wc_all_container.container():
                        generate_wordcloud(st.session_state["df"], "All", wc_all_container, "Accent")
                if not st.session_state["df_positive"].empty:
                    with wc_pos_container.container():
                        generate_wordcloud(st.session_state["df_positive"], "Positive", wc_pos_container, "viridis")
                if not st.session_state["df_negative"].empty:
                    with wc_neg_container.container():
                        generate_wordcloud(st.session_state["df_negative"], "Negative", wc_neg_container, "Pastel1")
        else:
            wc_container.empty()
            wc_all_container.empty() 
            wc_pos_container.empty() 
            wc_neg_container.empty() 

        # Print the most positive and most negative comments
        if extremes_check:
            df = st.session_state["df"]

            # Most Positive Comment (Maximum Score)
            idx_max = df['score'].idxmax() # .idxmax() returns the index of the highest value
            most_positive_comment = remove_html_tags(df.loc[idx_max]['comment'])
            max_score = df.loc[idx_max]['score']

            # Find Most Negative Comment (Minimum Score)
            idx_min = df['score'].idxmin()             # .idxmin() returns the index of the lowest value
            most_negative_comment = remove_html_tags(df.loc[idx_min]['comment'])
            min_score = df.loc[idx_min]['score']

            with extremes_container.container():
                section = st.container()  # one section container
                section.divider()
                section.header("Most Polarizing Comments")
                st.write(f"🟢 Most Positive Comment (Score: {max_score:.2f}/ 1.00): **{most_positive_comment}**")
                st.write(f"🔴 Most Negative Comment (Score: {min_score:.2f}/ 1.00): **{most_negative_comment}**")
        
        else:
            extremes_container.empty()

        # Show strip chart if selected
        if strip_check:

            scores = st.session_state['df']['score'].astype(float)
            sigma = float(scores.std(ddof=0)) # standard deviation of the scores

            # Interpret distribution shape
            if sigma < 0.10:
                spread_text = " Sentiment scores are tightly clustered, showing strong agreement among viewers. "
            elif sigma < 0.25:
                spread_text = " Scores show moderate variation, suggesting some differing comments. "
            else:
                spread_text = " Scores are widely dispersed, indicating that comments are polarized. "

            with strip_container.container():
                section = st.container()  # one section container
                section.divider()
                section.header("Score Distribution")
                section.markdown(
                f"The score indicates the strength of the positive sentiment expressed in the comment "
                f"(0 = very negative, 1 = very positive). "
                f"The mean score of all the comments is {mean_score:.2f}, with a standard deviation of {sigma:.2f}. {spread_text}"
                )
                show_strip_chart(st.session_state["df"], section)
        else:
            strip_container.empty()
        
    # Show no comments found message
    else:
        with no_comment_container.container():
            st.info("ℹ️ No comments found for analysis.")
