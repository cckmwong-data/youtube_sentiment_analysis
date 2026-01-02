# YouTube Sentiment Analysis with LSTM & Transformer Models

## Project Overview

This project presents an end-to-end sentiment analysis system for YouTube comments, combining a custom deep learning model and a transformer-based model, and deploying them through an interactive Streamlit web application.

The project consists of two main stages:

1. **Model Development**
   - Data preprocessing and exploratory analysis
   - Training a custom LSTM sentiment analysis model
   - Compare its performance with traditional approaches, TextBlob and VADER
   - Saving and exporting the trained model for deployment

3. **Model Deployment**
   - Hosting the trained LSTM model on Hugging Face
   - Integrating a transformer-based model (RoBERTa)
   - Deploying a Streamlit application that allows users to select and compare models

---

## Models

### Custom LSTM Model
- Trained on labeled sentiment data (Sentiment140)
- Captures sequential and contextual patterns in text
- Saved and uploaded to Hugging Face for inference

### Transformer Model (RoBERTa)
- Pre-trained transformer-based sentiment analysis model
- Provides robust contextual understanding and strong performance
- Used as an alternative to the custom LSTM model

---

## Streamlit Application

The final deliverable of this project is a [Streamlit web application](https://huggingface.co/spaces/cckmwong/youtube_sentiment_demo) deployed on Hugging Face Spaces. We do not consider Streamlit Community Cloud due to the large file size of the model and complexity of the project.

### Features of the Demo Application
- Model selection:
  - Custom LSTM
  - Transformer (RoBERTa)
- Sentiment analysis on preloaded YouTube comments
- Interactive and real-time result display

---

## Available Video Categories

The demonstration includes preloaded YouTube comments from the following categories:

- New product teaser
- Game trailer
- Social topic

---

## Use Cases and Value

### New Product Teasers
- Measure customer excitement before product launch
- Identify potential concerns or negative reactions
- Improve marketing strategies and product positioning

### Game Trailers
- Understand player expectations and engagement
- Anticipate audience reception
- Support promotional decision-making

### Social Topics
- Monitor public sentiment
- Assess reputational impact
- Inform communication and policy strategies

---

## API Limitation Note

In a full implementation, users would be able to input any YouTube video URL and extract comments in real time using the YouTube Data API.

Due to daily quota limitations of the YouTube Data API, this demonstration uses preloaded comments to ensure a stable and consistent user experience while effectively showcasing the system’s sentiment analysis capabilities.

---

## System Workflow

1. User selects a YouTube video category
2. User selects a sentiment analysis model (LSTM or RoBERTa)
3. Preloaded comments are analyzed
4. Sentiment predictions are generated and displayed in the Streamlit app

---

## Full Application

In a full implementation of this application, users would be able to input any YouTube video URL, allowing the system to extract comments in real time using the YouTube Data API. Please check out the following link for the full implementation of the application:

[Hugging Face Space – Streamlit App](<INSERT_LINK_HERE>)

---

## Future Improvements

- Multilingual sentiment analysis
- Advanced sentiment categorization
- Sentiment trend visualization

---

## Author

Developed by **[Carmen Wong]**
