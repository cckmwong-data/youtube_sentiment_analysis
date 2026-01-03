# Deep Learning–Based Sentiment Analysis for YouTube Comments

## Project Overview

This project presents an end-to-end sentiment analysis system for YouTube comments, combining a custom deep learning model and a transformer-based model, and deploying them through an interactive [Streamlit web application](https://huggingface.co/spaces/cckmwong/youtube_sentiment_demo). 

Due to daily quota limitations of the YouTube Data API, this demonstration uses preloaded comments to ensure a stable and consistent user experience while effectively showcasing the system’s sentiment analysis capabilities. **In a full implementation, users would be able to input any YouTube video URL and extract comments in real time using the YouTube Data API.**

---

## Stages of Development

The whole project consists of two main stages:

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
- Trained on labeled sentiment data ([Sentiment140](https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip))
- Captures sequential and contextual patterns in text
- Saved and uploaded to Hugging Face for inference
- Lightweight and fast to run

### Transformer Model (RoBERTa)
- Pre-trained transformer-based sentiment analysis model
- Provides robust contextual understanding and strong performance
- High accuracy on negations and subtle emotional cues
- Longer processing time due to larger model and heavier memory usage

---

## Streamlit Application

> **Note:**  
> *This Streamlit application is hosted on the free tier of Hugging Face Spaces. If the app has been idle for more than 24 hours, it may take some time to reactivate. In such cases, please click “Restart this Space” to relaunch the application. Thank you for your patience.*

The final deliverable of this project is a [Streamlit web application](https://huggingface.co/spaces/cckmwong/youtube_sentiment_demo) deployed on Hugging Face Spaces. We do not consider Streamlit Community Cloud due to the large file size of the model and complexity of the project.

![Hugginng Face](images/hugging_face.png)

### Features of the Demo Application
- Model selection:
  - Custom LSTM
  - Transformer (RoBERTa)
- Sentiment analysis on preloaded YouTube comments
- Interactive results display

### Full Application

In a full implementation of this application, users would be able to input any YouTube video URL, allowing the system to extract comments in real time using the YouTube Data API. Please check out [here](https://youtu.be/KlG7qcbPQD4) for the video demonstration of full implementation:

![Streamlit App Demo](images/full_app.png)

---

## Sentiment & Business Insight

### New Product Teaser - Everything New with GoPro HERO13 Black
#### Aim
- Measure customer excitement before product launch
- Identify potential concerns or negative reactions
- Improve marketing strategies and product positioning

#### Analysis
The overall sentiment analysis indicates a slightly positive but polarized user perception, with an average sentiment score of 0.58 / 1.00 across 402 comments. Positive feedback represents the largest share (44.8%), followed by neutral (28.9%) and negative (26.4%) comments. While positive sentiment dominates, the wide score dispersion highlights a clear divide between enthusiastic supporters and dissatisfied users.

From a business perspective, user sentiment is **strongly feature-driven rather than brand-driven**. Word cloud analysis shows that discussions are centered on product capabilities such as lens quality, sensor performance, battery life, software updates, and upgrades. Positive sentiment aligns closely with purchase intent and upgrade interest, indicating strong demand among early adopters and existing users.

The recurring presence of “upgrade” in both positive and negative contexts reflects high customer expectations, where incremental improvements risk backlash if they fail to deliver meaningful value.

Additionally, **frequent mentions of competitors including DJI and Osmo highlight a highly competitive market with low switching costs**, increasing the importance of clear differentiation and transparent communication.

![](images/teaser2.png)

### Game Trailer - Fallout 4: Anniversary Edition
#### Aim
- Understand player expectations and engagement
- Anticipate audience reception
- Support promotional decision-making

#### Analysis
The overall sentiment for this dataset skews negative and polarized, with an average sentiment score of 0.47 / 1.00 across 458 comments. Negative sentiment dominates the discussion (45.0%), outweighing positive (30.8%) and neutral (24.2%) feedback. The wide dispersion of sentiment scores confirms a strong divide between dissatisfied users and a smaller but vocal positive group.

From a business perspective, conversation is **heavily franchise- and release-driven rather than centered on gameplay mechanics alone**. Word cloud analysis shows dominant themes such as “Fallout,” “Bethesda,” “new,” “version,” “remaster,” “release,” and “DLC,” indicating that user sentiment is shaped largely by expectations around new releases, remakes, and updates.

Negative sentiment, however, is more prevalent and more specific. Common terms such as “old,” “nothing,” “mod,” “wait,” “money,” “need,” and “play Skyrim again” point to frustration with perceived lack of innovation, repetitive re-releases, and reliance on modding communities. Complaints appear less about technical failure and more about strategic direction and content freshness.

**The recurring appearance of “remaster” and “new version” in negative contexts suggests release fatigue.** Users expect substantive changes, not incremental updates to existing titles. When expectations are unmet, disappointment translates quickly into negative sentiment.

![](images/game2.png)

### Social Topics
- Monitor public sentiment
- Assess reputational impact
- Inform communication and policy strategies

---

## Future Improvements

- Multilingual sentiment analysis
- Advanced sentiment categorization
- Sentiment trend visualization

---

## Author

**Carmen Wong**
