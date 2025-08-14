# Pulseguard-lite 🚨📊

An end-to-end **NLP pipeline** for real-time **Customer Support Sentiment Analysis** using **Twitter data**, built with **RoBERTa**, and visualized in **Streamlit**.

---

## 🔧 Features

- Real-time tweet ingestion and cleaning
- RoBERTa + VADER hybrid sentiment scoring
- Alert generation on spikes/negative sentiment
- Trend visualization + Slack notifications
- Interactive dashboard in Streamlit

---

## 🛠️ Tech Stack

- Python
- RoBERTa (Transformers - HuggingFace)
- Streamlit
- Slack SDK
- Pandas, Plotly, Numpy, etc.

---

## 📂 Project Structure

pipeline/
├── 01_clean_and_prepare.py # Clean and structure Twitter data
├── 02_sentiment_roberta.py # Apply sentiment analysis models
├── 03_alert_generation.py # Create alerts from sentiment spikes


app/
├── app.py # CLI or batch app
├── streamlit_app.py # Web dashboard
