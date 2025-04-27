![github-submission-banner](https://github.com/user-attachments/assets/a1493b84-e4e2-456e-a791-ce35ee2bcf2f)

# 🚀 Project Title

> Stock market chatbot

---

## 📌 Problem Statement

Problem Statement 1 - Weave AI magic with Groq



---

## 🎯 Objective

To develop an intelligent stock market chatbot that automates data retrieval, technical and fundamental analysis, sentiment evaluation, and price forecasting — empowering users to make faster and smarter investment decisions.

---

## 🧠 Team & Approach

### Team Name:  
`NA`

### Team Members:  
- Anurag

### Your Approach:  
- Why I Chose This Problem: Simplify stock analysis by automating data retrieval, sentiment analysis, and forecasting for quicker, smarter investment decisions.
- Key Challenges: Addressed missing stock data, fine-tuned sentiment analysis with FinBERT, and improved forecast accuracy using multiple models.
- Breakthroughs: Combined Google News RSS with FinBERT for sentiment tracking

---

## 🛠️ Tech Stack
Data Handling & APIs:

- yfinance (for stock data)

- feedparser (for Google News RSS)

Analysis:

- pandas, numpy (data manipulation)

Sentiment Analysis:

- transformers (FinBERT for sentiment analysis)

Visualization:

- matplotlib (for charts)

Miscellaneous:

- python-dotenv (for environment variable management)

colorama (for terminal output styling)
### Core Technologies Used:
- Frontend:
- Backend:
- Database:
- APIs: Groq
- Hosting:

### Sponsor Technologies Used (if any):
- ✅ **Groq:** Groq is used to accelerate machine learning models by providing high-performance inference. Although it’s not directly involved in the core functionalities of stock data analysis and forecasting, it can be integrated for improving the processing speed of large-scale sentiment analysis or financial model predictions. 

---

## ✨ Key Features

✅ Real-time Stock Data Retrieval: Fetches the latest stock data with Yahoo Finance API, including historical prices, returns, and more.

✅ Technical & Fundamental Analysis: Calculates key indicators like Moving Averages, RSI, Beta, and retrieves financial metrics like P/E ratio and Dividend Yield.

✅ Sentiment Analysis on News: Analyzes stock-related news using FinBERT for sentiment, helping users understand market sentiment.

---

## 📽️ Demo & Deliverables

- **Demo Video Link:** https://www.loom.com/share/3a36f57378d541e2940d6a2e667facfc?sid=7883d117-426d-4265-a6e9-96c253383dec 

---

## ✅ Tasks & Bonus Checklist

- ✅ **All members of the team completed the mandatory task - Followed at least 2 of our social channels and filled the form** (Details in Participant Manual)  
- ✅ **All members of the team completed Bonus Task 1 - Sharing of Badges and filled the form (2 points)**  (Details in Participant Manual)
- ✅ **All members of the team completed Bonus Task 2 - Signing up for Sprint.dev and filled the form (3 points)**  (Details in Participant Manual)



---

## 🧪 How to Run the Project

### Requirements:
- Node.js / Python / Docker / etc.
- API Keys (if any)
- .env file setup (if needed)

### Local Setup:
```bash
# Clone the repo
git clone https://github.com/imanurag0202/Stock-market-chatbot.git

# Install dependencies
cd Stock-market-chatbot
python -m pip install -r requirements.txt

# Execution
python chatbot.py



# Provide any backend/frontend split or environment setup notes here.
We created a python virtual environment (python version=3.12.0) using mamba and installed the dependencies using pip. 

Please use the symbol for the name of stock company, for example googl for google, appl for apple, tatasteel.ns for tata steel and so on at the end of the user input. Action (rsi, roi, moving average, ama, sentiment, bolinger bands) followed by company symbol.  
---

## 🧬 Future Scope

List improvements, extensions, or follow-up features:

- We plan to integrate with fluvio to bsemlessly access real time stock data.  
- 🛡️ Security enhancements  
- Regarding accesibility we will work to make the user input more robust and efficient by catching the keywords describing action and stock company name more effectively.

---

## 📎 Resources / Credits

- GROQ APIs used 
- NASDAQ and NSE database
- FinBert and google news rss - open source python libraries for sentiment analysis 
- Acknowledgements  

---

## 🏁 Final Words

Share your hackathon journey — challenges, learnings, fun moments, or shout-outs!

---
