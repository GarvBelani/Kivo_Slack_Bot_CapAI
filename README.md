# 🧠 Slack-Jira GPT Bot

This Slack bot integrates with Jira and OpenAI to help you:
- Create Jira tickets using natural language
- Find solutions from similar historical tickets
- View assigned/watched/reported tickets in Slack
- Comment on Jira tickets and check ticket status—all from Slack

---

## 🔧 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/GarvBelani/Kivo_Slack_Bot_CapAI.git
cd slack-jira-gpt-bot
```

### 2. Create and activate virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Create a `.env` file in the root folder with the following keys:

```env
SLACK_BOT_TOKEN=your-slack-bot-token
SLACK_SIGNING_SECRET=your-slack-signing-secret
SLACK_APP_TOKEN=your-slack-app-level-token

JIRA_SERVER=https://your-jira-domain.atlassian.net
JIRA_USER=your-jira-email
JIRA_API_TOKEN=your-jira-api-token
JIRA_PROJECT_KEY=CJ
JIRA_PROJECT_KEY_SEARCH=CP

OPENAI_API_KEY=your-openai-api-key
```

---

## 🧠 How to Run

1. **Preprocess ticket data**  
   Run this script to embed and store existing tickets:

   ```bash
   python save_ticket_details.py
   python store_ticket_info.py
   ```


2. **Start the bot**  
   This will launch the Slack app with Socket Mode:

   ```bash
   python main.py
   ```

---

## 📁 File Structure

```
.
├── main.py                 # Slack bot app with all features
├── save_ticket_details.py # Script to embed & cache Jira tickets
├── stored_tickets.json    # Ticket data with embeddings
├── requirements.txt
└── .env                   # Your secrets (not tracked in Git)
```

---

## 📌 Features

- 🤖 Create Jira tickets from natural language
- 🔍 Get GPT-based summaries of similar past tickets
- 🏠 View assigned, watched, and reported tickets in the Slack Home tab
- 💬 Comment on and track the status of any Jira ticket from Slack
- ✨ Auto-generate summary and description using GPT when creating tickets

---

## 🛡️ Security

- `.env`, `stored_tickets.json` and `stored_ticket_details.json` are **excluded** from version control
- Keep your API tokens secret and never commit them to GitHub