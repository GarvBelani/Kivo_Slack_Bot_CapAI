from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
import os
import requests
from requests.auth import HTTPBasicAuth
import json
import re
import openai
import numpy as np
import time
import faiss

# Load environment variables
load_dotenv()

# Slack bot tokens and Jira details
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN")

JIRA_SERVER = os.getenv("JIRA_SERVER")
JIRA_EMAIL = os.getenv("JIRA_USER")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_CREATE_KEY = os.getenv("JIRA_PROJECT_KEY")
JIRA_SEARCH_KEY = os.getenv("JIRA_PROJECT_KEY_SEARCH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
pending_tickets = {}

# Global variables
faiss_index = None
ticket_lookup = []  # Stores ticket dicts aligned with index vectors
embedding_dim = 1536  # For text-embedding-3-small (adjust if using another model)


def parse_ticket_details(text):
    if "create ticket" not in text.lower():
        return None
    ticket_details = {
        "summary": "", "description": "", "priority": "",
        "brand": "", "environment": "", "components": "","issuetype": ""
    }
    patterns = {
        "summary": r"summary\s*[:=]\s*([^\;]+)",
        "description": r"description\s*[:=]\s*([^\;]+)",
        "priority": r"priority\s*[:=]\s*([^\;]+)",
        "brand": r"brand\s*[:=]\s*([^\;]+)",
        "environment": r"environment\s*[:=]\s*([^\;]+)",
        "components": r"components\s*[:=]\s*([^\;]+)",
        "issuetype": r"issuetype\s*[:=]\s*([^\;]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            ticket_details[key] = match.group(1).strip()
    return ticket_details if all(ticket_details.values()) else None

with open("stored_tickets.json", "r") as infile:
    tickets = json.load(infile)

def load_stored_tickets():
    with open("stored_tickets.json", "r") as f:
        return json.load(f)

# Normalize vectors
def normalize_vector(vec):
    vec = np.array(vec)
    return vec / np.linalg.norm(vec)

def initialize_faiss_index():
    global faiss_index, ticket_lookup

    stored_tickets = load_stored_tickets()
    ticket_embeddings = []
    ticket_lookup = []

    for ticket in stored_tickets:
        if "text_embedding" in ticket:
            emb = normalize_vector(ticket["text_embedding"])
            ticket_embeddings.append(emb)
            ticket_lookup.append(ticket)

    if not ticket_embeddings:
        print("⚠️ No ticket embeddings found to build FAISS index.")
        return

    ticket_embeddings = np.array(ticket_embeddings).astype('float32')

    # Build FAISS index
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    faiss_index.add(ticket_embeddings)

    print(f"✅ FAISS index initialized with {len(ticket_embeddings)} tickets.")

def search_similar_tickets_cached(problem_description, top_k=5):
    if faiss_index is None or not ticket_lookup:
        print("❌ FAISS index not initialized.")
        return []

    # Summarize query for contextual match
    summarized_query = summarize_user_query(problem_description)
    print(f"🔎 Summarized Query: {summarized_query}")

    # Get embedding
    query_embedding = get_embedding(summarized_query)
    query_embedding = normalize_vector(query_embedding).astype('float32')

    # Search FAISS
    D, I = faiss_index.search(np.array([query_embedding]), top_k)

    print(f"🔍 Top Similarity Scores: {D[0]}")

    # Dynamic threshold: keep results >= 80% of best score
    best_score = D[0][0]
    matched_tickets = []
    for idx, score in zip(I[0], D[0]):
        if score >= best_score * 0.8 and score > 0.4:
            matched_tickets.append(ticket_lookup[idx])

    return matched_tickets



# Summarize user query before embedding (contextual match)
def summarize_user_query(user_query):
    prompt = (
        "Summarize this user query into a concise problem statement for semantic search:\n\n"
        f"{user_query}"
    )
    try:

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ GPT summarization failed: {e}")
        return user_query.strip()  # Fallback to raw query

# Embed text using OpenAI
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.Embedding.create(input=[text], model=model)
    return response["data"][0]["embedding"]

def search_similar_tickets_faiss(problem_description, stored_tickets, top_k=5):
    # Summarize problem description
    summarized_query = summarize_user_query(problem_description)
    print(f"🔎 Summarized Query: {summarized_query}")

    # Get embedding for summarized query
    query_embedding = get_embedding(summarized_query)
    query_embedding = normalize_vector(query_embedding)

    # Prepare ticket embeddings
    ticket_embeddings = []
    valid_tickets = []
    for ticket in stored_tickets:
        if "text_embedding" in ticket:
            emb = normalize_vector(ticket["text_embedding"])
            ticket_embeddings.append(emb)
            valid_tickets.append(ticket)

    if not ticket_embeddings:
        print("⚠️ No valid ticket embeddings found.")
        return []

    ticket_embeddings = np.array(ticket_embeddings).astype('float32')

    # FAISS index for inner product (cosine similarity)
    index = faiss.IndexFlatIP(ticket_embeddings.shape[1])
    index.add(ticket_embeddings)

    # Search top_k
    D, I = index.search(np.array([query_embedding]).astype('float32'), top_k)

    print(f"🔍 Top Similarity Scores: {D[0]}")

    # Dynamic threshold: keep results >= 80% of best score
    best_score = D[0][0]
    matched_tickets = []
    for idx, score in zip(I[0], D[0]):
        if score >= best_score * 0.8 and score > 0.4:  # Lower hard floor to 0.4
            matched_tickets.append(valid_tickets[idx])

    return matched_tickets


updated_tickets = []  # <--- Add this line before the for loop

for idx, ticket in enumerate(tickets):
    if "text_embedding" not in ticket:
        text = f"{ticket.get('summary', '')} {ticket.get('description', '')}"
        embedding = get_embedding(text)
        if embedding:
            ticket["text_embedding"] = embedding
        else:
            print(f"Skipping ticket {ticket.get('key')} due to embedding failure")
    updated_tickets.append(ticket)
    print(f"[{idx+1}/{len(tickets)}] Processed {ticket.get('key')}")




def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def search_similar_tickets_semantic(problem_description, stored_tickets, top_n=3, threshold=0.6):
    query_embedding = get_embedding(problem_description)
    if not query_embedding:
        return []

    results = []
    for ticket in stored_tickets:
        if "text_embedding" not in ticket:
            continue
        sim = cosine_similarity(query_embedding, ticket["text_embedding"])
        results.append((sim, ticket))

    results.sort(key=lambda x: x[0], reverse=True)
    return [ticket for sim, ticket in results if sim > threshold][:top_n]

def search_similar_tickets(summary):
    keywords = " ".join(summary.split()[:5])
    jql = f'project = {JIRA_SEARCH_KEY} AND summary ~ "{keywords}" ORDER BY created DESC'
    url = f"{JIRA_SERVER}/rest/api/3/search"
    params = {"jql": jql, "fields": "summary", "maxResults": 5}
    response = requests.get(url, params=params, auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN))
    if response.status_code != 200:
        return []
    return [{"key": issue["key"], "summary": issue["fields"]["summary"]} for issue in response.json().get("issues", [])]


def extract_text_from_adf(adf):
    result = ""
    if isinstance(adf, dict):
        if adf.get("type") == "text":
            result += adf.get("text", "")
        elif "content" in adf:
            for item in adf["content"]:
                result += extract_text_from_adf(item)
    elif isinstance(adf, list):
        for item in adf:
            result += extract_text_from_adf(item)
    return result

def exact_match_check(problem_description, stored_tickets):
    query = problem_description.strip().lower()
    for ticket in stored_tickets:
        if query in ticket.get("summary", "").lower():
            return ticket
    return None

def get_all_comments(ticket_key):
    url = f"{JIRA_SERVER}/rest/api/3/issue/{ticket_key}/comment"
    response = requests.get(url, auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN))
    if response.status_code != 200:
        return []
    comments = response.json().get("comments", [])
    return [extract_text_from_adf(comment["body"]).strip() for comment in comments]

def summarize_comments_with_gpt(comments):
    if not comments:
        return "⚠️ No comments to summarize."
    prompt = (
            "Summarize the following Jira ticket comments in 50-100 words, "
            "highlighting key discussion points and any resolution:\n\n"
            + "\n\n".join(comments)
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Failed to summarize comments: {str(e)}"



@app.event("app_home_opened")
def update_home_tab(event, client, logger):
    user_id = event["user"]
    try:
        jira_email = JIRA_EMAIL  # Placeholder for actual user mapping
        assigned_tickets = fetch_user_jira_tickets(jira_email)
        watched_tickets = fetch_watched_jira_tickets(jira_email)
        reported_tickets = fetch_reported_jira_tickets(jira_email)

        def ticket_card(ticket):
            status_emoji = "🟢" if ticket.get("status") == "Open" else "🔵"
            return {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*<{JIRA_SERVER}/browse/{ticket['key']}|{ticket['key']}>* | {ticket['summary']}\nStatus: {status_emoji} {ticket['status']} | Priority: *{ticket['priority']}*"
                }
            }

        blocks = []

        # Welcome Section
        blocks.append({"type": "header", "text": {"type": "plain_text", "text": "👋 Welcome to Kivo"}})

        # Quick Actions
        blocks.append({
            "type": "actions",
            "elements": [
                {"type": "button", "text": {"type": "plain_text", "text": "➕ Create Ticket"}, "action_id": "open_jira_modal", "style": "primary"},
                {"type": "button", "text": {"type": "plain_text", "text": "🔍 Find Solution"}, "action_id": "open_find_solution_modal"}
            ]
        })

        # Assigned Tickets
        blocks.append({"type": "divider"})
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "🗂️ *Assigned to You*"}})
        if assigned_tickets:
            for ticket in assigned_tickets[:3]:
                blocks.append(ticket_card(ticket))
            if len(assigned_tickets) > 3:
                blocks.append({
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": f"View {len(assigned_tickets) - 3} more in Jira"},
                            "url": f"{JIRA_SERVER}/issues/?jql=assignee=currentUser()+AND+status!=Done"
                        }
                    ]
                })
        else:
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "No assigned tickets! 🎉"}})

        # Watching Tickets
        if watched_tickets:
            blocks.append({"type": "divider"})
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "👀 *Watching*"}})
            for ticket in watched_tickets[:3]:
                blocks.append(ticket_card(ticket))
            if len(watched_tickets) > 3:
                blocks.append({
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": f"View {len(watched_tickets) - 3} more in Jira"},
                            "url": f"{JIRA_SERVER}/issues/?jql=watcher=currentUser()+AND+status!=Done"
                        }
                    ]
                })

        # Reported Tickets
        if reported_tickets:
            blocks.append({"type": "divider"})
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "📝 *Reported by You*"}})
            for ticket in reported_tickets[:3]:
                blocks.append(ticket_card(ticket))
            if len(reported_tickets) > 3:
                blocks.append({
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": f"View {len(reported_tickets) - 3} more in Jira"},
                            "url": f"{JIRA_SERVER}/issues/?jql=reporter=currentUser()+AND+status!=Done"
                        }
                    ]
                })

        # Footer Info
        blocks.append({"type": "divider"})

        # Publish View
        client.views_publish(
            user_id=user_id,
            view={"type": "home", "blocks": blocks}
        )

    except Exception as e:
        logger.error(f"Failed to update home tab: {str(e)}")


# Add this new view handler for the Find Solution modal submission
@app.view("find_solution")
def handle_find_solution_submission(ack, body, client, view, logger):
    ack()
    user_id = body["user"]["id"]
    problem_description = view["state"]["values"]["problem_description_block"]["problem_description_input"]["value"]

    try:
        loading_msg = client.chat_postMessage(
            channel=user_id,
            text="🤖 I'm working on your request. Please wait..."
        )
        loading_ts = loading_msg["ts"]
        stored_tickets = load_stored_tickets()
        similar_tickets = search_similar_tickets_cached(problem_description, top_k=5)

        if not similar_tickets:
            message = "No similar tickets found for this problem. Consider creating a new ticket if needed."
            client.chat_postMessage(
                channel=user_id,
                text=message
            )
            return

        # Get comments from similar tickets and summarize resolutions
        solutions = []
        for ticket in similar_tickets[:3]:  # Limit to top 3 similar tickets
            comments = get_all_comments(ticket["key"])
            if comments:
                summary = summarize_solution_with_gpt(problem_description, ticket["summary"], comments)
                solutions.append({
                    "ticket": ticket,
                    "summary": summary
                })

        if not solutions:
            message = "Found similar tickets but no resolution comments available."
            client.chat_postMessage(
                channel=user_id,
                text=message
            )
            return

        # Prepare the response message with solutions
        blocks = [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"🔍 *Potential Solutions for your problem:*\n> {problem_description}"
                }
            },
            {"type": "divider"}
        ]

        for solution in solutions:
            ticket = solution["ticket"]
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*<{JIRA_SERVER}/browse/{ticket['key']}|{ticket['key']}>*: {ticket['summary']}\n\n{solution['summary']}"
                }
            })
            blocks.append({"type": "divider"})


        # Send the solutions to the user
        client.chat_postMessage(
            channel=user_id,
            blocks=blocks,
            text="Here are potential solutions for your problem"
        )

    except Exception as e:
        logger.error(f"Error finding solution: {e}")
        client.chat_postMessage(
            channel=user_id,
            text="⚠️ An error occurred while searching for solutions. Please try again."
        )


def summarize_solution_with_gpt(problem_description, ticket_summary, comments):
    joined_comments = "\n\n".join(comments)
    prompt = (
        f"A user described this issue:\n\n'{problem_description}'\n\n"
        f"We found this ticket summary:\n'{ticket_summary}'\n\n"
        "These are the ticket's comments:\n"
        f"{joined_comments}\n\n"
        "Based on the comments above, summarize the **solution or root cause** in 100-150 words. "
        "If no solution is clear, say so clearly."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Failed to summarize: {e}"




def fetch_watched_jira_tickets(jira_email):
    jql = f'watcher = "{jira_email}" AND assignee != "{jira_email}" ORDER BY updated DESC'
    url = f"{JIRA_SERVER}/rest/api/3/search"
    params = {"jql": jql, "fields": "summary,status,priority", "maxResults": 10}
    response = requests.get(url, params=params, auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN))
    if response.status_code != 200:
        return []
    data = response.json().get("issues", [])
    return [
        {
            "key": issue["key"],
            "summary": issue["fields"]["summary"],
            "status": issue["fields"]["status"]["name"],
            "priority": issue["fields"]["priority"]["name"] if issue["fields"].get("priority") else "None"
        }
        for issue in data
    ]


def fetch_user_jira_tickets(jira_email):
    jql = f'assignee = "{jira_email}" AND statusCategory != Done ORDER BY updated DESC'
    url = f"{JIRA_SERVER}/rest/api/3/search"
    params = {"jql": jql, "fields": "summary,status,priority", "maxResults": 10}
    response = requests.get(url, params=params, auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN))
    if response.status_code != 200:
        return []
    data = response.json().get("issues", [])
    return [
        {
            "key": issue["key"],
            "summary": issue["fields"]["summary"],
            "status": issue["fields"]["status"]["name"],
            "priority": issue["fields"]["priority"]["name"] if issue["fields"].get("priority") else "None"
        }
        for issue in data
    ]

@app.action("view_more_assigned")
def handle_view_more_assigned(ack, body, logger):
    ack()
    logger.info("Clicked: view_more_assigned")

@app.command("/comment")
def handle_comment_command(ack, body, client, logger):
    ack()

    user_id = body["user_id"]
    text = body.get("text", "").strip()

    if not text:
        client.chat_postMessage(channel=user_id, text="❌ Usage: `/comment CP-1234 This is the comment text`")
        return

    # Try to parse issue key and comment
    import re
    match = re.match(r"([A-Z]+-\d+)\s+(.+)", text)
    if not match:
        client.chat_postMessage(channel=user_id, text="⚠️ Please provide a ticket key and comment. Example:\n`/comment CP-1234 This is my update`")
        return

    issue_key, comment = match.groups()

    try:
        # Send comment to Jira
        url = f"{JIRA_SERVER}/rest/api/3/issue/{issue_key}/comment"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        data = {
            "body": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": comment}]
                    }
                ]
            }
        }

        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(data),
            auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
        )

        if response.status_code == 201:
            client.chat_postMessage(channel=user_id, text=f"✅ Comment added to <{JIRA_SERVER}/browse/{issue_key}|{issue_key}>")
        else:
            logger.error(f"Failed to add comment: {response.text}")
            client.chat_postMessage(channel=user_id, text="❌ Failed to add comment. Please check the ticket key or try again.")

    except Exception as e:
        logger.error(f"Exception in /comment: {e}")
        client.chat_postMessage(channel=user_id, text="❌ An error occurred while adding the comment.")


@app.command("/help")
def handle_help_command(ack, body, client, logger):
    ack()

    user_id = body["user_id"]

    help_text = """
👋 *Here’s how you can use this bot:*

🆕 *To create a new Jira ticket:*  
Click the *➕ Create Ticket* button in the *Home* tab.

🔧 `/ticket <description>`  
Create a new Jira ticket using natural language.

🔍 *To find a solution to a problem:*  
Click the *🔍 Find Solution* button in the *Home* tab and describe your issue.

💬 *To add a comment to a Jira ticket:*  
Use the `/comment <JIRA-KEY>` command. I'll help you write a smart update using GPT.

📋 *To check the status of a ticket:*  
Type `status CP-1234` in a message to me.

🏠 *Home tab also shows:*  
- Tickets assigned to you  
- Tickets you reported or are watching  
- Quick action buttons

💡 Need more help? Just ask!
"""

    try:
        client.chat_postMessage(channel=user_id, text=help_text)
    except Exception as e:
        logger.error(f"Error in /help: {e}")
        client.chat_postMessage(channel=user_id, text="❌ Failed to show help.")



@app.action("view_more_watching")
def handle_view_more_watching(ack, body, logger):
    ack()
    logger.info("Clicked: view_more_watching")

@app.action("open_jira_modal")
def open_modal(ack, body, client, logger):
    ack()
    try:
        client.views_open(
            trigger_id=body["trigger_id"],
            view={
                "type": "modal",
                "callback_id": "create_jira_ticket",
                "title": {
                    "type": "plain_text",
                    "text": "Create Jira Ticket"
                },
                "submit": {
                    "type": "plain_text",
                    "text": "Submit"
                },
                "close": {
                    "type": "plain_text",
                    "text": "Cancel"
                },
                "blocks": [
                    {
                        "type": "input",
                        "block_id": "summary_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "summary_input"
                        },
                        "label": {"type": "plain_text", "text": "Summary"}
                    },
                    {
                        "type": "input",
                        "block_id": "description_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "description_input",
                            "multiline": True
                        },
                        "label": {"type": "plain_text", "text": "Description"}
                    },
                    {
                        "type": "input",
                        "block_id": "priority_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "priority_input",
                            "options": [
                                {"text": {"type": "plain_text", "text": "Highest-P0"}, "value": "Highest-P0"},
                                {"text": {"type": "plain_text", "text": "High-P1"}, "value": "High-P1"},
                                {"text": {"type": "plain_text", "text": "Medium-P2"}, "value": "Medium-P2"},
                                {"text": {"type": "plain_text", "text": "Low-P3"}, "value": "Low-P3"}
                            ]
                        },
                        "label": {"type": "plain_text", "text": "Priority"}
                    },
                    {
                        "type": "input",
                        "block_id": "brand_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "brand_input",
                            "options": [
                                {"text": {"type": "plain_text", "text": "Abbott_HK_Demo"}, "value": "Abbott_HK_Demo"},
                                {"text": {"type": "plain_text", "text": "Abbott_HK_Prod"}, "value": "Abbott_HK_Prod"},
                                {"text": {"type": "plain_text", "text": "ADAC"}, "value": "ADAC"},
                                {"text": {"type": "plain_text", "text": "ADAC_UAT"}, "value": "ADAC_UAT"},
                                {"text": {"type": "plain_text", "text": "#1 Cochran"}, "value": "#1 Cochran"},
                                {"text": {"type": "plain_text", "text": "#1 Cochran Demo"}, "value": "#1 Cochran Demo"},
                                {"text": {"type": "plain_text", "text": "AB InBev"}, "value": "AB InBev"},
                                {"text": {"type": "plain_text", "text": "AB InBev Staging"},
                                 "value": "AB InBev Staging"},
                                {"text": {"type": "plain_text", "text": "Abbott"}, "value": "Abbott"},
                                {"text": {"type": "plain_text", "text": "Abbott All Demo"}, "value": "Abbott All Demo"},
                                {"text": {"type": "plain_text", "text": "Abbott Indonesia"},
                                 "value": "Abbott Indonesia"},
                                {"text": {"type": "plain_text", "text": "Abbott Malaysia"}, "value": "Abbott Malaysia"},
                                {"text": {"type": "plain_text", "text": "Abbott Malaysia Demo"},
                                 "value": "Abbott Malaysia Demo"},
                                {"text": {"type": "plain_text", "text": "Abbott MY"}, "value": "Abbott MY"},
                                {"text": {"type": "plain_text", "text": "Aldar"}, "value": "Aldar"},
                                {"text": {"type": "plain_text", "text": "ABFRL"}, "value": "ABFRL"},
                                {"text": {"type": "plain_text", "text": "AFG"}, "value": "AFG"},
                                {"text": {"type": "plain_text", "text": "Al Futtaim"}, "value": "Al Futtaim"},
                                {"text": {"type": "plain_text", "text": "Bata"}, "value": "Bata"},
                                {"text": {"type": "plain_text", "text": "Bata Indonesia"}, "value": "Bata Indonesia"},
                                {"text": {"type": "plain_text", "text": "Bata Malaysia"}, "value": "Bata Malaysia"},
                                {"text": {"type": "plain_text", "text": "Bata Singapore"}, "value": "Bata Singapore"},
                                {"text": {"type": "plain_text", "text": "BIRA"}, "value": "BIRA"},
                                {"text": {"type": "plain_text", "text": "BlueBird"}, "value": "BlueBird"},
                                {"text": {"type": "plain_text", "text": "Calvin Klein"}, "value": "Calvin Klein"},
                                {"text": {"type": "plain_text", "text": "Dominos IDN"}, "value": "Dominos IDN"},
                                {"text": {"type": "plain_text", "text": "GAP"}, "value": "GAP"}
                            ]
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Brand"
                        }
                    },
                    {
                        "type": "input",
                        "block_id": "environment_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "environment_input",
                            "options": [
                                {"text": {"type": "plain_text", "text": "Prod"}, "value": "Prod"},
                                {"text": {"type": "plain_text", "text": "Demo"}, "value": "Demo"},
                                {"text": {"type": "plain_text", "text": "Go-Live"}, "value": "Go-Live"},
                                {"text": {"type": "plain_text", "text": "UAT"}, "value": "UAT"},
                                {"text": {"type": "plain_text", "text": "Nightly"}, "value": "Nightly"},
                                {"text": {"type": "plain_text", "text": "Staging"}, "value": "Staging"}
                            ]
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Environment"
                        }
                    },
                    {
                        "type": "input",
                        "block_id": "components_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "components_input",
                            "options": [
                                {"text": {"type": "plain_text", "text": "Access"}, "value": "Access"},
                                {"text": {"type": "plain_text", "text": "admin-console"}, "value": "admin-console"},
                                {"text": {"type": "plain_text", "text": "aIRA"}, "value": "aIRA"},
                                {"text": {"type": "plain_text", "text": "alerts"}, "value": "alerts"},
                                {"text": {"type": "plain_text", "text": "API"}, "value": "API"},
                                {"text": {"type": "plain_text", "text": "artificial intelligence"},
                                 "value": "artificial intelligence"},
                                {"text": {"type": "plain_text", "text": "Arya"}, "value": "Arya"},
                                {"text": {"type": "plain_text", "text": "AST"}, "value": "AST"},
                                {"text": {"type": "plain_text", "text": "Async"}, "value": "Async"},
                                {"text": {"type": "plain_text", "text": "Audience Filter"}, "value": "Audience Filter"},
                                {"text": {"type": "plain_text", "text": "aws"}, "value": "aws"},
                                {"text": {"type": "plain_text", "text": "Backend-CRM"}, "value": "Backend-CRM"},
                                {"text": {"type": "plain_text", "text": "Backend-R+"}, "value": "Backend-R+"},
                                {"text": {"type": "plain_text", "text": "Badges"}, "value": "Badges"},
                                {"text": {"type": "plain_text", "text": "Behavioural events"},
                                 "value": "Behavioural events"},
                                {"text": {"type": "plain_text", "text": "Campaign Personalisation"},
                                 "value": "Campaign Personalisation"},
                                {"text": {"type": "plain_text", "text": "Campaigns"}, "value": "Campaigns"},
                                {"text": {"type": "plain_text", "text": "Campaigns UI"}, "value": "Campaigns UI"},
                                {"text": {"type": "plain_text", "text": "Campaigns, Iris API(apps)"},
                                 "value": "Campaigns, Iris API(apps)"},
                                {"text": {"type": "plain_text", "text": "Capillary Cloud"}, "value": "Capillary Cloud"},
                                {"text": {"type": "plain_text", "text": "CDP"}, "value": "CDP"},
                                {"text": {"type": "plain_text", "text": "Cloudflare"}, "value": "Cloudflare"},
                                {"text": {"type": "plain_text", "text": "Communication Engine"},
                                 "value": "Communication Engine"},
                                {"text": {"type": "plain_text", "text": "Connect+"}, "value": "Connect+"},
                                {"text": {"type": "plain_text", "text": "Connect+ UI"}, "value": "Connect+ UI"},
                                {"text": {"type": "plain_text", "text": "ConnectPlus-Hydra"},
                                 "value": "ConnectPlus-Hydra"},
                                {"text": {"type": "plain_text", "text": "Core UI"}, "value": "Core UI"},
                                {"text": {"type": "plain_text", "text": "Cortex Search"}, "value": "Cortex Search"},
                                {"text": {"type": "plain_text", "text": "Coupon-Gateway"}, "value": "Coupon-Gateway"},
                                {"text": {"type": "plain_text", "text": "Coupons"}, "value": "Coupons"},
                                {"text": {"type": "plain_text", "text": "Creatives"}, "value": "Creatives"},
                                {"text": {"type": "plain_text", "text": "Creatives UI"}, "value": "Creatives UI"},
                                {"text": {"type": "plain_text", "text": "Cron Scheduler"}, "value": "Cron Scheduler"},
                                {"text": {"type": "plain_text", "text": "Custom Events"}, "value": "Custom Events"},
                                {"text": {"type": "plain_text", "text": "Data Cleanup"}, "value": "Data Cleanup"},
                                {"text": {"type": "plain_text", "text": "data deletion"}, "value": "data deletion"},
                                {"text": {"type": "plain_text", "text": "Data Request"}, "value": "Data Request"},
                                {"text": {"type": "plain_text", "text": "Databricks"}, "value": "Databricks"},
                                {"text": {"type": "plain_text", "text": "DataTools"}, "value": "DataTools"},
                                {"text": {"type": "plain_text", "text": "Dracarys"}, "value": "Dracarys"},
                                {"text": {"type": "plain_text", "text": "eCommerce"}, "value": "eCommerce"},
                                {"text": {"type": "plain_text", "text": "EMF"}, "value": "EMF"},
                                {"text": {"type": "plain_text", "text": "Engage"}, "value": "Engage"},
                                {"text": {"type": "plain_text", "text": "Engage UI"}, "value": "Engage UI"},
                                {"text": {"type": "plain_text", "text": "Event Notification"},
                                 "value": "Event Notification"},
                                {"text": {"type": "plain_text", "text": "EventSDKConsumers"},
                                 "value": "EventSDKConsumers"},
                                {"text": {"type": "plain_text", "text": "Export Framework"},
                                 "value": "Export Framework"},
                                {"text": {"type": "plain_text", "text": "FFCR"}, "value": "FFCR"},
                                {"text": {"type": "plain_text", "text": "FT"}, "value": "FT"},
                                {"text": {"type": "plain_text", "text": "FT-Apps"}, "value": "FT-Apps"}
                            ]
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Components"
                        }
                    },
                    {
                        "type": "input",
                        "block_id": "issuetype_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "issuetype_input",
                            "options": [
                                {"text": {"type": "plain_text", "text": "Task"}, "value": "Task"},
                                {"text": {"type": "plain_text", "text": "Bug"}, "value": "Bug"},
                                {"text": {"type": "plain_text", "text": "Epic"}, "value": "Epic"},
                                {"text": {"type": "plain_text", "text": "Enhancement"}, "value": "Enhancement"},
                                {"text": {"type": "plain_text", "text": "Story"}, "value": "Story"},
                                {"text": {"type": "plain_text", "text": "Release"}, "value": "Release"},
                                {"text": {"type": "plain_text", "text": "Customer Request"}, "value": "Customer Request"},
                                {"text": {"type": "plain_text", "text": "InfraIssue"}, "value": "InfraIssue"},
                                {"text": {"type": "plain_text", "text": "Tech2Geo"}, "value": "Tech2Geo"},
                                {"text": {"type": "plain_text", "text": "Improvement"}, "value": "Improvement"},
                                {"text": {"type": "plain_text", "text": "Release Request"}, "value": "Release Request"},
                                {"text": {"type": "plain_text", "text": "Change Request"}, "value": "Change Request"}
                            ]
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Issue Type"
                        }
                    }
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error opening modal: {e}")

def suggest_fields_from_similar_tickets(tickets):
    try:
        # Use only valid components and issuetypes from modal setup
        valid_components = [
            "Access", "admin-console", "aIRA", "alerts", "API", "artificial intelligence", "Arya",
            "AST", "Async", "Audience Filter", "aws", "Backend-CRM", "Backend-R+", "Badges",
            "Behavioural events", "Campaign Personalisation", "Campaigns", "Campaigns UI",
            "Campaigns, Iris API(apps)", "Capillary Cloud", "CDP", "Cloudflare",
            "Communication Engine", "Connect+", "Connect+ UI", "ConnectPlus-Hydra", "Core UI",
            "Cortex Search", "Coupon-Gateway", "Coupons", "Creatives", "Creatives UI",
            "Cron Scheduler", "Custom Events", "Data Cleanup", "data deletion", "Data Request",
            "Databricks", "DataTools", "Dracarys", "eCommerce", "EMF", "Engage", "Engage UI",
            "Event Notification", "EventSDKConsumers", "Export Framework", "FFCR", "FT", "FT-Apps"
        ]

        valid_issuetypes = [
            "Task", "Bug", "Epic", "Enhancement", "Story", "Release", "Customer Request",
            "InfraIssue", "Tech2Geo", "Improvement", "Release Request", "Change Request"
        ]

        # Use component and issuetype fields from actual similar tickets to guide context
        prompt = (
            f"Choose the most relevant *component* and *issuetype* for the new Jira ticket. "
            f"You must pick from ONLY these:\n\n"
            f"Components: {', '.join(valid_components)}\n"
            f"IssueTypes: {', '.join(valid_issuetypes)}\n\n"
            f"Here are some example similar tickets:\n"
        )

        for t in tickets[:3]:
            prompt += f"- Summary: {t['summary']}\n  Components: {', '.join(t.get('components', []))}\n  IssueType: {t.get('issuetype', '')}\n\n"

        prompt += (
            "Now return a JSON object like this:\n"
            "{\"components\": \"<component from list>\", \"issuetype\": \"<issuetype from list>\"}"
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100
        )

        result = json.loads(response.choices[0].message.content.strip())

        # Validate output strictly
        if result.get("components") not in valid_components:
            result["components"] = "API"  # fallback
        if result.get("issuetype") not in valid_issuetypes:
            result["issuetype"] = "Bug"

        return result

    except Exception as e:
        print(f"❌ GPT field suggestion error: {e}")
        return {"components": "API", "issuetype": "Bug"}


@app.action("confirm_ticket_creation")
def handle_ticket_modal_open(ack, body, client, logger):
    ack()
    user_id = body["user"]["id"]
    trigger_id = body["trigger_id"]
    open_ticket_modal(user_id, client, trigger_id, logger)

def open_ticket_modal(user_id, client, trigger_id, logger):
    details = pending_tickets.get(user_id, {})
    try:
        client.views_open(
            trigger_id=trigger_id,
            view={
                "type": "modal",
                "callback_id": "create_jira_ticket",
                "title": {"type": "plain_text", "text": "Create Jira Ticket"},
                "submit": {"type": "plain_text", "text": "Submit"},
                "close": {"type": "plain_text", "text": "Cancel"},
                "blocks": [
                    {
                        "type": "input",
                        "block_id": "summary_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "summary_input",
                            "initial_value": details.get("summary", "")
                        },
                        "label": {"type": "plain_text", "text": "Summary"}
                    },
                    {
                        "type": "input",
                        "block_id": "description_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "description_input",
                            "initial_value": details.get("description", ""),
                            "multiline": True
                        },
                        "label": {"type": "plain_text", "text": "Description"}
                    },
                    {
                        "type": "input",
                        "block_id": "components_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "components_input",
                            "options": [
                                {"text": {"type": "plain_text", "text": "Access"}, "value": "Access"},
                                {"text": {"type": "plain_text", "text": "admin-console"}, "value": "admin-console"},
                                {"text": {"type": "plain_text", "text": "aIRA"}, "value": "aIRA"},
                                {"text": {"type": "plain_text", "text": "alerts"}, "value": "alerts"},
                                {"text": {"type": "plain_text", "text": "API"}, "value": "API"},
                                {"text": {"type": "plain_text", "text": "artificial intelligence"},
                                 "value": "artificial intelligence"},
                                {"text": {"type": "plain_text", "text": "Arya"}, "value": "Arya"},
                                {"text": {"type": "plain_text", "text": "AST"}, "value": "AST"},
                                {"text": {"type": "plain_text", "text": "Async"}, "value": "Async"},
                                {"text": {"type": "plain_text", "text": "Audience Filter"}, "value": "Audience Filter"},
                                {"text": {"type": "plain_text", "text": "aws"}, "value": "aws"},
                                {"text": {"type": "plain_text", "text": "Backend-CRM"}, "value": "Backend-CRM"},
                                {"text": {"type": "plain_text", "text": "Backend-R+"}, "value": "Backend-R+"},
                                {"text": {"type": "plain_text", "text": "Badges"}, "value": "Badges"},
                                {"text": {"type": "plain_text", "text": "Behavioural events"},
                                 "value": "Behavioural events"},
                                {"text": {"type": "plain_text", "text": "Campaign Personalisation"},
                                 "value": "Campaign Personalisation"},
                                {"text": {"type": "plain_text", "text": "Campaigns"}, "value": "Campaigns"},
                                {"text": {"type": "plain_text", "text": "Campaigns UI"}, "value": "Campaigns UI"},
                                {"text": {"type": "plain_text", "text": "Campaigns, Iris API(apps)"},
                                 "value": "Campaigns, Iris API(apps)"},
                                {"text": {"type": "plain_text", "text": "Capillary Cloud"}, "value": "Capillary Cloud"},
                                {"text": {"type": "plain_text", "text": "CDP"}, "value": "CDP"},
                                {"text": {"type": "plain_text", "text": "Cloudflare"}, "value": "Cloudflare"},
                                {"text": {"type": "plain_text", "text": "Communication Engine"},
                                 "value": "Communication Engine"},
                                {"text": {"type": "plain_text", "text": "Connect+"}, "value": "Connect+"},
                                {"text": {"type": "plain_text", "text": "Connect+ UI"}, "value": "Connect+ UI"},
                                {"text": {"type": "plain_text", "text": "ConnectPlus-Hydra"},
                                 "value": "ConnectPlus-Hydra"},
                                {"text": {"type": "plain_text", "text": "Core UI"}, "value": "Core UI"},
                                {"text": {"type": "plain_text", "text": "Cortex Search"}, "value": "Cortex Search"},
                                {"text": {"type": "plain_text", "text": "Coupon-Gateway"}, "value": "Coupon-Gateway"},
                                {"text": {"type": "plain_text", "text": "Coupons"}, "value": "Coupons"},
                                {"text": {"type": "plain_text", "text": "Creatives"}, "value": "Creatives"},
                                {"text": {"type": "plain_text", "text": "Creatives UI"}, "value": "Creatives UI"},
                                {"text": {"type": "plain_text", "text": "Cron Scheduler"}, "value": "Cron Scheduler"},
                                {"text": {"type": "plain_text", "text": "Custom Events"}, "value": "Custom Events"},
                                {"text": {"type": "plain_text", "text": "Data Cleanup"}, "value": "Data Cleanup"},
                                {"text": {"type": "plain_text", "text": "data deletion"}, "value": "data deletion"},
                                {"text": {"type": "plain_text", "text": "Data Request"}, "value": "Data Request"},
                                {"text": {"type": "plain_text", "text": "Databricks"}, "value": "Databricks"},
                                {"text": {"type": "plain_text", "text": "DataTools"}, "value": "DataTools"},
                                {"text": {"type": "plain_text", "text": "Dracarys"}, "value": "Dracarys"},
                                {"text": {"type": "plain_text", "text": "eCommerce"}, "value": "eCommerce"},
                                {"text": {"type": "plain_text", "text": "EMF"}, "value": "EMF"},
                                {"text": {"type": "plain_text", "text": "Engage"}, "value": "Engage"},
                                {"text": {"type": "plain_text", "text": "Engage UI"}, "value": "Engage UI"},
                                {"text": {"type": "plain_text", "text": "Event Notification"},
                                 "value": "Event Notification"},
                                {"text": {"type": "plain_text", "text": "EventSDKConsumers"},
                                 "value": "EventSDKConsumers"},
                                {"text": {"type": "plain_text", "text": "Export Framework"},
                                 "value": "Export Framework"},
                                {"text": {"type": "plain_text", "text": "FFCR"}, "value": "FFCR"},
                                {"text": {"type": "plain_text", "text": "FT"}, "value": "FT"},
                                {"text": {"type": "plain_text", "text": "FT-Apps"}, "value": "FT-Apps"}
                            ]
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Components"
                        }
                    },
                    {
                        "type": "input",
                        "block_id": "issuetype_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "issuetype_input",
                            "options": [
                                {"text": {"type": "plain_text", "text": "Task"}, "value": "Task"},
                                {"text": {"type": "plain_text", "text": "Bug"}, "value": "Bug"},
                                {"text": {"type": "plain_text", "text": "Epic"}, "value": "Epic"},
                                {"text": {"type": "plain_text", "text": "Enhancement"}, "value": "Enhancement"},
                                {"text": {"type": "plain_text", "text": "Story"}, "value": "Story"},
                                {"text": {"type": "plain_text", "text": "Release"}, "value": "Release"},
                                {"text": {"type": "plain_text", "text": "Customer Request"},
                                 "value": "Customer Request"},
                                {"text": {"type": "plain_text", "text": "InfraIssue"}, "value": "InfraIssue"},
                                {"text": {"type": "plain_text", "text": "Tech2Geo"}, "value": "Tech2Geo"},
                                {"text": {"type": "plain_text", "text": "Improvement"}, "value": "Improvement"},
                                {"text": {"type": "plain_text", "text": "Release Request"}, "value": "Release Request"},
                                {"text": {"type": "plain_text", "text": "Change Request"}, "value": "Change Request"}
                            ]
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Issue Type"
                        }
                    },
                    {
                        "type": "input",
                        "block_id": "brand_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "brand_input",
                            "options": [
                                {"text": {"type": "plain_text", "text": "Abbott_HK_Demo"}, "value": "Abbott_HK_Demo"},
                                {"text": {"type": "plain_text", "text": "Abbott_HK_Prod"}, "value": "Abbott_HK_Prod"},
                                {"text": {"type": "plain_text", "text": "ADAC"}, "value": "ADAC"},
                                {"text": {"type": "plain_text", "text": "ADAC_UAT"}, "value": "ADAC_UAT"},
                                {"text": {"type": "plain_text", "text": "#1 Cochran"}, "value": "#1 Cochran"},
                                {"text": {"type": "plain_text", "text": "#1 Cochran Demo"}, "value": "#1 Cochran Demo"},
                                {"text": {"type": "plain_text", "text": "AB InBev"}, "value": "AB InBev"},
                                {"text": {"type": "plain_text", "text": "AB InBev Staging"},
                                 "value": "AB InBev Staging"},
                                {"text": {"type": "plain_text", "text": "Abbott"}, "value": "Abbott"},
                                {"text": {"type": "plain_text", "text": "Abbott All Demo"}, "value": "Abbott All Demo"},
                                {"text": {"type": "plain_text", "text": "Abbott Indonesia"},
                                 "value": "Abbott Indonesia"},
                                {"text": {"type": "plain_text", "text": "Abbott Malaysia"}, "value": "Abbott Malaysia"},
                                {"text": {"type": "plain_text", "text": "Abbott Malaysia Demo"},
                                 "value": "Abbott Malaysia Demo"},
                                {"text": {"type": "plain_text", "text": "Abbott MY"}, "value": "Abbott MY"},
                                {"text": {"type": "plain_text", "text": "Aldar"}, "value": "Aldar"},
                                {"text": {"type": "plain_text", "text": "ABFRL"}, "value": "ABFRL"},
                                {"text": {"type": "plain_text", "text": "AFG"}, "value": "AFG"},
                                {"text": {"type": "plain_text", "text": "Al Futtaim"}, "value": "Al Futtaim"},
                                {"text": {"type": "plain_text", "text": "Bata"}, "value": "Bata"},
                                {"text": {"type": "plain_text", "text": "Bata Indonesia"}, "value": "Bata Indonesia"},
                                {"text": {"type": "plain_text", "text": "Bata Malaysia"}, "value": "Bata Malaysia"},
                                {"text": {"type": "plain_text", "text": "Bata Singapore"}, "value": "Bata Singapore"},
                                {"text": {"type": "plain_text", "text": "BIRA"}, "value": "BIRA"},
                                {"text": {"type": "plain_text", "text": "BlueBird"}, "value": "BlueBird"},
                                {"text": {"type": "plain_text", "text": "Calvin Klein"}, "value": "Calvin Klein"},
                                {"text": {"type": "plain_text", "text": "Dominos IDN"}, "value": "Dominos IDN"},
                                {"text": {"type": "plain_text", "text": "GAP"}, "value": "GAP"}
                            ]
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Brand"
                        }
                    },
                    {
                        "type": "input",
                        "block_id": "environment_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "environment_input",
                            "options": [
                                {"text": {"type": "plain_text", "text": "Prod"}, "value": "Prod"},
                                {"text": {"type": "plain_text", "text": "Demo"}, "value": "Demo"},
                                {"text": {"type": "plain_text", "text": "Go-Live"}, "value": "Go-Live"},
                                {"text": {"type": "plain_text", "text": "UAT"}, "value": "UAT"},
                                {"text": {"type": "plain_text", "text": "Nightly"}, "value": "Nightly"},
                                {"text": {"type": "plain_text", "text": "Staging"}, "value": "Staging"}
                            ]
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Environment"
                        }
                    },
                    {
                        "type": "input",
                        "block_id": "priority_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "priority_input",
                            "options": [
                                {"text": {"type": "plain_text", "text": "Highest-P0"}, "value": "Highest-P0"},
                                {"text": {"type": "plain_text", "text": "High-P1"}, "value": "High-P1"},
                                {"text": {"type": "plain_text", "text": "Medium-P2"}, "value": "Medium-P2"},
                                {"text": {"type": "plain_text", "text": "Low-P3"}, "value": "Low-P3"}
                            ]
                        },
                        "label": {"type": "plain_text", "text": "Priority"}
                    }
                    # Reuse existing brand, environment, priority dropdown blocks from your modal
                    # Suggestion: refactor them into reusable function if needed
                ]
            }
        )
    except Exception as e:
        logger.error(f"Failed to open ticket modal: {e}")

def extract_comment_text(body):
    text_parts = []

    if not body:
        return ""

    for block in body.get("content", []):
        if block.get("type") == "paragraph":
            for inline in block.get("content", []):
                if inline.get("type") == "text":
                    text_parts.append(inline.get("text", ""))

    return " ".join(text_parts).strip() if text_parts else "Unable to parse last comment."


def get_ticket_status(issue_key):
    url = f"{JIRA_SERVER}/rest/api/3/issue/{issue_key}"
    response = requests.get(url, auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN))

    if response.status_code != 200:
        return f"❗ Unable to fetch details for `{issue_key}`. Check the key or try again later."

    issue = response.json()
    fields = issue.get("fields", {})

    # Safely access nested fields
    status = fields.get("status", {}).get("name", "Unknown")
    assignee_data = fields.get("assignee")
    assignee = assignee_data.get("displayName") if assignee_data else "Unassigned"
    summary = fields.get("summary", "No summary")
    comments_data = fields.get("comment", {}).get("comments", [])

    if comments_data:
        last_comment_body = comments_data[-1].get("body", {})
        last_comment = extract_comment_text(last_comment_body)
    else:
        last_comment = "No comments yet."

    return (
        f"*📋 Ticket:* <{JIRA_SERVER}/browse/{issue_key}|{issue_key}>\n"
        f"*🔖 Summary:* {summary}\n"
        f"*👤 Assignee:* {assignee}\n"
        f"*📌 Status:* {status}\n"
        f"*💬 Last Comment:* {last_comment}"
    )


@app.event("app_mention")
def handle_app_mention(event, say):
    user = event["user"]
    text = event["text"]

    if user in pending_tickets:
        decision = text.strip().lower()
        if decision == "yes":
            create_jira_ticket(user, say)
        else:
            say(text="✅ Ticket creation canceled.", thread_ts=event["ts"])
            del pending_tickets[user]
        return

    ticket_details = parse_ticket_details(text)
    if not ticket_details:
        say(text="❌ Invalid command, please use a valid command (e.g., 'create ticket Summary: ... Description: ... Priority: ...').")
        return

    similar_tickets = search_similar_tickets(ticket_details["summary"])
    if similar_tickets:
        message_blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": ":warning: *Found similar tickets in project CP:*"}}]
        for ticket in similar_tickets:
            comments = get_all_comments(ticket["key"])
            summary = summarize_comments_with_gpt(comments)
            block = {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*<{JIRA_SERVER}/browse/{ticket['key']}|{ticket['key']}>*: *{ticket['summary']}*\n> {summary if summary else '⚠️ No comments to summarize.'}"
                }
            }
            message_blocks.append(block)

        message_blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": ":question: *Do you want to create a new ticket based on the above details?*"
            }
        })

        message_blocks.append({
            "type": "actions",
            "elements": [
                {"type": "button", "text": {"type": "plain_text", "text": "Yes, create ticket"}, "value": "yes", "action_id": "create_ticket_yes"},
                {"type": "button", "text": {"type": "plain_text", "text": "No, cancel"}, "value": "no", "action_id": "create_ticket_no"}
            ]
        })

        say(
            text="I found similar tickets. Do you want to create a new ticket?",
            blocks=message_blocks
        )
        pending_tickets[user] = ticket_details
    else:
        pending_tickets[user] = ticket_details
        create_jira_ticket(user, say)

@app.action("create_ticket_yes")
def handle_create_ticket_yes(ack, body, say, logger):
    ack()
    user = body["user"]["id"]
    if user in pending_tickets:
        create_jira_ticket(user, say)
    else:
        say(text="❌ No ticket details available to create a Jira ticket.", thread_ts=body["message"]["ts"])
    logger.info(f"User {user} confirmed ticket creation.")

@app.action("create_ticket_no")
def handle_create_ticket_no(ack, body, say, logger):
    ack()
    user = body["user"]["id"]
    if user in pending_tickets:
        del pending_tickets[user]
        say(text="✅ Ticket creation canceled.", thread_ts=body["message"]["ts"])
    else:
        say(text="❌ No pending ticket creation to cancel.", thread_ts=body["message"]["ts"])
    logger.info(f"User {user} canceled ticket creation.")

@app.view("create_jira_ticket")
def handle_submission(ack, body, client, logger, say=None):
    values = body["view"]["state"]["values"]
    user = body["user"]["id"]

    errors = {}

    # --- Validate Priority ---
    priority_input = values["priority_block"]["priority_input"]
    priority = priority_input.get("selected_option", {}).get("value")
    if not priority:
        errors["priority_block"] = "Please select a valid priority."

    # --- Validate Brand ---
    brand_input = values["brand_block"]["brand_input"]
    brand = brand_input.get("selected_option", {}).get("value")
    if not brand:
        errors["brand_block"] = "Please select a valid brand."

    # --- Validate Environment ---
    environment_input = values["environment_block"]["environment_input"]
    environment = environment_input.get("selected_option", {}).get("value")
    if not environment:
        errors["environment_block"] = "Please select a valid environment."

    # --- Validate Components ---
    components_input = values["components_block"]["components_input"]
    components = components_input.get("selected_option", {}).get("value")
    if not components or not components.strip():
        errors["components_block"] = "Please enter a valid component."

    # --Validate Issue Type
    issuetype_input = values["issuetype_block"]["issuetype_input"]
    issuetype = issuetype_input.get("selected_option", {}).get("value")
    if not issuetype or not issuetype.strip():
        errors["issuetype_block"] = "Please enter a valid Issue type."

    # --- If errors exist, respond with them ---
    if errors:
        ack(response_action="errors", errors=errors)
        return
    ack()
    ticket_details = {
        "summary": values["summary_block"]["summary_input"]["value"],
        "description": values["description_block"]["description_input"]["value"],
        "priority": values["priority_block"]["priority_input"]["selected_option"]["value"],
        "brand": values["brand_block"]["brand_input"]["selected_option"]["value"],
        "environment": values["environment_block"]["environment_input"]["selected_option"]["value"],
        "components": values["components_block"]["components_input"]["selected_option"]["value"],
        "issuetype": values["issuetype_block"]["issuetype_input"]["selected_option"]["value"]
    }

    # Store ticket temporarily to reuse existing create_jira_ticket function
    pending_tickets[user] = ticket_details



    # Say not directly available here — send confirmation via chat
    try:
        result = client.conversations_open(users=user)
        channel_id = result["channel"]["id"]
        def say_fn(message): client.chat_postMessage(channel=channel_id, text=message)
        create_jira_ticket(user, say_fn)
    except Exception as e:
        logger.error(f"Failed to notify user: {str(e)}")

def extract_ticket_fields_with_gpt(text):
    prompt = (
        "Extract a concise summary and description from the following problem statement. "
        "Return JSON like: {\"summary\": \"...\", \"description\": \"...\"}\n\n"
        f"Issue: {text}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        print(f"❌ GPT field extraction error: {e}")
        return {"summary": "", "description": ""}

@app.message(re.compile(r"(?i)^status\s+([A-Z]+-\d+)"))
def handle_ticket_status(message, say, context, client, logger):
    user_id = message["user"]
    issue_key = context["matches"][0]  # Extract matched ticket key like CP-1234

    response_text = get_ticket_status(issue_key)
    say(text=response_text)
    # notify_action(f"Checked status for *{issue_key}*", user_id)


@app.message(re.compile(r"(?i)^create ticket"))
def handle_direct_message_event(message, say, logger):
    user = message['user']
    text = message['text']

    # Parse ticket details from message
    ticket_details = parse_ticket_details(text)
    if not ticket_details:
        say("❌ Invalid format. Please use: 'create ticket Summary: ... Description: ... Priority: ... Brand: ... Environment: ... Components: ...'")
        return

    # Check for similar tickets
    similar_tickets = search_similar_tickets(ticket_details["summary"])
    if similar_tickets:
        message_blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": ":warning: *Found similar tickets in project CP:*"}}
        ]
        for ticket in similar_tickets:
            comments = get_all_comments(ticket["key"])
            summary = summarize_comments_with_gpt(comments)
            block = {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*<{JIRA_SERVER}/browse/{ticket['key']}|{ticket['key']}>*: *{ticket['summary']}*\n> {summary if summary else '⚠️ No comments to summarize.'}"
                }
            }
            message_blocks.append(block)

        message_blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": ":question: *Do you want to create a new ticket based on the above details?*"
            }
        })

        message_blocks.append({
            "type": "actions",
            "elements": [
                {"type": "button", "text": {"type": "plain_text", "text": "Yes, create ticket"}, "value": "yes",
                 "action_id": "create_ticket_yes"},
                {"type": "button", "text": {"type": "plain_text", "text": "No, cancel"}, "value": "no",
                 "action_id": "create_ticket_no"}
            ]
        })

        say(
            text="I found similar tickets. Do you want to create a new ticket?",
            blocks=message_blocks
        )
        pending_tickets[user] = ticket_details
    else:
        pending_tickets[user] = ticket_details
        create_jira_ticket(user, say)

@app.event("message")
def handle_message_events(event, say):
    user = event.get("user")
    text = event.get("text", "").strip().lower()
    if not user or user not in pending_tickets:
        return
    if text == "yes":
        create_jira_ticket(user, say)
    elif text == "no":
        say(text="✅ Ticket creation canceled.", thread_ts=event["ts"])
        del pending_tickets[user]
    else:
        say(text="❌ Please reply with 'yes' or 'no' to continue.")

@app.action("view_more_reported")
def handle_view_more_reported(ack, body, logger):
    ack()
    logger.info("Clicked: view_more_reported")

@app.action("open_find_solution_modal")
def open_find_solution_modal(ack, body, client, logger):
    # Acknowledge the action first
    ack()

    try:
        # Now open the modal
        client.views_open(
            trigger_id=body["trigger_id"],
            view={
                "type": "modal",
                "callback_id": "find_solution",
                "title": {
                    "type": "plain_text",
                    "text": "Find Solution"
                },
                "submit": {
                    "type": "plain_text",
                    "text": "Search"
                },
                "close": {
                    "type": "plain_text",
                    "text": "Cancel"
                },
                "blocks": [
                    {
                        "type": "input",
                        "block_id": "problem_description_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "problem_description_input",
                            "multiline": True
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Describe your problem"
                        }
                    }
                ]
            }
        )
    except Exception as e:
        logger.error(f"Error opening find solution modal: {e}")

def fetch_reported_jira_tickets(jira_email):
    jql = f'reporter = "{jira_email}" AND statusCategory != Done ORDER BY updated DESC'
    url = f"{JIRA_SERVER}/rest/api/3/search"
    params = {"jql": jql, "fields": "summary,status,priority", "maxResults": 10}
    response = requests.get(url, params=params, auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN))
    if response.status_code != 200:
        return []
    data = response.json().get("issues", [])
    return [
        {
            "key": issue["key"],
            "summary": issue["fields"]["summary"],
            "status": issue["fields"]["status"]["name"],
            "priority": issue["fields"]["priority"]["name"] if issue["fields"].get("priority") else "None"
        }
        for issue in data
    ]

@app.command("/ticket")
def handle_ticket_command(ack, body, client, logger):
    ack()
    user_id = body["user_id"]
    text = body.get("text", "").strip()

    if not text:
        client.chat_postMessage(channel=user_id, text="❌ Please describe your issue after `/ticket` command.")
        return

    try:
        loading_msg = client.chat_postMessage(
            channel=user_id,
            text="🤖 I'm working on your request. Please wait..."
        )
        loading_ts = loading_msg["ts"]
        # Extract ticket fields (summary & description) via GPT
        extracted = extract_ticket_fields_with_gpt(text)
        summary = extracted.get("summary", "")
        description = extracted.get("description", "")

        if not summary or not description:
            client.chat_postMessage(channel=user_id, text="⚠️ Couldn't extract ticket summary or description. Please rephrase.")
            return

        # Search similar tickets
        similar_tickets = search_similar_tickets_cached(text, top_k=5)

        if similar_tickets:
            blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": f"🔍 *Found similar tickets:*"}}]
            for ticket in similar_tickets:
                comments = get_all_comments(ticket["key"])
                summary_text = summarize_solution_with_gpt(text, ticket["summary"], comments)
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*<{JIRA_SERVER}/browse/{ticket['key']}|{ticket['key']}>*: *{ticket['summary']}*\n{summary_text}"
                    }
                })
                blocks.append({"type": "divider"})

            # Suggest components and issuetype
            suggestions = suggest_fields_from_similar_tickets(similar_tickets)

            # Store for later use (user-specific)
            pending_tickets[user_id] = {
                "summary": summary,
                "description": description,
                "components": suggestions.get("components", ""),
                "issuetype": suggestions.get("issuetype", "")
            }

            # Prompt to proceed
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Create Ticket"},
                        "action_id": "confirm_ticket_creation"
                    }
                ]
            })

            client.chat_postMessage(channel=user_id, blocks=blocks, text="Do you want to create a new ticket?")
        else:
            # No similar tickets, ask directly
            pending_tickets[user_id] = {"summary": summary, "description": description}
            trigger_id = body["trigger_id"]
            open_ticket_modal(user_id, client, trigger_id, logger)


    except Exception as e:
        logger.error(f"/ticket error: {e}")
        client.chat_postMessage(channel=user_id, text="❌ Something went wrong while processing your ticket.")



def create_jira_ticket(user, say):
    ticket_details = pending_tickets[user]
    del pending_tickets[user]
    url = f"{JIRA_SERVER}/rest/api/3/issue"
    auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    payload = {
        "fields": {
            "customfield_11997": [{"value": ticket_details["brand"]}],
            "customfield_11800": [{"value": ticket_details["environment"]}],
            "components": [{"name": ticket_details["components"]}],
            "description": {
                "content": [
                    {
                        "content": [{"text": ticket_details["description"], "type": "text"}],
                        "type": "paragraph"
                    }
                ],
                "type": "doc",
                "version": 1
            },
            "issuetype": {"name": ticket_details["issuetype"]},
            "priority": {"name": ticket_details["priority"]},
            "project": {"id": "11693"},
            "summary": ticket_details["summary"]
        }
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), auth=auth)
        if response.status_code == 201:
            issue_key = response.json()["key"]
            say(f"✅ JIRA ticket created: <{JIRA_SERVER}/browse/{issue_key}|{issue_key}>")
        else:
            error_data = response.json()
            errors = error_data.get("errors", {})
            messages = [
                f"Please provide a {field_map.get(field, 'valid value')}."
                for field, msg in errors.items()
                for field_map in [{"priority": "valid priority", "summary": "valid summary",
                                   "description": "valid description", "components": "valid components",
                                   "brand": "valid brand", "environment": "valid environment", "issuetype" : "valid issuetype" }]
            ]
            say("❌ " + " ".join(messages) if messages else "❌ Failed to create JIRA ticket due to unknown error.")
    except Exception as e:
        say(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    initialize_faiss_index()
    SocketModeHandler(app, SLACK_APP_TOKEN).start()

