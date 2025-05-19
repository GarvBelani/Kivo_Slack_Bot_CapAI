from dotenv import load_dotenv
import os
import json
import requests
from requests.auth import HTTPBasicAuth

# Load environment variables
load_dotenv()

# ==== CONFIG ====
JIRA_SERVER = os.getenv("JIRA_SERVER")
JIRA_EMAIL = os.getenv("JIRA_USER")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY_SEARCH")
OUTPUT_FILE = "stored_ticket_details.json"

# =================

def load_existing_tickets():
    try:
        with open(OUTPUT_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def fetch_new_tickets(existing_keys, max_results=100):
    jql = f'project = {PROJECT_KEY} ORDER BY created DESC'
    url = f"{JIRA_SERVER}/rest/api/3/search"

    start_at = 0
    all_tickets = []

    while True:
        params = {
            "jql": jql,
            "fields": "summary,description,components,issuetype",
            "maxResults": max_results,
            "startAt": start_at
        }

        print(f"Fetching tickets starting at {start_at}...")

        response = requests.get(url, params=params, auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN))
        if response.status_code != 200:
            raise Exception(f"Failed to fetch JIRA tickets: {response.status_code} - {response.text}")

        issues = response.json().get("issues", [])
        print(f"Fetched {len(issues)} tickets from this page...")

        new_tickets = []
        for issue in issues:
            key = issue["key"]
            if key in existing_keys:
                continue

            fields = issue["fields"]
            summary = fields.get("summary", "")
            description = extract_description(fields.get("description"))
            components = [comp.get("name") for comp in fields.get("components", [])]
            issuetype = fields.get("issuetype", {}).get("name", "")

            comments = fetch_ticket_comments(issue["id"])

            new_tickets.append({
                "key": key,
                "summary": summary,
                "description": description,
                "components": components,
                "issuetype": issuetype,
                "comments": comments
            })

        all_tickets.extend(new_tickets)

        if len(issues) < max_results:
            break

        start_at += max_results

    print(f"Total new tickets fetched: {len(all_tickets)}")
    return all_tickets

def extract_description(description_field):
    if description_field is None:
        return ""
    return extract_text_from_adf(description_field).strip()

def extract_text_from_adf(adf):
    result = ""
    if isinstance(adf, dict):
        if adf.get("type") == "text":
            result += adf.get("text", "")
        elif "content" in adf:
            for item in adf["content"]:
                result += extract_text_from_adf(item)
        if adf.get("type") == "paragraph":
            result += "\n"
    elif isinstance(adf, list):
        for item in adf:
            result += extract_text_from_adf(item)
    return result

def fetch_ticket_comments(issue_id):
    url = f"{JIRA_SERVER}/rest/api/3/issue/{issue_id}/comment"
    response = requests.get(url, auth=HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN))
    if response.status_code != 200:
        print(f"❌ Failed to fetch comments for issue {issue_id}: {response.status_code}")
        return []

    comments_data = response.json().get("comments", [])
    comments_text = []
    for comment in comments_data:
        body = comment.get("body")
        if body:
            comments_text.append(extract_text_from_adf(body))

    return comments_text

def update_ticket_store():
    existing_tickets = load_existing_tickets()
    existing_keys = {t["key"] for t in existing_tickets}

    new_tickets = fetch_new_tickets(existing_keys)

    all_tickets = existing_tickets + new_tickets

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_tickets, f, indent=2)

    print(f"✅ Added {len(new_tickets)} new tickets. Total stored: {len(all_tickets)}")

if __name__ == "__main__":
    update_ticket_store()
