import os
import pickle
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# SCOPES determines the level of access for Gmail API.
# 'readonly' is the safest for simple retrieval tasks.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Path to your OAuth 2.0 Client Credentials file (downloaded from Google Cloud Console)
CREDENTIALS_FILE = 'auth/credentials/credentials.json'

# Path to the token file where access/refresh tokens will be saved
TOKEN_FILE = 'auth/credentials/token.pickle'


def get_gmail_service():
    """
    Authenticates with the Gmail API and returns a service client for making API calls.

    Handles:
        - Loading existing OAuth credentials (if available)
        - Refreshing expired access tokens automatically
        - Launching browser authentication (first run only)
        - Saving new/updated tokens for future use
    Returns:
        gmail_api_service: Authorized Gmail API Python client
    """
    creds = None
    token_path = Path(TOKEN_FILE)
    # Attempt to load previously saved credentials
    if token_path.exists():
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

    # If no credentials or they're invalid/expired, initiate auth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # Attempt silent refresh
        else:
            # Initiate local server OAuth authentication
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save new credentials for next time
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
    # Build and return the Gmail API service object
    service = build('gmail', 'v1', credentials=creds)
    return service


if __name__ == "__main__":
    # Authenticate and create Gmail API service
    service = get_gmail_service()
    
    # Simple usage example: List the subject lines of your latest emails
    results = service.users().messages().list(userId='me', maxResults=5).execute()
    messages = results.get('messages', [])
    
    print("Latest email subjects:")
    for i, msg in enumerate(messages, 1):
        msg_detail = service.users().messages().get(userId='me', id=msg['id'], format='metadata', metadataHeaders=['Subject']).execute()
        headers = msg_detail['payload'].get('headers', [])
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '(No Subject)')
        print(f"{i}. {subject}")

