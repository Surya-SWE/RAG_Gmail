import base64
import quopri
import re
from datetime import datetime, timedelta
from typing import List, Dict
from googleapiclient.errors import HttpError

def clean_text(text: str) -> str:
    """
    Basic cleaning of email text:
    - Decode quoted-printable content
    - Remove HTML tags
    - Remove excessive whitespace
    """
    # Skip quoted-printable decoding if text is already a string
    # (it's already been decoded from base64)
    
    # Remove HTML tags (naive)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_email_body(payload: Dict) -> str:
    """
    Recursively extract plain text body from email payload parts.
    Handles multipart emails.
    """
    if 'parts' in payload:
        # Recursively check parts for text/plain content
        for part in payload['parts']:
            if part.get('mimeType') == 'text/plain' and 'data' in part['body']:
                data = part['body']['data']
                text = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                return clean_text(text)
            elif part.get('mimeType', '').startswith('multipart/'):
                # Recursive call
                text = extract_email_body(part)
                if text:
                    return text
    else:
        # Single part email
        if payload.get('mimeType') == 'text/plain' and 'data' in payload.get('body', {}):
            data = payload['body']['data']
            text = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            return clean_text(text)
    return ""

def get_date_query(days_back: int = 7) -> str:
    """
    Generate Gmail query for emails from the last N days.
    """
    date = (datetime.now() - timedelta(days=days_back)).strftime('%Y/%m/%d')
    # Ensure ASCII-only string
    return 'after:{}'.format(date)

def fetch_emails(service, user_id='me', query: str = '', max_results: int = 10) -> List[Dict]:
    """
    Fetch emails from Gmail API matching `query`.
    Parameters:
        - service: Authenticated Gmail API service object
        - user_id: 'me' for authenticated user
        - query: Gmail search query string (optional)
        - max_results: number of emails to fetch (limit)
    Returns:
        - List of emails with keys: 'id', 'threadId', 'subject', 'snippet', 'body'
    """
    emails = []
    try:
        # Ensure query is ASCII-encoded
        if query:
            query = query.encode('ascii', 'ignore').decode('ascii')
        
        results = service.users().messages().list(userId=user_id, q=query, maxResults=max_results).execute()
        messages = results.get('messages', [])
        
        for msg in messages:
            msg_detail = service.users().messages().get(userId=user_id, id=msg['id'], format='full').execute()
            payload = msg_detail.get('payload', {})
            headers = payload.get('headers', [])
            
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '(No Subject)')
            date = next((h['value'] for h in headers if h['name'] == 'Date'), '')
            from_email = next((h['value'] for h in headers if h['name'] == 'From'), '')
            snippet = msg_detail.get('snippet', '')
            body = extract_email_body(payload)
            
            emails.append({
                'id': msg['id'],
                'threadId': msg_detail.get('threadId', ''),
                'subject': subject,
                'date': date,
                'from': from_email,
                'snippet': snippet,
                'body': body
            })

    except HttpError as error:
        print(f'An error occurred: {error}')
    except Exception as e:
        print(f'Unexpected error: {type(e).__name__}: {e}')
        import traceback
        traceback.print_exc()
    return emails

def fetch_last_week_emails(service, user_id='me', max_results: int = 10) -> List[Dict]:
    """
    Convenience function to fetch emails from the last 7 days.
    """
    query = get_date_query(days_back=7)
    return fetch_emails(service, user_id, query, max_results)

