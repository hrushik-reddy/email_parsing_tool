#!/usr/bin/env python3
"""
Email Fetcher Service for Gmail and Microsoft Outlook
Fetches emails daily at 4:45pm and processes them through the REST API
"""

import os
import base64
import json
import tempfile
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import schedule
import time
import logging
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

# Gmail API imports
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Microsoft Graph API imports (for future use)
try:
    import msal
    from msgraph import GraphServiceClient
    from azure.identity import ClientSecretCredential
    from kiota_abstractions.base_request_configuration import RequestConfiguration
    MICROSOFT_AVAILABLE = True
except ImportError:
    MICROSOFT_AVAILABLE = False
    print("Microsoft Graph dependencies not installed. Gmail-only mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('email_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmailFetcher:
    def __init__(self, config_file="email_config.json"):
        self.config = self.load_config(config_file)
        self.gmail_service = None
        self.microsoft_client = None
        self.api_base_url = self.config.get('api_base_url', 'http://localhost:8000')
        
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            default_config = {
                "gmail": {
                    "credentials_file": "credentials.json",
                    "token_file": "token.json",
                    "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
                    "search_query": "from:reports@company.com OR subject:(daily report)",
                    "max_results": 10
                },
                "microsoft": {
                    "client_id": "",
                    "client_secret": "",
                    "tenant_id": "",
                    "mailbox": "user@company.com",
                    "search_query": "from:reports@company.com OR subject:(daily report)"
                },
                "schedule": {
                    "time": "16:45",
                    "timezone": "UTC"
                },
                "api_base_url": "http://localhost:8000",
                                 "notification": {
                     "enabled": False,
                     "smtp_server": "",
                     "smtp_port": 587,
                     "email": "",
                     "password": "",
                     "recipients": []
                 }
            }
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config file: {config_file}")
            return default_config

    def setup_gmail_auth(self) -> bool:
        """Setup Gmail API authentication"""
        try:
            gmail_config = self.config['gmail']
            creds = None
            
            # Load existing token
            if os.path.exists(gmail_config['token_file']):
                creds = Credentials.from_authorized_user_file(
                    gmail_config['token_file'], 
                    gmail_config['scopes']
                )
            
            # If no valid credentials, get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not os.path.exists(gmail_config['credentials_file']):
                        logger.error(f"Gmail credentials file not found: {gmail_config['credentials_file']}")
                        logger.info("Please download credentials.json from Google Cloud Console")
                        return False
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        gmail_config['credentials_file'], 
                        gmail_config['scopes']
                    )
                    creds = flow.run_local_server(port=0)
                
                # Save credentials for next run
                with open(gmail_config['token_file'], 'w') as token:
                    token.write(creds.to_json())
            
            self.gmail_service = build('gmail', 'v1', credentials=creds)
            logger.info("Gmail authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"Gmail authentication failed: {e}")
            return False

    def setup_microsoft_auth(self) -> bool:
        """Setup Microsoft Graph API authentication for personal accounts"""
        if not MICROSOFT_AVAILABLE:
            logger.warning("Microsoft Graph dependencies not available")
            return False
            
        try:
            ms_config = self.config['microsoft']
            
            if not ms_config.get('client_id'):
                logger.warning("Microsoft client_id not configured")
                return False
            
            # For personal accounts, use interactive authentication
            from azure.identity import InteractiveBrowserCredential
            
            # Use public client (no secret needed for delegated permissions)
            credential = InteractiveBrowserCredential(
                client_id=ms_config['client_id'],
                tenant_id='consumers',  # Use 'consumers' for personal Microsoft accounts only
                redirect_uri='http://localhost:8080'  # Local redirect for interactive auth
            )
            
            self.microsoft_client = GraphServiceClient(
                credentials=credential,
                scopes=['https://graph.microsoft.com/Mail.Read']
            )
            
            logger.info("Microsoft Graph authentication successful")
            return True
            
        except Exception as e:
            logger.error(f"Microsoft authentication failed: {e}")
            return False

    def fetch_gmail_emails(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Fetch emails from Gmail"""
        if not self.gmail_service:
            logger.error("Gmail service not initialized")
            return []
        
        try:
            gmail_config = self.config['gmail']
            
            # Calculate date filter (last 24 hours by default)
            since_date = datetime.now() - timedelta(hours=hours_back)
            date_filter = since_date.strftime('%Y/%m/%d')
            
            # Build search query - for testing, get any recent emails
            query = f"after:{date_filter}"  # Remove specific search criteria for testing
            
            # Search for messages
            results = self.gmail_service.users().messages().list(
                userId='me',
                q=query,
                maxResults=gmail_config['max_results']
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            for message in messages:
                try:
                    # Get full message
                    msg = self.gmail_service.users().messages().get(
                        userId='me',
                        id=message['id'],
                        format='raw'
                    ).execute()
                    
                    # Decode the raw email
                    raw_email = base64.urlsafe_b64decode(msg['raw']).decode('utf-8')
                    
                    # Get message metadata
                    msg_metadata = self.gmail_service.users().messages().get(
                        userId='me',
                        id=message['id'],
                        format='metadata'
                    ).execute()
                    
                    headers = {h['name']: h['value'] for h in msg_metadata['payload']['headers']}
                    
                    emails.append({
                        'id': message['id'],
                        'raw_content': raw_email,
                        'subject': headers.get('Subject', 'No Subject'),
                        'from': headers.get('From', 'Unknown'),
                        'date': headers.get('Date', 'Unknown'),
                        'source': 'gmail'
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing Gmail message {message['id']}: {e}")
                    continue
            
            logger.info(f"Fetched {len(emails)} emails from Gmail")
            return emails
            
        except HttpError as e:
            logger.error(f"Gmail API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching Gmail emails: {e}")
            return []

    async def fetch_microsoft_emails(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Fetch emails from Microsoft Outlook"""
        if not self.microsoft_client:
            logger.error("Microsoft client not initialized")
            return []
        
        try:
            ms_config = self.config['microsoft']
            
            # Calculate date filter
            since_date = datetime.now() - timedelta(hours=hours_back)
            date_filter = since_date.isoformat() + 'Z'
            
            # Build filter query for recent emails
            filter_query = f"receivedDateTime ge {date_filter}"
            
            # For delegated permissions, always use 'me' (current user)
            mailbox = 'me'
            
            # Fetch messages using Microsoft Graph API
            request_url = f"/users/{mailbox}/messages"
            
            # Use the Graph client to get messages
            request_config = RequestConfiguration()
            request_config.query_parameters = {
                "$filter": filter_query,
                "$top": ms_config.get('max_results', 10),
                "$select": "id,subject,from,receivedDateTime,body"
            }
            
            messages = await self.microsoft_client.users.by_user_id(mailbox).messages.get(request_configuration=request_config)
            
            emails = []
            
            if messages and messages.value:
                for message in messages.value:
                    try:
                        # Get the full message with MIME content
                        msg_request_config = RequestConfiguration()
                        msg_request_config.query_parameters = {
                            "$select": "id,subject,from,receivedDateTime,body,internetMessageHeaders"
                        }
                        
                        full_message = await self.microsoft_client.users.by_user_id(mailbox).messages.by_message_id(message.id).get(request_configuration=msg_request_config)
                        
                        # Convert to EML format (simplified)
                        raw_content = self.convert_outlook_to_eml(full_message)
                        
                        emails.append({
                            'id': message.id,
                            'raw_content': raw_content,
                            'subject': message.subject or 'No Subject',
                            'from': message.from_.email_address.address if message.from_ else 'Unknown',
                            'date': message.received_date_time.isoformat() if message.received_date_time else 'Unknown',
                            'source': 'outlook'
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing Outlook message {message.id}: {e}")
                        continue
            
            logger.info(f"Fetched {len(emails)} emails from Outlook")
            return emails
            
        except Exception as e:
            logger.error(f"Error fetching Microsoft emails: {e}")
            return []
    
    def convert_outlook_to_eml(self, message) -> str:
        """Convert Outlook message to EML format"""
        try:
            # Create basic EML structure
            eml_lines = []
            
            # Headers
            if message.from_:
                eml_lines.append(f"From: {message.from_.email_address.address}")
            if message.subject:
                eml_lines.append(f"Subject: {message.subject}")
            if message.received_date_time:
                eml_lines.append(f"Date: {message.received_date_time.strftime('%a, %d %b %Y %H:%M:%S %z')}")
            
            eml_lines.append("Content-Type: text/html; charset=utf-8")
            eml_lines.append("")  # Empty line between headers and body
            
            # Body
            if message.body and message.body.content:
                eml_lines.append(message.body.content)
            
            return '\n'.join(eml_lines)
            
        except Exception as e:
            logger.error(f"Error converting message to EML: {e}")
            return f"Subject: {getattr(message, 'subject', 'Error')}\n\nError converting message"

    def save_email_as_eml(self, email_data: Dict[str, Any], temp_dir: str) -> Optional[str]:
        """Save email content as .eml file"""
        try:
            filename = f"{email_data['source']}_{email_data['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.eml"
            filepath = os.path.join(temp_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(email_data['raw_content'])
            
            return filepath
        except Exception as e:
            logger.error(f"Error saving email as EML: {e}")
            return None

    def send_to_api(self, eml_file_path: str) -> Optional[Dict[str, Any]]:
        """Send EML file to the REST API for analysis"""
        try:
            with open(eml_file_path, 'rb') as f:
                files = {'file': (os.path.basename(eml_file_path), f, 'message/rfc822')}
                data = {
                    'model': 'gpt4-o',
                    'max_tokens': 1500
                }
                
                response = requests.post(
                    f"{self.api_base_url}/analyze",
                    files=files,
                    data=data,
                    timeout=120
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"API error {response.status_code}: {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error sending to API: {e}")
            return None

    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate a consolidated report from all analysis results"""
        report_lines = [
            f"# Daily Email Analysis Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total emails processed: {len(results)}",
            "",
            "---",
            ""
        ]
        
        for i, result in enumerate(results, 1):
            if result and result.get('success'):
                report_lines.extend([
                    f"## Email {i}: {result.get('metadata', {}).get('subject', 'Unknown Subject')}",
                    f"**From:** {result.get('metadata', {}).get('from', 'Unknown')}",
                    f"**Date:** {result.get('metadata', {}).get('date', 'Unknown')}",
                    "",
                    "### Analysis Results:",
                    result.get('analysis', 'No analysis available'),
                    "",
                    "---",
                    ""
                ])
            else:
                report_lines.extend([
                    f"## Email {i}: Processing Failed",
                    "Error occurred during analysis",
                    "",
                    "---",
                    ""
                ])
        
        return '\n'.join(report_lines)

    def send_notification(self, report: str, success_count: int, total_count: int):
        """Send email notification with the report"""
        try:
            notification_config = self.config.get('notification', {})
            
            if not notification_config.get('enabled', False):
                return
            
            smtp_server = notification_config.get('smtp_server')
            smtp_port = notification_config.get('smtp_port', 587)
            email = notification_config.get('email')
            password = notification_config.get('password')
            recipients = notification_config.get('recipients', [])
            
            if not all([smtp_server, email, password, recipients]):
                logger.warning("Notification settings incomplete")
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"Daily Email Analysis Report - {success_count}/{total_count} processed"
            
            # Add report as body
            msg.attach(MIMEText(report, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(email, password)
                server.send_message(msg)
            
            logger.info(f"Report sent to {len(recipients)} recipients")
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")

    async def process_emails(self):
        """Main function to fetch and process emails"""
        logger.info("Starting email processing...")
        
        all_emails = []
        
        # Fetch Gmail emails (commented out)
        # if self.setup_gmail_auth():
        #     gmail_emails = self.fetch_gmail_emails()
        #     all_emails.extend(gmail_emails)
        
        # Fetch Microsoft emails
        if self.setup_microsoft_auth():
            microsoft_emails = await self.fetch_microsoft_emails()
            all_emails.extend(microsoft_emails)
        
        if not all_emails:
            logger.info("No emails found to process")
            return
        
        logger.info(f"Processing {len(all_emails)} emails...")
        
        # Process each email
        results = []
        success_count = 0
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for email_data in all_emails:
                try:
                    # Save as EML file
                    eml_file = self.save_email_as_eml(email_data, temp_dir)
                    if not eml_file:
                        continue
                    
                    # Send to API for analysis
                    result = self.send_to_api(eml_file)
                    if result and result.get('success'):
                        success_count += 1
                        logger.info(f"Successfully processed: {email_data['subject']}")
                    else:
                        logger.error(f"Failed to process: {email_data['subject']}")
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing email: {e}")
                    results.append(None)
        
        # Generate and save report
        report = self.generate_report(results)
        
        # Save report to file
        report_filename = f"daily_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved: {report_filename}")
        logger.info(f"Processing complete: {success_count}/{len(all_emails)} emails successful")
        
        # Notification disabled - just save report locally

    def start_scheduler(self):
        """Start the scheduled email processing"""
        schedule_time = self.config.get('schedule', {}).get('time', '16:45')
        
        schedule.every().day.at(schedule_time).do(lambda: asyncio.run(self.process_emails()))
        
        logger.info(f"Scheduler started. Will run daily at {schedule_time}")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Email Fetcher Service')
    parser.add_argument('--run-now', action='store_true', help='Run email processing immediately')
    parser.add_argument('--config', default='email_config.json', help='Configuration file path')
    
    args = parser.parse_args()
    
    fetcher = EmailFetcher(args.config)
    
    if args.run_now:
        logger.info("Running email processing immediately...")
        asyncio.run(fetcher.process_emails())
    else:
        fetcher.start_scheduler()

if __name__ == "__main__":
    main() 