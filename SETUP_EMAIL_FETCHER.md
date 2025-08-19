# Email Fetcher Setup Guide

This guide will help you set up the automated email fetching service to process daily reports at 4:45 PM.

## Prerequisites

1. Python 3.8 or higher
2. Your existing REST API running on localhost:8000
3. Gmail account for testing
4. Google Cloud Console access

## Step 1: Install Dependencies

```bash
pip install -r requirements_email_fetcher.txt
```

## Step 2: Gmail API Setup

### 2.1 Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Gmail API:
   - Go to APIs & Services > Library
   - Search for "Gmail API"
   - Click "Enable"

### 2.2 Create OAuth2 Credentials

1. Go to APIs & Services > Credentials
2. Click "Create Credentials" > "OAuth client ID"
3. Choose "Desktop application"
4. Download the JSON file as `credentials.json`
5. Place `credentials.json` in your project directory

### 2.3 Configure OAuth Consent Screen

1. Go to APIs & Services > OAuth consent screen
2. Choose "External" user type
3. Fill required fields:
   - App name: "Email Fetcher"
   - User support email: your email
   - Developer contact: your email
4. Add test users (your Gmail account)

## Step 3: Configuration

### 3.1 Run Initial Setup

```bash
python email_fetcher.py --run-now
```

This will:
- Create `email_config.json` with default settings
- Prompt for Gmail authentication (first time only)
- Create `token.json` for future use

### 3.2 Edit Configuration

Edit `email_config.json` to customize:

```json
{
  "gmail": {
    "credentials_file": "credentials.json",
    "token_file": "token.json",
    "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
    "search_query": "from:reports@yourcompany.com OR subject:(daily report)",
    "max_results": 10
  },
  "schedule": {
    "time": "16:45",
    "timezone": "UTC"
  },
  "api_base_url": "http://localhost:8000",
  "notification": {
    "enabled": true,
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "email": "your-email@gmail.com",
    "password": "your-app-password",
    "recipients": ["recipient1@email.com", "recipient2@email.com"]
  }
}
```

### 3.3 Gmail Search Query Examples

Update the `search_query` field to match your emails:

- `"from:amazon-reports@amazon.com"` - Emails from specific sender
- `"subject:(Amazon Movers Shakers)"` - Emails with specific subject
- `"from:reports@company.com AND subject:(daily)"` - Combined filters
- `"has:attachment filename:*.xlsx"` - Emails with Excel attachments

## Step 4: Email Notification Setup (Optional)

To receive email notifications with the daily report:

### 4.1 Gmail App Password

1. Enable 2-factor authentication on your Gmail account
2. Go to Google Account settings > Security
3. Generate App Password for "Mail"
4. Use this password in the configuration

### 4.2 Update Notification Config

```json
"notification": {
  "enabled": true,
  "smtp_server": "smtp.gmail.com",
  "smtp_port": 587,
  "email": "your-email@gmail.com",
  "password": "your-16-char-app-password",
  "recipients": ["team@company.com"]
}
```

## Step 5: Usage

### Test Run (Immediate)
```bash
python email_fetcher.py --run-now
```

### Start Scheduler (Daily at 4:45 PM)
```bash
python email_fetcher.py
```

### Background Service (Linux/Mac)
```bash
nohup python email_fetcher.py > email_fetcher.out 2>&1 &
```

### Windows Service
```bash
python email_fetcher.py
# Run in background using Task Scheduler or pythonw
```

## Step 6: Microsoft Outlook Setup (Future)

For Microsoft Outlook integration:

1. Register app in Azure AD
2. Get client_id, client_secret, tenant_id
3. Update `microsoft` section in config
4. Grant Mail.Read permissions

## Troubleshooting

### Gmail Authentication Issues

1. **Invalid Credentials**: Ensure `credentials.json` is valid and from correct project
2. **Scope Error**: Delete `token.json` and re-authenticate
3. **API Quota**: Check Google Cloud Console for API usage limits

### API Connection Issues

1. **Connection Refused**: Ensure your REST API is running on localhost:8000
2. **Timeout**: Increase timeout in `send_to_api()` method
3. **File Format**: Ensure emails are saved as proper .eml format

### Email Search Issues

1. **No Emails Found**: Check search query syntax
2. **Date Filter**: Emails are filtered to last 24 hours by default
3. **Permissions**: Ensure Gmail API has read permissions

## Logs and Monitoring

- Check `email_fetcher.log` for detailed logs
- Reports saved as `daily_report_YYYYMMDD_HHMMSS.md`
- Monitor console output for real-time status

## Production Deployment

### Docker (Recommended)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements_email_fetcher.txt .
RUN pip install -r requirements_email_fetcher.txt
COPY . .
CMD ["python", "email_fetcher.py"]
```

### Systemd Service (Linux)

```ini
[Unit]
Description=Email Fetcher Service
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/email_tool
ExecStart=/usr/bin/python3 email_fetcher.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## Security Notes

1. Keep `credentials.json` and `token.json` secure
2. Use app passwords instead of main account passwords
3. Restrict API keys to specific IPs if possible
4. Regular credential rotation recommended 