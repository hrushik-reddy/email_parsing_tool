# Personal Outlook Account Setup Guide

## Required Changes for Personal Accounts

### 1. Update Azure App Registration

1. **Go to Azure Portal** → Your app → **Authentication**
2. **Add platform** → **Mobile and desktop applications**
3. **Add redirect URI**: `http://localhost:8080`
4. **Save changes**

### 2. Change API Permissions

1. **Go to API permissions**
2. **Remove** all current Application permissions
3. **Add permission** → **Microsoft Graph** → **Delegated permissions**
4. **Add**: `Mail.Read` (Delegated permission)
5. **Grant admin consent**

### 3. App Registration Type

**Change supported account types:**
1. **Go to Authentication**
2. **Supported account types**: Select **"Personal Microsoft accounts only"**
3. **Save**

### 4. Configuration Changes Made

The code now uses:
- **InteractiveBrowserCredential** (opens browser for login)
- **Delegated permissions** (user context)
- **No client secret needed** (public client)
- **tenant_id**: 'common' (for personal accounts)

## How It Works Now

1. **First run**: Browser opens for Microsoft login
2. **User consent**: Grant permission to read emails
3. **Token storage**: Credentials cached for future runs
4. **Automatic access**: No browser popup on subsequent runs

## Test the Setup

```bash
python email_fetcher.py --run-now
```

**Expected flow:**
1. Browser opens → Login to your personal Microsoft account
2. Consent screen → Click "Accept" 
3. Returns to terminal → Fetches emails automatically

## Troubleshooting

**Browser doesn't open:**
- Check firewall/antivirus blocking localhost:8080
- Try different port in redirect URI

**Login fails:**
- Ensure app supports "Personal Microsoft accounts"
- Check redirect URI matches exactly

**Permission denied:**
- Verify Mail.Read is delegated permission (not application)
- Re-grant consent if needed

## Security Note

The authentication token is cached locally. For production:
- Use environment variables
- Implement token encryption
- Set appropriate token expiration 