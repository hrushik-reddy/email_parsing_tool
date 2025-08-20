# Microsoft Outlook/365 Credentials Setup Guide

## Required Credentials

You need these 3 values from Azure:
1. **`client_id`** - Application (client) ID
2. **`client_secret`** - Client secret value  
3. **`tenant_id`** - Directory (tenant) ID

## Step-by-Step Setup

### 1. Register Application in Azure Portal

1. Go to [Azure Portal](https://portal.azure.com/)
2. Navigate to **Azure Active Directory** (or **Microsoft Entra ID**)
3. Go to **App registrations**
4. Click **+ New registration**

### 2. Configure Application

**Application Details:**
- **Name**: `Email Fetcher App`
- **Supported account types**: Choose based on your needs:
  - **Single tenant**: Only your organization
  - **Multi-tenant**: Any organizational directory
- **Redirect URI**: Leave blank for now
- Click **Register**

### 3. Get Client ID and Tenant ID

After registration, you'll see the **Overview** page:
- Copy **Application (client) ID** → This is your `client_id`
- Copy **Directory (tenant) ID** → This is your `tenant_id`

### 4. Create Client Secret

1. In your app, go to **Certificates & secrets**
2. Click **+ New client secret**
3. **Description**: `Email Fetcher Secret`
4. **Expires**: Choose duration (recommend 12-24 months)
5. Click **Add**
6. **IMPORTANT**: Copy the **Value** immediately → This is your `client_secret`
   ⚠️ You won't be able to see this value again!

### 5. Configure API Permissions

1. Go to **API permissions**
2. Click **+ Add a permission**
3. Choose **Microsoft Graph**
4. Select **Application permissions**
5. Add these permissions:
   - `Mail.Read` - Read mail in all mailboxes
   - `User.Read.All` - Read all users' profiles (if accessing specific users)
6. Click **Add permissions**
7. **IMPORTANT**: Click **Grant admin consent** (requires admin rights)

### 6. Update Configuration

Edit your `email_config.json`:

```json
{
  "microsoft": {
    "client_id": "your-application-client-id-here",
    "client_secret": "your-client-secret-value-here",
    "tenant_id": "your-directory-tenant-id-here",
    "mailbox": "user@yourcompany.com",
    "max_results": 10
  }
}
```

## Example Values

```json
{
  "microsoft": {
    "client_id": "12345678-1234-1234-1234-123456789012",
    "client_secret": "abcDEF123~hijKLM456_nopQRS789",
    "tenant_id": "87654321-4321-4321-4321-210987654321",
    "mailbox": "reports@company.com",
    "max_results": 10
  }
}
```

## Mailbox Options

**For specific user's mailbox:**
```json
"mailbox": "john.doe@company.com"
```

**For current authenticated user:**
```json
"mailbox": "me"
```

## Testing Credentials

Run this to test your setup:
```bash
python email_fetcher.py --run-now
```

## Troubleshooting

### Common Errors:

**1. Authentication Failed**
- Check `client_id`, `client_secret`, `tenant_id` are correct
- Ensure client secret hasn't expired

**2. Access Denied / 403 Error**
- Admin consent not granted
- Missing `Mail.Read` permission
- User doesn't have access to the specified mailbox

**3. Invalid Mailbox**
- User doesn't exist
- Mailbox name format incorrect
- Use UPN format: `user@domain.com`

### Permission Types:

**Application Permissions** (recommended for automation):
- `Mail.Read` - Read mail in all mailboxes
- Requires admin consent
- Works without user interaction

**Delegated Permissions** (for user context):
- `Mail.Read` - Read user's mail
- Requires user login
- Limited to specific user's mailbox

## Security Notes

1. **Store credentials securely** - Use environment variables in production
2. **Rotate secrets regularly** - Set expiration dates
3. **Principle of least privilege** - Only grant necessary permissions
4. **Monitor usage** - Check Azure logs for unauthorized access

## Production Environment Variables

Instead of storing in config file:

```bash
export MICROSOFT_CLIENT_ID="your-client-id"
export MICROSOFT_CLIENT_SECRET="your-client-secret"  
export MICROSOFT_TENANT_ID="your-tenant-id"
export MICROSOFT_MAILBOX="user@company.com"
```

Update code to read from environment:
```python
ms_config = {
    "client_id": os.getenv('MICROSOFT_CLIENT_ID'),
    "client_secret": os.getenv('MICROSOFT_CLIENT_SECRET'),
    "tenant_id": os.getenv('MICROSOFT_TENANT_ID'),
    "mailbox": os.getenv('MICROSOFT_MAILBOX', 'me')
}
``` 