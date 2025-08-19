#!/usr/bin/env python3
"""
Quick Start Script for Email Fetcher Service
Guides through initial setup and testing
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_step(step_num, title):
    """Print step header"""
    print(f"\n📌 Step {step_num}: {title}")
    print("-" * 40)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    try:
        # Check if requirements file exists
        if not os.path.exists("requirements_email_fetcher.txt"):
            print("❌ requirements_email_fetcher.txt not found")
            return False
        
        # Install dependencies
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements_email_fetcher.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print("❌ Failed to install dependencies")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_api_running():
    """Check if the REST API is running"""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ REST API is running on localhost:8000")
            return True
        else:
            print(f"❌ REST API returned status: {response.status_code}")
            return False
    except ImportError:
        print("⚠️  Cannot test API - requests not installed yet")
        return False
    except:
        print("❌ REST API is not running on localhost:8000")
        print("   Please start your API with: python api.py")
        return False

def create_sample_config():
    """Create sample configuration file"""
    print("Creating sample configuration...")
    
    config = {
        "gmail": {
            "credentials_file": "credentials.json",
            "token_file": "token.json",
            "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
            "search_query": "from:amazon-reports@amazon.com OR subject:(Amazon Movers)",
            "max_results": 10
        },
        "microsoft": {
            "client_id": "",
            "client_secret": "",
            "tenant_id": "",
            "mailbox": "user@company.com",
            "search_query": "from:reports@company.com"
        },
        "schedule": {
            "time": "16:45",
            "timezone": "UTC"
        },
        "api_base_url": "http://localhost:8000",
        "notification": {
            "enabled": False,
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "email": "",
            "password": "",
            "recipients": []
        }
    }
    
    with open("email_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration file created: email_config.json")
    return True

def show_gmail_setup_instructions():
    """Show Gmail API setup instructions"""
    print_step("", "Gmail API Setup Required")
    
    print("""
🔧 To use Gmail integration, you need to:

1. Go to Google Cloud Console (https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Gmail API:
   - APIs & Services > Library
   - Search "Gmail API" and enable it
4. Create OAuth2 credentials:
   - APIs & Services > Credentials
   - Create Credentials > OAuth client ID
   - Choose "Desktop application"
   - Download as 'credentials.json'
5. Place credentials.json in this directory

📋 OAuth Consent Screen setup:
   - APIs & Services > OAuth consent screen
   - Choose "External" user type
   - Fill app name: "Email Fetcher"
   - Add your email as test user

⚠️  Without credentials.json, the email fetcher won't work!
""")

def customize_search_query():
    """Help user customize the search query"""
    print_step("", "Customize Email Search")
    
    print("""
📧 Edit email_config.json to customize your email search:

Example search queries:
• from:amazon-reports@amazon.com
• subject:(daily report)
• from:reports@company.com AND subject:(Amazon)
• has:attachment filename:*.xlsx

Current config is set for Amazon Movers & Shakers emails.
""")
    
    response = input("\nWould you like to edit the search query now? (y/n): ").lower()
    
    if response == 'y':
        print("\nOpening email_config.json for editing...")
        config_path = Path("email_config.json")
        
        if sys.platform == "darwin":  # macOS
            os.system(f"open {config_path}")
        elif sys.platform == "win32":  # Windows
            os.system(f"start {config_path}")
        else:  # Linux
            os.system(f"xdg-open {config_path}")
        
        input("Press Enter after editing and saving the file...")

def run_test_suite():
    """Run the test suite"""
    print_step("", "Running Tests")
    
    try:
        result = subprocess.run([sys.executable, "test_email_fetcher.py"], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def show_next_steps():
    """Show next steps to user"""
    print_step("", "Next Steps")
    
    print("""
🚀 Your email fetcher is ready! Here's what you can do:

📋 Test immediately:
   python email_fetcher.py --run-now

⏰ Start scheduled service (daily at 4:45 PM):
   python email_fetcher.py

🧪 Run tests anytime:
   python test_email_fetcher.py

📁 Files created:
   • email_config.json - Configuration file
   • email_fetcher.log - Logs (when running)
   • daily_report_*.md - Generated reports

🔧 Customize:
   • Edit email_config.json for search queries
   • Add notification settings for email alerts
   • Modify schedule time as needed

📚 Documentation:
   • Read SETUP_EMAIL_FETCHER.md for detailed setup
   • Check logs in email_fetcher.log for troubleshooting
""")

def main():
    """Main quick start function"""
    print_header("Email Fetcher Quick Start")
    print("Welcome! This script will help you set up automated email fetching.")
    
    # Step 1: Check Python version
    print_step(1, "Check Python Version")
    if not check_python_version():
        print("\nPlease upgrade Python and try again.")
        return
    
    # Step 2: Check if API is running
    print_step(2, "Check REST API")
    api_running = check_api_running()
    if not api_running:
        print("\n⚠️  Please start your REST API first:")
        print("   python api.py")
        
        response = input("\nStart API now and press Enter to continue, or 'q' to quit: ")
        if response.lower() == 'q':
            return
        
        # Check again
        if not check_api_running():
            print("❌ API still not running. Please start it manually.")
            return
    
    # Step 3: Install dependencies
    print_step(3, "Install Dependencies")
    if not install_dependencies():
        print("\nPlease install dependencies manually:")
        print("pip install -r requirements_email_fetcher.txt")
        return
    
    # Step 4: Create configuration
    print_step(4, "Create Configuration")
    if not os.path.exists("email_config.json"):
        create_sample_config()
    else:
        print("✅ Configuration file already exists")
    
    # Step 5: Gmail setup instructions
    print_step(5, "Gmail API Setup")
    if not os.path.exists("credentials.json"):
        show_gmail_setup_instructions()
        
        response = input("\nHave you downloaded credentials.json? (y/n): ").lower()
        if response != 'y':
            print("\n⚠️  Please complete Gmail API setup before continuing.")
            print("   Run this script again after downloading credentials.json")
            return
    else:
        print("✅ credentials.json found")
    
    # Step 6: Customize search
    print_step(6, "Customize Search Query")
    customize_search_query()
    
    # Step 7: Run tests
    print_step(7, "Run Tests")
    print("Running comprehensive test suite...")
    
    if run_test_suite():
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        response = input("\nContinue anyway? (y/n): ").lower()
        if response != 'y':
            return
    
    # Final step: Show next steps
    show_next_steps()
    
    print_header("Setup Complete!")
    print("🎉 Your email fetcher is ready to use!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("Please check the error and try again.") 