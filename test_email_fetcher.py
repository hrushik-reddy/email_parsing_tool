#!/usr/bin/env python3
"""
Test script for Email Fetcher Service
Tests individual components before running the full automation
"""

import os
import sys
import json
import requests
from datetime import datetime
from email_fetcher import EmailFetcher

def test_api_connection():
    """Test connection to the REST API"""
    print("🔌 Testing API connection...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is running - Status: {data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is it running on localhost:8000?")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_config_file():
    """Test configuration file"""
    print("\n📋 Testing configuration...")
    
    config_file = "email_config.json"
    
    if not os.path.exists(config_file):
        print(f"⚠️  Config file not found. Creating default: {config_file}")
        fetcher = EmailFetcher()  # This will create default config
        return True
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check required sections
        required_sections = ['gmail', 'schedule', 'api_base_url']
        missing_sections = [s for s in required_sections if s not in config]
        
        if missing_sections:
            print(f"❌ Missing config sections: {missing_sections}")
            return False
        
        print("✅ Configuration file is valid")
        
        # Check Gmail credentials
        gmail_config = config['gmail']
        creds_file = gmail_config.get('credentials_file', 'credentials.json')
        
        if os.path.exists(creds_file):
            print(f"✅ Gmail credentials file found: {creds_file}")
        else:
            print(f"⚠️  Gmail credentials file missing: {creds_file}")
            print("   Download from Google Cloud Console")
        
        return True
        
    except json.JSONDecodeError:
        print("❌ Configuration file is not valid JSON")
        return False
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_gmail_auth():
    """Test Gmail authentication"""
    print("\n🔐 Testing Gmail authentication...")
    
    try:
        fetcher = EmailFetcher()
        success = fetcher.setup_gmail_auth()
        
        if success:
            print("✅ Gmail authentication successful")
            return True
        else:
            print("❌ Gmail authentication failed")
            print("   Check credentials.json and run authentication flow")
            return False
            
    except Exception as e:
        print(f"❌ Gmail auth test failed: {e}")
        return False

def test_email_search():
    """Test email search functionality"""
    print("\n📧 Testing email search...")
    
    try:
        fetcher = EmailFetcher()
        
        if not fetcher.setup_gmail_auth():
            print("❌ Cannot test email search - authentication failed")
            return False
        
        emails = fetcher.fetch_gmail_emails(hours_back=168)  # Last 7 days
        
        if emails:
            print(f"✅ Found {len(emails)} emails in the last 7 days")
            for i, email in enumerate(emails[:3], 1):  # Show first 3
                print(f"   {i}. {email['subject'][:50]}... from {email['from'][:30]}")
            if len(emails) > 3:
                print(f"   ... and {len(emails) - 3} more")
            return True
        else:
            print("⚠️  No emails found matching search criteria")
            print("   Check search_query in config or expand date range")
            return True  # Not necessarily an error
            
    except Exception as e:
        print(f"❌ Email search test failed: {e}")
        return False

def test_sample_eml_file():
    """Test processing a sample EML file"""
    print("\n📄 Testing EML file processing...")
    
    # Look for existing EML files in the directory
    eml_files = [f for f in os.listdir('.') if f.endswith('.eml')]
    
    if not eml_files:
        print("⚠️  No EML files found for testing")
        print("   You can test with an actual email after fetching")
        return True
    
    try:
        sample_file = eml_files[0]
        print(f"   Testing with: {sample_file}")
        
        fetcher = EmailFetcher()
        result = fetcher.send_to_api(sample_file)
        
        if result and result.get('success'):
            print("✅ EML file processed successfully")
            return True
        else:
            print("❌ EML file processing failed")
            print(f"   Result: {result}")
            return False
            
    except Exception as e:
        print(f"❌ EML processing test failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("\n📦 Testing dependencies...")
    
    required_packages = [
        'google.auth',
        'googleapiclient',
        'requests',
        'schedule'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("   Install with: pip install -r requirements_email_fetcher.txt")
        return False
    
    print("✅ All required dependencies installed")
    return True

def run_all_tests():
    """Run all tests"""
    print("🧪 Email Fetcher Test Suite")
    print("=" * 40)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Configuration", test_config_file),
        ("API Connection", test_api_connection),
        ("Gmail Authentication", test_gmail_auth),
        ("Email Search", test_email_search),
        ("EML Processing", test_sample_eml_file)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print("\n\n⏹️  Tests interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your email fetcher is ready to run.")
        print("\nNext steps:")
        print("1. Customize search_query in email_config.json")
        print("2. Run: python email_fetcher.py --run-now (for immediate test)")
        print("3. Run: python email_fetcher.py (for scheduled daily run)")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before proceeding.")
    
    return passed == len(results)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Email Fetcher Service')
    parser.add_argument('--api-only', action='store_true', help='Test only API connection')
    parser.add_argument('--gmail-only', action='store_true', help='Test only Gmail functionality')
    parser.add_argument('--config-only', action='store_true', help='Test only configuration')
    
    args = parser.parse_args()
    
    if args.api_only:
        test_api_connection()
    elif args.gmail_only:
        test_gmail_auth()
        test_email_search()
    elif args.config_only:
        test_config_file()
    else:
        run_all_tests()

if __name__ == "__main__":
    main() 