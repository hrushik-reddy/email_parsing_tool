#!/usr/bin/env python3
"""
Test script for the EML Parser REST API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_models():
    """Test the models endpoint"""
    print("Testing /models endpoint...")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    print(f"Available models: {list(response.json()['available_models'].keys())}")
    print()

def test_parse(eml_file_path):
    """Test the parse endpoint"""
    print(f"Testing /parse endpoint with {eml_file_path}...")
    
    with open(eml_file_path, 'rb') as f:
        files = {'file': (eml_file_path, f, 'application/octet-stream')}
        response = requests.post(f"{BASE_URL}/parse", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Success: {data['success']}")
        print(f"Tables found: {data['tables_found']}")
        for i, table in enumerate(data['tables']):
            print(f"  {i+1}. {table['section']} - {table['row_count']} rows")
    else:
        print(f"Error: {response.text}")
    print()

def test_analyze(eml_file_path, model="gpt4-o"):
    """Test the analyze endpoint"""
    print(f"Testing /analyze endpoint with {eml_file_path} using {model}...")
    
    with open(eml_file_path, 'rb') as f:
        files = {'file': (eml_file_path, f, 'application/octet-stream')}
        data = {'model': model, 'max_tokens': 1000}
        response = requests.post(f"{BASE_URL}/analyze", files=files, data=data)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Model used: {result['model_used']}")
        print(f"Tokens used: {result['tokens_used']}")
        print(f"Execution time: {result['execution_time']:.2f}s")
        if result['analysis']:
            print(f"Analysis preview: {result['analysis'][:200]}...")
    else:
        print(f"Error: {response.text}")
    print()

def main():
    """Main test function"""
    print("🚀 EML Parser API Test Suite\n")
    
    # Test basic endpoints
    test_health()
    test_models()
    
    # Test with sample EML files
    eml_files = [
        "Fw_ Amazon Movers & Shakers, Top 100 (1).eml",
        "Amazon Movers & Shakers, Top 100 _missing (1).eml",
        "Amazon Movers & Shakers, Top 100_missing24hoursales.eml"
    ]
    
    for eml_file in eml_files:
        try:
            # Test parsing
            test_parse(eml_file)
            
            # Test analysis (only for first file to avoid hitting API limits)
            if eml_file == eml_files[0]:
                print("⚠️  Skipping AI analysis test - uncomment to test with Azure OpenAI")
                # test_analyze(eml_file)
                
        except FileNotFoundError:
            print(f"File {eml_file} not found, skipping...")
        except Exception as e:
            print(f"Error testing {eml_file}: {e}")

if __name__ == "__main__":
    main() 