#!/usr/bin/env python3
"""
Single-page Streamlit app for parsing EML files and analyzing with Azure OpenAI
"""

import streamlit as st
import email
from email import policy
from email.parser import BytesParser
import re
from pathlib import Path
import os
import tempfile
import time
from typing import Dict, Any, Optional, List, Tuple
from io import BytesIO
from openai import AzureOpenAI
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="HC email parsing tool demo",
    page_icon="📧",
    layout="wide"
)

# ============== AZURE OPENAI HELPER ==============
def interact_with_azure_gpt(system_prompt: str, 
                            user_prompt: str,
                            deployment_name: str = "gpt4-o",
                            temperature: float = 0.3,
                            max_tokens: int = 1000,
                            timeout: int = 30) -> Dict[str, Any]:
    """
    Interact with GPT model using Azure OpenAI API.
    """
    start_time = datetime.now()
    
    try:
        # Check if this is a GPT-5 model first
        is_gpt5 = 'gpt-5' in deployment_name.lower()
        
        # Adjust timeout significantly for GPT-5 models (research shows they need 45s+ vs 20s for GPT-4)
        if is_gpt5:
            timeout = max(timeout * 6, 180)  # At least 3 minutes for GPT-5
            print(f"[DEBUG] GPT-5 detected: Using extended timeout = {timeout} seconds")
        else:
            print(f"[DEBUG] GPT-4 model: Using standard timeout = {timeout} seconds")
        
        # Initialize Azure OpenAI client
        # Use a newer API version for better GPT-5 support
        api_version = '2025-04-01-preview' if is_gpt5 else os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        
        client = AzureOpenAI(
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            api_version=api_version,
            azure_endpoint=os.getenv('AZURE_OPENAI_API_BASE'),
            timeout=timeout  # Apply the adjusted timeout to the client
        )
        
        # Adjust temperature for GPT-5 models
        if is_gpt5:
            temperature = 1.0
        
        # Build the API call parameters
        api_params = {
            "model": deployment_name,  # Azure uses deployment name as model parameter
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "timeout": timeout
        }
        
        # Use max_completion_tokens for GPT-5, max_tokens for others
        if is_gpt5:
            # GPT-5 needs much higher token limits - increase significantly
            max_tokens_adjusted = max(max_tokens * 3, 4000)  # At least 3x or 4000 tokens
            api_params["max_completion_tokens"] = max_tokens_adjusted
            print(f"[DEBUG] GPT-5 detected: Using max_completion_tokens = {max_tokens_adjusted} (original: {max_tokens})")
        else:
            api_params["max_tokens"] = max_tokens
            print(f"[DEBUG] GPT-4 model: Using max_tokens = {max_tokens}")
        
        # Debug: Print API parameters
        print(f"[DEBUG] API params: {api_params}")
        
        # Implement retry logic for GPT-5 models due to their higher latency
        max_retries = 3 if is_gpt5 else 1
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt  # Exponential backoff: 2s, 4s, 8s
                    print(f"[DEBUG] Retry attempt {attempt + 1}/{max_retries} after {wait_time}s wait...")
                    time.sleep(wait_time)
                
                # Call Azure OpenAI API with deployment name
                response = client.chat.completions.create(**api_params)
                break  # Success, exit retry loop
                
            except Exception as api_error:
                if "timeout" in str(api_error).lower() or "timed out" in str(api_error).lower():
                    print(f"[DEBUG] Timeout on attempt {attempt + 1}: {api_error}")
                    if attempt == max_retries - 1:  # Last attempt
                        raise api_error
                    continue  # Retry
                else:
                    # Non-timeout error, don't retry
                    raise api_error
        
        # Debug logging for troubleshooting
        print(f"[DEBUG] Model: {deployment_name}")
        print(f"[DEBUG] Response choices length: {len(response.choices)}")
        print(f"[DEBUG] Response choice 0 message: {response.choices[0].message}")
        print(f"[DEBUG] Raw content type: {type(response.choices[0].message.content)}")
        print(f"[DEBUG] Raw content repr: {repr(response.choices[0].message.content)}")
        
        # Check finish reason for potential issues
        finish_reason = response.choices[0].finish_reason
        print(f"[DEBUG] Finish reason: {finish_reason}")
        
        # Handle content filtering issues
        if finish_reason == 'content_filter':
            return {
                'success': False,
                'content': None,
                'model_used': deployment_name,
                'tokens_used': response.usage.total_tokens if response.usage else 0,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'error': f"Content was filtered by Azure's content filtering system for model {deployment_name}."
            }
        
        # Handle length limit issues (common with GPT-5)
        if finish_reason == 'length':
            return {
                'success': False,
                'content': None,
                'model_used': deployment_name,
                'tokens_used': response.usage.total_tokens if response.usage else 0,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'error': f"Model {deployment_name} hit token limit. Input too long or max_tokens too small. Current tokens used: {response.usage.total_tokens if response.usage else 'unknown'}. Try reducing input size or increasing max_tokens."
            }
        
        # Extract response content with proper null checking
        raw_content = response.choices[0].message.content
        
        # Handle cases where content might be None or empty
        if raw_content is None:
            return {
                'success': False,
                'content': None,
                'model_used': deployment_name,
                'tokens_used': response.usage.total_tokens if response.usage else 0,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'error': f"Model {deployment_name} returned null content. This might be due to content filtering or model configuration issues."
            }
        
        content = raw_content.strip()
        
        # Additional check for empty content after stripping
        if not content:
            return {
                'success': False,
                'content': '',
                'model_used': deployment_name,
                'tokens_used': response.usage.total_tokens if response.usage else 0,
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'error': f"Model {deployment_name} returned empty content after processing. Check input length, content filtering settings, or model parameters."
            }
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'success': True,
            'content': content,
            'model_used': deployment_name,
            'tokens_used': response.usage.total_tokens if response.usage else 0,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        return {
            'success': False,
            'content': None,
            'model_used': deployment_name,
            'tokens_used': 0,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

# ============== EML PARSING FUNCTIONS ==============
def extract_body_content(msg):
    """Extract body content from email message, prioritizing HTML table extraction."""
    import base64
    
    html_content = ""
    plain_content = ""
    
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            
            if "attachment" in content_disposition:
                continue
            
            if content_type == "text/plain":
                plain_content = part.get_payload(decode=True).decode(errors='ignore')
            elif content_type == "text/html":
                html_content = part.get_payload(decode=True).decode(errors='ignore')
    else:
        # Handle single-part messages (could be base64 encoded)
        content_type = msg.get_content_type()
        transfer_encoding = msg.get('Content-Transfer-Encoding', '').lower()
        
        if transfer_encoding == 'base64':
            # Base64 encoded content
            try:
                raw_payload = msg.get_payload()
                # Remove whitespace and newlines from base64 string
                clean_b64 = ''.join(raw_payload.split())
                decoded_content = base64.b64decode(clean_b64).decode('utf-8', errors='ignore')
                
                if content_type == "text/html" or '<html' in decoded_content.lower() or '<table' in decoded_content.lower():
                    html_content = decoded_content
                else:
                    plain_content = decoded_content
            except Exception as e:
                print(f"Error decoding base64 content: {e}")
                # Fallback to regular decoding
                content = msg.get_payload(decode=True).decode(errors='ignore')
                if '<html' in content.lower():
                    html_content = content
                else:
                    plain_content = content
        else:
            # Regular content
            content = msg.get_payload(decode=True).decode(errors='ignore')
            if '<html' in content.lower():
                html_content = content
            else:
                plain_content = content
    
    # If we have HTML content, extract tables more intelligently
    if html_content:
        return extract_tables_from_html(html_content)
    else:
        return plain_content

def extract_tables_from_html(html_content):
    """Extract table data from HTML content and convert to structured text."""
    # Find table sections and their headers
    extracted_text = []
    
    # Look for section headers that appear before tables
    section_patterns = [
        r'Movers\s*&\s*Shakers',
        r'Top\s*100\s*Print',
        r'Kindle',
        r'Audible'
    ]
    
    # Split content by table tags to process each table separately
    table_matches = re.finditer(r'<table[^>]*>(.*?)</table>', html_content, re.DOTALL | re.IGNORECASE)
    
    for i, table_match in enumerate(table_matches):
        table_html = table_match.group(1)
        
        # Find section header before this table
        section_header = f"Table {i+1}"
        start_pos = max(0, table_match.start() - 1000)  # Look back 1000 chars
        preceding_text = html_content[start_pos:table_match.start()]
        
        # Look for section indicators
        if 'Amazon Rank (AM)' in preceding_text:
            section_header = "Amazon Rank (AM)"
        elif 'Amazon Rank (PM)' in preceding_text:
            section_header = "Amazon Rank (PM)"
        elif 'Top 100 Print' in preceding_text:
            section_header = "Top 100 Print"
        elif 'Audible' in preceding_text:
            section_header = "Audible"
        elif 'Kindle' in preceding_text:
            section_header = "Kindle"
        
        extracted_text.append(f"\n{section_header}\n")
        
        # Extract table headers
        header_matches = re.findall(r'<th[^>]*>(.*?)</th>', table_html, re.DOTALL | re.IGNORECASE)
        if header_matches:
            headers = []
            for header in header_matches:
                # Remove HTML tags and clean up
                clean_header = re.sub(r'<[^>]+>', '', header).strip()
                clean_header = re.sub(r'\s+', ' ', clean_header)
                if clean_header:
                    headers.append(clean_header)
            
            if headers:
                extracted_text.append('\t'.join(headers))
        
        # Extract table rows
        row_matches = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
        for row_html in row_matches:
            # Skip header rows (they contain <th> tags)
            if '<th' in row_html.lower():
                continue
                
            # Extract cell data
            cell_matches = re.findall(r'<td[^>]*>(.*?)</td>', row_html, re.DOTALL | re.IGNORECASE)
            if cell_matches:
                cells = []
                for cell in cell_matches:
                    # Remove HTML tags and clean up
                    clean_cell = re.sub(r'<[^>]+>', '', cell).strip()
                    clean_cell = re.sub(r'\s+', ' ', clean_cell)
                    # Handle empty cells
                    if not clean_cell:
                        clean_cell = ""
                    cells.append(clean_cell)
                
                if cells and any(cell.strip() for cell in cells):  # Skip empty rows
                    extracted_text.append('\t'.join(cells))
    
    return '\n'.join(extracted_text)

def detect_multiple_tables(lines):
    """Detect multiple table structures and convert to markdown with section headers."""
    tables = []
    current_table_data = []
    current_headers = None
    current_section = None
    
    # Known section indicators for your reports
    section_indicators = [
        'Amazon Rank (AM)', 'Amazon Rank (PM)', 'Top 100 Print', 
        'Kindle', 'Audible', 'Movers & Shakers'
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this line indicates a new section
        is_section_header = any(indicator in line for indicator in section_indicators)
        if is_section_header:
            # Save previous table if it exists
            if current_headers and current_table_data:
                table_markdown = build_markdown_table(current_headers, current_table_data)
                if table_markdown:
                    tables.append({
                        'section': current_section or 'Unknown Section',
                        'markdown': table_markdown,
                        'headers': current_headers,
                        'data': current_table_data
                    })
            
            # Start new section
            current_section = line
            current_headers = None
            current_table_data = []
            continue
        
        # Detect potential table rows
        if '\t' in line or '|' in line or re.search(r'\s{2,}', line):
            # Split by various delimiters
            if '\t' in line:
                cells = [cell.strip() for cell in line.split('\t')]
            elif '|' in line:
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            else:
                cells = [cell.strip() for cell in re.split(r'\s{2,}', line)]
            
            # Filter out empty cells
            cells = [cell for cell in cells if cell]
            
            # Skip lines that don't look like table data
            if len(cells) < 2:
                continue
            
            # Detect headers (lines with common table column names)
            header_keywords = [
                'rank', 'title', 'author', 'imprint', 'asin', 'isbn', 
                'sales', 'stock', 'price', 'am', 'pm', 'promotion'
            ]
            
            if not current_headers and len(cells) > 3:
                # Check if this looks like a header row
                is_header = any(any(keyword in cell.lower() for keyword in header_keywords) 
                              for cell in cells)
                if is_header:
                    current_headers = cells
                    continue
            
            # Add data row if we have headers
            if current_headers and len(cells) >= 2:
                current_table_data.append(cells)
    
    # Don't forget the last table
    if current_headers and current_table_data:
        table_markdown = build_markdown_table(current_headers, current_table_data)
        if table_markdown:
            tables.append({
                'section': current_section or 'Final Section',
                'markdown': table_markdown,
                'headers': current_headers,
                'data': current_table_data
            })
    
    return tables

def build_markdown_table(headers, table_data):
    """Build a markdown table from headers and data."""
    if not headers or not table_data:
        return None
    
    max_cols = len(headers)
    markdown_lines = []
    
    # Header
    markdown_lines.append('| ' + ' | '.join(headers) + ' |')
    # Separator
    markdown_lines.append('|' + '|'.join([' --- ' for _ in range(max_cols)]) + '|')
    
    # Data rows
    for row in table_data:
        # Pad row if needed
        while len(row) < max_cols:
            row.append('')
        # Truncate if too many columns
        row = row[:max_cols]
        markdown_lines.append('| ' + ' | '.join(row) + ' |')
    
    return '\n'.join(markdown_lines)

def parse_eml_content(eml_content: bytes) -> Tuple[str, dict, List]:
    """Parse EML content and return markdown, metadata, and tables data."""
    # Parse the .eml content
    msg = BytesParser(policy=policy.default).parse(BytesIO(eml_content))
    
    # Extract metadata
    metadata = {
        'from': msg.get('From', 'N/A'),
        'to': msg.get('To', 'N/A'),
        'subject': msg.get('Subject', 'N/A'),
        'date': msg.get('Date', 'N/A')
    }
    
    # Build markdown output
    markdown_output = []
    
    # Email headers
    markdown_output.append("## Email Headers\n")
    markdown_output.append(f"**From:** {metadata['from']}  ")
    markdown_output.append(f"**To:** {metadata['to']}  ")
    markdown_output.append(f"**Subject:** {metadata['subject']}  ")
    markdown_output.append(f"**Date:** {metadata['date']}  \n")
    
    # Extract and process body
    markdown_output.append("## Report Content\n")
    
    body_content = extract_body_content(msg)
    tables = []
    
    if body_content:
        lines = body_content.split('\n')
        
        # Look for multiple tables
        tables = detect_multiple_tables(lines)
        
        if tables:
            for i, table_info in enumerate(tables):
                markdown_output.append(f"### {table_info['section']}\n")
                markdown_output.append(table_info['markdown'])
                markdown_output.append("\n")
        else:
            # Add raw content if no tables detected
            markdown_output.append("### Raw Content\n")
            for line in lines:
                if line.strip():
                    markdown_output.append(line.strip())
    
    markdown_content = '\n'.join(markdown_output)
    
    return markdown_content, metadata, tables

# ============== STREAMLIT APP ==============
def main():
    st.title("📧 EML Report Analyzer")
    
    # Initialize session state
    if 'markdown_content' not in st.session_state:
        st.session_state.markdown_content = None
    if 'ai_response' not in st.session_state:
        st.session_state.ai_response = None
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    
    # Create two columns for layout
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        # Upload section
        st.markdown("### Upload")
        uploaded_file = st.file_uploader("Choose an EML file", type=['eml'], label_visibility="collapsed")
        
        # Configuration section
        st.markdown("### Configure Model")
        
        # Check for Azure OpenAI credentials
        has_azure_creds = all([
            os.getenv('AZURE_OPENAI_API_KEY'),
            os.getenv('AZURE_OPENAI_API_BASE')
        ])
        
        if not has_azure_creds:
            st.warning("⚠️ Azure OpenAI not configured. Set environment variables.")
        
        # Model selection - using your Azure deployment names
        model_options = {
            "GPT-4o": "gpt4-o",
            "GPT-4o Mini": "gpt-4o-mini", 
            "GPT-4 Turbo": "gpt4-turbo",
            "GPT-5": "gpt-5-use2",
            "GPT-5 Chat": "gpt-5-chat-use2",
            "GPT-5 Mini": "gpt-5-mini-use2"
        }
        
        model_display = st.selectbox(
            "Model",
            list(model_options.keys()),
            label_visibility="collapsed",
            help="GPT-4o: Balanced performance | GPT-5: Latest capabilities but slower (1-3 min response time) | Mini versions: Faster & cheaper"
        )
        model_choice = model_options[model_display]
        
        # Show temperature info for GPT-5 models
        if 'GPT-5' in model_display:
            st.info("ℹ️ GPT-5 models use temperature=1.0, max_completion_tokens parameter, and take 1-3 minutes to respond")
        
        # Advanced settings - initialize with default
        max_tokens_input = 1000
        with st.expander("Advanced Settings"):
            # Set higher default for GPT-5 models
            if 'GPT-5' in model_display:
                default_max_tokens = 4000
                help_text = "GPT-5 models typically need higher token limits. Recommended: 4000+"
            else:
                default_max_tokens = 1000
                help_text = "Maximum number of tokens to generate in the response"
                
            max_tokens_input = st.slider(
                "Max Tokens",
                min_value=100,
                max_value=8000,
                value=default_max_tokens,
                help=help_text
            )
        
        # System prompt
        default_system_prompt = """You are a data quality specialist for analyzing email reports containing MULTIPLE SEPARATE TABLES.

This report contains several distinct table sections (e.g., "Amazon Rank (AM)", "Amazon Rank (PM)", "Top 100 Print", "Kindle", "Audible"). 
Analyze EACH TABLE SECTION INDIVIDUALLY and identify:

1. **Per-Table Analysis**: For each table section, identify missing values in critical columns
2. **Data Quality Issues**: Inconsistencies, formatting problems, or incomplete data within each table
3. **Missing Table Sections**: Expected sections that are completely absent or empty
4. **Cross-Table Comparison**: Compare data between different table sections to identify gaps
5. **Section-Specific Issues**: Table sections with headers but no data rows
6. **Critical Data Gaps**: Missing information in key columns like Sales, Stock Status, Title, ISBN

Provide analysis organized BY TABLE SECTION with specific actionable insights for data completeness and quality."""
        
        system_prompt = st.text_area(
            "System Prompt",
            value=default_system_prompt,
            height=225,
            label_visibility="visible"
        )
        
        # User prompt
        default_user_prompt = """Analyze this book sales report with MULTIPLE TABLE SECTIONS for missing values and data quality issues.

The report should contain these sections:
- Amazon Rank (AM) table
- Amazon Rank (PM) table  
- Top 100 Print table
- Kindle table
- Audible table

For EACH table section, focus on critical columns: Past 24 Hour Sales, Amz in stock?, Title, and ISBN.
Identify which sections are missing entirely and which have incomplete data."""
        
        user_prompt = st.text_area(
            "User Prompt",
            value=default_user_prompt,
            height=300,
            label_visibility="visible"
        )
        
        # Process button
        if uploaded_file and has_azure_creds:
            if st.button("🔍 Analyze Report", type="primary", use_container_width=True):
                try:
                    # Read and parse EML
                    eml_content = uploaded_file.read()
                    with st.spinner("Parsing EML file..."):
                        markdown_content, metadata, tables = parse_eml_content(eml_content)
                        st.session_state.markdown_content = markdown_content
                        st.session_state.file_uploaded = True
                        
                        # Save parsed markdown content to a text file for accuracy checking
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_filename = f"parsed_eml_{timestamp}.txt"
                        try:
                            with open(output_filename, 'w', encoding='utf-8') as f:
                                f.write("="*50 + "\n")
                                f.write("PARSED EML CONTENT - MARKDOWN FORMAT\n")
                                f.write("="*50 + "\n\n")
                                f.write(markdown_content)
                                f.write("\n\n" + "="*50 + "\n")
                                f.write("END OF PARSED CONTENT\n")
                                f.write("="*50 + "\n")
                            st.success(f"✅ Parsed content saved to: {output_filename}")
                        except Exception as save_error:
                            st.warning(f"⚠️ Could not save file: {save_error}")
                    
                    # Run AI analysis
                    with st.spinner("Running AI analysis..."):
                        full_user_prompt = f"{user_prompt}\n\nREPORT CONTENT:\n{markdown_content}"
                        
                        # Set temperature based on model type
                        temp = 1.0 if 'gpt-5' in model_choice.lower() else 0.3
                        
                        # Show different spinner message for GPT-5 models
                        if 'gpt-5' in model_choice.lower():
                            st.info("⏱️ GPT-5 models take longer to process (1-3 minutes). Please wait...")
                        
                        result = interact_with_azure_gpt(
                            system_prompt=system_prompt,
                            user_prompt=full_user_prompt,
                            deployment_name=model_choice,
                            temperature=temp,
                            max_tokens=max_tokens_input
                        )
                        print(result)
                        
                        if result['success']:
                            st.session_state.ai_response = result['content']
                        else:
                            # Provide specific guidance for timeout errors
                            error_msg = result.get('error', 'Unknown error')
                            if 'timeout' in error_msg.lower() or 'timed out' in error_msg.lower():
                                st.session_state.ai_response = f"❌ Timeout Error: {error_msg}\n\n💡 **Suggestions:**\n- GPT-5 models need more time - this is normal\n- Try reducing your input text length\n- Increase max tokens in Advanced Settings\n- Consider using GPT-4o for faster responses"
                            else:
                                st.session_state.ai_response = f"❌ Error: {error_msg}"
                    
                    st.success(f"✅ Analysis complete using {model_display}!")
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    with right_col:
        # Parsed EML → Markdown section
        st.markdown("### Parsed EML → Markdown")
        
        if st.session_state.markdown_content:
            # Display in a scrollable container
            with st.container():
                st.markdown(
                    f"""<div style="height: 300px; overflow-y: scroll; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
                    <pre style="white-space: pre-wrap; font-family: monospace; font-size: 12px;">{st.session_state.markdown_content}</pre>
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Download button
                st.download_button(
                    label="📥 Download Markdown",
                    data=st.session_state.markdown_content,
                    file_name="parsed_report.md",
                    mime="text/markdown"
                )
        else:
            st.info("Upload an EML file to see parsed content")
        
        # AI Response section
        st.markdown("### AI Response")
        
        if st.session_state.ai_response:
            # Display AI response
            with st.container():
                st.markdown(
                    f"""<div style="height: 300px; overflow-y: scroll; border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: #f8f9fa;">
                    {st.session_state.ai_response}
                    </div>""",
                    unsafe_allow_html=True
                )
                
                # Download button
                st.download_button(
                    label="📥 Download Analysis",
                    data=st.session_state.ai_response,
                    file_name="ai_analysis.md",
                    mime="text/markdown"
                )
        else:
            st.info("AI analysis will appear here after processing")

if __name__ == "__main__":
    main()
