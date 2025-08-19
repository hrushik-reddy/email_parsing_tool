#!/usr/bin/env python3
"""
REST API for EML file parsing and AI analysis using FastAPI
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import tempfile
import os
from datetime import datetime
import base64

# Import functions from the existing app
from app import parse_eml_content, interact_with_azure_gpt

# Initialize FastAPI app
app = FastAPI(
    title="EML Email Parser API",
    description="API for parsing EML files and analyzing with Azure OpenAI",
    version="1.0.0"
)

# Response models
class ParseResponse(BaseModel):
    success: bool
    message: str
    tables_found: int
    tables: List[Dict[str, Any]]
    metadata: Dict[str, str]
    markdown_content: str

class AnalysisResponse(BaseModel):
    success: bool
    message: str
    analysis: Optional[str]
    model_used: str
    tokens_used: int
    execution_time: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

# Parse EML endpoint
@app.post("/parse", response_model=ParseResponse)
async def parse_eml_file(file: UploadFile = File(...)):
    """
    Parse an EML file and extract table data
    """
    try:
        # Validate file type
        if not file.filename.endswith('.eml'):
            raise HTTPException(status_code=400, detail="File must be an .eml file")
        
        # Read file content
        content = await file.read()
        
        # Parse the EML content
        markdown_content, metadata, tables = parse_eml_content(content)
        
        # Format tables for response
        formatted_tables = []
        for table in tables:
            formatted_tables.append({
                'section': table['section'],
                'headers': table['headers'],
                'row_count': len(table['data']),
                'data': table['data'][:5]  # Return first 5 rows to avoid large responses
            })
        
        return ParseResponse(
            success=True,
            message=f"Successfully parsed EML file with {len(tables)} tables",
            tables_found=len(tables),
            tables=formatted_tables,
            metadata=metadata,
            markdown_content=markdown_content[:1000]  # Truncate for response
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing EML file: {str(e)}")

# Analyze endpoint
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_eml_file(
    file: UploadFile = File(...),
    model: str = Form("gpt4-o"),
    system_prompt: str = Form(None),
    user_prompt: str = Form(None),
    max_tokens: int = Form(1000)
):
    """
    Parse and analyze an EML file with AI
    """
    try:
        # Validate file type
        if not file.filename.endswith('.eml'):
            raise HTTPException(status_code=400, detail="File must be an .eml file")
        
        # Read and parse file
        content = await file.read()
        markdown_content, metadata, tables = parse_eml_content(content)
        
        # Default prompts if not provided
        if not system_prompt:
            system_prompt = """You are a data quality specialist for analyzing email reports containing MULTIPLE SEPARATE TABLES.

This report contains several distinct table sections (e.g., "Amazon Rank (AM)", "Amazon Rank (PM)", "Top 100 Print", "Kindle", "Audible"). 
Analyze EACH TABLE SECTION INDIVIDUALLY and identify:

1. **Per-Table Analysis**: For each table section, identify missing values in critical columns
2. **Data Quality Issues**: Inconsistencies, formatting problems, or incomplete data within each table
3. **Missing Table Sections**: Expected sections that are completely absent or empty
4. **Cross-Table Comparison**: Compare data between different table sections to identify gaps
5. **Section-Specific Issues**: Table sections with headers but no data rows
6. **Critical Data Gaps**: Missing information in key columns like Sales, Stock Status, Title, ISBN

Provide analysis organized BY TABLE SECTION with specific actionable insights for data completeness and quality."""

        if not user_prompt:
            user_prompt = """Analyze this book sales report with MULTIPLE TABLE SECTIONS for missing values and data quality issues.

The report should contain these sections:
- Amazon Rank (AM) table
- Amazon Rank (PM) table  
- Top 100 Print table
- Kindle table
- Audible table

For EACH table section, focus on critical columns: Past 24 Hour Sales, Amz in stock?, Title, and ISBN.
Identify which sections are missing entirely and which have incomplete data."""
        
        # Combine prompts with content
        full_user_prompt = f"{user_prompt}\n\nREPORT CONTENT:\n{markdown_content}"
        
        # Set temperature based on model
        temperature = 1.0 if 'gpt-5' in model.lower() else 0.3
        
        # Call AI analysis
        result = interact_with_azure_gpt(
            system_prompt=system_prompt,
            user_prompt=full_user_prompt,
            deployment_name=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if result['success']:
            return AnalysisResponse(
                success=True,
                message="Analysis completed successfully",
                analysis=result['content'],
                model_used=result['model_used'],
                tokens_used=result['tokens_used'],
                execution_time=result['execution_time'],
                timestamp=result['timestamp']
            )
        else:
            return AnalysisResponse(
                success=False,
                message=f"Analysis failed: {result.get('error', 'Unknown error')}",
                analysis=None,
                model_used=result['model_used'],
                tokens_used=result['tokens_used'],
                execution_time=result['execution_time'],
                timestamp=result['timestamp']
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing EML file: {str(e)}")

# List available models endpoint
@app.get("/models")
async def list_models():
    """List available AI models"""
    models = {
        "gpt4-o": "GPT-4o - Balanced performance",
        "gpt-4o-mini": "GPT-4o Mini - Fast and efficient", 
        "gpt4-turbo": "GPT-4 Turbo - High performance",
        "gpt-5-use2": "GPT-5 - Latest capabilities (slower)",
        "gpt-5-chat-use2": "GPT-5 Chat - Conversational AI",
        "gpt-5-mini-use2": "GPT-5 Mini - Fast GPT-5 variant"
    }
    return {"available_models": models}

# Test endpoint with sample data
@app.post("/test")
async def test_api():
    """Test endpoint to verify API is working"""
    return {
        "message": "EML Parser API is working!",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "parse": "POST /parse - Parse EML file",
            "analyze": "POST /analyze - Parse and analyze EML file",
            "models": "GET /models - List available models",
            "health": "GET /health - Health check"
        }
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 