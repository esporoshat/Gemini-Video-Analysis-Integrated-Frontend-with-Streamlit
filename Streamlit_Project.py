import streamlit as st
import json
import requests
from google.oauth2 import service_account
import google.auth.transport.requests
import pandas as pd
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import glob

SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", os.path.join(os.path.dirname(__file__), "service_account.json"))
PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
LOCATION = os.getenv("LOCATION", "us-central1")
MODEL_ID = os.getenv("MODEL_ID", "gemini-2.5-flash")

# Performance Configuration - BALANCED FOR SPEED & RELIABILITY
API_TIMEOUT = 120  
IMAGE_TIMEOUT = 60    
MAX_CONCURRENT_REQUESTS = 5  
RATE_LIMIT_DELAY = 0.8  
BATCH_SIZE = 8  
MAX_RETRIES = 2  

# Bulk Processing Configuration (for large batches)
BULK_MAX_CONCURRENT = 12  
BULK_BATCH_SIZE = 25  
BULK_RATE_DELAY = 0.2  
BULK_THRESHOLD = 20  #


# ===================== BRAND-SPECIFIC PROMPTS =====================

# Function to load prompts from text files
def load_prompt_from_file(filename):
    """Load prompt content from a text file in the prompts directory"""
    try:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", filename)
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        st.error(f"❌ Prompt file not found: {filename}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading prompt file {filename}: {e}")
        return None

# Default/General prompt for cosmetic media analysis

DEFAULT_PROMPT = load_prompt_from_file("general_prompt.txt") or """Analyze cosmetic/skincare media and extract product information. 
Target brands: CeraVe, L'Oréal Paris, Garnier, Dove, Mixa, No-Cosmetics.

Output JSON format:
{
  "gezeigte_produkte": [{"produkte": "Product Name", "kategorie_I": "Category"}],
  "eingeblendeter_text": "Text in video",
  "kategorie_II": "klas. Kampagne | co-creation | Event",
  "thematisierte_inhaltsstoffe": ["Ingredient1"]
}

Categories: Sun, Face Care, Face Cleansing, Body&Hand, X-Cat, Brand
Product names in German Title Case. Only include products from target brands.
"""




# Brand-specific prompt files mapping
BRAND_PROMPT_FILES = {
    "general": "general_prompt.txt",
    "cerave": "cerave_prompt.txt",
    "loreal": "loreal_prompt.txt",
    "garnier": "garnier_prompt.txt",
    "dove": "dove_prompt.txt",
    "mixa": "mixa_prompt.txt",
    "no-cosmetics": "no_cosmetics_prompt.txt"
}



def get_brand_specific_prompt(selected_brand=None):
    """Get the appropriate prompt from text files"""
    if selected_brand and selected_brand in BRAND_PROMPT_FILES:
        # Load the specific brand prompt from file
        prompt_file = BRAND_PROMPT_FILES[selected_brand]
        prompt = load_prompt_from_file(prompt_file)
        if prompt:
            return prompt
        else:
            st.warning(f"⚠️ Failed to load {selected_brand} prompt file. Using general prompt as fallback.")
            return DEFAULT_PROMPT
    # Return general prompt (either from file or fallback)
    return DEFAULT_PROMPT

def detect_primary_brand_from_url(media_url):
    """Try to detect brand from URL patterns (optional enhancement)"""
    url_lower = media_url.lower()
    
    # URL pattern detection 
    if 'cerave' in url_lower:
        return 'cerave'
    elif 'loreal' in url_lower or 'l-oreal' in url_lower:
        return 'loreal'
    elif 'garnier' in url_lower:
        return 'garnier'
    elif 'dove' in url_lower:
        return 'dove'
    elif 'mixa' in url_lower:
        return 'mixa'
    elif 'no-cosmetics' in url_lower or 'nocosmetics' in url_lower:
        return 'no-cosmetics'
    
    return None  

# Page config
st.set_page_config(page_title="Video & Image Analyzer", layout="wide")
st.title("Cosmetic Media Analyzer")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []
if 'csv_file_content' not in st.session_state:
    st.session_state.csv_file_content = None
if 'csv_url_column' not in st.session_state:
    st.session_state.csv_url_column = None
if 'csv_file_name' not in st.session_state:
    st.session_state.csv_file_name = None


# Function to get access token
def get_access_token():
    try:
        # Check if service account file exists
        if not os.path.exists(SERVICE_ACCOUNT_FILE):
            st.error(f"Service account file not found: {SERVICE_ACCOUNT_FILE}")
            st.info(" **Setup Instructions:**")
            st.markdown("""
            1. Download your Google Cloud service account JSON key file
            2. Place it in the project root directory as `service_account.json`
            3. Or set the `SERVICE_ACCOUNT_FILE` environment variable to point to your key file
            4. Make sure the service account has Vertex AI API access enabled
            """)
            return None
        
        # Create credentials
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, 
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        # Check if token needs refresh
        if not credentials.valid:
            credentials.refresh(google.auth.transport.requests.Request())
        
        # Verify we have a token
        if not credentials.token:
            st.error(" Failed to obtain access token")
            return None
            
        return credentials.token
        
    except Exception as e:
        st.error(f"Error getting access token: {e}")
        st.info(" **Troubleshooting:**")
        st.markdown("""
        - Verify your service account JSON file is valid
        - Check that the service account has Vertex AI permissions
        - Ensure PROJECT_ID is set correctly (currently: `{}`)
        - Make sure you have enabled the Vertex AI API in Google Cloud Console
        """.format(PROJECT_ID))
        return None

# Function to detect media type from URL
def detect_media_type(media_url):
    url_lower = media_url.lower()
    
    
    if any(ext in url_lower for ext in ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv']):
        if '.mp4' in url_lower:
            return "video/mp4"
        elif '.avi' in url_lower:
            return "video/avi"
        elif '.mov' in url_lower:
            return "video/quicktime"
        elif '.wmv' in url_lower:
            return "video/x-ms-wmv"
        elif '.flv' in url_lower:
            return "video/x-flv"
        elif '.webm' in url_lower:
            return "video/webm"
        elif '.mkv' in url_lower:
            return "video/x-matroska"
    
    # Image extensions
    elif any(ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']):
        if '.jpg' in url_lower or '.jpeg' in url_lower:
            return "image/jpeg"
        elif '.png' in url_lower:
            return "image/png"
        elif '.gif' in url_lower:
            return "image/gif"
        elif '.bmp' in url_lower:
            return "image/bmp"
        elif '.webp' in url_lower:
            return "image/webp"
    
    # Audio extensions
    elif any(ext in url_lower for ext in ['.mp3', '.wav', '.aac', '.ogg', '.flac']):
        if '.mp3' in url_lower:
            return "audio/mpeg"
        elif '.wav' in url_lower:
            return "audio/wav"
        elif '.aac' in url_lower:
            return "audio/aac"
        elif '.ogg' in url_lower:
            return "audio/ogg"
        elif '.flac' in url_lower:
            return "audio/flac"
    
    # For URLs without extensions, try to detect based on URL patterns

    if 's3.eu-central-1.amazonaws.com' in url_lower and 'story-elements' in url_lower:
        return "image/jpeg"  # Default to JPEG for S3 story elements
    
    
    return "image/jpeg"

# Function to get media type category for display
def get_media_category(mime_type):
    if mime_type.startswith('video/'):
        return "Video"
    elif mime_type.startswith('image/'):
        return "Image"
    elif mime_type.startswith('audio/'):
        return "Audio"
    else:
        return "Unknown"

# Function to extract JSON from Gemini response
def extract_json_from_response(response_text):
    """
    Extract JSON from Gemini response which might be wrapped in markdown or have extra text
    """
    if not response_text:
        return None
    

    
    # 1. Look for JSON wrapped in markdown code blocks
    markdown_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL | re.IGNORECASE)
    if markdown_match:
        try:
            return json.loads(markdown_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 1b. Look for JSON wrapped in markdown code blocks with more flexible matching
    markdown_match2 = re.search(r'```.*?(\{.*?\}).*?```', response_text, re.DOTALL | re.IGNORECASE)
    if markdown_match2:
        try:
            return json.loads(markdown_match2.group(1))
        except json.JSONDecodeError:
            pass
    
    # 2. Look for JSON wrapped in backticks (single line)
    backtick_match = re.search(r'`(\{.*?\})`', response_text, re.DOTALL)
    if backtick_match:
        try:
            return json.loads(backtick_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 3. Look for JSON at the beginning of the response
    start_match = re.search(r'^\s*(\{.*?\})\s*', response_text, re.DOTALL)
    if start_match:
        try:
            return json.loads(start_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 4. Look for JSON anywhere in the response (most permissive)
    anywhere_match = re.search(r'(\{.*?\})', response_text, re.DOTALL)
    if anywhere_match:
        try:
            return json.loads(anywhere_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 5. Try to find the largest JSON-like structure
    json_blocks = re.findall(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', response_text, re.DOTALL)
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            # Check if it has the expected structure
            if any(key in parsed for key in ['gezeigte_produkte', 'thema_des_videos', 'eingeblendeter_text']):
                return parsed
        except json.JSONDecodeError:
            continue
    
    return None

# Function to create payload for different media types
def create_media_payload(media_url, prompt, mime_type):
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "file_data": {
                            "file_uri": media_url,
                            "mime_type": mime_type
                        }
                    }
                ]
            }
        ]
    }

# Function to analyze media with Gemini

def analyze_media(media_url, prompt, mime_type):
    access_token = get_access_token()
    if not access_token:
        return "Error: Failed to get access token"
    
    endpoint = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"
    
    payload = create_media_payload(media_url, prompt, mime_type)
    
    # Choose timeout based on media type
    timeout = IMAGE_TIMEOUT if mime_type.startswith('image/') else API_TIMEOUT
    
    # Retry logic for failed requests
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.post(
                endpoint, 
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                }, 
                data=json.dumps(payload),
                timeout=timeout
            )
            
            if response.status_code == 200:
                response_data = response.json()
                if "candidates" in response_data and len(response_data["candidates"]) > 0:
                    result_text = response_data["candidates"][0]["content"]["parts"][0]["text"]
                    return result_text
                else:
                    return "No response text found"
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                
               
                if 400 <= response.status_code < 500:
                    if response.status_code == 400:
                       
                        error_text = response.text.lower()
                        if "cannot fetch content" in error_text or "robots.txt" in error_text or "not accessible" in error_text:
                            detailed_error = (
                                "Error: URL Access Failed (400)\n"
                                "Vertex AI cannot access the URL. Common reasons:\n"
                                "• URL is not publicly accessible (requires authentication/login)\n"
                                "• URL is blocked by robots.txt\n"
                                "• URL is a private S3 bucket or cloud storage (needs public access)\n"
                                "• URL requires special headers or authentication\n"
                                "• URL is behind a firewall or VPN\n"
                                "• URL format is incorrect\n\n"
                                "Solutions:\n"
                                "• Ensure the URL is publicly accessible (no login required)\n"
                                "• For S3/cloud storage: Make the file/bucket publicly readable\n"
                                "• Check if the URL works in a browser (incognito mode)\n"
                                "• Verify robots.txt allows access for Vertex AI\n"
                                f"• URL: {media_url[:100]}..."
                            )
                            return detailed_error
                        else:
                            return f"Error: Bad Request (400) - Check URL format and media type. {response.text[:200]}"
                    elif response.status_code == 401:
                        return "Error: Authentication failed (401) - Check service account credentials"
                    elif response.status_code == 403:
                        return "Error: Access forbidden (403) - Check API permissions and quotas"
                    elif response.status_code == 429:
                        return "Error: Rate limit exceeded (429) - Too many requests, try reducing concurrency"
                    else:
                        return error_msg
                        
                
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)  
                    continue
                return error_msg
                
        except requests.exceptions.Timeout as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)  
                continue
            media_type = "image" if mime_type.startswith('image/') else "video"
            return f"Error: Request timed out after {timeout} seconds. The {media_type} analysis with the detailed prompt is taking longer than expected. This can happen with complex prompts or large files."
        except requests.exceptions.ConnectionError as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)  
                continue
            return f"Error: Connection failed - {str(e)}. Please check your internet connection."
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)  
                continue
            return f"Error: {str(e)}"
    
    return "Error: Maximum retry attempts exceeded"

def flatten_json_response(parsed_result):
    """Flatten JSON response to extract all fields dynamically"""
    if parsed_result is None:
        return {}
    
    flattened = {}
    
    # Helper function to recursively flatten nested structures
    def flatten_dict(d, parent_key='', sep='_'):
        items = {}
        if d is None:
            return items
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if v is None:
                items[new_key] = "N/A"
            elif isinstance(v, dict):
                # Recursively flatten nested dicts
                nested_items = flatten_dict(v, new_key, sep=sep)
                items.update(nested_items)
            elif isinstance(v, list):
                # Handle lists - convert to string or process each item
                if len(v) > 0 and isinstance(v[0], dict):
                    # If list contains dicts, flatten each dict
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            nested_items = flatten_dict(item, f"{new_key}_{i}", sep=sep)
                            items.update(nested_items)
                        else:
                            items[f"{new_key}_{i}"] = str(item) if item is not None else "N/A"
                else:
                    # Simple list - join with semicolon
                    items[new_key] = "; ".join(str(x) for x in v if x is not None) if v else "N/A"
            else:
                items[new_key] = v
        return items
    
    # Flatten the entire response
    flattened = flatten_dict(parsed_result)
    
    # Convert all values to strings for CSV compatibility
    flattened = {k: str(v) if v is not None else "N/A" for k, v in flattened.items()}
    
    return flattened

def process_single_url(url, prompt, progress_callback=None):
    """Process a single URL with progress callback"""
    try:
        if not url.startswith(('http://', 'https://')):
            return {
                "url": url,
                "error": "Invalid URL format",
                "success": False
            }

        # Auto-detect media type
        detected_type = detect_media_type(url)
        media_category = get_media_category(detected_type)

        # Analyze the media
        result = analyze_media(url, prompt, detected_type)

        if result and not result.startswith("Error"):
       
            parsed_result = extract_json_from_response(result)

            if parsed_result:
                # Start with metadata fields that are always needed
                result_data = {
                    "Media URL": url,
                    "Media Type": media_category,
                    "Raw AI Response": result
                }
                
                # Extract all fields dynamically from the JSON response
                dynamic_fields = flatten_json_response(parsed_result)
                
                # Add all dynamic fields from the AI response
                # Convert keys to readable format (replace underscores with spaces, title case)
                for key, value in dynamic_fields.items():
                    # Use readable format as primary key
                    readable_key = key.replace('_', ' ').title()
                    # Only add if not already present (avoid duplicates)
                    if readable_key not in result_data:
                        result_data[readable_key] = value
                
                # Only add standard fields if they exist in the parsed response 
                # This ensures we don't add empty columns when using custom prompts
                if "gezeigte_produkte" in parsed_result and len(parsed_result["gezeigte_produkte"]) > 0:
                    produkt_data = parsed_result["gezeigte_produkte"][0]
                    produkt_name = produkt_data.get("produkte", "N/A")
                    product_category = produkt_data.get("kategorie_I", "N/A")
                    
                    # Only add if not already in result_data from dynamic_fields
                    if "Produkt" not in result_data and produkt_name != "N/A":
                        result_data["Produkt"] = produkt_name
                    if "Kategorie I" not in result_data and product_category != "N/A":
                        result_data["Kategorie I"] = product_category
                
                # Add Kategorie II if it exists
                kategorie_ii_raw = parsed_result.get("kategorie_II")
                if kategorie_ii_raw and "Kategorie Ii" not in result_data and "Kategorie II" not in result_data:
                    kategorie_ii_mapping = {
                        "klas.Kampagne": "Klas. Kampagne",
                        "co-creation": "Co-Creation",
                        "Event": "Event"
                    }
                    kategorie_ii = kategorie_ii_mapping.get(kategorie_ii_raw, kategorie_ii_raw)
                    result_data["Kategorie II"] = kategorie_ii
                
                # Add Eingeblendeter Text if it exists
                eingeblendeter_text = parsed_result.get("eingeblendeter_text")
                if eingeblendeter_text and "Eingeblendeter Text" not in result_data:
                    result_data["Eingeblendeter Text"] = eingeblendeter_text
                
                # Add Inhaltsstoffe if it exists
                thematisierte_inhaltsstoffe = parsed_result.get("thematisierte_inhaltsstoffe")
                if thematisierte_inhaltsstoffe and "Inhaltsstoffe" not in result_data and "Thematisierte Inhaltsstoffe" not in result_data:
                    result_data["Inhaltsstoffe"] = "; ".join(thematisierte_inhaltsstoffe) if thematisierte_inhaltsstoffe else "N/A"
                
                # Add Influencer Kategorie I only if we have the data to determine it
                eingeblendeter_text_lower = (eingeblendeter_text or "").lower()
                kategorie_ii_raw_lower = (kategorie_ii_raw or "").lower()
                kategorie_ii_raw_str = str(kategorie_ii_raw) if kategorie_ii_raw else ""
                if "Influencer Kategorie I" not in result_data:
                    if kategorie_ii_raw and "Event" in kategorie_ii_raw_str:
                        influencer_kategorie = "Lifestyle"
                    elif kategorie_ii_raw and "co-creation" in kategorie_ii_raw_lower:
                        influencer_kategorie = "Beauty Influencer"
                    elif eingeblendeter_text and any(keyword in eingeblendeter_text_lower for keyword in ["skincare", "hautpflege", "serum", "creme"]):
                        influencer_kategorie = "Skinfluencer"
                    elif eingeblendeter_text and any(keyword in eingeblendeter_text_lower for keyword in ["medizin", "dermatolog", "arzt", "medizinisch"]):
                        influencer_kategorie = "Medfluencer"
                    else:
                        influencer_kategorie = None  # Don't add if we can't determine
                    
                    if influencer_kategorie:
                        result_data["Influencer Kategorie I"] = influencer_kategorie

                return {
                    "url": url,
                    "success": True,
                    "data": result_data
                }
            else:
                return {
                    "url": url,
                    "success": False,
                    "error": "JSON parsing failed",
                    "data": {
                        "Media URL": url,
                        "Media Type": media_category,
                        "Raw AI Response": result,
                        "Error": "JSON parsing failed"
                    }
                }
        else:
            return {
                "url": url,
                "success": False,
                "error": result,
                "data": {
                    "Media URL": url,
                    "Media Type": media_category if 'media_category' in locals() else "N/A",
                    "Raw AI Response": result,
                    "Error": result if result.startswith("Error") else f"Error: {result}"
                }
            }
    except Exception as e:
        return {
            "url": url,
            "success": False,
            "error": str(e),
            "data": {
                "Media URL": url,
                "Media Type": "N/A",
                "Raw AI Response": f"Error: {str(e)}",
                "Error": str(e)
            }
        }

def process_multiple_urls(urls_list, prompt):
    """Process multiple URLs with parallel processing and progress tracking"""
    results = []
    total_urls = len(urls_list)
    
    # Determine if we should use bulk processing settings
    is_bulk_processing = total_urls >= BULK_THRESHOLD
    
    if is_bulk_processing:
        max_workers = BULK_MAX_CONCURRENT
        batch_size = BULK_BATCH_SIZE
        rate_delay = BULK_RATE_DELAY
        st.info(f" **Bulk Processing Mode Activated**: {total_urls} URLs detected. Using optimized settings for large batches.")
    else:
        max_workers = MAX_CONCURRENT_REQUESTS
        batch_size = BATCH_SIZE
        rate_delay = RATE_LIMIT_DELAY
    
    # Initialize progress tracking
    if 'progress' not in st.session_state:
        st.session_state.progress = {'completed': 0, 'total': total_urls, 'results': []}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    time_status = st.empty()
    
    # Time estimation
    start_time = time.time()
    
    # Process URLs in batches for better performance
    for batch_start in range(0, total_urls, batch_size):
        batch_end = min(batch_start + batch_size, total_urls)
        batch_urls = urls_list[batch_start:batch_end]
        
        # Calculate time estimates
        if st.session_state.progress['completed'] > 0:
            elapsed_time = time.time() - start_time
            avg_time_per_url = elapsed_time / st.session_state.progress['completed']
            remaining_urls = total_urls - st.session_state.progress['completed']
            estimated_remaining = remaining_urls * avg_time_per_url
            time_status.text(f" Estimated time remaining: {estimated_remaining/60:.1f} minutes")
        
        status_text.text(f"Processing batch {batch_start//batch_size + 1}: URLs {batch_start + 1}-{batch_end}")
        
        # Process batch with parallel execution while preserving order
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all URLs in the batch with their original indices
            future_to_info = {
                executor.submit(process_single_url, url, prompt): {
                    'url': url, 
                    'original_index': batch_start + i
                }
                for i, url in enumerate(batch_urls)
            }
            
            # Collect results as they complete, but store with original index
            batch_results = {}
            for future in as_completed(future_to_info):
                info = future_to_info[future]
                url = info['url']
                original_index = info['original_index']
                
                try:
                    result = future.result()
                    # Store result with its original index and add URL order info
                    result_data = result['data'].copy()
                    result_data['URL Order'] = original_index + 1  # 1-based indexing for users
                    batch_results[original_index] = result_data
                    
                    # Update progress
                    st.session_state.progress['completed'] += 1
                    progress_value = min(st.session_state.progress['completed'] / total_urls, 1.0)
                    progress_bar.progress(progress_value)
                    
                    # Add rate limiting delay
                    time.sleep(rate_delay)
                    
                except Exception as e:
               
                    
                    batch_results[original_index] = {
                        "URL Order": original_index + 1,  
                        "Media URL": url,
                        "Media Type": "N/A",
                        "Raw AI Response": f"Error: {str(e)}",
                        "Error": str(e)
                    }
                    
                    st.session_state.progress['completed'] += 1
                    progress_value = min(st.session_state.progress['completed'] / total_urls, 1.0)
                    progress_bar.progress(progress_value)
            
      
            for i in range(batch_start, batch_end):
                if i in batch_results:
                    results.append(batch_results[i])
    
    status_text.text("Processing complete!")
    return results

def process_csv_file(uploaded_file, prompt, url_column_name):
    """Process uploaded CSV file with single column containing URLs and groups"""
    try:
        # Ensure file pointer is at the start
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)
        
        # Verify file has content
        if hasattr(uploaded_file, 'read'):
            # Check file size
            current_pos = uploaded_file.tell() if hasattr(uploaded_file, 'tell') else 0
            uploaded_file.seek(0, 2)  # Seek to end
            file_size = uploaded_file.tell() if hasattr(uploaded_file, 'tell') else 0
            uploaded_file.seek(0)  # Reset to start
            
            if file_size == 0:
                st.error("❌ The CSV file appears to be empty.")
                return None
        
        # Read CSV file with row 1 as header (standard CSV format)
        df = pd.read_csv(uploaded_file, header=0)
        st.info(f"📁 Loaded CSV: {len(df)} rows and {len(df.columns)} columns")
        st.info(f"📋 **Column**: `{url_column_name}`")
        
        # Use the selected column name
        target_column = url_column_name
        
        if target_column not in df.columns:
            st.error(f"❌ Column '{target_column}' not found in the CSV file.")
            st.info(f"📋 **Available columns**: {', '.join(df.columns.tolist())}")
            st.error(f"Please ensure your CSV file has a column named exactly '{target_column}' containing the URLs.")
            return None
        
        st.success(f"✅ **Target column found**: {target_column}")
        url_column = target_column
        
        # Function to extract multiple URLs from a cell
        def extract_urls_from_cell(cell_content):
            """Extract all URLs from a cell that may contain multiple URLs separated by commas, semicolons, newlines, or spaces"""
            if pd.isna(cell_content) or str(cell_content).strip() == '' or str(cell_content) == 'nan':
                return []
            
            cell_str = str(cell_content).strip()
            urls = []
            
            # Use regex to find all URLs in the cell 
            url_pattern = r'https?://[^\s,;]+'
            found_urls = re.findall(url_pattern, cell_str)
            
            # Clean up URLs (remove trailing punctuation that might have been captured)
            for url in found_urls:
                # Remove trailing punctuation that might be part of the separator
                url = url.rstrip('.,;')
                if url.startswith(('http://', 'https://')):
                    urls.append(url)
            
            # If regex didn't find URLs but cell starts with http/https, treat as single URL
            if not urls and cell_str.startswith(('http://', 'https://')):
                urls.append(cell_str)
            
            return urls
        
        # Extract URLs from the single column - process ALL rows in order
        # Store row information for every row to maintain exact CSV order
        row_data_list = []  # List of (row_index, urls_list, original_cell_content)
        all_urls = []  # Only valid URLs for processing
        url_to_row_map = {}  # Map URL to its CSV row index
        
        for idx, row in df.iterrows():
            cell_content = row[url_column]
            
            # Extract all URLs from this cell
            urls_in_cell = extract_urls_from_cell(cell_content)
            
            if not urls_in_cell:
                # No valid URLs found
                if pd.isna(cell_content) or str(cell_content).strip() == '' or str(cell_content) == 'nan':
                    row_data_list.append((idx, [], None))
                else:
                    row_data_list.append((idx, [], str(cell_content).strip()))
            else:
                # Found URLs in this cell
                row_data_list.append((idx, urls_in_cell, str(cell_content).strip()))
                for url in urls_in_cell:
                    all_urls.append(url)
                    if url not in url_to_row_map:
                        url_to_row_map[url] = idx
        
        # Show warning if no valid URLs, but continue processing all rows
        if not all_urls:
            st.warning("⚠️ No valid URLs found in the CSV file. All rows will be returned with N/A values.")
            
            # Show sample of what was actually found
            sample_values = []
            for idx, row in df.iterrows():
                cell_content = row[url_column]
                if not pd.isna(cell_content) and str(cell_content).strip() != '' and str(cell_content) != 'nan':
                    sample_values.append(str(cell_content).strip()[:100])  # First 100 chars
                    if len(sample_values) >= 5:  # Show up to 5 samples
                        break
            
            if sample_values:
                with st.expander("What was found in your CSV (sample)"):
                    for i, val in enumerate(sample_values, 1):
                        st.write(f"{i}. `{val}`")
                    st.warning("These values don't start with 'http://' or 'https://', so they're not recognized as URLs.")
            
            st.info(" **Expected format**: CSV file with a column containing URLs (starting with http:// or https://). Example:")
            st.code("""URLs
https://example.com/video1.mp4
https://example.com/video2.mp4
https://example.com/image1.jpg
https://example.com/image2.jpg""")
            st.info("**Note**: This app analyzes media files (images/videos) from URLs. All rows will still be returned in the same order.")
        
        total_url_count = len(all_urls)
        total_csv_rows = len(df)
        rows_with_urls = sum(1 for _, urls, _ in row_data_list if urls)
        empty_rows = sum(1 for _, urls, content in row_data_list if not urls and content is None)
        invalid_url_rows = sum(1 for _, urls, content in row_data_list if not urls and content is not None)
        rows_with_multiple_urls = sum(1 for _, urls, _ in row_data_list if len(urls) > 1)
        
        # Show statistics
        st.info(f" **CSV Analysis**: {total_csv_rows} total rows")
        st.info(f" **URL Summary**: {total_url_count} total URLs found | {rows_with_urls} rows with URLs | {rows_with_multiple_urls} rows with multiple URLs")
        st.info(f" **Row Summary**: {empty_rows} empty rows | {invalid_url_rows} rows with invalid URLs")
        
        if not all_urls:
            st.warning(" No valid URLs found. All rows will be returned with N/A values.")
        
        # Show preview of URLs and row structure
        with st.expander(" Preview of URLs to be processed"):
            st.write(f"**Total URLs found**: {len(all_urls)}")
            st.write(f"**Total CSV rows**: {len(row_data_list)}")
            st.write("**First 10 rows with their URLs:**")
            for i, (row_idx, urls, content) in enumerate(row_data_list[:10]):
                if urls:
                    st.write(f"Row {row_idx + 1}: {len(urls)} URL(s) - {', '.join(urls[:2])}{'...' if len(urls) > 2 else ''}")
                else:
                    st.write(f"Row {row_idx + 1}: No valid URLs")
            if len(row_data_list) > 10:
                st.write(f"... and {len(row_data_list) - 10} more rows")
        
        # Process only valid URLs with Gemini
        analysis_results = []
        if all_urls:
            st.info(" Starting Gemini analysis...")
            analysis_results = process_multiple_urls(all_urls, prompt)
            
            # Verify results match input order
            if len(analysis_results) != len(all_urls):
                st.warning(f" Warning: Expected {len(all_urls)} results but got {len(analysis_results)}. Order may not be preserved.")
        else:
            st.warning(" No valid URLs found to process. All rows will be marked as N/A.")
        
        # Create results dataframe maintaining exact CSV row order
        results_data = []
        
        # Map valid URLs to their analysis results (by index in all_urls)
        url_results_map = {}
        for i, result_data in enumerate(analysis_results):
            if i < len(all_urls):
                url_results_map[all_urls[i]] = result_data
        
        # Dynamically collect all columns from all results 
        all_columns = set()
        for result_data in analysis_results:
            if isinstance(result_data, dict):
                all_columns.update(result_data.keys())
        
        # Define preferred column order (metadata first, then standard fields if they exist, then dynamic fields)
        preferred_order = ['Media URL', 'Media Type', 'Kategorie I', 'Kategorie II', 'Produkt', 
                          'Anzahl Produkt', 'Inhaltsstoffe', 'Influencer Kategorie I', 
                          'Eingeblendeter Text', 'Raw AI Response']
        
        # Build column list: preferred columns first (if they exist), then any other columns
        analysis_columns = []

        for col in preferred_order:
            if col in all_columns:
                analysis_columns.append(col)
        # Add any remaining columns that aren't in preferred list (sorted alphabetically)
        remaining_cols = sorted([col for col in all_columns if col not in preferred_order])
        analysis_columns.extend(remaining_cols)
        
        # Function to merge multiple analysis results into one row
        def merge_results(results_list):
            """Merge multiple analysis results into a single row - fully dynamic, handles any columns"""
            if not results_list:
                return {col: 'N/A' for col in analysis_columns}
            
            merged = {}
            
            # Collect all columns from all results
            all_result_columns = set()
            for result in results_list:
                if isinstance(result, dict):
                    all_result_columns.update(result.keys())
            
            # Merge each column dynamically
            for col in all_result_columns:
                values = []
                for result in results_list:
                    if isinstance(result, dict) and col in result:
                        val = result[col]
                        # Only collect non-empty, non-N/A values
                        if val and val != 'N/A' and str(val).strip() and str(val).strip() != 'N/A':
                            values.append(str(val).strip())
                
                if values:
                    # Special handling for specific columns
                    if col == 'Anzahl Produkt':
                        # Sum numeric values
                        try:
                            total = sum(int(v) for v in values if v.isdigit())
                            merged[col] = str(total) if total > 0 else 'N/A'
                        except:
                            merged[col] = '; '.join(set(values))
                    elif col == 'Raw AI Response':
                        # Join with separator for raw responses
                        merged[col] = ' | '.join(values)
                    elif col == 'Eingeblendeter Text':
                        # Join with separator for text
                        merged[col] = ' | '.join(values)
                    else:
                        # For other columns, join unique values with semicolon
                        merged[col] = '; '.join(set(values))
                else:
                    merged[col] = 'N/A'
            
            # Ensure all expected columns exist 
            for col in analysis_columns:
                if col not in merged:
                    merged[col] = 'N/A'
            
            return merged
        
        # Process each CSV row in the exact same order
        for csv_row_idx, (original_row_idx, urls_in_row, original_cell_content) in enumerate(row_data_list):
            row_data = {
                'CSV_Row_Number': original_row_idx + 1,  
                'Row_Order': csv_row_idx + 1  
            }
            
            # Store original cell content exactly as it was in the CSV
            row_data['Original_URLs'] = original_cell_content if original_cell_content else ''
            
            if urls_in_row:
                # Row has valid URLs (process all of them and merge into ONE row)
                row_results = []
                
                for url in urls_in_row:
                    if url in url_results_map:
                        row_results.append(url_results_map[url])
                    else:
                        # URL was processed but result not found - create empty result
                        empty_result = {col: 'N/A' for col in analysis_columns}
                        row_results.append(empty_result)
                
                # Combine all URLs from this row into a single string 
                row_data['URL'] = '; '.join(urls_in_row)
                row_data['URL_Count'] = len(urls_in_row)
                
                if row_results:
                    # Merge results from all URLs in this row into ONE result
                    merged_result = merge_results(row_results)
                    for col in analysis_columns:
                        row_data[col] = merged_result.get(col, 'N/A')
                else:
                    # No results found 
                    for col in analysis_columns:
                        row_data[col] = 'N/A'
            elif original_cell_content:
                # Invalid URL (not starting with http/https) - mark as N/A
                row_data['URL'] = original_cell_content
                row_data['URL_Count'] = 0
                row_data['Error'] = 'Invalid URL format'
                # Original_URLs already set above
                for col in analysis_columns:
                    row_data[col] = 'N/A'
            else:
                # Empty row - mark as N/A
                row_data['URL'] = ''
                row_data['URL_Count'] = 0
                row_data['Error'] = 'Empty row'
                for col in analysis_columns:
                    row_data[col] = 'N/A'
            
            results_data.append(row_data)
        
        # Create enhanced dataframe with results
        enhanced_df = pd.DataFrame(results_data)
        
        # Show processing statistics
        total_results = len(enhanced_df)
        # Count successful results (any row that doesn't have an Error column or Error is N/A)
        if 'Error' in enhanced_df.columns:
            successful_results = len(enhanced_df[(enhanced_df['Error'].isna()) | (enhanced_df['Error'] == 'N/A')])
        else:
            # Fallback: count rows where Raw AI Response doesn't start with "Error"
            if 'Raw AI Response' in enhanced_df.columns:
                successful_results = len(enhanced_df[~enhanced_df['Raw AI Response'].astype(str).str.startswith('Error')])
            else:
                successful_results = total_results
        st.info(f"📊 **Processing Summary**: {total_results} URLs processed | {successful_results} successful analyses")
        
        return enhanced_df, len(analysis_results), url_column
        
    except Exception as e:
        st.error(f"❌ Error processing CSV file: {str(e)}")
        return None

# Main interface
st.header("Analyze Cosmetic Videos/Images with Gemini AI")

# Configuration info
st.info(f" Using model: **{MODEL_ID}** in **{LOCATION}**")

# Input form
with st.form("media_analysis_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input")
        
        # Single URL input
        media_url = st.text_input("Single Media URL:", 
                                 placeholder="https://example.com/image.jpg or https://example.com/video.mp4")
        
        st.markdown("**OR**")
        
        # Multiple URLs input
        multiple_urls = st.text_area("Multiple URLs (one per line):", 
                                    placeholder="https://example.com/image1.jpg\nhttps://example.com/video1.mp4\nhttps://example.com/image2.jpg",
                                    height=150)
        
        st.markdown("**OR**")
        
        # CSV file upload - simplified for one column with only URLs
        uploaded_file = st.file_uploader("Upload CSV file with URLs:", 
                                       type=['csv'],
                                       help="Upload a CSV file with ONE column containing only URLs. Row 1 will be used as column header.")
        
        # Automatically use the first (and ideally only) column
        url_column_name = None
        if uploaded_file is not None:
            # Store file content in session state so it persists after form submission
            try:
                uploaded_file.seek(0)  # Reset file pointer
                file_content = uploaded_file.read()
                # Ensure we store as bytes
                if isinstance(file_content, str):
                    st.session_state.csv_file_content = file_content.encode('utf-8')
                else:
                    st.session_state.csv_file_content = file_content
                st.session_state.csv_file_name = uploaded_file.name
            except Exception as e:
                st.error(f"❌ Error reading uploaded file: {str(e)}")
                st.session_state.csv_file_content = None
            
            # Try to read the CSV to get column names using stored content
            try:
                # Use stored content to read CSV (create fresh BytesIO)
                if st.session_state.csv_file_content is not None:
                    temp_file = io.BytesIO(st.session_state.csv_file_content)
                    temp_file.seek(0)
                    temp_df = pd.read_csv(temp_file, header=0, nrows=0)  # Use row 1 as header
                    available_columns = temp_df.columns.tolist()
                    
                    if len(available_columns) == 1:
                        url_column_name = available_columns[0]
                        st.session_state.csv_url_column = url_column_name
                        st.success(f"✅ **Single column detected**: `{url_column_name}` - Perfect for URL processing!")
                    else:
                        st.warning(f"⚠️ **Multiple columns detected**: {len(available_columns)} columns found. Using first column: `{available_columns[0]}`")
                        url_column_name = available_columns[0]
                        st.session_state.csv_url_column = url_column_name
                        st.info("💡 **Tip**: For best results, use a CSV with only ONE column containing URLs and group labels")
                    
                    st.info(f"📋 **Column to process**: `{url_column_name}`")
                else:
                    st.error("❌ Could not read file content")
                
            except Exception as e:
                st.error(f"❌ Could not read CSV file: {str(e)}")
                st.info("💡 **Tip**: Make sure your CSV file has a header row and contains valid data")
        else:
            # Use stored values from session state if file was previously uploaded
            if st.session_state.csv_url_column:
                url_column_name = st.session_state.csv_url_column
                st.info(f"📋 **Using previously selected column**: `{url_column_name}`")
                if st.session_state.csv_file_name:
                    st.info(f"📁 **File**: `{st.session_state.csv_file_name}`")
        
        # Brand selection for targeted analysis
        brand_selection = st.selectbox(
            "Select Brand Focus (optional):",
            options=["General (All Brands)", "CeraVe", "L'Oréal", "Garnier", "Dove", "Mixa", "No-Cosmetics"],
            help="Choose a specific brand for targeted analysis, or use General for all brands"
        )
        
        # Get appropriate prompt based on selection
        if brand_selection == "General (All Brands)":
            selected_prompt = DEFAULT_PROMPT
            selected_brand = None
        else:
            brand_map = {
                "CeraVe": "cerave",
                "L'Oréal": "loreal", 
                "Garnier": "garnier",
                "Dove": "dove",
                "Mixa": "mixa",
                "No-Cosmetics": "no-cosmetics"
            }
            selected_brand = brand_map[brand_selection]
            selected_prompt = get_brand_specific_prompt(selected_brand)
        
        prompt = st.text_area("Analysis Prompt:", 
                             value=selected_prompt,
                             height=200,
                             help=f"Using {'brand-specific' if selected_brand else 'general'} prompt for {brand_selection}. Loaded from: prompts/{BRAND_PROMPT_FILES.get(selected_brand or 'general', 'general_prompt.txt')}")
        
        if selected_brand:
            st.info(f"🎯 **Brand-Focused Analysis**: Using {brand_selection}-specific prompt for enhanced detection")
        else:
            st.info(f"📝 **General Analysis**: Using general prompt that detects all supported brands")
        
        # Show prompt file information
        st.expander("📁 Prompt File Information").markdown(f"""
        **Current prompt loaded from:** `prompts/{BRAND_PROMPT_FILES.get(selected_brand or 'general', 'general_prompt.txt')}`
        
        **To customize prompts:**
        1. Navigate to the `prompts/` directory in your project
        2. Edit the appropriate `.txt` file for your brand
        3. Save the file and restart the app to load your changes
        
        **Available prompt files:**
        - `general_prompt.txt` - General prompt for all brands
        - `cerave_prompt.txt` - CeraVe-specific prompt
        - `loreal_prompt.txt` - L'Oréal-specific prompt  
        - `garnier_prompt.txt` - Garnier-specific prompt
        - `dove_prompt.txt` - Dove-specific prompt
        - `mixa_prompt.txt` - Mixa-specific prompt
        - `no_cosmetics_prompt.txt` - No-Cosmetics-specific prompt
        """)
        
        # Processing options
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            submit_single = st.form_submit_button("Analyze Single URL")
        with col1b:
            submit_multiple = st.form_submit_button("Analyze Multiple URLs")
        with col1c:
            submit_csv = st.form_submit_button("Process CSV File")
    
    with col2:
        st.subheader("Instructions")
        st.markdown("""
        **Cosmetic Media Analysis Tool**
        
        This tool analyzes cosmetic/skincare media and extracts:
        - **Kategorie I**: Product categories (Brand, Face Care, Sun, etc.)
        - **Kategorie II**: Content types (Klas. Kampagne, Co-Creation, Event)
        - **Produkt**: Product names as detected by Gemini AI
        - **Anzahl Produkt**: Product count
        - **Inhaltsstoffe**: Ingredients mentioned
        - **Influencer Kategorie I**: Influencer types (Lifestyle, Beauty Influencer, Skinfluencer, Medfluencer)
        
        **🎯 Processing Options:**
        
        **1. Single URL**: Analyze one media file
        - Paste a single URL and get instant analysis
        
        **2. Multiple URLs**: Batch process multiple URLs
        - Paste multiple URLs (one per line) for batch processing
        
        **3. CSV Upload**: Bulk process from CSV file
        - Upload a CSV with ONE column containing only URLs
        - Simple format - just URLs, no group labels needed
        
        **📁 CSV Format (Simple):**
        ```
        URLs
        https://example.com/video1.mp4
        https://example.com/video2.mp4
        https://example.com/image1.jpg
        https://example.com/image2.jpg
        ```
        
        **📊 Supported Media:**
        - **Images**: JPG, JPEG, PNG, GIF, BMP, WebP
        - **Videos**: MP4, AVI, MOV, WMV, FLV, WebM, MKV
        
        **🚀 Performance Features:**
        - **Standard Mode**: Up to 5 URLs simultaneously (< 20 URLs)
        - **Bulk Mode**: Up to 12 URLs simultaneously (≥ 20 URLs)
        - **Smart Timeouts**: Optimized for reliability
        - **Progress Tracking**: Real-time updates during processing
        - **Retry Logic**: Handles temporary failures automatically
        
        **✅ What you get:**
        - **Complete analysis** for each URL
        - **CSV download** with all results
        - **Clean data structure** ready for Excel
        - **Fast processing** with progress tracking
        """)

# Process single URL analysis
if submit_single and media_url and prompt:
    if not media_url.startswith(('http://', 'https://')):
        st.error("Please enter a valid URL starting with http:// or https://")
    else:
        with st.spinner("Analyzing cosmetic media with Gemini AI..."):
            # Auto-detect media type
            detected_type = detect_media_type(media_url)
            media_category = get_media_category(detected_type)
            st.info(f"Detected media type: {detected_type} ({media_category})")
            
            result = analyze_media(media_url, prompt, detected_type)
            
            if result and not result.startswith("Error"):
                # Try to parse the JSON response
                parsed_result = extract_json_from_response(result)
                
                if parsed_result:
                    st.success(f"Analysis completed for {media_category.lower()}!")
                    st.subheader("Analysis Result:")
                    st.json(parsed_result)
                else:
                    st.warning("AI response couldn't be parsed as JSON.")
                    st.subheader("Raw AI Response:")
                    st.write(result)
            else:
                st.error(f"Analysis failed: {result}")

# Process multiple URLs analysis
if submit_multiple and multiple_urls and prompt:
    urls_list = [url.strip() for url in multiple_urls.split('\n') if url.strip()]
    
    if not urls_list:
        st.error("Please enter at least one URL")
    else:
        st.info(f"Processing {len(urls_list)} URLs...")
        
        # Process all URLs
        batch_results = process_multiple_urls(urls_list, prompt)
        
        st.success(f"Successfully processed {len(batch_results)} URLs!")
        
        # Display results
        if batch_results:
            st.subheader("Analysis Results")
            results_df = pd.DataFrame(batch_results)
            st.dataframe(results_df, use_container_width=True)
            
            # Download option
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False, sep=';')
            csv_str = csv_buffer.getvalue()
            
            st.download_button(
                label="Download Results as CSV",
                data=csv_str,
                file_name="multiple_urls_analysis_results.csv",
                mime="text/csv"
            )

# Process CSV file analysis
# Use session state values if file was uploaded but reset after form submission
csv_file_to_process = uploaded_file if uploaded_file is not None else None
if csv_file_to_process is None and st.session_state.csv_file_content is not None:
    # Recreate file-like object from session state
    try:
        # Ensure we have valid content
        file_content = st.session_state.csv_file_content
        if file_content is None or (isinstance(file_content, bytes) and len(file_content) == 0):
            st.error("❌ Stored file content is empty. Please upload the file again.")
            csv_file_to_process = None
        else:
            if isinstance(file_content, bytes):
                csv_file_to_process = io.BytesIO(file_content)
            else:
                # If it's a string, encode it to bytes
                csv_file_to_process = io.BytesIO(file_content.encode('utf-8'))
            csv_file_to_process.seek(0)  # Ensure file pointer is at the start
            csv_file_to_process.name = st.session_state.csv_file_name or "uploaded_file.csv"
    except Exception as e:
        st.error(f"❌ Error recreating file from session state: {str(e)}")
        csv_file_to_process = None

csv_url_column = url_column_name if url_column_name else st.session_state.csv_url_column

if submit_csv and csv_file_to_process and prompt and csv_url_column:
    with st.spinner("Processing CSV file..."):
        # Ensure file is ready for reading
        try:
            if hasattr(csv_file_to_process, 'seek'):
                csv_file_to_process.seek(0)
            # Process the CSV file
            csv_result = process_csv_file(csv_file_to_process, prompt, csv_url_column)
        except Exception as e:
            st.error(f"❌ Error preparing file for processing: {str(e)}")
            st.info("💡 Try uploading the file again.")
            csv_result = None
        
        if csv_result:
            enhanced_df, processed_count, url_column = csv_result
            
            st.success(f"🎉 Successfully processed {processed_count} URLs from CSV!")
            
            # Show summary statistics
            total_products_found = len(enhanced_df[enhanced_df['Produkt'] != 'N/A'])
            st.info(f"📊 **Analysis Summary**: {total_products_found}/{processed_count} URLs contained detectable products")
            
            # Display results
            st.subheader("📋 Analysis Results")
            
            # Display the enhanced dataframe
            st.dataframe(enhanced_df, use_container_width=True)
            
            # Download options
            st.subheader("📥 Download Results")
            
            # Reorder columns for better readability
            download_columns = ['CSV_Row_Number', 'Row_Order', 'Original_URLs', 'URL', 'URL_Count', 'Kategorie I', 'Kategorie II', 'Produkt', 'Anzahl Produkt', 
                              'Inhaltsstoffe', 'Influencer Kategorie I', 'Media Type', 'Eingeblendeter Text', 'Raw AI Response', 'Error']
            
            # Create download dataframe with available columns
            available_download_columns = [col for col in download_columns if col in enhanced_df.columns]
            download_df = enhanced_df[available_download_columns].copy()
            
            # Results are already in exact CSV order, but sort by Row_Order to ensure order
            if 'Row_Order' in enhanced_df.columns:
                download_df = download_df.sort_values('Row_Order')
            elif 'CSV_Row_Number' in enhanced_df.columns:
                download_df = download_df.sort_values('CSV_Row_Number')
            
            csv_buffer = io.StringIO()
            download_df.to_csv(csv_buffer, index=False, sep=';')
            csv_str = csv_buffer.getvalue()
            
            st.download_button(
                label="📊 Download Complete Analysis Results",
                data=csv_str,
                file_name=f"analysis_results_{csv_file_to_process.name if csv_file_to_process else 'results.csv'}",
                mime="text/csv",
                help="Download complete analysis results",
                use_container_width=True
            )
else:
    if submit_csv:
        error_msg = "❌ No CSV file uploaded or processing failed."
        if not csv_file_to_process:
            error_msg += " Please upload a CSV file."
        elif not csv_url_column:
            error_msg += " Could not determine column name."
        elif not prompt:
            error_msg += " Please ensure prompt is loaded."
        st.error(error_msg)

# Footer
st.markdown("---")
st.markdown("*Powered by Google Gemini AI - Specialized for Cosmetic Video Analysis*")

