import base64
import pathlib
import requests
import os
import io
import json
import time
import hashlib
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import PyPDF2
import queue
import random
import threading

# Folder paths
PDF_FILE_PATH = "D:\\Work_GMO\\Scripts\\coman_data\\COMAN.pdf"
PDF_FOLDER_PATH = os.path.dirname(PDF_FILE_PATH)
OUTPUT_DIR = f"{PDF_FOLDER_PATH}\\output"

# Gemini Model and API settings
MODEL = "gemini-2.5-pro"  # 'gemini-2.5-flash' or 'gemini-2.5-pro' or 'gemini-2.5-flash-lite' are also valid
API_ENDPOINT_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"

# Multithreading and API retry settings
RETRY_ATTEMPTS = 3
RETRY_DELAY = 30  # in seconds

PDF_FILE_MAX_SIZE_MB = 2  # Maximum size of each split PDF file in MB

# ==============================================================================
#                             CORE FUNCTIONS
# ==============================================================================

# Lock to handle concurrent writes to the combined results file
consolidate_lock = threading.Lock()

# Create a queue to hold API keys
api_keys_queue = queue.Queue()


def log_with_timestamp(message, status="INFO", file_context=None):
    """
    Function to print a log message with a timestamp and optional file context.
    Adds a newline for better readability.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if file_context:
        print(f"[{timestamp}][{status}][{file_context}]\n{message}")
    else:
        print(f"[{timestamp}][{status}] {message}")

def load_api_keys(file_name="Gemini_Keys.txt"):
    """
    Loads API keys from a specified text file, one key per line.
    Shuffles them and puts them into a queue.
    """
    keys = []
    try:
        filepath = os.path.join(os.path.dirname(__file__), file_name)
        with open(filepath, 'r') as f:
            for line in f:
                key = line.strip()
                if key:
                    keys.append(key)
        
        if not keys:
            raise ValueError("No API keys found in the file.")
            
        random.shuffle(keys)
        for key in keys:
            api_keys_queue.put(key)
        return keys
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found. Please create it and add your API keys.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading API keys: {e}")
        exit()

def calculate_file_checksum(file_path):
    """
    Calculate MD5 checksum of a file.
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        log_with_timestamp(f"Error calculating checksum for {file_path}: {e}", "ERROR")
        return None

def get_tracking_file_path(original_filename):
    """
    Generates the path for the file tracking JSON based on the original PDF filename.
    """
    base_filename = os.path.splitext(original_filename)[0]
    return os.path.join(OUTPUT_DIR, f"{base_filename}_file_tracking.json")

def get_combined_results_path(original_filename):
    """
    Generates the path for the combined results JSON based on the original PDF filename.
    """
    base_filename = os.path.splitext(original_filename)[0]
    return os.path.join(OUTPUT_DIR, f"{base_filename}_combined_results.json")

def load_file_tracking(original_filename):
    """
    Load file tracking data from JSON file for a specific PDF.
    """
    tracking_path = get_tracking_file_path(original_filename)
    if os.path.exists(tracking_path):
        try:
            with open(tracking_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            log_with_timestamp("File tracking file is corrupted. Starting fresh.", "WARN", file_context=original_filename)
    return {}

def save_file_tracking(original_filename, tracking_data):
    """
    Save file tracking data to JSON file for a specific PDF.
    """
    tracking_path = get_tracking_file_path(original_filename)
    with open(tracking_path, 'w', encoding='utf-8') as f:
        json.dump(tracking_data, f, indent=2, ensure_ascii=False)

def verify_split_files_integrity(original_pdf_path, tracking_data):
    """
    Verify if split files exist and have correct checksums.
    Returns True if all files are valid, False otherwise.
    """
    original_filename = os.path.basename(original_pdf_path)
    if original_filename not in tracking_data:
        return False
    
    file_info = tracking_data[original_filename]
    split_files_data = file_info.get('split_files', [])
    
    if not split_files_data:
        return False
    
    log_with_timestamp(f"Verifying integrity of {len(split_files_data)} split files for '{original_filename}'...", file_context=original_filename)
    
    for file_data in split_files_data:
        file_path = file_data['file_path']
        expected_checksum = file_data['checksum']
        
        if not os.path.exists(file_path):
            log_with_timestamp(f"Split file missing: {file_path}", "WARN", file_context=original_filename)
            return False
        
        current_checksum = calculate_file_checksum(file_path)
        if current_checksum != expected_checksum:
            log_with_timestamp(f"Checksum mismatch for {file_path}. Expected: {expected_checksum}, Got: {current_checksum}", "WARN", file_context=original_filename)
            return False
    
    log_with_timestamp(f"All split files for '{original_filename}' are valid.", "INFO", file_context=original_filename)
    return True

def call_gemini_api_with_pdf(pdf_path, prompt_text, json_schema, api_key):
    """
    Calls the Gemini API with a PDF file and a specific API key.
    """
    headers = {
        "Content-Type": "application/json"
    }

    if not os.path.exists(pdf_path):
        log_with_timestamp(f"PDF file not found: {pdf_path}", "ERROR", file_context=os.path.basename(pdf_path))
        return None, "FILE_NOT_FOUND", {}

    try:
        filepath = pathlib.Path(pdf_path)
        pdf_data_base64 = base64.b64encode(filepath.read_bytes()).decode('utf-8')
        
        parts = [
            {"text": prompt_text},
            {
                "inlineData": {
                    "mimeType": "application/pdf",
                    "data": pdf_data_base64
                }
            }
        ]
        
        payload = {
            "contents": [
                {"parts": parts}
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "response_schema": json_schema,
                "temperature": 0.0,
                "maxOutputTokens": 1000000
            }
        }
        
        api_endpoint = API_ENDPOINT_TEMPLATE.format(MODEL)
        response = requests.post(
            f"{api_endpoint}?key={api_key}",
            json=payload,
            headers=headers
        )
        response.raise_for_status()

        gemini_response_json = response.json()
        finish_reason = gemini_response_json.get("candidates", [{}])[0].get("finishReason", "UNKNOWN_REASON")
        
        if finish_reason == "STOP":
            content_part = gemini_response_json["candidates"][0]["content"]["parts"][0]["text"]
            try:
                result_json = json.loads(content_part)
                return result_json, finish_reason, {}
            except json.JSONDecodeError:
                log_with_timestamp(f"Response from Gemini is not a valid JSON for {os.path.basename(pdf_path)}.", "ERROR", file_context=os.path.basename(pdf_path))
                return None, finish_reason, {}
        else:
            log_with_timestamp(f"Gemini API did not finish with STOP for {os.path.basename(pdf_path)}. Key: {api_key[-4:]}. Reason: {finish_reason}", "WARN", file_context=os.path.basename(pdf_path))
            return None, finish_reason, {}

    except requests.exceptions.RequestException as e:
        log_with_timestamp(f"API request error for {os.path.basename(pdf_path)}: {e}", "ERROR", file_context=os.path.basename(pdf_path))
        return None, "API_ERROR", {}
    except Exception as e:
        log_with_timestamp(f"Unexpected error for {os.path.basename(pdf_path)}: {e}", "ERROR", file_context=os.path.basename(pdf_path))
        return None, "UNKNOWN_ERROR", {}

def call_gemini_with_queue(split_file_data, user_prompt, json_schema, used_keys_for_file=None):
    """
    Worker function to get a key from the queue, call the API, and then return the key.
    Includes exponential backoff for retries and tracking of used keys.
    This function now updates the split_file_data directly.
    """
    if used_keys_for_file is None:
        used_keys_for_file = set()
    
    pdf_path = split_file_data['file_path']
    filename = split_file_data['filename']
    
    current_time = datetime.now().isoformat()
    current_delay = RETRY_DELAY
    
    for i in range(RETRY_ATTEMPTS):
        api_key = None
        
        # Try to get a new key that hasn't been used for this file yet
        # This part of the code is now more robust. It iterates through the queue
        # to find a key that is not in the `used_keys_for_file` set.
        keys_to_check = []
        while not api_keys_queue.empty():
            keys_to_check.append(api_keys_queue.get())
        
        for key in keys_to_check:
            if key not in used_keys_for_file:
                api_key = key
                keys_to_check.remove(key)
                break
        
        # Put remaining keys back into the queue
        for key in keys_to_check:
            api_keys_queue.put(key)

        # If no new key is found, get the next available one from the queue (even if used before)
        if api_key is None and not api_keys_queue.empty():
            api_key = api_keys_queue.get()

        if api_key is None:
            log_with_timestamp(f"No API key available to process {filename}. Skipping.", "WARN", file_context=filename)
            split_file_data['status_request'] = 'Failed'
            return None, None
            
        log_with_timestamp(f"Worker assigned key ending in ...{api_key[-4:]} for {filename}. Keys in queue: {api_keys_queue.qsize()}", "DEBUG", file_context=filename)

        try:
            result, finish_reason, _ = call_gemini_api_with_pdf(pdf_path, user_prompt, json_schema, api_key)
            
            if 'requests' not in split_file_data:
                split_file_data['requests'] = []
            
            split_file_data['requests'].append({
                'attempt': i + 1,
                'timestamp': current_time,
                'api_key_suffix': api_key[-4:],
                'status': 'success' if result and finish_reason == "STOP" else 'failed',
                'finish_reason': finish_reason
            })
            
            if result and finish_reason == "STOP":
                split_file_data['status_request'] = 'Success'
                return result, api_key
            else:
                # Add the failed key to the used keys list for this file
                used_keys_for_file.add(api_key)
                if i < RETRY_ATTEMPTS - 1:
                    log_with_timestamp(f"Attempt {i+1} failed for {filename} (key: ...{api_key[-4:]}). Retrying in {current_delay}s with a different key...", "WARN", file_context=filename)
                    time.sleep(current_delay)
                    current_delay *= 2
                else:
                    log_with_timestamp(f"All retry attempts failed for {filename} (last key: ...{api_key[-4:]}).", "ERROR", file_context=filename)

        finally:
            if api_key:
                api_keys_queue.put(api_key)
                log_with_timestamp(f"Worker released key ending in ...{api_key[-4:]}. Keys in queue: {api_keys_queue.qsize()}", file_context=filename)
    
    split_file_data['status_request'] = 'Failed'
    return None, None

def save_json(full_path, data):
    """
    Saves a single JSON data to a file.
    """
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log_with_timestamp(f"Saved result to {full_path}")

def split_pdf_by_size(pdf_path, max_size_mb=4, tracking_data=None):
    """
    Splits a large PDF file into multiple smaller PDF files, ensuring each part
    does not exceed the specified maximum size. Returns a list of paths
    to the generated smaller PDF files.
    """
    if tracking_data is None:
        tracking_data = {}
    
    max_size_bytes = max_size_mb * 1024 * 1024
    split_dir = os.path.join(os.path.dirname(pdf_path), f"{os.path.splitext(os.path.basename(pdf_path))[0]}_parts")
    original_filename = os.path.basename(pdf_path)
    
    if verify_split_files_integrity(pdf_path, tracking_data):
        return [file_data for file_data in tracking_data[original_filename]['split_files']]
    
    if os.path.exists(split_dir):
        log_with_timestamp(f"Cleaning up old split directory for '{original_filename}'.", "INFO", file_context=original_filename)
        for f in os.listdir(split_dir):
            os.remove(os.path.join(split_dir, f))
        os.rmdir(split_dir)

    log_with_timestamp(f"Splitting PDF '{original_filename}' into parts (max {max_size_mb} MB per part)...", file_context=original_filename)
    os.makedirs(split_dir, exist_ok=True)

    pdf_reader = PyPDF2.PdfReader(pdf_path)
    pdf_writer = PyPDF2.PdfWriter()
    split_files_data = []
    current_part_size = 0
    part_number = 1

    if os.path.getsize(pdf_path) <= max_size_bytes:
        log_with_timestamp("PDF file is already smaller than max size, no splitting needed.", file_context=original_filename)
        checksum = calculate_file_checksum(pdf_path)
        split_files_data.append({
            'file_path': pdf_path,
            'filename': original_filename,
            'checksum': checksum,
            'part_number': 1,
            'created_at': datetime.now().isoformat(),
            'status_request': '',
            'requests': []
        })
        
        tracking_data[original_filename] = {
            'original_path': pdf_path,
            'split_files': split_files_data,
            'total_parts': 1,
            'created_at': datetime.now().isoformat()
        }
        
        return split_files_data

    for page_num in tqdm(range(len(pdf_reader.pages)), desc=f"Splitting {original_filename}"):
        page = pdf_reader.pages[page_num]
        
        page_stream = io.BytesIO()
        temp_writer = PyPDF2.PdfWriter()
        temp_writer.add_page(page)
        temp_writer.write(page_stream)
        page_size = len(page_stream.getvalue())

        if current_part_size + page_size > max_size_bytes and current_part_size > 0:
            part_filename = f"{os.path.splitext(original_filename)[0]}_part{part_number}.pdf"
            part_filepath = os.path.join(split_dir, part_filename)
            with open(part_filepath, "wb") as output_file:
                pdf_writer.write(output_file)
            
            checksum = calculate_file_checksum(part_filepath)
            split_files_data.append({
                'file_path': part_filepath,
                'filename': part_filename,
                'checksum': checksum,
                'part_number': part_number,
                'created_at': datetime.now().isoformat(),
                'status_request': '',
                'requests': []
            })

            pdf_writer = PyPDF2.PdfWriter()
            pdf_writer.add_page(page)
            current_part_size = page_size
            part_number += 1
        else:
            pdf_writer.add_page(page)
            current_part_size += page_size

    if pdf_writer.pages:
        part_filename = f"{os.path.splitext(original_filename)[0]}_part{part_number}.pdf"
        part_filepath = os.path.join(split_dir, part_filename)
        with open(part_filepath, "wb") as output_file:
            pdf_writer.write(output_file)
        
        checksum = calculate_file_checksum(part_filepath)
        split_files_data.append({
            'file_path': part_filepath,
            'filename': part_filename,
            'checksum': checksum,
            'part_number': part_number,
            'created_at': datetime.now().isoformat(),
            'status_request': '',
            'requests': []
        })
    
    tracking_data[original_filename] = {
        'original_path': pdf_path,
        'split_files': split_files_data,
        'total_parts': len(split_files_data),
        'created_at': datetime.now().isoformat()
    }
    
    log_with_timestamp(f"PDF '{original_filename}' split into {len(split_files_data)} parts.", file_context=original_filename)
    return split_files_data

def consolidate_results(data_entries):
    """
    Consolidates shareholder data from multiple PDF parts into a single entry
    if they share the same corporate_registration_number, as_of_date, and share_value.
    Also sorts shareholders_list by sequence_number in ascending order.
    """
    consolidated_data = {}
    
    for data_entry in data_entries:
        company_info_key = (
            data_entry.get('corporate_registration_number'),
            data_entry.get('as_of_date'),
            data_entry.get('share_value')
        )
        
        if company_info_key not in consolidated_data:
            consolidated_data[company_info_key] = {
                "company_name": data_entry.get('company_name'),
                "corporate_registration_number": data_entry.get('corporate_registration_number'),
                "as_of_date": data_entry.get('as_of_date'),
                "share_value": data_entry.get('share_value'),
                "shareholders_list": []
            }
        
        consolidated_data[company_info_key]["shareholders_list"].extend(data_entry.get('shareholders_list', []))
    
    for company_data in consolidated_data.values():
        shareholders_list = company_data.get("shareholders_list", [])
        shareholders_list.sort(key=lambda x: x.get('sequence_number', 0) if x.get('sequence_number') is not None else 0)
        company_data["shareholders_list"] = shareholders_list
    
    final_output = list(consolidated_data.values())
    return {"datas": final_output}

def load_combined_results(original_filename):
    """
    Loads existing consolidated results from the output file if it exists.
    Returns the loaded data or an empty list.
    """
    combined_path = get_combined_results_path(original_filename)
    if os.path.exists(combined_path):
        try:
            with open(combined_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'datas' in data:
                    return data.get('datas', [])
                else:
                    log_with_timestamp("Combined output file is not in the expected format. Starting fresh.", "WARN", file_context=original_filename)
                    return []
        except (json.JSONDecodeError, FileNotFoundError):
            log_with_timestamp("Combined output file is corrupted or empty. Starting with no previous data.", "WARN", file_context=original_filename)
            return []
    return []

def save_new_result_incrementally(new_result_datas, original_filename):
    """
    Saves a new result incrementally to the combined results file.
    It loads existing data, adds the new data, consolidates, and saves it back.
    Uses a lock to prevent race conditions in multi-threaded environment.
    """
    with consolidate_lock:
        existing_datas = load_combined_results(original_filename)
        
        all_data_entries = existing_datas + new_result_datas
        
        if all_data_entries:
            final_output = consolidate_results(all_data_entries)
            combined_path = get_combined_results_path(original_filename)
            save_json(combined_path, final_output)
        else:
            log_with_timestamp("No data found to consolidate.", "WARN", file_context=original_filename)

# ==============================================================================
#                               MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    API_KEYS = load_api_keys()
    MAX_WORKERS = len(API_KEYS)
    log_with_timestamp(f"Loaded {len(API_KEYS)} API keys.")

    user_prompt = """You are an expert data extraction bot. Your task is to extract all data from ALL shareholder tables in this document. This document may contain multiple shareholder tables, each with a different "as of date". The output MUST be in the JSON format as defined in the response schema.

**Instructions for Data Extraction:**

1.  **Source of Data:** Identify and extract information ONLY from ALL main shareholder tables. You must process every table found in the document, regardless of its "as of date". IGNORE all other text outside of these tables, including headers, footers, and surrounding text.
2.  **Data Integrity:** All extracted values must be an exact replication of the source document.
    * **company_name**: Extract the full name exactly as it appears.
    * **corporate_registration_number**: Extract the number exactly as it appears.
    * **as_of_date**: Extract the date exactly as it appears, in the format "DD/MM/YYYY".
    * **share_value**: Extract the value and currency (e.g., "0.50 บาท") exactly as it appears.
3.  **Handling Missing Data:** If a cell in the table is empty or the information is not present, the corresponding value in the JSON must be `null` or an empty string (`""`). **DO NOT guess, infer, or create data.**
4.  **Shareholder List:** Iterate through each row of every table to extract shareholder details.
    * **sequence_number**: Convert to a numeric type.
    * **name**: Extract the shareholder's full name.
    * **address**: Extract the full address.
    * **nationality**: Extract the nationality.
    * **number_of_shares_held**: Extract "common_shares", "preferred_shares", and "share_certificate_number" from their respective columns. Ensure all numeric values in this section have NO commas.
    """

    json_schema = {
        "type": "object",
        "properties": {
            "datas": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company_name": {"type": "string"},
                        "corporate_registration_number": {"type": "string"},
                        "as_of_date": {"type": "string"},
                        "share_value": {"type": "string"},
                        "shareholders_list": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sequence_number": {"type": "number"},
                                    "name": {"type": "string"},
                                    "address": {"type": "string"},
                                    "nationality": {"type": "string"},
                                    "number_of_shares_held": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "common_shares": {"type": "string"},
                                                "preferred_shares": {"type": "string"},
                                                "share_certificate_number": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "required": ["datas"]
    }
    
    pdf_files_to_process = [os.path.join(PDF_FOLDER_PATH, f) for f in os.listdir(PDF_FOLDER_PATH) if f.endswith('.pdf')]

    for pdf_file_path in pdf_files_to_process:
        original_filename = os.path.basename(pdf_file_path)

        file_tracking_data = load_file_tracking(original_filename)
        log_with_timestamp(f"Loaded file tracking data for {original_filename}.", file_context=original_filename)

        split_files_data = split_pdf_by_size(pdf_file_path, max_size_mb=PDF_FILE_MAX_SIZE_MB, tracking_data=file_tracking_data)

        files_to_run_data = []
        for file_info in split_files_data:
            if file_info.get('status_request') != 'Success':
                files_to_run_data.append(file_info)

        if not files_to_run_data:
            log_with_timestamp(f"All parts of {original_filename} have been processed successfully. Skipping.", "INFO", file_context=original_filename)
            continue

        log_with_timestamp(f"Found {len(files_to_run_data)} files to process for {original_filename}.", file_context=original_filename)
        log_with_timestamp(f"Using {MAX_WORKERS} workers to process files.", file_context=original_filename)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {}
            for split_file_data in files_to_run_data:
                used_keys = set()
                # Populate used_keys from previous failed requests
                for req in split_file_data.get('requests', []):
                    key_suffix = req['api_key_suffix']
                    for full_key in API_KEYS:
                        if full_key.endswith(key_suffix):
                            used_keys.add(full_key)
                            break
                
                future = executor.submit(call_gemini_with_queue, split_file_data, user_prompt, json_schema, used_keys)
                future_to_file[future] = split_file_data

            for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc=f"Processing {original_filename} parts"):
                split_file_data = future_to_file[future]
                try:
                    result_tuple = future.result()
                    if result_tuple and len(result_tuple) == 2:
                        result, used_key = result_tuple
                        if result and 'datas' in result:
                            # Save result incrementally to the combined results file
                            save_new_result_incrementally(result['datas'], original_filename)
                            log_with_timestamp(f"Successfully processed {split_file_data['filename']} with key ...{used_key[-4:]}", "SUCCESS", file_context=split_file_data['filename'])
                        else:
                            log_with_timestamp(f"Failed to process {split_file_data['filename']}: Invalid result or no 'datas' key found.", "ERROR", file_context=split_file_data['filename'])
                    else:
                        log_with_timestamp(f"Invalid result format for {split_file_data['filename']}", "ERROR", file_context=split_file_data['filename'])
                except Exception as exc:
                    log_with_timestamp(f"'{split_file_data['filename']}' generated an exception: {exc}", "ERROR", file_context=split_file_data['filename'])
                
                # Save file_tracking data after each file is processed (status is updated)
                save_file_tracking(original_filename, file_tracking_data)
        
        log_with_timestamp(f"Finished processing all parts for {original_filename}.", "INFO", file_context=original_filename)
    
    log_with_timestamp("Processing of all PDF files complete.")
