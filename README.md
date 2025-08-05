# Gemini PDF Shareholder Extractor

This Python script is a powerful tool for extracting shareholder data from PDF documents. It's built to work with the **Gemini AI API**, using a robust, multithreaded approach to handle large or multiple PDF files efficiently.

## Key Features

  * **Intelligent Data Extraction**: Uses the Gemini 2.5 Pro model to accurately pull shareholder details—including company name, registration number, and individual shareholder information—from complex PDF tables.
  * **Multithreaded Processing**: Processes different parts of a large PDF at the same time, which dramatically cuts down on processing time.
  * **API Key Management**: Supports using multiple API keys, distributing tasks to help avoid rate limits and increase overall speed.
  * **Automatic PDF Splitting**: Automatically breaks large PDF files into smaller, manageable chunks to fit within API size limits.
  * **Data Consolidation**: Merges all extracted data from the smaller PDF parts into a single, comprehensive JSON file, with all shareholder lists sorted correctly.
  * **Robust Error Handling**: Comes with built-in retry logic and exponential backoff for failed API calls, and handles corrupted or missing files gracefully.
  * **State Persistence**: Keeps track of the processing status for each PDF part, so you can stop and restart the script without losing progress.

## Requirements

  * Python 3.x
  * The following Python libraries:
      * `requests`
      * `PyPDF2`
      * `tqdm`

## How to Use

1.  **Clone the Repository**:

    ```sh
    git clone https://github.com/your-username/Gemini-PDF-Shareholder-Extractor.git
    cd Gemini-PDF-Shareholder-Extractor
    ```

2.  **Install Dependencies**:

    ```sh
    pip install requests PyPDF2 tqdm
    ```

3.  **Set up API Keys**:
    Create a file named `Gemini_Keys.txt` in the root directory. Add your Gemini API keys to this file, with one key on each line.

    ```
    YOUR_GEMINI_API_KEY_1
    YOUR_GEMINI_API_KEY_2
    YOUR_GEMINI_API_KEY_3
    ```

4.  **Place PDF Files**:
    Put the PDF files you want to process into the `coman_data` folder.

5.  **Run the Script**:
    Execute the main Python script.

    ```sh
    python your_main_script.py
    ```

    The script will create an `output` directory inside the `coman_data` folder and save the extracted JSON results there.

## Output

The final output is a JSON file named `[original_filename]_combined_results.json`. This file contains the consolidated shareholder data for every company found in the PDF.

**Example Output:**

```json
{
  "datas": [
    {
      "company_name": "บริษัท ตัวอย่าง จำกัด",
      "corporate_registration_number": "1234567890",
      "as_of_date": "01/01/2025",
      "share_value": "0.50 บาท",
      "shareholders_list": [
        {
          "sequence_number": 1,
          "name": "นาย ก. นามสกุล",
          "address": "123 ถนนบางกอก กรุงเทพฯ",
          "nationality": "ไทย",
          "number_of_shares_held": [
            {
              "common_shares": "10000",
              "preferred_shares": "0",
              "share_certificate_number": "0001"
            }
          ]
        }
      ]
    }
  ]
}
```
