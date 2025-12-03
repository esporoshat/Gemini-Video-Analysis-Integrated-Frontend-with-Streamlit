# Cosmetic Media Analyzer

A Streamlit application that uses Google's Gemini AI (via Vertex AI) to analyze cosmetic and skincare media (images and videos) and extract product information, categories, and influencer classifications.

## Features

- **Single URL Analysis**: Analyze individual media files
- **Batch Processing**: Process multiple URLs simultaneously
- **CSV Bulk Processing**: Upload CSV files with URLs for large-scale analysis
- **Brand-Specific Analysis**: Custom prompts for different cosmetic brands (CeraVe, L'Oréal, Garnier, Dove, Mixa, No-Cosmetics)
- **Smart Media Detection**: Automatically detects media type from URLs
- **Progress Tracking**: Real-time progress updates for batch processing
- **Export Results**: Download analysis results as CSV files

## Prerequisites

- Python 3.8 or higher
- Google Cloud Project with Vertex AI API enabled
- Google Cloud Service Account with Vertex AI permissions
- Service Account JSON key file

## Installation

1. **Clone the repository** (or download the project files)

2. **Install dependencies**:
   ```bash
   pip3 install -r requirements_project.txt
   ```

3. **Set up Google Cloud credentials**:
   - Create a service account in Google Cloud Console
   - Download the JSON key file
   - Place it in the project root as `service_account.json`
   - Or set the `SERVICE_ACCOUNT_FILE` environment variable

4. **Configure environment variables** (optional):
   - Copy `.env.example` to `.env`
   - Fill in your Google Cloud project details:
     ```
     SERVICE_ACCOUNT_FILE=service_account.json
     PROJECT_ID=your-project-id
     LOCATION=us-central1
     MODEL_ID=gemini-2.5-flash
     ```

## Usage

1. **Start the Streamlit app**:
   ```bash
   streamlit run Streamlit_Project.py
   ```

2. **Access the app**:
   - Open your browser to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Analyze media**:
   - Enter a single URL, multiple URLs, or upload a CSV file
   - Select a brand focus (optional) the Prompts variations not inlcuded here.
   - Click the appropriate analyze button
   - Download results as CSV

## CSV Format

The CSV file should have one column containing URLs:

```csv
URLs
https://example.com/video1.mp4
https://example.com/video2.mp4
https://example.com/image1.jpg
```

## Supported Media Types

- **Images**: JPG, JPEG, PNG, GIF, BMP, WebP
- **Videos**: MP4, AVI, MOV, WMV, FLV, WebM, MKV

## Project Structure

```
.
├── Streamlit_Project.py      # Main application file
├── requirements_project.txt   # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore rules
├── prompts/                 # Brand-specific prompt files
│   ├── general_prompt.txt
│   ├── cerave_prompt.txt
│   ├── loreal_prompt.txt
│   └── ...
└── README_GITHUB.md         # This file
```

## Configuration

### Environment Variables

- `SERVICE_ACCOUNT_FILE`: Path to Google Cloud service account JSON file
- `PROJECT_ID`: Your Google Cloud project ID
- `LOCATION`: Vertex AI location (default: `us-central1`)
- `MODEL_ID`: Gemini model to use (default: `gemini-2.5-flash`)

### Customizing Prompts

Edit the prompt files in the `prompts/` directory to customize analysis behavior for different brands.

## Troubleshooting

### Service Account File Not Found

- Ensure `service_account.json` is in the project root
- Or set `SERVICE_ACCOUNT_FILE` environment variable to the correct path

### Authentication Errors

- Verify your service account has Vertex AI API access
- Check that the Vertex AI API is enabled in Google Cloud Console
- Ensure the service account has the necessary IAM roles

### URL Access Errors

- URLs must be publicly accessible (no authentication required)
- For S3/cloud storage, ensure files/buckets are publicly readable
- Check that robots.txt allows access for Vertex AI

## License

This is a personal project. Use at your own discretion.

## Contributing

This is a personal project, but suggestions and improvements are welcome!


