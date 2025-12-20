# Chat History Database - Streamlit App

This is a Streamlit web application for retrieving and analyzing chat history data from AWS S3.

## Project Structure

```
chatbot_database/
├── app.py                 # Main Streamlit application (entry point)
├── local_database.py      # Database class for S3 operations
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Features

- Retrieve chat history from S3 by date range
- Filter by user ID
- Display statistics and analytics
- Mode detection (education vs consultation)
- Topic extraction for education mode conversations
- Word cloud visualization
- Export data to JSON

## Setup for Streamlit Cloud

### 1. Environment Variables

Set the following environment variables in Streamlit Cloud:

**Required:**
- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_DEFAULT_REGION`: AWS region (e.g., `us-east-1`)
- `APP_PASSWORD`: Password to access the application (set a strong password!)

**Optional:**
- `HTTPS_PROXY`: Proxy URL if needed

**Note:** For Streamlit Cloud, you can also set `APP_PASSWORD` in the Secrets section (`.streamlit/secrets.toml`) instead of environment variables. 

### 2. Deploy to Streamlit Cloud

1. Push this `chatbot_database` folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to: `app.py`
5. Set the Python version (recommended: 3.9 or higher)
6. Add the environment variables listed above
7. Deploy!

## Local Development

### Installation

```bash
cd chatbot_database
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

### Environment Setup

Create a `.env` file in the `chatbot_database` directory:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
APP_PASSWORD=your_secure_password_here
```

**Important:** Change `APP_PASSWORD` to a strong password. This password will be required to access the application.

## Configuration

Default S3 settings (can be modified in `local_database.py`):

- Bucket: `chatbot-content-storage`
- Path: `chat_history_library`

## Dependencies

- `streamlit`: Web framework
- `boto3`: AWS SDK
- `pandas`: Data processing
- `wordcloud`: Word cloud visualization (optional)
- `matplotlib`: Plotting (optional)

