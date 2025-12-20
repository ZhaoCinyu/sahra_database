import boto3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

# Default S3 bucket and path constants
DEFAULT_S3_BUCKET = "chatbot-content-storage"
DEFAULT_S3_CHAT_HISTORY_PATH = "chat_history_library"


def get_aws_credentials():
    """
    Get AWS credentials from multiple sources in order of priority:
    1. Streamlit secrets (for Streamlit Cloud)
    2. Environment variables
    3. Default AWS credentials file (~/.aws/credentials)
    """
    try:
        import streamlit as st
        # Try to get from Streamlit secrets
        if hasattr(st, 'secrets'):
            aws_access_key = st.secrets.get("AWS_ACCESS_KEY_ID")
            aws_secret_key = st.secrets.get("AWS_SECRET_ACCESS_KEY")
            aws_region = st.secrets.get("AWS_DEFAULT_REGION")
            
            if aws_access_key and aws_secret_key:
                return {
                    'aws_access_key_id': aws_access_key,
                    'aws_secret_access_key': aws_secret_key,
                    'region_name': aws_region or os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
                }
    except (ImportError, RuntimeError):
        # Streamlit not available or not in Streamlit context
        pass
    
    # Fall back to environment variables
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    
    if aws_access_key and aws_secret_key:
        return {
            'aws_access_key_id': aws_access_key,
            'aws_secret_access_key': aws_secret_key,
            'region_name': aws_region
        }
    
    # Return None to use default boto3 credential chain
    return None


class LocalDatabase:
    """
    Local database for retrieving chat history data from S3 bucket.
    Supports filtering by time period and caching data locally.
    """
    
    def __init__(
        self,
        bucket_name: str = DEFAULT_S3_BUCKET,
        s3_path: str = DEFAULT_S3_CHAT_HISTORY_PATH,
        cache_dir: Optional[str] = None,
        proxy: Optional[str] = None
    ):
        """
        Initialize the local database.
        
        Args:
            bucket_name: Name of the S3 bucket (default: from config)
            s3_path: Path prefix in S3 bucket for chat history (default: from config)
            cache_dir: Optional directory to cache downloaded files locally
            proxy: Optional proxy URL (e.g., "http://127.0.0.1:7890") or None to use environment variables
        """
        # Configure proxy - boto3 will automatically use HTTP_PROXY and HTTPS_PROXY env vars
        if proxy:
            os.environ['HTTPS_PROXY'] = proxy
            os.environ['HTTP_PROXY'] = proxy
            os.environ['https_proxy'] = proxy
            os.environ['http_proxy'] = proxy
        
        self.bucket_name = bucket_name
        self.s3_path = s3_path.rstrip('/')  # Remove trailing slash if present
        
        # Get AWS credentials and create S3 client
        credentials = get_aws_credentials()
        if credentials:
            self.s3_client = boto3.client("s3", **credentials)
        else:
            # Use default boto3 credential chain (environment variables, ~/.aws/credentials, IAM role)
            self.s3_client = boto3.client("s3")
        
        self.cache_dir = cache_dir
        self._cached_data = []  # In-memory cache
        
        # Create cache directory if specified
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def list_files(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        List all chat history files from S3, optionally filtered by date range.
        
        Args:
            start_date: Optional start date for filtering (inclusive)
            end_date: Optional end date for filtering (inclusive)
            
        Returns:
            List of file metadata dictionaries with keys: Key, LastModified, Size
        """
        try:
            # List all objects under the specified prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.s3_path + "/"
            )
            
            all_files = response.get("Contents", [])
            
            # Filter JSON files
            json_files = [f for f in all_files if f["Key"].endswith(".json")]
            
            # Filter by date range if provided
            if start_date or end_date:
                filtered_files = []
                for file in json_files:
                    last_modified = file["LastModified"]
                    # Handle timezone-aware datetime
                    if hasattr(last_modified, 'replace'):
                        last_modified = last_modified.replace(tzinfo=None)
                    
                    # Apply date filters
                    if start_date and last_modified < start_date:
                        continue
                    if end_date and last_modified > end_date:
                        continue
                    
                    filtered_files.append(file)
                
                return filtered_files
            
            return json_files
            
        except Exception as e:
            print(f"Error listing files from S3: {e}")
            return []
    
    def _parse_timestamp(self, time_str: str) -> Optional[datetime]:
        """
        Parse timestamp string from chat history file.
        Handles multiple timestamp formats.
        
        Args:
            time_str: Timestamp string in various formats
            
        Returns:
            datetime object or None if parsing fails
        """
        if not time_str:
            return None
            
        # Try different timestamp formats
        formats = [
            "%Y-%m-%d %H:%M:%S",  # Standard format from save_chat_history_to_s3
            "%Y%m%d_%H%M%S",      # Format from file naming
            "%Y-%m-%dT%H:%M:%S",  # ISO format
            "%Y-%m-%dT%H:%M:%S.%f",  # ISO format with microseconds
            "%Y-%m-%dT%H:%M:%SZ",  # ISO format with Z
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _extract_file_timestamp(self, file_metadata: Dict, file_data: Dict) -> Optional[datetime]:
        """
        Extract timestamp from file metadata or file data.
        Prioritizes file data timestamp, falls back to S3 LastModified.
        
        Args:
            file_metadata: S3 file metadata
            file_data: Parsed JSON data from file
            
        Returns:
            datetime object or None
        """
        # First, try to get timestamp from file data
        if "time" in file_data:
            parsed = self._parse_timestamp(file_data["time"])
            if parsed:
                return parsed
        
        # Try timestamp field (different format)
        if "timestamp" in file_data:
            parsed = self._parse_timestamp(file_data["timestamp"])
            if parsed:
                return parsed
        
        # Fall back to S3 LastModified
        if "LastModified" in file_metadata:
            last_modified = file_metadata["LastModified"]
            if hasattr(last_modified, 'replace'):
                return last_modified.replace(tzinfo=None)
            return last_modified
        
        return None
    
    def _download_file(self, file_key: str) -> Optional[Dict]:
        """
        Download and parse a single JSON file from S3.
        
        Args:
            file_key: S3 key of the file to download
            
        Returns:
            Parsed JSON data as dictionary, or None if error
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_key
            )
            file_content = response["Body"].read().decode("utf-8")
            return json.loads(file_content)
        except Exception as e:
            print(f"Error downloading file {file_key}: {e}")
            return None
    
    def retrieve_by_time_period(
        self,
        start_date: datetime,
        end_date: datetime,
        use_cache: bool = True
    ) -> List[Dict]:
        """
        Retrieve all chat history data within a specified time period.
        
        Args:
            start_date: Start date for the time period (inclusive)
            end_date: End date for the time period (inclusive)
            use_cache: Whether to use cached data if available
            
        Returns:
            List of chat history records (dictionaries) within the time period
        """
        # Validate date range
        if start_date > end_date:
            raise ValueError("start_date must be before or equal to end_date")
        
        results = []
        
        # Get list of files in the date range
        files = self.list_files(start_date=start_date, end_date=end_date)
        
        print(f"Found {len(files)} files in the specified time period")
        
        # Download and filter each file
        for file_metadata in files:
            file_key = file_metadata["Key"]
            
            # Download file
            file_data = self._download_file(file_key)
            if not file_data:
                continue
            
            # Extract timestamp
            file_timestamp = self._extract_file_timestamp(file_metadata, file_data)
            
            if not file_timestamp:
                # If we can't determine timestamp, skip or include based on S3 metadata
                file_timestamp = file_metadata.get("LastModified")
                if hasattr(file_timestamp, 'replace'):
                    file_timestamp = file_timestamp.replace(tzinfo=None)
                if not file_timestamp or not (start_date <= file_timestamp <= end_date):
                    continue
            else:
                # Filter by timestamp from file data
                if not (start_date <= file_timestamp <= end_date):
                    continue
            
            # Add file key and metadata to the record
            file_data["_file_key"] = file_key
            file_data["_retrieved_timestamp"] = file_timestamp.isoformat() if file_timestamp else None
            file_data["_s3_last_modified"] = file_metadata.get("LastModified")
            
            results.append(file_data)
            
            # Cache if enabled
            if use_cache:
                self._cached_data.append(file_data)
        
        print(f"Retrieved {len(results)} records within the time period")
        return results
    
    def retrieve_all(self, use_cache: bool = True) -> List[Dict]:
        """
        Retrieve all chat history data from S3.
        
        Args:
            use_cache: Whether to cache the retrieved data
            
        Returns:
            List of all chat history records
        """
        return self.retrieve_by_time_period(
            start_date=datetime(2000, 1, 1),  # Very early date
            end_date=datetime.now() + timedelta(days=365),  # Far future date
            use_cache=use_cache
        )
    
    def get_by_user_id(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Retrieve chat history for a specific user ID within optional time period.
        
        Args:
            user_id: User ID to filter by
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            
        Returns:
            List of chat history records for the specified user
        """
        if start_date and end_date:
            all_data = self.retrieve_by_time_period(start_date, end_date)
        else:
            all_data = self.retrieve_all()
        
        return [record for record in all_data if record.get("user_id") == user_id]
    
    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cached_data = []
    
    def get_cached_data(self) -> List[Dict]:
        """
        Get all currently cached data.
        
        Returns:
            List of cached chat history records
        """
        return self._cached_data.copy()
    
    def save_to_local_json(
        self,
        data: List[Dict],
        output_file: str,
        exclude_metadata: bool = True
    ):
        """
        Save retrieved data to a local JSON file.
        
        Args:
            data: List of chat history records to save
            output_file: Path to output JSON file
            exclude_metadata: Whether to exclude internal metadata fields
        """
        data_to_save = data.copy()
        
        if exclude_metadata:
            # Remove internal metadata fields
            for record in data_to_save:
                record.pop("_file_key", None)
                record.pop("_retrieved_timestamp", None)
                record.pop("_s3_last_modified", None)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Saved {len(data_to_save)} records to {output_file}")
    
    def test_s3_connection(self) -> bool:
        """
        Test connection to S3 bucket.
        
        Returns:
            True if connection successful, False otherwise
        """
        print("="*60)
        print("Testing S3 Connection")
        print("="*60)
        print(f"Bucket: {self.bucket_name}")
        print(f"S3 Path: {self.s3_path}")
        
        # Check proxy settings
        proxy_env = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
        if proxy_env:
            print(f"Using proxy: {proxy_env}")
        else:
            print("No proxy configured")
        
        print()
        
        try:
            # Test 1: Check if bucket exists and is accessible
            print("Test 1: Checking bucket access...")
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print("✓ Bucket access successful")
            
            # Test 2: List objects (limit to 1 for test)
            print("Test 2: Testing list_objects_v2 operation...")
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.s3_path + "/",
                MaxKeys=1
            )
            file_count = len(response.get("Contents", []))
            print(f"✓ List operation successful (found {file_count} file(s) in test)")
            
            # Test 3: Get bucket location
            print("Test 3: Getting bucket location...")
            try:
                location = self.s3_client.get_bucket_location(Bucket=self.bucket_name)
                region = location.get('LocationConstraint', 'us-east-1')
                if region is None:
                    region = 'us-east-1'
                print(f"✓ Bucket region: {region}")
            except Exception as e:
                print(f"⚠ Could not get bucket location: {e}")
            
            print()
            print("="*60)
            print("✓ S3 Connection Test Successful")
            print("="*60)
            return True
            
        except Exception as e:
            print()
            print("="*60)
            print(f"✗ S3 Connection Test Failed: {e}")
            print("="*60)
            import traceback
            traceback.print_exc()
            return False

