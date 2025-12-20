import streamlit as st
from datetime import datetime, timedelta
from local_database import LocalDatabase
import json
import pandas as pd
import os
from dotenv import load_dotenv
import re
from collections import Counter
import io

# Try to import wordcloud and matplotlib, with fallback if not available
try:
    from wordcloud import WordCloud
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for Streamlit
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    # Note: Install wordcloud and matplotlib for word cloud functionality
    # pip install wordcloud matplotlib

# Load environment variables (for AWS credentials)
load_dotenv()

st.set_page_config(
    page_title="SARHAchat Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Password authentication - Improved version
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def verify_password():
        """Checks whether a password entered by the user is correct."""
        # Get password from secrets (Streamlit Cloud) or environment variable
        correct_password = st.secrets.get("APP_PASSWORD") or os.environ.get("APP_PASSWORD", "default_password_change_me")
        
        if st.session_state.get("password_input", "") == correct_password:
            st.session_state["password_correct"] = True
            st.session_state["password_input"] = ""  # Clear password
            st.session_state["password_error"] = False  # Clear error
        else:
            st.session_state["password_correct"] = False
            st.session_state["password_error"] = True  # Set error flag
    
    # Initialize session state
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
        st.session_state["password_error"] = False
    
    # Return True if password is correct
    if st.session_state["password_correct"]:
        return True
    
    # Custom CSS to remove top padding and make login form top-aligned
    st.markdown("""
    <style>
    /* Remove top padding/margin */
    .main > div {
        padding-top: 0rem !important;
    }
    
    /* Remove Streamlit header spacing */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* Login container styling */
    .login-container {
        padding: 2rem 1rem;
        max-width: 500px;
        margin: 0 auto;
    }
    
    .login-title {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .password-input-container {
        margin-bottom: 1rem;
    }
    
    .enter-button-container {
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Login form - positioned at top
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    st.markdown('<div class="login-title">', unsafe_allow_html=True)
    st.title("🔐 SARHAchat Analytics")
    st.markdown("---")
    st.markdown("### Please enter your password to continue")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Password input
    st.markdown('<div class="password-input-container">', unsafe_allow_html=True)
    password = st.text_input(
        "Password", 
        type="password", 
        key="password_input",
        label_visibility="visible",
        placeholder="Enter your password"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enter button
    st.markdown('<div class="enter-button-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Enter", type="primary", use_container_width=True, on_click=verify_password):
            pass  # Verification happens in verify_password callback
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show warning only if password was attempted and incorrect
    if st.session_state.get("password_error", False):
        st.warning("❌ Incorrect password. Please try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return False

# Check authentication before showing the app
if not check_password():
    st.stop()  # Stop execution if password is incorrect

# Initialize database
@st.cache_resource
def init_database(proxy=None):
    if proxy:
        return LocalDatabase(proxy=proxy)
    return LocalDatabase()

# Topic keywords for education mode conversations
EDUCATION_TOPICS = {
    "abortion": ["abortion", "abort", "terminat", "pregnancy termination"],
    "birth_control": ["birth control", "contraception", "contraceptive", "pill", "iud", "implant", "condom", "ring", "patch"],
    "pregnancy": ["pregnancy", "pregnant", "gestation", "trimester", "fetus", "fetal"],
    "sti": ["sti", "std", "sexually transmitted", "chlamydia", "gonorrhea", "syphilis", "hiv", "herpes", "hpv"],
    "periods": ["period", "menstrual", "menstruation", "cycle", "pms", "cramps", "bleeding"],
    "fertility": ["fertility", "fertile", "ovulation", "ovulate", "conceive", "conception"],
    "sexual_health": ["sexual health", "sex", "sexual", "intercourse", "safe sex"],
    "reproductive_health": ["reproductive", "reproduction", "uterus", "ovary", "cervix"],
    "emergency_contraception": ["emergency contraception", "plan b", "morning after", "ella"],
    "testing": ["test", "testing", "screen", "screening", "diagnosis"]
}

# Stop words for word cloud
STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'from', 'up', 'about', 'into', 'through', 'during', 'including', 'until', 'against',
    'among', 'throughout', 'despite', 'towards', 'upon', 'concerning', 'to', 'of', 'in',
    'for', 'on', 'at', 'by', 'with', 'from', 'up', 'about', 'into', 'through', 'during',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may',
    'might', 'must', 'can', 'cannot', 'can\'t', 'don\'t', 'doesn\'t', 'didn\'t', 'won\'t',
    'wouldn\'t', 'shouldn\'t', 'couldn\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t',
    'haven\'t', 'hasn\'t', 'hadn\'t', 'what', 'which', 'who', 'whom', 'whose', 'where',
    'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 'just', 'now', 'then', 'here', 'there', 'when', 'where', 'why', 'how',
    'hi', 'hello', 'hey', 'thanks', 'thank', 'please', 'yes', 'no', 'ok', 'okay',
    'can', 'help', 'question', 'questions', 'answer', 'answers', 'ask', 'asking'
}

def extract_topic_from_chat_history(chat_history):
    """
    Extract topic from education mode chat history using keyword matching.
    Returns the most relevant topic or 'other' if no match found.
    """
    if not chat_history or not isinstance(chat_history, list):
        return "other"
    
    # Combine all content from chat_history
    all_text = ""
    for msg in chat_history:
        if isinstance(msg, dict):
            content = msg.get("content", "")
        else:
            content = str(msg)
        all_text += " " + content.lower()
    
    # Count topic matches
    topic_scores = {}
    for topic, keywords in EDUCATION_TOPICS.items():
        score = 0
        for keyword in keywords:
            # Count occurrences of keyword in text
            score += len(re.findall(r'\b' + re.escape(keyword) + r'\b', all_text))
        if score > 0:
            topic_scores[topic] = score
    
    # Return topic with highest score, or 'other' if no matches
    if topic_scores:
        return max(topic_scores.items(), key=lambda x: x[1])[0]
    return "other"

def clean_text_for_wordcloud(text):
    """
    Clean text for word cloud generation - remove HTML tags, URLs, and normalize.
    """
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove special characters but keep spaces and basic punctuation
    text = re.sub(r'[^\w\s\'-]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_wordcloud(text, title="Word Cloud"):
    """
    Generate a word cloud from text and return as image.
    """
    if not WORDCLOUD_AVAILABLE:
        return None
    
    if not text or len(text.strip()) < 10:
        return None
    
    # Clean text
    cleaned_text = clean_text_for_wordcloud(text)
    
    # Filter out stop words and short words
    words = cleaned_text.lower().split()
    filtered_words = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
    
    if not filtered_words:
        return None
    
    # Join filtered words
    text_for_cloud = ' '.join(filtered_words)
    
    # Generate word cloud
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            relative_scaling=0.5,
            colormap='viridis'
        ).generate(text_for_cloud)
        
        # Convert to image
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=20)
        
        # Convert to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        return None

# Function to determine mode based on chat_history
def determine_mode(record):
    """
    Determine the mode (education or consultation) based on chat_history[2]
    Checks the content of the third message (index 2) in the chat history.
    """
    chat_history = record.get("chat_history", [])
    
    # Check if chat_history has at least 3 items (index 0, 1, 2)
    if len(chat_history) < 3:
        return "unknown"
    
    # Get the third message (index 2)
    third_message = chat_history[2]
    
    # Extract content from the message
    # Handle different possible structures: dict with 'content' key, or direct string
    if isinstance(third_message, dict):
        content = third_message.get("content", "")
        if not content:
            # Try alternative keys
            content = third_message.get("message", "") or third_message.get("text", "")
    elif isinstance(third_message, str):
        content = third_message
    else:
        content = str(third_message)
    
    # Normalize content for comparison (strip whitespace)
    content = content.strip()
    
    # Check for education mode
    if content.startswith("Great! I can help answer questions about sexual and reproductive health."):
        return "education"
    
    # Check for consultation mode
    if content.startswith("Hey there! I'd like to start by asking—are you thinking about using contraception?"):
        return "consultation"
    
    return "unknown"

# Initialize database - proxy is optional and only used if set in environment
# Streamlit Cloud typically doesn't need proxy
proxy_url = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
db = init_database(proxy=proxy_url if proxy_url else None)

st.title("SARHAchat Analytics")
st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    
    # Date range selection - exactly like test.ipynb
    st.subheader("Date Range")
    
    # Simple: just select number of days, exactly like test.ipynb
    # test.ipynb: end_date = datetime.now(), start_date = end_date - timedelta(days=3)
    days_options = [1, 3, 7, 14, 30, 90, 365]
    selected_days = st.selectbox(
        "Retrieve last N days",
        days_options,
        index=2,  # Default to 7 days
        format_func=lambda x: f"Last {x} days" if x < 365 else "Last year"
    )
    
    # Show preview of date range
    preview_end = datetime.now()
    preview_start = preview_end - timedelta(days=selected_days)
    st.info(f"Will retrieve from {preview_start.strftime('%Y-%m-%d %H:%M:%S')} to {preview_end.strftime('%Y-%m-%d %H:%M:%S')}")
    
    st.markdown("---")
    st.subheader("Filters")
    
    filter_user_id = st.text_input("Filter by User ID (optional)", "")
    
    st.markdown("---")
    
    # Retrieve button - calculate datetime exactly like test.ipynb when clicked
    if st.button("Retrieve Data", type="primary", use_container_width=True):
        # Exactly match test.ipynb: end_date = datetime.now(), start_date = end_date - timedelta(days=3)
        end_datetime = datetime.now()
        start_datetime = end_datetime - timedelta(days=selected_days)
        
        # Debug: print like test.ipynb
        print(f"[Web] end_date=datetime.now()={end_datetime}, start_date=end_date-timedelta(days={selected_days})={start_datetime}")
        print(f"[Web] bucket={db.bucket_name}, path={db.s3_path}")
        
        st.session_state.retrieve_clicked = True
        st.session_state.start_date = start_datetime
        st.session_state.end_date = end_datetime
        st.session_state.filter_user_id = filter_user_id
        st.session_state.selected_days = selected_days
    
    # Test connection button
    if st.button("Test S3 Connection", use_container_width=True):
        try:
            # Test by listing files without date filter
            files = db.list_files()
            st.success(f"Connection successful! Found {len(files)} total files in S3")
            if len(files) > 0:
                st.write(f"Sample files (first 5):")
                for f in files[:5]:
                    st.write(f"- {f.get('Key', 'N/A')}")
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")
            st.exception(e)
    
    # Clear cache button
    if st.button("Clear Cache", use_container_width=True):
        db.clear_cache()
        st.cache_resource.clear()
        st.success("Cache cleared!")
        st.rerun()
    
    st.markdown("---")
    st.subheader("Info")
    st.info(f"Bucket: {db.bucket_name}")
    st.info(f"Path: {db.s3_path}")

# Main content area
if "retrieve_clicked" not in st.session_state:
    st.session_state.retrieve_clicked = False

if st.session_state.retrieve_clicked:
    start_datetime = st.session_state.start_date
    end_datetime = st.session_state.end_date
    filter_user_id = st.session_state.filter_user_id
    selected_days = st.session_state.get('selected_days', 7)
    last_range = st.session_state.get("last_range")
    needs_fetch = last_range != (start_datetime, end_datetime) or "retrieved_data" not in st.session_state
    
    if needs_fetch:
        with st.spinner(f"Retrieving data from {start_datetime.date()} to {end_datetime.date()}..."):
            try:
                # Show debug info - exactly like test.ipynb
                with st.expander("Debug Info", expanded=False):
                    st.write(f"**Exactly like test.ipynb:**")
                    st.code(f"end_date = datetime.now()\nstart_date = end_date - timedelta(days={selected_days})")
                    st.write(f"Start datetime: {start_datetime}")
                    st.write(f"End datetime: {end_datetime}")
                    st.write(f"Bucket: {db.bucket_name}")
                    st.write(f"S3 Path: {db.s3_path}")
                    
                    # First, list files to see what we're working with
                    files = db.list_files(start_date=start_datetime, end_date=end_datetime)
                    st.write(f"Files found in S3: {len(files)}")
                    if len(files) > 0 and len(files) <= 10:
                        for f in files:
                            st.write(f"- {f.get('Key', 'N/A')} (Modified: {f.get('LastModified', 'N/A')})")
                
                # Retrieve data - exactly like test.ipynb: data = db.retrieve_by_time_period(start_date, end_date)
                print(f"[Web] Calling retrieve_by_time_period({start_datetime}, {end_datetime})")
                retrieved_data = db.retrieve_by_time_period(start_datetime, end_datetime)
                print(f"[Web] Result: {len(retrieved_data)} records")
                
                # Filter by user ID if specified
                if filter_user_id:
                    retrieved_data = [record for record in retrieved_data if record.get("user_id") == filter_user_id]
                
                # Determine mode for each record based on chat_history[2]
                for record in retrieved_data:
                    record["mode"] = determine_mode(record)
                
                # Store in session state
                st.session_state.retrieved_data = retrieved_data
                st.session_state.retrieved_count = len(retrieved_data)
                st.session_state.last_range = (start_datetime, end_datetime)
                
            except Exception as e:
                st.error(f"Error retrieving data: {str(e)}")
                st.exception(e)
                st.session_state.retrieved_data = []
                st.session_state.retrieved_count = 0

# Display results
if "retrieved_data" in st.session_state and st.session_state.retrieved_data:
    data = st.session_state.retrieved_data
    count = st.session_state.retrieved_count
    
    # Summary statistics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Records", count)
    
    unique_users = len(set(record.get("user_id", "unknown") for record in data))
    with col2:
        st.metric("Unique Users", unique_users)
    
    total_messages = sum(len(record.get("chat_history", [])) for record in data)
    with col3:
        st.metric("Total Messages", total_messages)
    
    # Calculate average messages per record
    avg_messages = total_messages / count if count > 0 else 0
    with col4:
        st.metric("Avg Messages/Record", f"{avg_messages:.1f}")
    
    # Mode statistics
    education_count = sum(1 for record in data if record.get("mode") == "education")
    consultation_count = sum(1 for record in data if record.get("mode") == "consultation")
    with col5:
        st.metric("Mode Distribution", f"Edu: {education_count}, Con: {consultation_count}")
    
    st.markdown("---")
    
    # Table View only
    table_data = []
    for i, record in enumerate(data):
        table_data.append({
            "User ID": record.get("user_id", "N/A"),
            "Timestamp": record.get("time") or record.get("timestamp", "N/A"),
            "Messages": len(record.get("chat_history", [])),
            "Mode": record.get("mode", "unknown")
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, height=400)
    
    start_datetime = st.session_state.start_date
    end_datetime = st.session_state.end_date
    
    # Custom CSS to make button more prominent
    st.markdown("""
    <style>
    .download-button-container .stDownloadButton > button {
        font-size: 18px !important;
        font-weight: bold !important;
        height: 60px !important;
        background-color: #FF4B4B !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        width: 100% !important;
    }
    .download-button-container .stDownloadButton > button:hover {
        background-color: #FF6B6B !important;
        transform: scale(1.02);
        transition: all 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Export Data section

    
    # Prepare data for download
    # Remove internal metadata fields before exporting
    export_data = []
    for record in data:
        export_record = record.copy()
        # Remove internal metadata
        export_record.pop("_file_key", None)
        export_record.pop("_retrieved_timestamp", None)
        export_record.pop("_s3_last_modified", None)
        export_data.append(export_record)
    
    # Convert to JSON string
    json_data = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
    
    # Generate filename
    json_filename = f"retrieved_chat_history_{start_datetime.date()}_to_{end_datetime.date()}.json"
    
    # Download button
    st.markdown('<div class="download-button-container">', unsafe_allow_html=True)
    st.download_button(
        label="💾 Download Data",
        data=json_data,
        file_name=json_filename,
        mime="application/json",
        type="primary",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Statistics for consultation mode contraceptives_prescribed
    st.markdown("---")
    st.subheader("[Consultation] Contraceptives Prescribed Statistics")
    
    # Filter consultation mode records
    consultation_records = [record for record in data if record.get("mode") == "consultation"]
    
    if consultation_records:
        # Count contraceptives_prescribed methods
        method_counts = {}
        for record in consultation_records:
            contraceptives = record.get("contraceptives_prescribed", [])
            if isinstance(contraceptives, list):
                for method in contraceptives:
                    if method:  # Skip empty strings/None
                        method_counts[method] = method_counts.get(method, 0) + 1
        
        if method_counts:
            # Create DataFrame for display
            stats_data = {
                "Method": list(method_counts.keys()),
                "Times": list(method_counts.values())
            }
            stats_df = pd.DataFrame(stats_data)
            stats_df = stats_df.sort_values("Times", ascending=False)
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No contraceptives prescribed found in consultation mode records.")
    else:
        st.info("No consultation mode records found.")
    
    # Statistics for education mode
    st.markdown("---")
    st.subheader("[Education] Topic Statistics")
    
    # Filter education mode records
    education_records = [record for record in data if record.get("mode") == "education"]
    
    if education_records:
        # Extract topics from each education conversation
        topic_counts = {}
        for record in education_records:
            chat_history = record.get("chat_history", [])
            topic = extract_topic_from_chat_history(chat_history)
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        if topic_counts:
            # Create DataFrame for display
            stats_data = {
                "Topic": [t.replace("_", " ").title() for t in topic_counts.keys()],
                "Times": list(topic_counts.values())
            }
            stats_df = pd.DataFrame(stats_data)
            stats_df = stats_df.sort_values("Times", ascending=False)
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.info("No topics extracted from education mode records.")
        
        # Word clouds for education mode
        st.markdown("---")
        st.subheader("[Education] Word Clouds")
        
        if not WORDCLOUD_AVAILABLE:
            st.warning("WordCloud library not available. Install with: `pip install wordcloud matplotlib`")
        else:
            # Collect all system and user messages
            system_texts = []
            user_texts = []
            
            for record in education_records:
                chat_history = record.get("chat_history", [])
                for msg in chat_history:
                    if isinstance(msg, dict):
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "system" and content:
                            system_texts.append(content)
                        elif role == "user" and content:
                            user_texts.append(content)
            
            # Generate word clouds
            col1, col2 = st.columns(2)
            
            with col1:
                if system_texts:
                    combined_system_text = " ".join(system_texts)
                    wordcloud_img = generate_wordcloud(combined_system_text, "System Messages")
                    if wordcloud_img:
                        st.image(wordcloud_img, use_container_width=True)
                    else:
                        st.info("Not enough text for system word cloud.")
                else:
                    st.info("No system messages found.")
            
            with col2:
                if user_texts:
                    combined_user_text = " ".join(user_texts)
                    wordcloud_img = generate_wordcloud(combined_user_text, "User Messages")
                    if wordcloud_img:
                        st.image(wordcloud_img, use_container_width=True)
                    else:
                        st.info("Not enough text for user word cloud.")
                else:
                    st.info("No user messages found.")
    else:
        st.info("No education mode records found.")

elif st.session_state.retrieve_clicked:
    st.info("No data found for the specified criteria. Please adjust your date range or filters.")

else:
    st.info("Please select a date range and click 'Retrieve Data' to start.")



