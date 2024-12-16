from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import logging
from airflow.models import Variable

# Configure logging
logger = logging.getLogger(__name__)

# List of book IDs - you can modify this or load from external source
BOOK_IDS = ['2704', '2705', '2706']  # Example book IDs

# Configuration
API_KEY = 'magentic-secret-key'
BASE_URL = 'http://localhost:8000'

def ingest_books_task(book_id, **context):
    """
    Calls the ingest endpoint with a specific book ID using the required format
    """
    task_instance = context['task_instance']
    execution_date = context['execution_date']
    
    logger.info(f"Starting ingestion for book_id: {book_id}")
    logger.info(f"Task execution date: {execution_date}")
    
    url = f'{BASE_URL}/ingest/{book_id}'
    headers = {
        'X-API-Key': API_KEY
    }
    
    # Log request details
    logger.info(f"Making POST request to: {url}")
    logger.info(f"Request headers: {headers}")
    
    start_time = datetime.now()
    
    try:
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        
        # Calculate request duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Log success details
        logger.info(f"Successfully ingested book_id: {book_id}")
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response time: {duration:.2f} seconds")
        
        # Store metrics in XCom for monitoring
        task_instance.xcom_push(key=f'response_time_{book_id}', value=duration)
        task_instance.xcom_push(key=f'status_code_{book_id}', value=response.status_code)
        
        if response.text:
            logger.info(f"Response body: {response.text[:200]}...")  # Log first 200 chars of response
        
        return f"Successfully ingested book_id: {book_id}"
    
    except requests.exceptions.RequestException as e:
        # Detailed error logging
        error_msg = str(e)
        status_code = e.response.status_code if hasattr(e, 'response') and e.response else 'N/A'
        
        logger.error(f"Failed to ingest book_id {book_id}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {error_msg}")
        logger.error(f"Status code: {status_code}")
        
        if hasattr(e, 'response') and e.response and e.response.text:
            logger.error(f"Error response body: {e.response.text[:200]}...")
            
        # Store error information in XCom
        task_instance.xcom_push(key=f'error_{book_id}', value={
            'error_type': type(e).__name__,
            'error_message': error_msg,
            'status_code': status_code
        })
        
        raise Exception(f"Failed to ingest book_id {book_id}: {error_msg}")

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,  # Enable email notifications on failure
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    'ingest_books_task_dag',
    default_args=default_args,
    description='DAG to ingest books via web endpoint',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 12, 16),
    catchup=False,
    tags=['ingest', 'books']  # Add tags for better organization
) as dag:

    # Create a task for each book ID
    for book_id in BOOK_IDS:
        task = PythonOperator(
            task_id=f'ingest_book_{book_id}',
            python_callable=ingest_books_task,
            op_args=[book_id],
            provide_context=True,  # This ensures we get the context dictionary
        )
        