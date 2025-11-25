from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# --- DAG Configuration ---
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    "titanic_pipeline",
    default_args=default_args,
    description="Preprocess -> Train -> Evaluate Titanic Model",
    schedule_interval="*/5 * * * *",  # every 5 minutes
    start_date=datetime(2025, 11, 25),
    catchup=False,
)

# --- File Paths ---
RAW_DATA = "/opt/airflow/data/titanic.csv"
CLEAN_DATA = "/opt/airflow/data/cleaned_data.csv"

# --- Tasks ---
preprocess_task = BashOperator(
    task_id="preprocess_data",
    bash_command=f"python /opt/airflow/scripts/preprocess.py {RAW_DATA} {CLEAN_DATA}",
    dag=dag,
)

train_task = BashOperator(
    task_id="train_model",
    bash_command=f"python /opt/airflow/scripts/train_model.py {CLEAN_DATA}",
    dag=dag,
)

evaluate_task = BashOperator(
    task_id="evaluate_model",
    bash_command=f"python /opt/airflow/scripts/evaluate_model.py {CLEAN_DATA}",
    dag=dag,
)

# --- Task Dependencies ---
preprocess_task >> train_task >> evaluate_task
