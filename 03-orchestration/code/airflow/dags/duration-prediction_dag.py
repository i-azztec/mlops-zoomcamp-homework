from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from duration_prediction_linear_reg import run  

default_args = {
    "owner": "airflow",
    "start_date": datetime(2023, 1, 1),
}

def train_model_task(year, month, **context):
    run(year=year, month=month)

with DAG(
    dag_id="duration_prediction_linear_reg_dag",
    schedule=None,  
    default_args=default_args,
    catchup=False,
    tags=["mlops", "linear_regression"],
) as dag:

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=train_model_task,
        op_kwargs={"year": 2023, "month": 3},  
    )