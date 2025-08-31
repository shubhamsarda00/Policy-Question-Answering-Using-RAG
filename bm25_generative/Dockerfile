FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r /app/requirements.txt


EXPOSE 8000


RUN python /app/bm25_generative_app.py & \
	echo 'Waiting for API to be up...' && \
    sleep 240 && \
	echo 'Running tests...' && \
    pytest -v /app/test.py && \
	echo 'Running the evaluations...' && \
	python /app/evaluate.py && \
	echo 'Evaluation results written to evaluation_results.txt' 


CMD ["python", "bm25_generative_app.py"]
