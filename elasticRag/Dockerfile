# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install the required packages for Elasticsearch
RUN apt-get update && apt-get install -y \
    wget \
    gnupg2 \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Download and install Elasticsearch
RUN wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.10-linux-x86_64.tar.gz \
    && tar -xzf elasticsearch-7.17.10-linux-x86_64.tar.gz \
    && rm elasticsearch-7.17.10-linux-x86_64.tar.gz

# Create a non-root user
RUN adduser --disabled-password --gecos '' elasticsearch_user

# Change ownership of the Elasticsearch directory
RUN chown -R elasticsearch_user:elasticsearch_user /app/elasticsearch-7.17.10


# Copy the elasticsearch.yml configuration into the Elasticsearch config directory
COPY elasticsearch.yml /app/elasticsearch-7.17.10/config/elasticsearch.yml


# Copy the current directory contents into the container
COPY . /app

RUN chown -R elasticsearch_user:elasticsearch_user /app

# Install the required packages
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the port the app runs on (adjust based on your app)
EXPOSE 8080 9200

# Switch to the non-root user
USER elasticsearch_user


# Start Elasticsearch and Flask app, then run the tests and evaluations
RUN echo 'Starting Elasticsearch...' && \
    /app/elasticsearch-7.17.10/bin/elasticsearch & \
    echo 'Waiting for Elasticsearch to be up...' && \
    sleep 45 && \
    echo 'Starting elasticRag_app.py...' && \
 	python /app/elasticRag_app.py & \
	echo 'Waiting for API to be up...' && \
    sleep 240 && \
	echo 'Running tests...' && \
    pytest -v /app/test.py && \
	echo 'Running the evaluations...' && \
	python /app/evaluate.py && \
	echo 'Evaluation results written to evaluation_results.txt' 

# Start Elasticsearch and Flask app when the container is run
CMD sh -c "\
    /app/elasticsearch-7.17.10/bin/elasticsearch & \
    echo 'Waiting for Elasticsearch to be up...' && \
    sleep 45 && \
    python /app/elasticRag_app.py"
