# MeetingIntelligence

This project is a speech-to-text pipeline that can transcribe audio files and generate a transcript of the speech. The project can also answer default and custom question asked by the user using the GPT-3.5 Turbo model.

### Overview:

This application is designed to process and transcribe audio files, then answer questions about the transcribed audio using GPT-3.5 Turbo. The application is built using Streamlit and deployed on a Amazon Web Service (AWS) instance. The project utilizes Amazon S3 for storage and Apache Airflow for managing the transcription workflow.

# Prerequisites

To run this project, you will need:

- Amazon Web Service account
- Docker
- AWS access and secret keys
- OpenAI API key
- .env file containing the AWS and OpenAI keys in the same directory as the Airflow DAGs Docker Compose and the web application (Streamlit) Docker Compose files

# Installation

- Clone the repository.
- Install the required packages by running pip install -r requirements.txt.
- Create a VM instance in Amazon Web Service.
- Create a new directory named "app" (Copy the contents of the Airflow folder from this repository into the "app" directory).
- Build Docker images for the Streamlit app and push them to Docker Hub.
- Set up the Streamlit app in the VM instance:
- Create a new directory called 'feapps3' for the Streamlit app (Copy the docker-compose.yml file from the 'feapps3' folder in this repository into the new directory).
- Ensure the .env file containing the AWS and OpenAI keys is present in both the "app" directory created for Airflow and the directory created for the Streamlit app.
- Pass the AWS and OpenAI keys as environment variables in the Docker Compose file.
- Run the Docker Compose file to start the Streamlit app.


### .env file for streamlit:
- AWS_ACCESS_KEY=<aws_access_key> <- should be given in double quotes ("")
- AWS_SECRET_KEY=<aws_secret_key> <- should be given in double quotes ("")
- AWS_LOGS_ACCESS_KEY=<aws_logs_access_key> <- should be given in double quotes ("")
- AWS_LOGS_SECRET_KEY=<aws_logs_secret_key> <- should be given in double quotes ("")
- OPENAI_API_KEY=<openai_api_key> <- should be given in double quotes ("")
- USER_BUCKET_NAME=<USER_BUCKET_NAME> <- should be given in double quotes ("")
