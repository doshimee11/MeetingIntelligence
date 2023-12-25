import os
import io
import time
import boto3
import openai
import shutil
import textwrap
import urllib.parse
from PIL import Image
import streamlit as st
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

# Create an AWS S3 client to store in user bucket
s3Client = boto3.client('s3',
                    region_name = 'us-east-1',
                    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY'),
                    aws_secret_access_key = os.environ.get('AWS_SECRET_KEY')
                    )

# Create an AWS S3 Resource to access resources available in user bucket
s3Res = boto3.resource('s3',
                        region_name = 'us-east-1',
                        aws_access_key_id = os.environ.get('AWS_ACCESS_KEY'),
                        aws_secret_access_key = os.environ.get('AWS_SECRET_KEY'))

# Create an AWS S3 Clients to log the user activity on the application
clientLogs = boto3.client('logs',
                        region_name='us-east-1',
                        aws_access_key_id = os.environ.get('AWS_LOGS_ACCESS_KEY'),
                        aws_secret_access_key = os.environ.get('AWS_LOGS_SECRET_KEY')
                        )

# Defining User Bucket to store file
user_s3_bucket = os.environ.get('USER_BUCKET_NAME')
user_bucket_access = s3Res.Bucket(user_s3_bucket)
openai.api_key = os.environ.get('OPENAI_API_KEY')

def write_process_logs(message: str):
    """
    Write's user activity logs to Audio-File-Process log stream in Interview-Summary-Application log group.
    
    Parameters:
    - message: The message to be logged in the application.
    
    Returns:
    - None
    """
    
    clientLogs.put_log_events(
        logGroupName = "Interview-Summary-Application",
        logStreamName = "Audio-File-Process",
        logEvents = [
            {
                'timestamp' : int(time.time() * 1e3),
                'message' : message
            }
        ]
    )


def write_qna_logs(message: str):
    """
    Write's user activity logs to Question-Answer log stream in Interview-Summary-Application log group.
    
    Parameters:
    - message: The message to be logged in the application.
    
    Returns:
    - None
    """
    
    clientLogs.put_log_events(
        logGroupName = "Interview-Summary-Application",
        logStreamName = "Question-Answer",
        logEvents = [
            {
                'timestamp' : int(time.time() * 1e3),
                'message' : message
            }
        ]
    )


def upload_mp3(audio_file):
    """
    Uploads the MP3 file to the user's S3 bucket.
    
    Parameters:
    - audio_file: The MP3 file to be uploaded.
    
    Returns:
    - audio_file (if the format is supported)
    - None (if the format is unsupported)
    """
    
    # Check if the audio file format is supported (MP3)
    if audio_file.type == "audio/mpeg" or "audio/x-mpeg" or "audio/mpeg3" or "audio/x-mpeg-3":
        
        # Define the S3 object key for storing the file in the user's bucket
        file = (audio_file.name.replace('.mp3', ''))
        s3_object_key = f'{file}/{audio_file.name}'
        
        # Upload the audio file to the S3 bucket
        s3Res.Bucket(user_s3_bucket).put_object(Key=s3_object_key, Body=audio_file.read())
        write_process_logs(f'Audio file {audio_file.name} is being uploaded to bucket {user_s3_bucket}')

        # Create the new folder
        os.makedirs('audio_files', exist_ok=True)
        
        # Create the file path by joining the temporary folder path with the file name
        file_path = os.path.join('audio_files', f"{audio_file.name}")

        # Save the file to the temporary folder
        with open(file_path, "wb") as f:
            f.write(audio_file.getbuffer())
        
        # return file_path
        return open(file_path, "rb")

    else:
        # Display an error message for unsupported audio format
        st.error('Unsupported audio format. Please upload an MP3 file.')
        st.write('You can convert your file from the following link:')
        st.write('https://online-audio-converter.com/')
        write_process_logs(f'Unsupported audio file format')
        
        return None


def convert_mp3_chunks(audio_file):
    """
    Converts the MP3 file to chunks of 20 seconds.
    
    Parameters:
    - audio_file: The uploaded MP3 file to be converted.
    
    Returns:
    - None
    """
    
    # Define the duration of each chunk in milliseconds
    chunk_duration = 20 * 60 * 1000
    
    # Load the audio file using the pydub library
    audio = AudioSegment.from_mp3(audio_file.name)
    
    # Calculate the total duration of the audio file
    total_duration = len(audio)
    
    # Iterate over the audio file in chunks of the specified duration
    for start_time in range(0, total_duration, chunk_duration):
        end_time = min(start_time + chunk_duration, total_duration)
        
        # Extract the chunk from the audio
        chunk = audio[start_time:end_time]

        # Generate a unique filename for each chunk
        chunk_filename = f'chunk_{start_time}.mp3'
        
        # Define the S3 object key for storing the chunk in the user's bucket
        file_name_split = (audio_file.name).split("/")[1]
        file = (file_name_split.replace('.mp3', ''))
        s3_object_key_chunks = f'{file}/20MinProcessed/{chunk_filename}'
        
        # Upload the chunk to the S3 bucket
        s3Res.Bucket(user_s3_bucket).put_object(Key=s3_object_key_chunks, Body=chunk.export())


def process_file():
    """
    Processes the uploaded audio file.

    Returns:
    - None
    """

    # Display the section header
    st.subheader('Processing Audio File')

    # Allow the user to upload an audio file (MP3)
    file = st.file_uploader('Please attach an audio file', type=['mp3'])
    
    if file is not None:
        # Display information about the uploaded file
        st.write('Filename:', file.name)
        st.write('File size:', round((file.size) / (1024 * 1024), 2), 'MB')
        st.write('File Type:', file.type)

        # Upload the audio file to S3 and convert it
        with st.spinner('Uploading file...'):
            write_process_logs(f'Uploading file in process...')
            audio_file = upload_mp3(file)
        write_process_logs(f'Successfully uploaded audio file')
        
        if audio_file:
            # Convert the audio file into chunks
            with st.spinner('Converting audio file into chunks...'):
                write_process_logs(f'Converting audio file chunks in process...')
                convert_mp3_chunks(audio_file)
            write_process_logs(f'Successfully converted audio file to process')
            
            # Transcribe the chunks using OpenAI Whisper model
            with st.spinner('Transcribing...'):
                write_process_logs(f'Transcribing in process...')
                transcript = []
                objects = []
                paginator = s3Client.get_paginator('list_objects_v2')
                folder_file = (file.name.replace('.mp3', ''))
                for result in paginator.paginate(Bucket=user_s3_bucket, Prefix=f'{folder_file}/20MinProcessed/'):
                    if 'Contents' in result:
                        objects.extend(result['Contents'])

                # Process each chunk with the Whisper model
                for obj in objects:
                    # Retrieve the object data from S3
                    media_file = s3Client.get_object(Bucket=user_s3_bucket, Key=obj['Key'])
                    audio_data = media_file['Body'].read()

                    # Create a temporary file-like object with a name attribute
                    media_file = io.BytesIO(audio_data)
                    media_file.name = obj['Key']

                    # Transcribe the chunk using OpenAI Whisper model
                    response = openai.Audio.transcribe("whisper-1", media_file)
                    transcripts = response['text']
                    transcript.append(transcripts)

            # Merge the transcriptions into a single transcript
            merged_transcript = " ".join(transcript)
            write_process_logs(f'Transcripts merged for each chunk of audio file processed')

            # Save the transcript as a text file
            file_name = file.name.replace('/', '_') + '.txt'
            with open(file_name, "w") as transcript_file:
                transcript_file.write(merged_transcript)

            # Upload the transcript file to S3
            script_name = (file.name.replace('.mp3', ''))
            transcript_object_key = f'{script_name}/Transcript/{script_name}_Interview_Transcript.txt'
            s3Res.Bucket(user_s3_bucket).upload_file(file_name, transcript_object_key)
            st.success('Transcript generated successfully')
            write_process_logs(f'Transcript generated successfully and uploaded at the location: {transcript_object_key}')

            generated_transcript_url = 'https://transcribeproject.s3.amazonaws.com/' + transcript_object_key

            # Encode the URL to replace spaces with %20
            generated_encoded_url = urllib.parse.quote(generated_transcript_url, safe=':/')
            
            st.write('')
            st.write('Link to download the Transcript generated:')
            write_process_logs(f'Link for generated transcript: {generated_encoded_url}')
            st.write(generated_encoded_url)

            # Remove the generated transcript file locally
            os.remove(file_name)
            
            # Remove the "audio_files" directory and its contents
            shutil.rmtree('audio_files')
        
        else:
            st.error('Error converting your audio file...')
            write_process_logs(f'Error converting your audio file...')

    else:
        st.warning('Please upload the audio file to process it...')


def get_key_from_value(dictionary, value):
    """
    Get the key from a dictionary based on a given value.

    Args:
    - dictionary (dict): The dictionary to search.
    - value: The value to find in the dictionary.

    Returns:
    - The key corresponding to the given value, or None if not found.
    """

    for key, val in dictionary.items():
        if val == value:
            return key

    return None


# Use the st.cache decorator to enable caching
@st.cache_data
def summarize_transcript(transcript):
    """
    Generates an AI-powered response using OpenAI's GPT-3.5 Turbo model.

    Parameters:
    prompt (str): The text prompt to generate a response for.

    Returns:
    str: The generated response from the GPT-3.5 Turbo model.
    """

    prompt_chunks = textwrap.wrap(transcript, 4096)
    generated_text = ""
    
    for chunk in prompt_chunks:
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": chunk},
            ],
            max_tokens = 2148,
            n = 1,
            stop = None,
            temperature = 0.6,
        )

        generated_chunk = response.choices[0].message["content"].strip()
        generated_text += generated_chunk + ''
    
    # Combine all the generated chunks into a final complete answer
    final_answer = "".join(generated_text)
    
    return final_answer


def get_gpt_answer(prompt):
    """
    Generates an AI-powered response using OpenAI's GPT-3.5 Turbo model.

    Parameters:
    prompt (str): The text prompt to generate a response for.

    Returns:
    str: The generated response from the GPT-3.5 Turbo model.
    """

    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens = 1960,
        n = 1,
        stop = None,
        temperature = 0.6,
    )
    
    return response.choices[0].message["content"].strip()


def render_conversation(qa, role):
    """
    Renders a conversation message in a Streamlit app.

    Parameters:
    qa (str): The message to render.
    role (str): The role of the conversation participant (either 'user' or 'system').

    Returns:
    None
    """

    if role == "user":
        st.markdown(
            f'<p style="color:#1abc9c;font-size:18px;font-weight:bold;margin-bottom:2px;">{role.capitalize()}:</p>',
            unsafe_allow_html = True,
        )
        st.markdown(
            f'<p style="color:#ffffff;font-size:16px;margin-top:0px;">{qa[len(role) + 2:]}</p>',
            unsafe_allow_html = True,
        )
    else:
        st.markdown(
            f'<p style="color:#f63366;font-size:18px;font-weight:bold;margin-bottom:2px;">{role.capitalize()}:</p>',
            unsafe_allow_html = True,
        )
        st.markdown(
            f'<p style="color:#ffffff;font-size:16px;margin-top:0px;">{qa[len(role) + 2:]}</p>',
            unsafe_allow_html = True,
        )


def question_answer():
    """
    Gives the summary of the audio files processed

    Returns:
    - None
    """

    st.subheader('Question & Answer')
    st.write('')

    # Retrieve the list of objects in the bucket
    response = s3Client.list_objects(Bucket = user_s3_bucket)
    
    transcript_dict = {}
    if 'Contents' in response:
        objects = response['Contents']

        # Iterate over the objects and append their keys to the list
        for obj in objects:
            object_key = obj['Key']
            if object_key.endswith('_Transcript.txt'):
                transcript_dict[object_key] = object_key.split('/')[-1]
    
    if "previous_selected_file" not in st.session_state:
        st.session_state.previous_selected_file = None

    if "file_conversation_history" not in st.session_state:
        st.session_state.file_conversation_history = {}

    selected_transcript = st.selectbox('Please select the transcript from below list', [''] + list(transcript_dict.values()))

    if st.session_state.previous_selected_file != selected_transcript:
        # Save the current conversation history
        if st.session_state.previous_selected_file is not None:
            st.session_state.file_conversation_history[
                st.session_state.previous_selected_file
            ] = st.session_state.conversation_history
        
        # Retrieve the conversation history for the newly selected file or create an empty list
        st.session_state.conversation_history = (
            st.session_state.file_conversation_history.get(selected_transcript, [])
        )

        st.session_state.previous_selected_file = selected_transcript
        st.experimental_rerun()

    if selected_transcript == '':
        st.write('')

    elif selected_transcript:
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
        
        selected_transcript_key = get_key_from_value(transcript_dict, selected_transcript)
        transcript_url = 'https://transcribeproject.s3.amazonaws.com/' + selected_transcript_key

        # Encode the URL to replace spaces with %20
        encoded_url = urllib.parse.quote(transcript_url, safe=':/')
        
        st.write('Download the transcript from the link below:')
        st.write(encoded_url)

        # Read the file from the S3 bucket
        s3_object = s3Res.Object(user_s3_bucket, selected_transcript_key)
        transcript_content = s3_object.get()['Body'].read().decode('utf-8')
        st.text_area('Transcript', value = transcript_content, height = 200, disabled = True)
        summarize_content = summarize_transcript(transcript_content)
        write_qna_logs(f'Generated summary of the transcript {selected_transcript}')

        question = st.text_input('Please enter your Question !!!')
        for i, qna in enumerate(st.session_state.conversation_history):
            role = "user" if i % 2 == 0 else "gpt-3.5-turbo"
            render_conversation(qna, role)

        if st.button("Submit New Question"):
            prompt = f"{summarize_content}\n\n"
            for qna in st.session_state.conversation_history:
                prompt += f"{qna}\n"
            prompt += f"User: {question}\ngpt-3.5-turbo:"
            answer = get_gpt_answer(prompt)
            
            st.session_state.conversation_history.append(f"User: {question}")
            write_qna_logs(f'User: {question}')
            st.session_state.conversation_history.append(f"gpt-3.5-turbo: {answer}\n")
            write_qna_logs(f'gpt-3.5-turbo: {answer}')

            st.text_area(f"gpt-3.5-turbo:", value=answer)

        if st.button("Save as Document"):
            # Save the transcript as a text file
            answer_file_name = selected_transcript.replace('_Interview_Transcript.txt','_Answers.txt')
            conversation_history_str = '\n'.join(st.session_state.conversation_history)

            with open(answer_file_name, "w") as answer_file:
                answer_file.write(conversation_history_str)

            # Upload the transcript file to S3
            script_name = (selected_transcript.replace('_Interview_Transcript.txt', ''))
            answers_object_key = f'{script_name}/{script_name}_Answers.txt'
            s3Res.Bucket(user_s3_bucket).upload_file(answer_file_name, answers_object_key)
            st.write('Answers Documented successfully')
            write_qna_logs(f'Document generated successfully and uploaded at the location: {answers_object_key}')

            answer_url = 'https://transcribeproject.s3.amazonaws.com/' + answers_object_key

            # Encode the URL to replace spaces with %20
            encoded_answer_url = urllib.parse.quote(answer_url, safe=':/')
            
            st.write('Download the Answers document from the link below:')
            write_qna_logs(f'Link for question answer document: {encoded_answer_url}')
            st.write(encoded_answer_url)
            
            # Remove the generated question answers document file locally
            os.remove(answer_file_name)


def main():
    """
    Main function to run the Streamlit application.

    Returns:
    - None
    """

    st.set_page_config(
        page_title="Interview Summary",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Display the application title
    st.title('Summary of Interview & QnA')
    st.write('')

    # Allow the user to choose the page
    page = st.radio("Choose Page 👇", ["Home Page", "Process File", "QnA"], horizontal=True)

    if page == "Home Page":
        # Display the Home page
        st.write('')
        st.image(Image.open('Home_Page.jpg'))
    
    elif page == "Process File":
        # Display the Process File page
        st.write('')
        process_file()

    elif page == "QnA":
        # Display the QnA page
        st.write('')
        question_answer()

    else:
        st.write('Please select a page from the options above...')


if __name__ == '__main__':
    main()
