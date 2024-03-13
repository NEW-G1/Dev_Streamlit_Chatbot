from moviepy.editor import AudioFileClip

import openai
import os
import sys
import streamlit as st
from audiorecorder import audiorecorder

from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


os.environ["OPENAI_API_KEY"] = st.secrets.OPENAI_API_KEY


#####################################################################################################################
# record_audio
# 오디오 녹음
#####################################################################################################################
def record_audio():    
    
    audio_recoder = audiorecorder("Record", "Record")

    if len(audio_recoder)>0:
        st.warning("버튼을 누르면 녹음을 종료합니다.")
        audio_file_path = "recorded_audio.wav"
        audio_recoder.export(audio_file_path, format="wav")

        # 저장된 오디오 파일을 표시
        st.audio(audio_file_path, format="audio/wav")
        st.success("Recording Compliete!")    
        return audio_file_path
    else:
        st.warning("버튼을 클릭하여 녹음을 시작하세요.")
        return None
        


#####################################################################################################################
# generate_transcript
# 오디오 파일 텍스트로 변환
#####################################################################################################################
def generate_transcript(audio_file):
    """
    Generate a transcript from an audio file using OpenAI's audio transcription service.

    Parameters:
        audio_file (str): Path to the input audio file.

    Returns:
        str: Path to the generated transcript text file.

    Raises:
        FileNotFoundError: If the input audio file cannot be found or accessed.
        openai.error.OpenAIError: If an error occurs during the transcription process.
    """
    # Import the OpenAI module
    from openai import OpenAI

    # Initialize the OpenAI client
    client = OpenAI()

    # Check if audio_file is not None
    if audio_file is not None:
        try:
            # Open the audio file in binary read mode
            audio = open(audio_file, "rb")

            # Request transcription from OpenAI
            transcript = client.audio.transcriptions.create(
                file=audio,
                model="whisper-1",
                language="ko",
                response_format="text",
                temperature=0.0,
            )

            # Extract the file name without extension
            name = os.path.splitext(audio_file)[0]

            # Save the transcript to a text file
            with open(f"{name}.txt", "w") as f:
                f.write(transcript)

            # Return the path to the generated transcript text file
            return f"{name}.txt"
        except FileNotFoundError:
            print(f"Audio file '{audio_file}' not found or cannot be accessed.")
        except Exception as e:
            print(f"An error occurred during transcription: {e}")
    else:
        print("Audio file path is None.")

    return None



#####################################################################################################################
# summarize_text_from_file
#####################################################################################################################
def summarize_text_from_file(file_path):
    """
    Summarize the text content of a file using LangChain with GPT-4.

    Parameters:
        file_path (str): Path to the input text file.

    Returns:
        str: Summarized text content.

    Raises:
        FileNotFoundError: If the input text file cannot be found or accessed.
    """
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter()

    try:
        # Read text content from the input file
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    except FileNotFoundError:
        raise FileNotFoundError("Input file not found or cannot be accessed.")

    # Initialize ChatOpenAI instance with GPT-4 model
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    # Split text into smaller chunks for processing
    texts = text_splitter.split_text(text)

    # Convert text chunks into LangChain Document objects
    docs = [Document(page_content=t) for t in texts]

    # Load summarization chain with the specified language model
    chain = load_summarize_chain(llm, chain_type="map_reduce")

    # Run summarization chain on the documents
    output = chain.run(docs)

    # Concatenate summarized text chunks into a single string
    summarized_text = "".join(output)

    return summarized_text



#####################################################################################################################
# get_key_points_from_file
#####################################################################################################################
def get_key_points_from_file(file_path):
    """
    Extract key points from a text file using LangChain with GPT-4.

    Parameters:
        file_path (str): Path to the input text file.

    Returns:
        list: List of key points extracted from the text.

    Raises:
        FileNotFoundError: If the input text file cannot be found or accessed.
    """
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter()

    # Check if file_path is not None
    if file_path is not None:
        try:
            # Read text content from the input file
            with open(file_path, "r", encoding='cp949') as file:
                text = file.read()
        except FileNotFoundError:
            raise FileNotFoundError("Input file not found or cannot be accessed.")
    else:
        print("File path is None.")
        return []

    # Initialize ChatOpenAI instance with GPT-4 model
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=['context'],
        template="Identify the keypoints for meeting minutes in the following: {context} \n\n Key points:\n-",
    )

    # Split text into smaller chunks for processing
    texts = text_splitter.split_text(text)

    # Convert text chunks into LangChain Document objects
    docs = [Document(page_content=t) for t in texts]

    # Initialize an empty list to store key points
    key_points = []

    # Iterate over each document and extract key points
    for doc in docs:
        # Initialize LLMChain
        chain = LLMChain(llm=llm, prompt=prompt, verbose=1)

        # Run LLMChain to extract key points
        key_point = chain.run(doc.page_content)

        # Append key point to the list
        key_points.append(key_point)

    return key_points


#####################################################################################################################
# generate_meeting_minutes
#####################################################################################################################
def generate_meeting_minutes(key_points):
    """
    Generate meeting minutes from key points using LangChain with GPT-4.

    Parameters:
        key_points (list): List of key points for the meeting.

    Returns:
        str: Generated meeting minutes.
    """
    # Initialize ChatOpenAI instance with GPT-4 model
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    # Define the prompt template for meeting minutes
    prompt_mom = PromptTemplate(
        input_variables=['key_points'],
        template="Beware!! Write Korean.Below are the pointers for a meeting. Generate a meeting minutes include section for key things discussed and action items accordingly.\n{key_points}.",
    )

    # Initialize LLMChain with the prompt template
    chain = LLMChain(llm=llm, prompt=prompt_mom, verbose=1)

    # Convert key points into a single string
    key_points_text = "\n".join(key_points)

    # Run LLMChain to generate meeting minutes
    mom = chain.run(key_points_text)

    return mom


        
#####################################################################################################################
# Main
#####################################################################################################################
def main():
    
    # 비디오파일 오디로 변환
    audio_file_path = record_audio()
    print(f"*****convert video to_audio: {audio_file_path}*****")
    
    # 오디오 파일 텍스트로 변환
    transcript = generate_transcript(audio_file_path)
    print(f"*****generate_transcript: {transcript}*****")
    
    # 텍스트 파일에서 keypoint 추출
    # input_file = "[GPTs로 꿀빨기] PPT 제작 자동화.txt"
    key_points = get_key_points_from_file(transcript)
    print(f"*****get_key_points_from_file: {key_points}*****")
    
    # 회의록 작성
    meeting_minutes = generate_meeting_minutes(key_points)
    
    st.subheader(meeting_minutes)

if __name__ == "__main__":
    main()        