import streamlit as st
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI
import os
import time

from langchain.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import Vectara
from langchain.schema.runnable import RunnablePassthrough

os.environ["OPENAI_API_KEY"] = st.secrets.OPENAI_API_KEY

os.environ["VECTARA_CUSTOMER_ID"] = st.secrets.VECTARA_CUSTOMER_ID
os.environ["VECTARA_CORPUS_ID"]   = st.secrets.VECTARA_CORPUS_ID
os.environ["VECTARA_API_KEY"]     = st.secrets.VECTARA_API_KEY

vectara = Vectara(
        vectara_customer_id = os.getenv("VECTARA_CUSTOMER_ID"),
        vectara_corpus_id   = os.getenv("VECTARA_CORPUS_ID"),
        vectara_api_key     = os.getenv("VECTARA_API_KEY")
    )

def record_audio():
    
    start_time = time.time()
    
    audio_bytes = audio_recorder()
    
    if audio_bytes:
        audio_file_path = "recorded_audio.wav"
        with open(audio_file_path, "wb") as f:
            f.write(audio_bytes)

        # 저장된 오디오 파일을 표시
        st.audio(audio_bytes, format="audio/wav")
        
        end_time = time.time()
        st.write(f"record_audio 실행 시간: {end_time - start_time} 초")
        
        return audio_file_path
    else:
        st.warning("오디오 녹음이 실패했습니다.")
        return None

def perform_stt_and_translation(audio_file_path):
    
    start_time = time.time()
    
    client = OpenAI()

    if audio_file_path:
        audio_file = open(audio_file_path, "rb")

        # STT Transcription
        transcription_result = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        
        # STT 결과 확인
        if hasattr(transcription_result, 'text'): 
            transcription_text = transcription_result.text
            st.write("Transcription Result:", transcription_text)

            # STT Translation
            translation_result = client.audio.translations.create(
                model="whisper-1", 
                file=audio_file
            )
            
            # STT Translation 결과 확인
            if hasattr(translation_result, 'text'): 
                translation_text = translation_result.text
                st.write("Translation Result:", translation_text)

                end_time = time.time()
                st.write(f"perform_stt_and_translation 실행 시간: {end_time - start_time} 초")

                return transcription_text, translation_text
            else:
                st.warning("STT Translation 결과에서 텍스트를 찾을 수 없습니다.")
                return None, None
        else:
            st.warning("STT 결과에서 텍스트를 찾을 수 없습니다.")
            return None, None
    else:
        st.warning("오디오 파일이 없습니다.")
        return None, None

def perform_tts(translation_text):
    
    start_time = time.time()
    
    client = OpenAI()

    if translation_text:
        # TTS 생성
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=translation_text,
        )
        
        # TTS 결과로 생성된 오디오 데이터를 표시
        try:
            audio_data = response.read()
            st.audio(audio_data, format="audio/mp3")
        except Exception as e:
            st.warning(f"오디오 데이터를 추출하는 중에 오류가 발생했습니다: {e}")
        finally:
            end_time = time.time()
            st.write(f"perform_tts 실행 시간: {end_time - start_time} 초")    
    else:
        st.warning("번역된 텍스트가 없습니다.")



#langchain test용 함수
def create_langchain():
    
    start_time = time.time()
    
    # # ChatPromptTemplate를 사용하여 prompt 생성
    # prompt = ChatPromptTemplate.from_messages([
	# 		("system", "You are a helpful assistant"),
	# 		("user", "{input}")
    # ])

    # Create KnowledgeBase_Prompt
    knowledgeBase_template = """
    SYSTEM
    You are an expert researcher and writer, tasked with answering any question.

    Generate a comprehensive and informative, yet concise answer of 250 words or less for the given question based solely on the provided search results (URL and content).
    You must only use information from the provided search results. Use an unbiased and journalistic tone. Combine search results together into a coherent answer.
    Do not repeat text. Cite search results using [${{number}}] notation. Only cite the most relevant results that answer the question accurately.
    Place these citations at the end of the sentence or paragraph that reference them - do not put them all at the end.
    If different results refer to different entities within the same name, write separate answers for each entity.
    If you want to cite multiple results for the same sentence, format it as `[${{number1}}] [${{number2}}]`.
    However, you should NEVER do this with the same number - if you want to cite `number1` multiple times for a sentence, only do `[${{number1}}]` not `[${{number1}}] [${{number1}}]`

    You should use bullet points in your answer for readability. Put citations where they apply rather than putting them all at the end.
    If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure." Don't try to make up an answer.
    Anything between the following `context` html blocks is retrieved from a knowledge bank, not part of the conversation with the user.
    You must answer in Korean.

    <context>
        {context}
    <context/>

    HUMAN
    {question}
    """
    retriever = vectara.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    knowledgeBase_prompt = ChatPromptTemplate.from_template(knowledgeBase_template)

    # ChatOpenAI를 사용하여 모델 생성
    # model = ChatOpenAI(model="gpt-4")
    model = ChatOpenAI(model='gpt-3.5-turbo-16k')

    # StrOutputParser를 사용하여 결과 파싱
    output_parser = StrOutputParser()

    # 생성된 요소들을 연결하여 Langchain 생성
    # chain = knowledgeBase_prompt | model | output_parser
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | knowledgeBase_prompt 
        | model 
        | output_parser
    )

    end_time = time.time()
    st.write(f"create_langchain 실행 시간: {end_time - start_time} 초")

    return chain

def invoke_langchain(chain, input):
    
    start_time = time.time()
    
    # Langchain에 주제를 전달하여 실행
    result = chain.invoke(input)

    end_time = time.time()
    st.write(f"invoke_langchain 실행 시간: {end_time - start_time} 초")

    return result


# def translate_text(input, target_language, translate_language): 
  
#   start_time = time.time()  
    
#   # ChatPromptTemplate를 사용하여 prompt 생성
#   prompt = ChatPromptTemplate.from_messages([
#     ("system", f"Translate the following text into {translate_language}: {input}"),
#   ])

#   # ChatOpenAI를 사용하여 모델 생성
#   model = ChatOpenAI(model="gpt-3.5-turbo-16k")

#   # StrOutputParser를 사용하여 결과 파싱
#   output_parser = StrOutputParser()

#   # 생성된 요소들을 연결하여 Langchain 생성
#   chain = prompt | model | output_parser

#   end_time = time.time()
#   st.write(f"translate_text 실행 시간: {end_time - start_time} 초")  

#   return chain.invoke({"target_language": target_language,"input":input})

        
# Streamlit 앱
def main():
    
    # 오디오 녹음 및 표시
    audio_file_path = record_audio()
    
    # STT 및 번역 수행
    transcription_text, translation_text = perform_stt_and_translation(audio_file_path)
    # st.write(f"**********************translation_text: {translation_text}")
    
    langchain = create_langchain()
    
    answer = invoke_langchain(langchain, translation_text)
    st.subheader(answer)
    
    # result = translate_text(result,"korean", "English")
    # st.write(result)
    
    # TTS 수행
    perform_tts(answer)

if __name__ == "__main__":
    main()        