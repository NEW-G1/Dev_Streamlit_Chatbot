import streamlit as st
from langchain.llms.openai import OpenAI
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory 
from langchain.schema import Document
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Vectara
from langchain.tools.retriever import create_retriever_tool
from langchain.chains import RetrievalQA
from langchain.tools.render import format_tool_to_openai_function
from langchain.schema.agent import AgentFinish
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableConfig

from langchain.agents import AgentType
from langchain.agents import initialize_agent, tool
import os
import wikipedia


st.set_page_config(page_title="LangChain: Chat with Multi Tools", page_icon="🐾")
st.title("😺 LangChain: Chat with Multi Tools")

os.environ["OPENAI_API_KEY"]    = st.secrets.OPENAI_API_KEY
os.environ["DB_URI"]            = st.secrets.DB_URL
os.environ["VECTARA_CUSTOMER_ID"] = st.secrets.VECTARA_CUSTOMER_ID
os.environ["VECTARA_CORPUS_ID"]   = st.secrets.VECTARA_CORPUS_ID
os.environ["VECTARA_API_KEY"]     = st.secrets.VECTARA_API_KEY

# User inputs
db_uri = os.getenv("DB_URI")
openai_api_key = os.getenv("OPENAI_API_KEY")
vectara_api_key = os.getenv("VECTARA_API_KEY")

# Check user inputs
if not db_uri:
    st.info("Please enter database URI to connect to your database.")
    st.stop()

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

if not vectara_api_key:
    st.info("Please add your Vectara API key to continue.")

#Setup Function
def faiss_retriever():
  few_shots = {
  "결함이 가장 많은 공급업체는 어디인가?": "SELECT TOP 1 VEND_NM, SUM(DEF_QTY) as TOTAL_DEFECTS FROM TEMP_TB_QMS_DYN_VEND_RESULT GROUP BY VEND_NM ORDER BY TOTAL_DEFECTS DESC",
  "입고월 기준 벤더(이지테크)의 결함 추이를 알고싶다.": "SELECT VEND_NM,ITEM_NM,SUBSTRING(IN_DATE, 1, 6) AS RECEIPT_MONTH, SUM(DEF_QTY) AS TOTAL_DEFECTS FROM TEMP_TB_QMS_DYN_VEND_RESULT WHERE VEND_NM LIKE '%이지테크%' AND DEF_QTY > 0 GROUP BY VEND_NM,ITEM_NM,SUBSTRING(IN_DATE, 1, 6)",
  "입고일/지시수량 기준 벤더(이지테크)의 결함 추이를 알고싶다.": "SELECT VEND_NM, ITEM_NM,IN_DATE, ORD_QTY, SUM(DEF_QTY) AS DEF_QTY FROM TEMP_TB_QMS_DYN_VEND_RESULT WHERE VEND_NM LIKE '%이지테크%' AND DEF_QTY > 0 GROUP BY VEND_NM, ITEM_NM,IN_DATE, ORD_QTY ORDER BY IN_DATE",
  "입고일/지시수량/입고수량 기준 벤더(이지테크)의 결함 추이를 알고싶다.결함이 있는 데이터만 보여달라.":"SELECT VEND_NM, ITEM_NM,IN_DATE, ORD_QTY,IN_QTY, SUM(DEF_QTY) AS DEF_QTY FROM TEMP_TB_QMS_DYN_VEND_RESULT WHERE VEND_NM LIKE '%이지테크%' AND DEF_QTY > 0 GROUP BY VEND_NM, ITEM_NM,IN_DATE, ORD_QTY,IN_QTY ORDER BY IN_DATE",
  "지시수량 대비 결함이 가장 많은 공급업체는 어디인가?":"SELECT  TOP 1 VEND_NM, ITEM_NM, SUM(DEF_QTY / ORD_QTY) as DEFECT_RATIO FROM TEMP_TB_QMS_DYN_VEND_RESULT GROUP BY VEND_NM, ITEM_NM ORDER BY DEFECT_RATIO DESC",
  "결함이 가장 많은 입고일은 언제인가?": "SELECT TOP 1 IN_DATE, SUM(DEF_QTY) as TOTAL_DEFECTS FROM TEMP_TB_QMS_DYN_VEND_RESULT GROUP BY IN_DATE ORDER BY SUM(DEF_QTY) DESC",
  "결함이 가장 많은 입고월은 언제인가?": "SELECT SUBSTRING(IN_DATE, 1, 6) AS Month, SUM(DEF_QTY) AS Total_Defects FROM TEMP_TB_QMS_DYN_VEND_RESULT GROUP BY SUBSTRING(IN_DATE, 1, 6) ORDER BY Total_Defects DESC",
  }

  embeddings = OpenAIEmbeddings()
  few_shot_docs = [
      Document(page_content=question, metadata={"soruce_table":"TEMP_TB_QMS_DYN_VEND_RESULT","sql_query": few_shots[question]})
      for question in few_shots.keys()
  ]

  vector_db = FAISS.from_documents(few_shot_docs, embeddings)

  tool_description = """
  This tool will help you understand similar examples to adapt them to the user question.
  Input to this tool should be the user question.
  """
  retriever = vector_db.as_retriever()

  retriever_tool = create_retriever_tool(
    retriever, name="sql_get_similar_examples", description=tool_description
  )
  return retriever_tool


@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    return SQLDatabase.from_uri(database_uri=db_uri, include_tables=[ "TB_QMS_DYN_FINAL_INSP","TB_QMS_DYN_VEND_RESULT","TB_QMS_MAP_BP_ITEM" ])


db = configure_db(db_uri)

@tool
def sql_agent_retriever(query: str) -> str:
  """
  I should first get the similar examples I know.
  If the examples are enough to construct the query, I can build it.
  Otherwise, I can then look at the tables in the database to see what I can query.
  Then I should query the schema of the most relevant tables
  """
  #Include tables
  include_tables=[ "TEMP_TB_QMS_DYN_VEND_RESULT" ]
  db = configure_db(db_uri)
  llm = ChatOpenAI(model_name='gpt-3.5-turbo-1106',temperature=0)
  toolkit = SQLDatabaseToolkit(db=db, llm=llm)

  # custom_suffix = """
  # I should first get the similar examples I know.
  # If the examples are enough to construct the query, I can build it.
  # Otherwise, I can then look at the tables in the database to see what I can query.
  # Then I should query the schema of the most relevant tables
  # """
  agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    extra_tools=[faiss_retriever()],
    #suffix=custom_suffix,
  )

  response = agent.run(query)
  return response

@tool  # tool 데코레이터를 통해 openai function 형태로 쉽게 변환 가능
def vectara_agent_retriever(query: str) -> str:
  """Search for KnowledgeBase First,"""
  vectara = Vectara(
      vectara_customer_id = os.getenv("VECTARA_CUSTOMER_ID"),
      vectara_corpus_id   = os.getenv("VECTARA_CORPUS_ID"),
      vectara_api_key     = os.getenv("VECTARA_API_KEY")
  )

  llm = ChatOpenAI(model_name='gpt-3.5-turbo-1106',temperature=0)

  retriever = vectara.as_retriever()
  qa = RetrievalQA.from_llm(llm=llm, retriever=retriever)
  response = qa({"query":query})
  return response['result']

@tool
def wikipedia_agent_retriever(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)

functions = [
	format_tool_to_openai_function(f) for f in [
		vectara_agent_retriever,
		sql_agent_retriever,
		wikipedia_agent_retriever
		]
]

# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
     Understood. Let me explain the approach to handling user questions.

1. **Utilizing the Knowledge Base:**
   Initially, the OpenAI GPT model is trained on a Knowledge Base. Thus, it first extracts and utilizes information relevant to the user's question from this Knowledge Base.

2. **Searching for Similar Entries:**
   Subsequently, the GPT model performs additional searches to provide more insights on similar topics. This search can leverage external databases, the internet, or custom data sources.

3. **Composing a Comprehensive Response:**
   Lastly, the model combines information from both the Knowledge Base and the search results to craft a comprehensive response. This integration of diverse sources aims to deliver a more enriched and accurate answer to the user.

This sequential process allows us to offer the best possible response to the user. Feel free to ask if you need more detailed information or have specific questions.
     """),
    ("user", "{input}"),
])

# Setup LLM
#model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", streaming=True, temperature=0).bind(functions=functions)
model = ChatOpenAI(model_name="gpt-4-0613", streaming=True, temperature=0).bind(functions=functions)

def route(result):
	if isinstance(result, AgentFinish): # 함수를 쓰지 않기로 결정한다면 -> content 반환
		return result.return_values['output']
	else: # 함수를 쓰기로 결정한다면 -> 함수명에 따라 어떤 함수를 쓸지 결정해주고, argument를 넣은 값 반환
		tools = {
			"sql_agent_retriever": sql_agent_retriever,
			"vectara_agent_retriever" : vectara_agent_retriever,
			"wikipedia_agent_retriever" : wikipedia_agent_retriever

		}
		return tools[result.tool].run(result.tool_input)

config = RunnableConfig(callbacks=[StreamlitCallbackHandler(st.container())])

chain = prompt | model | OpenAIFunctionsAgentOutputParser() | route 
chain = chain.with_config(config=config)
# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True) 
 
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        #st_cb = StreamlitCallbackHandler(st.container())
        #response = chain.invoke(user_query, callbacks=[st_cb])
        response = chain.invoke({"input":user_query})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)