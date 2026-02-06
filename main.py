# ----- Import packages ----- #
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# ----- Get env variables ----- #
load_dotenv()

todoist_api_key = os.getenv( "TODOIST_API_KEY" )
gemini_api_key = os.getenv( "GEMINI_API_KEY" )

# ----- Langchain ----- #
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    google_api_key = gemini_api_key,
    temperature = 0.3
)

system_prompt = "You are a helpful assistant. You will help the user at tasks"
user_input = "Tell me how best to manage my time"

prompt = ChatPromptTemplate([
    ( "system", system_prompt ),
    ( "user", user_input )
])

chain = prompt | llm | StrOutputParser()

response = chain.invoke( { "input" : "user_input" })
print( response )
