# ----- Import packages ----- #
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor

# ----- Get env variables ----- #
load_dotenv()

todoist_api_key = os.getenv( "TODOIST_API_KEY" )
gemini_api_key = os.getenv( "GEMINI_API_KEY" )

# ----- Define Functions ----- #
@tool
def add_task(task):
    '''
    Adds a new task to the users task list. Use this when the user wants to add or create a task
    '''
    print( f"Adding {task}!" )
    print( f"{task} added" )

# ----- Langchain ----- #
tools = [ add_task ]
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    google_api_key = gemini_api_key,
    temperature = 0.3
)

system_prompt = "You are a helpful assistant. You will help the user at tasks"
user_input = "Add a new task to buy milk"

prompt = ChatPromptTemplate([
    ( "system", system_prompt ),
    ( "user", user_input ),
    MessagesPlaceholder("agent_scratchpad")
])

# chain = prompt | llm | StrOutputParser()
agent = create_openai_tools_agent( llm, tools, prompt )
agent_exec = AgentExecutor( agent = agent, tools = tools, verbose = True )

response = agent_exec.invoke( { "input" : user_input } )
print( response )
