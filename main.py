# ----- Import packages ----- #
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from todoist_api_python.api import TodoistAPI
from datetime import datetime

# ----- Get env variables ----- #
load_dotenv()

todoist_api_key = str( os.getenv( "TODOIST_API_KEY" ) )
gemini_api_key = os.getenv( "GEMINI_API_KEY" )

# ----- Initialise Todoist API ----- #
todoist = TodoistAPI( todoist_api_key )

# ----- Define Functions ----- #
@tool
def add_task(task, task_description = None):
    '''
    Adds a new task to the users task list. Use this when the user wants to add or create a task
    '''
    print( f"Adding {task}!" )
    
    todoist.add_task( content = task, description = task_description )
    
    print( f"{task} added" )

# ----- Langchain ----- #
tools = [ add_task ]
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    google_api_key = gemini_api_key,
    temperature = 0.3
)

system_prompt = "You are a helpful assistant. You will help the user at tasks"

prompt = ChatPromptTemplate([
    ( "system", system_prompt ),
    MessagesPlaceholder( "history" ),
    ( "user", "{input}" ),
    MessagesPlaceholder( "agent_scratchpad" )
])

# chain = prompt | llm | StrOutputParser()
agent = create_openai_tools_agent( llm, tools, prompt )
agent_exec = AgentExecutor( agent = agent, tools = tools, verbose = True )

history = []
while True:
    user_input = input( "You: " )
    
    if user_input == "stop":
        break
    
    response = agent_exec.invoke( { "input": user_input, "history": history } )
    print( response["output"] )
    history.append( HumanMessage( content = user_input ) )
    history.append( AIMessage( content = response["output"] ) )
