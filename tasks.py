from crewai import Task
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agents import Agent
load_dotenv()

research_task = Task(
    name='Research',
    description='Extract actionable insights from data',
    agent=Agent,
    
)