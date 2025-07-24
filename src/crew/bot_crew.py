#!/usr/bin/env python3
import os
import yaml
from crewai import Agent, Task, Crew, Process, LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from tools.search import pinecone_search_tool
from tools.serper import SerperTool
from tools.context import get_from_memory

# Load environment variables
load_dotenv()

@CrewBase
class BotCrew:
    """BotCrew for handling user queries about KartavyaAI"""
    
    def __init__(self):
        # Check for API key when crew is initialized
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY is not set. Please set your Google API key.")
        
        # Initialize Gemini LLM
        self.llm = LLM(
            model="gemini/gemini-2.0-flash",
        )
        
        # Load configuration files
        self.agents_config = self._load_config('config/agents.yaml')
        self.tasks_config = self._load_config('config/tasks.yaml')
    
    def _load_config(self, file_path):
        """Load YAML configuration file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Configuration file {file_path} not found.")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file {file_path}: {e}")
            return {}

    @agent
    def memory_recaller(self) -> Agent:
        """
        This agent is responsible for recalling past interactions and information related to kartavya.
        It assists users by retrieving relevant memories and providing context for their current queries.
        """
        return Agent(
            config=self.agents_config.get('Memory_recaller', {}),
            llm=self.llm,
            verbose=True
        )

    @agent
    def research_agent(self) -> Agent:
        """
        This agent is responsible for searching information in the vector database or the internet.
        It uses the `search_kartavya_pdf` tool to search the vector database and `SerperTool` for internet searches.
        """
        return Agent(
            config=self.agents_config.get('Researcher_Agent', {}),
            tools=[pinecone_search_tool, SerperTool],
            llm=self.llm,
            verbose=True
        )

    @agent
    def final_reply_agent(self) -> Agent:
        """
        This agent is responsible for generating the final response based on research findings.
        It synthesizes information and provides comprehensive answers to user queries.
        """
        return Agent(
            config=self.agents_config.get('Final_reply', {}),
            llm=self.llm,
            verbose=True
        )
    
    @task
    def recall_memory_task(self) -> Task:
        """
        Task for recalling past interactions and information related to kartavya.
        It uses the `get_from_memory` tool to retrieve relevant memories.
        """
        return Task(
            config=self.tasks_config.get('Reacall_memory', {}),
            agent=self.memory_recaller(),
            tools=[get_from_memory]
        )

    @task
    def search_task(self) -> Task:
        """
        Task for searching and gathering information from various sources.
        """
        return Task(
            config=self.tasks_config.get('Search_task', {}),
            agent=self.research_agent()
        )

    @task
    def reply_task(self) -> Task:
        """
        Task for generating the final reply based on search results.
        """
        return Task(
            config=self.tasks_config.get('replier', {}),
            agent=self.final_reply_agent(),
            context=[self.search_task()]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the BotCrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )