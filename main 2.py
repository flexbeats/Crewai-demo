import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

os.environ["OPENAI_API_KEY"] = 'NA'
os.environ["SERPER_API_KEY"] = "yourd" # serper.dev API key

# You can choose to use a local model through Ollama for example. See https://docs.crewai.com/how-to/LLM-Connections/ for more information.

os.environ["OPENAI_API_BASE"] = 'http://localhost:11434/v1'
os.environ["OPENAI_MODEL_NAME"] ='crewai-mistral'  # Adjust based on available model
# os.environ["OPENAI_API_KEY"] ='sk-111111111111111111111111111111111111111111111111'

# You can pass an optional llm attribute specifying what model you wanna use.
# It can be a local model through Ollama / LM Studio or a remote
# model like OpenAI, Mistral, Antrophic or others (https://docs.crewai.com/how-to/LLM-Connections/)
#
# import os
# os.environ['OPENAI_MODEL_NAME'] = 'gpt-3.5-turbo'
#
# OR
#
# from langchain_openai import ChatOpenAI

search_tool = SerperDevTool()

# Define your agents with roles and goals
journalist = Agent(
  role='Debate organizer',
  goal='Compare politicals arguments from different sides to build a clear view of 2024 European elections political opinions',
  backstory="""You are an experienced european political journalist.
  Your expertise lies in asking general questions to candidates during a debate. In this crew, candidates are co-workers, so you need to delegate work to 
  co-workers to have answers from the candidates. Most important thing is that you have to collect opinions of each candidates: far-right, far-left and ecological.
  You encourages candidates to explain their opinions with details and examples. You ask them to keep short answers as everyone has to speak.
  When you collected responses, you ask different candidates to react and provides arguments against each others in order to converge to common solutions for humanity sake.
  To Search the internet, you have to use Search the internet(search_query: 'string').
  Action 'Search the internet (something here)' don't exist, these are the only available Actions:
  Search the internet: Search the internet(search_query: 'string') - A tool that can be used to search the internet with a search_query.
  Action 'Modify question structure' don't exist.
  In order to delegate work to co-worker, please never forget context string argument, this is very important. Delegate task to co-worker(task: str, context: str, coworker: Optional[str] = None, **kwargs).
  The input to this tool should be the co-worker, the task you want them to do, and ALL necessary context to execute the task, they know nothing about the task, so share absolute everything you know, don't reference things but instead explain them.
  This is the same for asking questions. Tool Ask question to co-worker accepts these inputs: Ask question to co-worker(question: str, context: str, coworker: Optional[str] = None, **kwargs).
  Please never forget context string argument, this is very important.
  The input to this tool should be the co-worker, the question you have for them, and ALL necessary context to ask the question properly, they know nothing about the question, so share absolute everything you know, don't reference things but instead explain them.""",
  verbose=True,
  allow_delegation=True,
  # You can pass an optional llm attribute specifying what model you wanna use.
  # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7),
  tools=[search_tool]
)
fact_checking_journalist = Agent(
  role='Fact Checking Journalist',
  goal='Fact checking of candidates arguments',
  backstory="""You are an experienced political journalist.
  Your expertise lies in fact checking political arguments during the debate.
  You answer to Debate Organizer by telling him if the candidate's argument is True/False/Partially""",
  verbose=True,
  allow_delegation=True,
  # You can pass an optional llm attribute specifying what model you wanna use.
  # llm=ChatOpenAI(model_name="gpt-3.5", temperature=0.7),
  tools=[search_tool]
)
candidate1 = Agent(
  role='European Far-left political candidate',
  goal='Prove that your opinion is the best one in order to convince people.',
  backstory="""As a first step, before answering the first question, you search your own backstory from the internet.
  You construct yourself a far left political candidate personnality by choosing a personal name.
  To Search the internet, you have to use Search the internet(search_query: 'string')
  Action 'Search the internet (something here)' don't exist, never put anything in brackets after the action name, these are the only available Actions:
  Search the internet: Search the internet(search_query: 'string') - A tool that can be used to search the internet with a search_query.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool]
)

candidate2 = Agent(
  role='European Far-right political candidate',
  goal='Prove that your opinion is the best one in order to convince people.',
  backstory="""As a first step, before answering the debate organizer, you search your own backstory from the internet.
  You construct you a far right political candidate personnality by choosing a personal name.
  To Search the internet, you have to use Search the internet(search_query: 'string')
  Action 'Search the internet (something here)' don't exist, never put anything in brackets after the action name, these are the only available Actions:
  Search the internet: Search the internet(search_query: 'string') - A tool that can be used to search the internet with a search_query.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool]
)

candidate3 = Agent(
  role='European Ecological political candidate ',
  goal='Prove that your opinion is the best one in order to convince people.',
  backstory="""As a first step, before answering the debate organizer, you search your own backstory from the internet.
  You construct you an ecological political candidate personnality by choosing a personal name.
  To Search the internet, you have to use Search the internet(search_query: 'string')
  Action 'Search the internet (something here)' don't exist, never put anything in brackets after the action name, these are the only available Actions:
  Search the internet: Search the internet(search_query: 'string') - A tool that can be used to search the internet with a search_query.""",
  verbose=True,
  allow_delegation=False,
  tools=[search_tool]
)

# Create tasks for your agents
task1 = Task(
  description="""We are just before 2024 European Elections, you have to organize a good political debate focused on current condition of planet Earth in 2024, specially concerning climate changes implications.
  Provide an extensive report of more than 4000 words, organized into different categories like : energy, jobs, protection of population ... The report must contain all point of view 
  from all different candidates, if you don't have all point of views, you have to collect more opinions""",
  expected_output="Fully detailed debate report with points of view from different candidates and comparison of arguments organized into categories.",
  agent=journalist,
  human_input=True,
)

task2 = Task(
  description="""Using the insights provided, verify all arguments provided by candidates""",
  expected_output="Full debate report verified",
  agent=fact_checking_journalist
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[journalist,candidate1,candidate2,candidate3],
  tasks=[task1],
  verbose=2, # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
