from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process

model = Ollama(model = 'dolphin-llama3')

email = 'you owe amazon a 1000$ payment'

classifier = Agent(
    role = 'email classifier',
    goal = 'classify the email. Give a brief description of whether the email is spam or not and why.',
    backstory = 'You are an AI model trained to classify emails as spam or not spam. You have been tasked to help user better manage their email inbox',
    verbose = True,
    allow_delegation = False,
    llm = model
)

responder = Agent(
    role = 'email responder',
    goal = 'respond to the email',
    backstory = 'You are an AI model trained to respond to emails. You have been tasked to help user better manage their email inbox',
    verbose = True,
    allow_delegation = False,
    llm = model
)
classify_task = Task(
    description = f"classify the email: '{email}'",
    agent = classifier,
    expected_output = 'One of these three classes: priority, casual, spam',
)

respond_task = Task(
    description = f"respond to the email: '{email}'",
    agent = responder,
    expected_output = "A response to the email with a short description based on the output of the 'classifier' agent",
)


crew = Crew(
    name = 'email crew',
    agents = [classifier, responder],
    tasks = [classify_task, respond_task],
    verbose = 0,
    processes = Process.sequential
)

output = crew.kickoff()
print(output)