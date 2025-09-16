# Example: Try the agent directly
from app import PocketFlowWebSearchAgent

agent = PocketFlowWebSearchAgent()
question = "Who won the Nobel Prize in Physics 2024?"
answer = agent(question)
print("Agent answer:", answer)