from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama.chat_models import ChatOllama

simple_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} to a complete beginner in 2-3 sentences.",
)

# formatted = simple_prompt.format(topic="AI Agents")
# print("=== Formatted Prompt ===")
# print(formatted)

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful coding tutor, keep answer short and clear."),
        ("human", "Explain {concept} with simple Python example."),
    ]
)

# messages = chat_template.format_messages(concept="list comprehension")
# print("=== Chat Messages ===")
# for msg in messages:
#     print(f" [{msg.type}]: {msg.content[:80]}")


model = ChatOllama(model="gemma3:1b", validate_model_on_init=True)

chain = chat_template | model | StrOutputParser()

response = chain.invoke({"concept": "AI Agents"})

print("=== Chain Response ===")
print(response)
print()
