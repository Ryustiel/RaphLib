from graph import *
from raphlib import LLMFunction
from graph import LLM

thread = {"configurable": {"thread_id": "3"}}
GRAPH.update_state(config=thread, values=DEFAULT_STATE)

# for i in range(10):  # Use double async for statements

#     VALUES = GRAPH.get_state(thread).values  # The state snapshot contains lots of metadata
#     for message in VALUES["out"]:
#         print("AI >", message)
#     VALUES["out"] = list()  # Clears "out" after printing
#     inp = input("YOU > ")
#     VALUES["CHAT"].append(inp)
#     GRAPH.update_state(config=thread, values=VALUES)

#     for event in GRAPH.stream(None, thread, stream_mode="values"):
#         if event["out"]: break

fnc = LLMFunction(LLM,
    """Create a mask that represents for each word if it is a verb : {message}""",
    words=["word", ...],
    mask=[True, ...]
)

fnc.prompt.pretty_print({"message": "Test"})

messages = ["I want to eat cakes", "How many cakes are in that truck ?", "Shit won't hold that cow", "Two trucks are necessary to keep on working."]
responses = fnc.run_many(message=messages)
print("\n".join([f"{message}\t\t\t>> {classe.words} {classe.mask}" for (message, classe) in zip(messages, responses)]))