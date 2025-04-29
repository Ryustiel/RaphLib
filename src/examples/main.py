from packages.raphlib.src.examples.alternatives.old_graph import *

thread = {"configurable": {"thread_id": "3"}}
GRAPH.update_state(config=thread, values=DEFAULT_STATE)

for i in range(10):  # Use double async for statements

    VALUES = GRAPH.get_state(thread).values  # The state snapshot contains lots of metadata
    for message in VALUES["out"]:
        print("AI >", message)
    VALUES["out"] = list()  # Clears "out" after printing
    inp = input("YOU > ")
    VALUES["CHAT"].append(inp)
    GRAPH.update_state(config=thread, values=VALUES)

    for event in GRAPH.stream(None, thread, stream_mode="values"):
        if event["out"]: break
