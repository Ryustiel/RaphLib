from graph import *
import asyncio

thread = {"configurable": {"thread_id": "28"}}
GRAPH.update_state(config=thread, values=DEFAULT_STATE)

async def main():
    state = await GRAPH.aget_state(thread)
    values = state.values

    while True:

        chat = values["chat"]
        out = values["out"]

        for message in out:
            print("AI >", message)
        
        new_state = {
            'chat': chat.append(input("YOU > ")), 
            'out': list()
        }
        
        GRAPH.update_state(config=thread, values=new_state)

        i = 0
        async for event in GRAPH.astream(None, thread, stream_mode="values"):

            if event["out"]: 
                values = event
                break

            elif i > 10:
                raise Exception("Exceeded the safe number of calls without an output event.")
            
            i += 1


asyncio.run(main())
