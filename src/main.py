import asyncio
from vaults.text_vault import TextVault
from vaults.sql_vault import SqlVault


async def user_prompt(setup_task):
    prompt = ""
    user_prompt = "What do you want to know? (enter exit to exit) -> "
    while True:

        if setup_task.done():
            prompt = f"[setup done] {user_prompt}"
        else:
            prompt = f"[setup runs] {user_prompt}"

        user_input = input(prompt)
        if user_input == "":
            continue

        if user_input.lower() == "exit":
            return

        vault = await setup_task

        if user_input.lower().startswith("insert: "):
            payload: str = user_input[len("Insert: ") :]
            await vault.insert_knowledge(payload)
            continue

        response = await vault.ask_model(user_input)
        print(response)
        print("\n\n")


async def create(system_prompt):
    # return await TextVault.create(system_prompt)
    return await SqlVault.create(system_prompt)


async def main():
    system_prompt: str = (
        """You are a helpful human assistant who likes to talk. When the user makes a question answer based on snippets of text provided in 'Context' or based on snippets of text provided in 'Chat History'. Always prefer to use the context and chat history during the talk. Never make up information. If you don't know the answer, just say so. You should respond to the user in a human like manner. Never mention that you are a computer program or that you have a given context. Never repeat yourself. Do not respond with the same answer twice in a row. Do not jude the user. Always be polite and follow the instructions.""".strip()
    )

    setup_task = asyncio.create_task(create(system_prompt))

    try:
        await user_prompt(setup_task)
    except asyncio.CancelledError:
        print("Setup task cancelled")
    except EOFError:
        print("Exiting...")
        setup_task.cancel()
    except KeyboardInterrupt:
        print("Exiting...")
        setup_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
