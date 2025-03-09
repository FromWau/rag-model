import asyncio
from vaults.text_vault import TextVault


async def user_prompt(setup_task):
    prompt = ""
    user_prompt = "What do you want to know? (enter exit to exit) -> "
    while True:

        if setup_task.done():
            prompt = f"[setup done] {user_prompt}"
        else:
            prompt = f"[setup runs] {user_prompt}"

        user_input = input(prompt)

        if user_input.lower() == "exit":
            return

        vault = await setup_task

        response = await vault.ask_model(user_input)
        print("\n\n")
        print(response["message"]["content"])


async def create(system_prompt):
    return await TextVault.create(system_prompt)


async def main():
    system_prompt = """You are a helpful human assistant who answers questions
        based on snippets of text provided in content. Answer only using the context provided,
        being as concise as possible. If you are unable to provide an answer, just say so.
        Context:
    """

    setup_task = asyncio.create_task(create(system_prompt))

    try:
        await user_prompt(setup_task)
    except asyncio.CancelledError:
        print("Setup task cancelled")
    except KeyboardInterrupt:
        print("Exiting...")
        setup_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
