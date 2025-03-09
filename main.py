import ollama
import os
import json
import numpy as np
from numpy.linalg import norm
import asyncio


vault_name = "vault"
vault_path = f"{vault_name}.vault"
embedding_path = f"embeddings/{vault_name}.json"


def parse_vault():
    if not os.path.exists(vault_path):
        print("Vault file not found")
        exit(1)

    with open(vault_path, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
        return paragraphs


def save_embeddings(embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")

    with open(embedding_path, "w") as f:
        json.dump(embeddings, f)


def need_to_update():
    vault_mtime = os.path.getmtime(vault_path)
    embedding_mtime = os.path.getmtime(embedding_path)
    return vault_mtime > embedding_mtime


def load_embeddings():
    if not os.path.exists(embedding_path):
        return False

    if need_to_update():
        print("Need to update")
        return False

    with open(embedding_path, "r") as f:
        return json.load(f)


def get_embeddings(modelname, chunks):
    if (embeddings := load_embeddings()) is not False:
        return embeddings

    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]

    save_embeddings(embeddings)
    return embeddings


def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)


async def ask_model(prompt):
    SYSTEM_PROMPT = """You are a helpful human assistant who answers questions
        based on snippets of text provided in content. Answer only using the context provided,
        being as concise as possible. If you are unable to provide an answer, just say so.
        Context:
    """

    if setup_task.done() is False:
        print("Waiting for setup to complete...")

    result = await setup_task
    vault = result[0]
    embeddings = result[1]

    prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)[
        "embedding"
    ]
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\n".join(vault[item[1]] for item in most_similar_chunks),
            },
            {"role": "user", "content": prompt},
        ],
    )
    print("\n\n")
    print(response["message"]["content"])


async def user_prompt():
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

        await ask_model(user_input)


async def setup():
    vault = parse_vault()
    embeddings = get_embeddings(modelname="nomic-embed-text", chunks=vault)
    return vault, embeddings


async def main():
    global setup_task
    setup_task = asyncio.create_task(setup())

    await user_prompt()

    try:
        await setup_task
    except asyncio.CancelledError:
        print("Setup task cancelled")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting...")
        setup_task.cancel()
        exit(0)
