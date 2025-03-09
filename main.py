import ollama
import os
import json
import numpy as np
from numpy.linalg import norm


def parse_vault(filename):
    with open(filename, encoding="utf-8-sig") as f:
        map = []
        for line in f.readlines():
            line = line.strip()
            if line:
                map.append(line)

        return map


def save_embeddings(filename, embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")

    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)


def load_embeddings(filename):
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False

    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)


def get_embeddings(filename, modelname, chunks):
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings

    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]

    save_embeddings(filename, embeddings)
    return embeddings


def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)


def main():
    SYSTEM_PROMPT = """You are a helpful human assistant who answers questions
        based on snippets of text provided in content. Answer only using the context provided,
        being as concise as possible. If you are unable to provide an answer, just say so.
        Context:
    """

    filename = "vault.txt"
    vault = parse_vault(filename)

    embeddings = get_embeddings(filename=filename, modelname="mistral", chunks=vault)

    prompt = input("What do you want to know? -> ")
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


if __name__ == "__main__":
    main()
