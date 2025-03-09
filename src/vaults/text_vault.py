import ollama
import os
import json
import numpy as np
from numpy.linalg import norm
from vaults.vault import Vault


class TextVault(Vault):
    async def get_knowledge(self):
        if not os.path.exists(self.vault_path):
            print("Vault file not found")
            exit(1)

        with open(self.vault_path, encoding="utf-8-sig") as f:
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

    def save_embeddings(self, embeddings):
        if not os.path.exists("embeddings"):
            os.makedirs("embeddings")

        with open(self.embedding_path, "w") as f:
            json.dump(embeddings, f)

    def need_to_update(self):
        vault_mtime = os.path.getmtime(self.vault_path)
        embedding_mtime = os.path.getmtime(self.embedding_path)
        return vault_mtime > embedding_mtime

    def load_embeddings(self):
        if not os.path.exists(self.embedding_path):
            return False

        if self.need_to_update():
            print("Need to update")
            return False

        with open(self.embedding_path, "r") as f:
            return json.load(f)

    async def get_embeddings(self, modelname):
        if (embeddings := self.load_embeddings()) is not False:
            return embeddings

        embeddings = [
            ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
            for chunk in self.vault
        ]

        self.save_embeddings(embeddings)
        return embeddings

    async def ask_model(self, user_input):
        prompt_embedding = ollama.embeddings(
            model="nomic-embed-text", prompt=user_input
        )["embedding"]
        most_similar_chunks = find_most_similar(prompt_embedding, self.embeddings)[:5]

        return ollama.chat(
            model="mistral",
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                    + "\n".join(self.vault[item[1]] for item in most_similar_chunks),
                },
                {"role": "user", "content": user_input},
            ],
        )

    def __init__(self, system_prompt):
        self.vault_name = "vault"
        self.vault_path = f"{self.vault_name}.vault"
        self.embedding_path = f"embeddings/{self.vault_name}.json"
        self.system_prompt = system_prompt

        self.vault = []
        self.embeddings = []

    @classmethod
    async def create(cls, system_prompt):
        instance = cls(system_prompt)
        instance.vault = await instance.get_knowledge()
        instance.embeddings = await instance.get_embeddings(
            modelname="nomic-embed-text"
        )
        return instance


def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)
