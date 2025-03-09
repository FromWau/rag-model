from typing_extensions import Self
from vaults.vault import Role, Message, Vault
from typing import Union
from utils import find_most_similar
import ollama
import os
import json


class TextVault(Vault):
    def save_embeddings(self, embeddings: list[str]) -> None:
        if not os.path.exists("embeddings"):
            os.makedirs("embeddings")

        with open(self.embedding_path, "w") as f:
            json.dump(embeddings, f)

    def need_to_update(self):
        vault_mtime = os.path.getmtime(self.vault_path)
        embedding_mtime = os.path.getmtime(self.embedding_path)
        return vault_mtime > embedding_mtime

    def load_embeddings(self) -> Union[list[str], bool]:
        if not os.path.exists(self.embedding_path):
            return False

        if self.need_to_update():
            return False

        with open(self.embedding_path, "r") as f:
            return json.load(f)

    async def get_embeddings(self, modelname: str) -> list[str]:
        knowledge_cache = self.load_embeddings()
        if isinstance(knowledge_cache, list):
            return knowledge_cache

        embeddings: list[str] = [
            ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
            for chunk in self.vault
        ]

        self.save_embeddings(embeddings)
        return embeddings

    async def ask_model(self, user_input: str) -> str:
        message: Message = Message(content=user_input, role=Role.USER)
        self.messages.append(message)

        prompt_embedding: list[str] = ollama.embeddings(
            model="nomic-embed-text", prompt=user_input
        )["embedding"]
        most_similar_chunks = find_most_similar(prompt_embedding, self.embeddings)[:5]

        messages: list = [
            Message(
                content=self.system_prompt
                + "\n".join(self.vault[item[1]] for item in most_similar_chunks),
                role=Role.SYSTEM,
            ).to_json(),
        ]
        messages.extend(msg.to_json() for msg in self.messages)

        response = ollama.chat(
            model="mistral",
            messages=messages,
        )

        msg: str = response["message"]["content"].strip()

        self.messages.append(Message(content=msg, role=Role.SYSTEM))
        return msg

    def __init__(self, system_prompt: str):
        super().__init__()
        self.vault_name: str = "vault"
        self.vault_path: str = f"{self.vault_name}.vault"
        self.embedding_path: str = f"embeddings/{self.vault_name}.json"
        self.system_prompt: str = system_prompt

        self.vault: list[str] = []
        self.embeddings: list[str] = []

    @classmethod
    async def create(cls, system_prompt: str) -> Self:
        instance = cls(system_prompt)
        instance.vault = await instance.get_knowledges()
        instance.embeddings = await instance.get_embeddings(
            modelname="nomic-embed-text"
        )
        return instance

    async def insert_knowledge(self, knowledge: str) -> bool:
        if not os.path.exists(self.vault_path):
            print("Vault file not found")
            exit(1)

        with open(self.vault_path, "a", encoding="utf-8-sig") as f:
            f.write(f"{knowledge}\n")

        self.vault = await self.get_knowledges()
        self.embeddings = await self.get_embeddings(modelname="nomic-embed-text")

        return True

    async def get_knowledges(self) -> list[str]:
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
