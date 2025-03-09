from typing import Union
from typing_extensions import Self
from vaults.vault import Role, Message, Vault
import ollama
import sqlite3
from datetime import datetime
import json
from utils import find_most_similar


class SqlVault(Vault):
    def save_embeddings(self, embeddings: list[str]) -> bool:
        json_embeddings = json.dumps(embeddings)
        self.cursor.execute("DELETE FROM embeddings")
        self.cursor.execute(
            "INSERT INTO embeddings (content, last_modified) VALUES (?, ?)",
            (json_embeddings, datetime.utcnow()),
        )
        self.vault_conn.commit()
        return self.cursor.rowcount == 1

    def need_to_update(self) -> bool:
        self.cursor.execute(
            """
            SELECT Max(last_modified) FROM embeddings
            """
        )
        last_modified_embeddings = self.cursor.fetchone()[0]

        self.cursor.execute(
            """
            SELECT Max(last_modified) FROM knowledge
            """
        )
        last_modified_knowledge = self.cursor.fetchone()[0]

        return last_modified_knowledge > last_modified_embeddings

    def load_embeddings(self) -> Union[list[str], bool]:
        self.cursor.execute("SELECT content FROM embeddings")
        rows = self.cursor.fetchall()
        if len(rows) == 0:
            return False

        if self.need_to_update():
            return False

        embeddings = [row[0] for row in rows]

        return embeddings

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
        self.vault_conn = sqlite3.connect("vault.db")
        self.cursor = self.vault_conn.cursor()

        self.system_prompt: str = system_prompt

        self.vault: list[str] = []
        self.embeddings: list[str] = []

    @classmethod
    async def create(cls, system_prompt: str) -> Self:
        instance = cls(system_prompt)

        instance.cursor.execute(
            """
            DROP TABLE IF EXISTS knowledge
            """
        )

        instance.cursor.execute(
            """
            DROP TABLE IF EXISTS embeddings
            """
        )

        instance.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
            )"""
        )
        instance.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
            )"""
        )
        instance.vault_conn.commit()

        instance.vault = await instance.get_knowledges()
        instance.embeddings = await instance.get_embeddings(
            modelname="nomic-embed-text"
        )

        return instance

    async def insert_knowledge(self, knowledge: str) -> bool:
        self.cursor.execute(
            "INSERT INTO knowledge (content, last_modified) VALUES (?,?)",
            (knowledge, datetime.utcnow()),
        )
        self.vault_conn.commit()

        self.vault = await self.get_knowledges()
        self.embeddings = await self.get_embeddings(modelname="nomic-embed-text")

        return self.cursor.rowcount == 1

    async def get_knowledges(self):
        self.cursor.execute("SELECT content FROM knowledge")
        results = self.cursor.fetchall()
        return [result[0] for result in results]
