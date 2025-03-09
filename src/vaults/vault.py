from abc import ABC, abstractmethod
from datetime import datetime
from typing_extensions import Self
from enum import Enum


class Role(Enum):
    USER = "user"
    SYSTEM = "system"


class Message:
    def __init__(self, content: str, role: Role):
        self.timestamp = datetime.utcnow().isoformat()
        self.content = content
        self.role = role

    def to_json(self):
        return {
            "timestamp": self.timestamp,
            "role": self.role.value,
            "content": self.content,
        }


class Vault(ABC):

    def __init__(self):
        self.messages: list[Message] = []

    @classmethod
    @abstractmethod
    async def create(cls, system_prompt: str) -> Self:
        """Factory method to create a new instance of the Vault class"""
        pass

    @abstractmethod
    async def insert_knowledge(self, knowledge: str) -> bool:
        """Insert a new knowledge into the vault"""
        pass

    @abstractmethod
    async def get_knowledges(self) -> list[str]:
        """Get all the knowledge from the vault"""
        pass

    @abstractmethod
    async def ask_model(self, user_input: str) -> str:
        """Ask the model a question based on the user input"""
        pass
