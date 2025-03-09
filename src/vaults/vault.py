from abc import ABC, abstractmethod


class Vault(ABC):

    @classmethod
    @abstractmethod
    async def create(cls, system_prompt):
        pass
