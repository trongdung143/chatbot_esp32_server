from src.agents.base import BaseAgent
from src.agents.chat.prompt import prompt_chat
from src.agents.state import State
from src.agents.utils import logger, clean_txt


class ChatAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_name="chat", state=State)

        self._prompt = prompt_chat

        self._chain = self._prompt | self._model

    async def process(self, state: State) -> State:
        try:
            messages = state.get("messages")
            response = await self._chain.ainvoke(
                {"messages": messages, "input": messages[-1].content}
            )
            raw = response.content
            answer = clean_txt(raw)
        except Exception as e:
            logger.exception(e)
        finally:
            state.update(answer=answer)
            logger.info("[Chat] process executed")
        return state
