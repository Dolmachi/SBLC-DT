from __future__ import annotations

import gc
import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import TypedDict

import torch
from langchain_chroma import Chroma
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    trim_messages,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline,
)
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph, add_messages
from typing_extensions import Annotated

from src.llm.base import LLM
from src.llm.system_prompts import get_system_prompt_template
from src.utils.profile_config import build_profile_paths, load_profile_config
from src.utils.output_silencers import (
    configure_hf_quiet_env,
    quiet_transformers_logging,
)

configure_hf_quiet_env()
quiet_transformers_logging()

from transformers import AutoTokenizer, Qwen3_5ForCausalLM, pipeline


RAG_COLLECTION_NAME = "dialogs"

DEFAULT_K = 3
DEFAULT_MAX_HISTORY_MESSAGES = 8

DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 0.65
DEFAULT_TOP_P = 0.8
DEFAULT_TOP_K = 20
DEFAULT_REPETITION_PENALTY = 1.0


class Qwen3_5ChatNonThinking(ChatHuggingFace):
    """
    ChatHuggingFace wrapper для Qwen3.5 в non-thinking mode.
    """

    def _to_chat_prompt(self, messages: list[BaseMessage]) -> str:
        if not messages:
            raise ValueError("At least one HumanMessage must be provided.")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("Last message must be a HumanMessage.")

        messages_dicts = [self._to_chatml_format(message) for message in messages]

        return self.tokenizer.apply_chat_template(
            messages_dicts,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )


class GraphState(TypedDict, total=False):
    """
    State LangGraph.

    messages:
        История диалога. Копится через add_messages.

    rag_context:
        RAG-фрагменты для текущего user turn.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    rag_context: str


class QwenCloneLLM(LLM):
    """
    LLM-модуль клона на базе:
    - Qwen3.5;
    - LangGraph;
    - Chroma RAG;
    - SQLite checkpointer.
    """

    def __init__(
        self,
        profile_dir: Path,
        model_id: str,
        embeddings: Embeddings,
        logger: logging.Logger,
        k: int = DEFAULT_K,
        max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    ) -> None:
        self.profile_dir = Path(profile_dir).resolve()
        self.model_id = model_id
        self.embeddings = embeddings
        self.logger = logger

        self.k = k
        self.max_history_messages = max_history_messages

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

        self.cfg = load_profile_config(self.profile_dir)
        self.paths = build_profile_paths(self.profile_dir, self.cfg)

        self.profile_text = self._load_profile_text()

        self.paths.memory_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = self.paths.memory_sqlite_path

        self.retriever = self._load_retriever()
        self.chat_model = self._load_chat_model()
        self.app = self._build_app()

    def ask(
        self,
        user_text: str,
        thread_id: str,
    ) -> str:
        """
        Полный non-streaming ответ.

        Нужен для тестов, debug и возможных режимов без аватара.
        """
        result = self.app.invoke(
            {
                "messages": [HumanMessage(user_text)],
            },
            config={
                "configurable": {
                    "thread_id": thread_id,
                },
            },
        )

        return str(result["messages"][-1].content).strip()

    def stream(
        self,
        user_text: str,
        thread_id: str,
    ) -> Iterator[str]:
        """
        Streaming ответ через LangGraph.
        """
        for chunk in self.app.stream(
            {
                "messages": [HumanMessage(user_text)],
            },
            config={
                "configurable": {
                    "thread_id": thread_id,
                },
            },
            stream_mode="messages",
            version="v2",
        ):
            message_chunk, _metadata = chunk["data"]
            if message_chunk.content:
                yield message_chunk.content
                
    def warm_up(self, runs: int = 5) -> None:
        """
        Прогревает LLM/RAG.
        """
        runs = max(0, int(runs))

        if runs == 0:
            return

        self.logger.info("Прогреваю LLM/RAG: %d прогонов", runs)

        for index in range(runs):
            thread_id = f"__warmup_llm_{index + 1}__"

            for _delta in self.stream(
                user_text="Ответь одним коротким словом: ок.",
                thread_id=thread_id,
            ):
                pass

            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def close(self) -> None:
        """
        Освобождает ресурсы LLM.
        """
        self.logger.debug("Выгружаю QwenCloneLLM")

        self.app = None
        self.chat_model = None
        self.retriever = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        self.logger.debug("QwenCloneLLM выгружен")

    def _load_profile_text(self) -> str:
        """
        Загружает profile.txt, который пользователь положил во входные данные.

        Этот текст вставляется в system prompt сразу после задания роли.
        """
        if not self.paths.source_profile_txt.exists():
            raise FileNotFoundError(
                f"Не найден profile.txt профиля: {self.paths.source_profile_txt}"
            )

        text = self.paths.source_profile_txt.read_text(encoding="utf-8").strip()

        if not text:
            raise RuntimeError(
                f"profile.txt пуст: {self.paths.source_profile_txt}"
            )

        return text

    def _load_retriever(self):
        """
        Загружает Chroma retriever из artifacts/rag/chroma.
        """
        chroma_dir = self.paths.artifacts_rag_chroma_dir

        if not chroma_dir.exists():
            raise FileNotFoundError(
                f"RAG база не найдена: {chroma_dir}\n"
                "Сначала запусти train pipeline."
            )

        self.logger.info("Загружаю RAG retriever: %s", chroma_dir)

        vectordb = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(chroma_dir),
            collection_name=RAG_COLLECTION_NAME,
        )

        return vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.k,
            },
        )

    def _load_chat_model(self) -> Qwen3_5ChatNonThinking:
        """
        Загружает Qwen3.5 + LangChain wrapper.
        """
        self.logger.info("Загружаю LLM: %s", self.model_id)

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        model = Qwen3_5ForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
        )

        gen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
        )

        def configure_generation_config(gen_cfg) -> None:
            gen_cfg.max_new_tokens = self.max_new_tokens
            gen_cfg.max_length = None
            gen_cfg.do_sample = True
            gen_cfg.temperature = self.temperature
            gen_cfg.top_p = self.top_p
            gen_cfg.top_k = self.top_k
            gen_cfg.repetition_penalty = self.repetition_penalty

        configure_generation_config(gen_pipe.generation_config)

        hf_llm = HuggingFacePipeline(
            pipeline=gen_pipe,
        )

        chat_model = Qwen3_5ChatNonThinking(
            llm=hf_llm,
            streaming=True,
        )

        self.logger.info("LLM загружена: %s", self.model_id)

        return chat_model

    def _build_app(self):
        """
        Собирает LangGraph:
            retrieve -> chat
        """
        graph = StateGraph(GraphState)

        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("chat", self._chat_node)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "chat")
        graph.add_edge("chat", END)

        conn = sqlite3.connect(
            self.sqlite_path.as_posix(),
            check_same_thread=False,
        )
        memory = SqliteSaver(conn)

        return graph.compile(checkpointer=memory)

    def _retrieve_node(self, state: GraphState) -> GraphState:
        """
        Достаёт релевантные RAG-фрагменты.
        """
        query = str(state["messages"][-1].content)

        docs = self.retriever.invoke(query)

        rag_context = "\n\n".join(doc.page_content for doc in docs)

        return {
            "rag_context": rag_context,
        }

    def _chat_node(self, state: GraphState) -> GraphState:
        """
        Генерирует ответ клона.
        """
        messages = trim_messages(
            state["messages"],
            max_tokens=self.max_history_messages,
            token_counter=len,
            strategy="last",
            start_on="human",
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", get_system_prompt_template(self.cfg.lang)),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt_template.invoke(
            {
                "target_name": self.cfg.name,
                "profile_text": self.profile_text,
                "rag_context": state["rag_context"],
                "messages": messages,
            }
        ).to_messages()

        self.logger.debug("LLM prompt messages count: %d", len(prompt))

        ai_message = self.chat_model.invoke(prompt)

        return {
            "messages": [ai_message],
        }
        