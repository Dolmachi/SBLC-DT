from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

from src.inference.local_window import LocalAvatarWindow
from src.inference.audio_io import SoundDeviceAudioIO
from src.inference.pipeline import CloneInferencePipeline
from src.inference.text_stream import SentenceStreamBuffer


class CloneInferenceSession:
    """
    Runtime-сессия виртуального клона.

    Схема ответа:
        mic -> ASR -> LLM text stream -> sentence buffer
        -> TTS целым sentence-level фрагментом
        -> FlashHead chunked talking-head inference
        -> local window.
    """

    def __init__(
        self,
        pipeline: CloneInferencePipeline,
        audio_io: SoundDeviceAudioIO,
        window: LocalAvatarWindow,
        logger: logging.Logger,
    ) -> None:
        self.pipeline = pipeline
        self.audio_io = audio_io
        self.window = window
        self.logger = logger

    @classmethod
    def load(
        cls,
        profile_dir: Path,
        logger: logging.Logger,
        llm_model_id: str = "Qwen/Qwen3.5-4B",
        asr_model_size: str = "large-v3-turbo",
        tts_model_id: str = "openbmb/VoxCPM2",
    ) -> "CloneInferenceSession":
        pipeline = CloneInferencePipeline.load(
            profile_dir=profile_dir,
            llm_model_id=llm_model_id,
            asr_model_size=asr_model_size,
            tts_model_id=tts_model_id,
            logger=logger,
        )

        audio_io = SoundDeviceAudioIO(
            logger=logger,
            input_sample_rate=16000,
        )

        window = LocalAvatarWindow(
            avatar=pipeline.avatar,
            logger=logger,
            window_name=f"Virtual Clone: {pipeline.ctx.cfg.name}",
        )

        return cls(
            pipeline=pipeline,
            audio_io=audio_io,
            window=window,
            logger=logger,
        )

    def run(self) -> None:
        self.pipeline.avatar.start()
        self.window.start()

        self._print_intro()

        try:
            while True:
                cmd = input("\n[Шаг] Enter — говорить, q — выход: ").strip()

                if cmd.lower() in {"q", "quit", "exit"}:
                    break

                self._run_one_turn_streaming()

        except KeyboardInterrupt:
            print("\n[Выход] Остановка по Ctrl+C.")

    def close(self) -> None:
        try:
            self.window.stop()
        finally:
            self.pipeline.close()

    def _run_one_turn_streaming(self) -> None:
        recorded = self.audio_io.record_until_enter()

        tts_queue: queue.Queue[str | None] = queue.Queue()
        tts_errors: list[BaseException] = []

        def tts_worker() -> None:
            try:
                while True:
                    text_unit = tts_queue.get()
                    if text_unit is None:
                        break

                    text_unit = text_unit.strip()
                    if not text_unit:
                        continue

                    wav = self.pipeline.synthesize_tts_audio(text_unit)
                    self.pipeline.avatar.push_audio(
                        audio=wav,
                        sample_rate=self.pipeline.tts.sample_rate,
                    )

            except BaseException as error:
                tts_errors.append(error)
            finally:
                self.pipeline.avatar.end_speech()

        worker: threading.Thread | None = None

        try:
            if recorded.audio.size < int(recorded.sample_rate * 0.25):
                print("\n[ASR] Запись слишком короткая. Попробуй ещё раз.")
                return

            user_text = self.pipeline.transcribe_audio(
                audio=recorded.audio,
                sample_rate=recorded.sample_rate,
            ).strip()

            if not user_text:
                print("\n[ASR] Речь не распознана. Попробуй ещё раз.")
                return

            print(f"\n[Ты / ASR]: {user_text}")
            print("\n[Клон]: ", end="", flush=True)

            self.pipeline.avatar.begin_speech()

            worker = threading.Thread(
                target=tts_worker,
                name="CloneSentenceTTSWorker",
                daemon=True,
            )
            worker.start()

            sentence_buffer = SentenceStreamBuffer(min_chars=40, max_chars=220)

            for text_delta in self.pipeline.stream_answer_text(user_text):
                print(text_delta, end="", flush=True)

                if tts_errors:
                    raise tts_errors[0]

                for text_unit in sentence_buffer.push(text_delta):
                    tts_queue.put(text_unit)

            tail = sentence_buffer.flush()
            if tail:
                tts_queue.put(tail)

            tts_queue.put(None)
            worker.join()
            worker = None

            if tts_errors:
                raise tts_errors[0]

            print()

        except Exception as error:
            self.logger.exception("Streaming inference turn failed")
            print(f"\n[ОШИБКА] Streaming inference turn failed: {error}")

        finally:
            if worker is not None and worker.is_alive():
                tts_queue.put(None)
                worker.join(timeout=10.0)

    @staticmethod
    def _print_intro() -> None:
        print("Виртуальный клон запущен.")
        print("На каждом шаге:")
        print("  1) Нажми Enter, чтобы начать запись.")
        print("  2) Говори.")
        print("  3) Нажми Enter, чтобы остановить запись.")
        print("  4) Клон начнёт отвечать голосом и аватаром.")