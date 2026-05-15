from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np
import sounddevice as sd

from src.avatar.base import StreamingAvatar


class LocalAvatarWindow:
    """
    Локальный renderer inference pipeline.

    Показывает видео через OpenCV окно.
    Аудио проигрывает через sounddevice OutputStream
    """

    def __init__(
        self,
        avatar: StreamingAvatar,
        logger: logging.Logger,
        window_name: str = "Virtual Clone",
    ) -> None:
        self.avatar = avatar
        self.logger = logger
        self.window_name = window_name

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._audio_stream: sd.OutputStream | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        self._stop_event.clear()

        self._audio_stream = sd.OutputStream(
            samplerate=self.avatar.sample_rate,
            channels=1,
            dtype="float32",
        )
        self._audio_stream.start()

        self._thread = threading.Thread(
            target=self._render_loop,
            name="LocalAvatarWindow",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._audio_stream is not None:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None

    def _render_loop(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        idle_frame = self.avatar.idle_frame()
        if idle_frame is not None:
            cv2.imshow(self.window_name, idle_frame)
            cv2.waitKey(1)

        frame_delay_sec = 1.0 / self.avatar.fps

        try:
            while not self._stop_event.is_set():
                started_at = time.perf_counter()

                avatar_frame = self.avatar.read_frame(timeout_sec=0.01)

                if avatar_frame is None:
                    idle_frame = self.avatar.idle_frame()
                    if idle_frame is not None:
                        cv2.imshow(self.window_name, idle_frame)
                    cv2.waitKey(1)
                else:
                    cv2.imshow(self.window_name, avatar_frame.frame)

                    if self._audio_stream is not None:
                        for chunk in avatar_frame.audio_chunks:
                            if chunk.size > 0:
                                self._audio_stream.write(
                                    chunk.astype(np.float32).reshape(-1, 1)
                                )

                    cv2.waitKey(1)

                elapsed = time.perf_counter() - started_at
                sleep_sec = frame_delay_sec - elapsed
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

        finally:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass