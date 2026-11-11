"""Silero VAD and the turn segmenter (32 ms frames)."""

from collections import deque

import numpy as np
import torch

RATE = 16000
FRAME = 512  # samples per Silero frame at 16 kHz
FRAME_SECONDS = FRAME / RATE


class Silero:
    def __init__(self):
        from silero_vad import load_silero_vad
        self.model = load_silero_vad(onnx=True)

    def __call__(self, frame):
        return float(self.model(torch.from_numpy(frame.astype(np.float32) / 32768), RATE))

    def reset(self):
        self.model.reset_states()


class Segmenter:
    """Feeds PCM16 audio, returns turn events in order:

    ("start",)                 VAD onset: 2 consecutive speech frames
    ("early", audio)           silence reached `earlyTranscribe`: start the final ASR now
    ("resume",)                speech came back after "early": that result is stale
    ("end", audio, length_s)   end of turn; length_s = seconds since onset
    """

    def __init__(self, turn):
        self.turn = turn
        self.vad = Silero()
        self.reset()

    def reset(self):
        self.vad.reset()
        self.pending = np.zeros(0, np.int16)
        # Pre-roll before onset plus the two onset frames: a soft onset ("Hey there") fires VAD late.
        self.preroll = deque(maxlen=round(self.turn["preRoll"] / FRAME_SECONDS) + 2)
        self.run = 0  # consecutive speech frames while idle
        self.frames = None  # the utterance, None while idle
        self.question = False  # latest partial ends with a question mark: shorter end of turn

    @property
    def recording(self):
        return self.frames is not None

    @property
    def elapsed(self):
        """Seconds since onset."""
        return self.since_onset * FRAME_SECONDS

    @property
    def silence_seconds(self):
        return self.silence * FRAME_SECONDS

    def audio(self):
        return np.concatenate(self.frames)

    def feed(self, pcm):
        samples = np.concatenate([self.pending, np.frombuffer(pcm, "<i2")])
        whole = len(samples) // FRAME * FRAME
        self.pending = samples[whole:]
        events = []
        for frame in samples[:whole].reshape(-1, FRAME):
            speech = self.vad(frame) > self.turn["vadThreshold"]
            if self.frames is None:
                self.preroll.append(frame)
                self.run = self.run + 1 if speech else 0
                if self.run >= 2:
                    self.frames, self.silence = list(self.preroll), 0
                    self.since_onset, self.early, self.question = 0, False, False
                    self.preroll.clear()
                    events.append(("start",))
                continue
            self.frames.append(frame)
            self.since_onset += 1
            if speech:
                self.silence = 0
                if self.early:
                    self.early = False
                    events.append(("resume",))
                continue
            self.silence += 1
            turn = self.turn
            limit = (turn["questionConfirm"] + turn["questionSilence"] if self.question
                     else turn["confirm"] + turn["silence"])
            if self.silence_seconds >= limit:
                events.append(("end", self.audio(), (self.since_onset + 2) * FRAME_SECONDS))
                self.frames, self.run = None, 0
            elif not self.early and self.silence_seconds >= turn["earlyTranscribe"]:
                self.early = True
                events.append(("early", self.audio()))
        return events
