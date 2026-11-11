"""Voice store: Breeze clone references in <data-dir>/voices/<id>/{ref.wav,ref.txt,meta.json}."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
import shutil
import subprocess

import numpy as np
import soundfile as sf

RATE = 24000
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def decode(path, rate, seconds=None):
    """Any audio ffmpeg can read -> mono PCM16 at `rate`, at most the first `seconds`."""
    limit = ["-t", str(seconds)] if seconds else []
    result = subprocess.run([FFMPEG, "-nostdin", "-loglevel", "error", "-i", str(path), *limit,
                             "-f", "s16le", "-ar", str(rate), "-ac", "1", "pipe:1"], capture_output=True)
    if result.returncode != 0:
        raise ValueError("无法解码音频：" + result.stderr.decode(errors="replace").strip()[-300:])
    return np.frombuffer(result.stdout, "<i2")


class Voices:
    def __init__(self, data_dir):
        self.root = Path(data_dir) / "voices"
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, voice_id):
        if not re.fullmatch(r"[0-9a-f]{8}", voice_id or "") or not (self.root / voice_id / "meta.json").exists():
            raise KeyError(f"音色不存在：{voice_id}")
        return self.root / voice_id

    def list(self):
        metas = [json.loads(path.read_text(encoding="utf-8")) for path in self.root.glob("*/meta.json")]
        return sorted(metas, key=lambda meta: meta["created"], reverse=True)

    def reference(self, voice_id):
        """(ref.wav path, transcript) for Breeze."""
        folder = self.dir(voice_id)
        return str(folder / "ref.wav"), (folder / "ref.txt").read_text(encoding="utf-8").strip()

    def add(self, name, language, source, pcm, text, instruction=None):
        voice_id = secrets.token_hex(4)
        folder = self.root / voice_id
        folder.mkdir()
        sf.write(folder / "ref.wav", pcm, RATE, subtype="PCM_16")
        (folder / "ref.txt").write_text(text + "\n", encoding="utf-8")
        meta = {"id": voice_id, "name": name, "language": language, "source": source,
                "duration": round(len(pcm) / RATE, 2), "text": text,
                "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "instruction": instruction}
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    def delete(self, voice_id):
        shutil.rmtree(self.dir(voice_id))
