import importlib.util
from pathlib import Path
import tempfile
import unittest
import wave


@unittest.skipUnless(importlib.util.find_spec("av") and importlib.util.find_spec("numpy"), "optional PyAV/numpy not installed")
class AudioTests(unittest.TestCase):
    def test_public_decoder_preserves_duration_and_amplitude(self):
        import numpy as np
        from clipscout.transcribe import decode_audio
        scratch = Path(__file__).resolve().parents[1] / ".cache" / "audio-tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            path = Path(temporary) / "tone.wav"
            rate = 48000
            signal = (12000 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)).astype("<i2")
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(rate)
                audio.writeframes(signal.tobytes())
            result = decode_audio(path)
            self.assertEqual(len(result), 16000)
            self.assertEqual(result.dtype, np.float32)
            self.assertTrue(0.2 < np.sqrt(np.mean(result**2)) < 0.3)


if __name__ == "__main__":
    unittest.main()
