"""Music2Video - local audio-to-prompt conversion for ComfyUI.

The package deliberately keeps heavy imports out of module scope so the pure
formatting/timing modules can be imported (and tested) without torch, librosa
or a running ComfyUI.
"""

__version__ = "0.1.0"
