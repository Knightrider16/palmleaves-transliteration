"""Model registry.  Importing this package registers every architecture."""
from . import crnn_ctc       # noqa: F401
from . import cnn_ctc        # noqa: F401
from . import vit_ctc        # noqa: F401
from . import crnn_attn      # noqa: F401
from . import trocr          # noqa: F401
from . import conformer      # noqa: F401

from ._base import REGISTRY, build, LineRecognizer  # noqa: F401
