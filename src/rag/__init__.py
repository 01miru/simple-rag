"""Local Markdown search exposed through MCP."""

from ._shared import model as model
from .indexing import chunks as chunks
from .indexing import index as index
from .search import (
    get_document as get_document,
)
from .search import (
    list_documents as list_documents,
)
from .search import (
    search as search,
)
from .server import create_server as create_server
from .server import main as main
