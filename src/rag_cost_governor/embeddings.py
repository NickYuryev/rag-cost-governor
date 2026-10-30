from __future__ import annotations

from collections.abc import Iterable, Iterator

from fastembed import TextEmbedding


class LocalEmbedder:
    """fastembed wrapper. onnx under the hood, no api calls, no key needed.

    Model and dimension are discovered once; the dimension is what the qdrant
    collection needs to be configured with.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", threads: int = 0) -> None:
        # threads=0 lets ONNX Runtime use all cores; fastembed otherwise defaults to 1.
        self._model = TextEmbedding(model_name=model_name, threads=threads)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            [v] = list(self._model.embed(["probe"]))
            self._dim = len(v)
        return self._dim

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        return [list(v) for v in self._model.embed(list(texts))]

    def embed_iter(self, texts: list[str]) -> Iterator[list[float]]:
        for v in self._model.embed(texts):
            yield list(v)

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
