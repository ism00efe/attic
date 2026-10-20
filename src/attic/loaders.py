"""Turn a file on disk into plain text.

Every loader returns a string or raises :class:`LoadError`. A file that cannot be read
is recorded as failed in the manifest and retried only once it changes, so a corrupt PDF
costs one attempt, not one per sync.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path


class LoadError(Exception):
    pass


_PLAIN_TEXT = {
    ".md", ".markdown", ".txt", ".rst", ".org", ".tex", ".csv", ".tsv", ".json", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".conf", ".xml",
}
_CODE = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".java", ".kt", ".c", ".h", ".cpp", ".cc",
    ".hpp", ".cs", ".go", ".rs", ".swift", ".rb", ".php", ".sql", ".sh", ".ps1", ".bat",
    ".r", ".m", ".scala", ".lua", ".dart", ".vue", ".svelte", ".css", ".scss",
}
_HTML = {".html", ".htm"}
_RICH = {".pdf", ".docx", ".pptx", ".xlsx", ".ipynb"}

SUPPORTED_SUFFIXES = frozenset(_PLAIN_TEXT | _CODE | _HTML | _RICH)

# Tried in order. cp1254 is Windows-Turkish: a .txt saved by an older Windows tool is
# far more likely to be that than anything else, and latin-1 never fails, so it is last.
_ENCODINGS = ("utf-8-sig", "cp1254", "latin-1")


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix in _PLAIN_TEXT or suffix in _CODE:
            return _read_text(path)
        if suffix in _HTML:
            return _html_to_text(_read_text(path))
        if suffix == ".pdf":
            return _pdf(path)
        if suffix == ".docx":
            return _docx(path)
        if suffix == ".pptx":
            return _pptx(path)
        if suffix == ".xlsx":
            return _xlsx(path)
        if suffix == ".ipynb":
            return _ipynb(path)
    except LoadError:
        raise
    except Exception as exc:  # parser libraries raise anything on a malformed file
        raise LoadError(f"{type(exc).__name__}: {exc}") from exc
    raise LoadError(f"unsupported file type {suffix!r}")


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        raise LoadError("looks binary")
    for enc in _ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise LoadError("undecodable")  # unreachable: latin-1 decodes anything


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return "".join(parser.parts)


def _pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        raise LoadError("encrypted PDF")
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    text = "\n\n".join(p for p in pages if p)
    if not text:
        raise LoadError("no extractable text (scanned PDF?)")
    return text


def _docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _pptx(path: Path) -> str:
    from pptx import Presentation

    slides = []
    for number, slide in enumerate(Presentation(str(path)).slides, start=1):
        texts = [
            shape.text_frame.text.strip()
            for shape in slide.shapes
            if getattr(shape, "has_text_frame", False) and shape.text_frame.text.strip()
        ]
        if texts:
            slides.append(f"Slide {number}\n" + "\n".join(texts))
    return "\n\n".join(slides)


def _xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), read_only=True, data_only=True)
    try:
        sheets = []
        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                cells = ["" if v is None else str(v) for v in row]
                if any(cells):
                    rows.append("\t".join(cells).rstrip())
            if rows:
                sheets.append(f"Sheet {sheet.title}\n" + "\n".join(rows))
        return "\n\n".join(sheets)
    finally:
        workbook.close()


def _ipynb(path: Path) -> str:
    notebook = json.loads(_read_text(path))
    cells = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        text = "".join(source) if isinstance(source, list) else str(source)
        if text.strip():
            cells.append(text.strip())
    return "\n\n".join(cells)
