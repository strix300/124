"""Сохраняет исходные PDF под управляемыми сервером путями."""

import hashlib
import tempfile
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

from app.core.config import Settings
from app.core.errors import ApplicationError


class FileStore:
    """Хранит оригиналы до явного удаления документа."""

    def __init__(self, settings: Settings) -> None:
        self.root = settings.storage_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = settings.max_upload_bytes

    def stage(self, stream: BinaryIO) -> tuple[Path, str]:
        """Потоково проверяет размер, сигнатуру и хеш, не загружая весь PDF в память."""
        digest = hashlib.sha256()
        total = 0
        with tempfile.NamedTemporaryFile(
            dir=self.root,
            prefix="upload-",
            suffix=".tmp",
            delete=False,
        ) as target:
            path = Path(target.name)
            try:
                while block := stream.read(1024 * 1024):
                    if total == 0 and not block.startswith(b"%PDF-"):
                        raise ApplicationError(
                            "unsupported_format",
                            "Поддерживаются только PDF.",
                            415,
                        )
                    total += len(block)
                    if total > self.max_bytes:
                        raise ApplicationError("upload_too_large", "Превышен размер загрузки.", 413)
                    digest.update(block)
                    target.write(block)
                if total == 0:
                    raise ApplicationError("empty_upload", "Загружен пустой файл.")
                target.flush()
            except BaseException:
                path.unlink(missing_ok=True)
                raise
        return path, digest.hexdigest()

    def path(self, doc_id: UUID) -> Path:
        """Получает путь только из проверенного UUID, без частей пользовательского имени."""
        return self.root / f"{doc_id}.pdf"

    def commit(self, staged: Path, doc_id: UUID) -> None:
        """Атомарно завершает загрузку после определения логического документа."""
        destination = self.path(doc_id)
        if destination.exists():
            staged.unlink()
        else:
            staged.replace(destination)

    def delete(self, doc_id: UUID) -> None:
        """Удаляет оригинал только при явном удалении документа."""
        self.path(doc_id).unlink(missing_ok=True)
