"""Создаёт небольшой воспроизводимый текстовый PDF для проверки локального pipeline."""

import argparse
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def create_example(path: Path) -> None:
    """Добавляет настоящие текстовые страницы со стандартным PDF-шрифтом."""
    pages = [
        [
            "Supply agreement",
            "Supplier: Alpha LLC. Buyer: Beta LLC.",
            "The delivery date is 15 December 2026.",
        ],
        [
            "4.2 Late delivery penalty",
            "The Supplier pays a penalty of 0.1% of the goods price",
            "for each day of late delivery.",
            "The agreement does not specify any other penalty rate.",
        ],
    ]
    writer = PdfWriter()
    for lines in pages:
        page = writer.add_blank_page(width=595, height=842)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            }
        )
        operations = ["BT /F1 12 Tf 50 780 Td 18 TL"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            operations.append(f"({escaped}) Tj T*")
        operations.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(operations).encode("ascii"))
        page[NameObject("/Contents")] = stream
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as target:
        writer.write(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    create_example(arguments.output)
