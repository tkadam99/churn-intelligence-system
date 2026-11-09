import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUIDE_DIR = PROJECT_ROOT / "docs" / "guide"
OUTPUT_FILE = PROJECT_ROOT / "PROJECT_GUIDE.docx"
CODE_BACKGROUND = "F2F2F2"


def add_bold_text(paragraph, text):
    """Write plain text while keeping **bold** spans bold."""
    for part in re.split(r"(\*\*.*?\*\*)", text):
        if part.startswith("**") and part.endswith("**"):
            paragraph.add_run(part[2:-2]).bold = True
        else:
            paragraph.add_run(part)


def table_cells(line):
    """Split one Markdown table row into its cell text."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_table_separator(line):
    cells = table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def add_markdown_table(document, lines):
    rows = [table_cells(line) for line in lines if not is_table_separator(line)]
    if not rows:
        return

    column_count = max(len(row) for row in rows)
    table = document.add_table(rows=0, cols=column_count)
    table.style = "Table Grid"
    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for column_index, cell in enumerate(cells):
            value = values[column_index] if column_index < len(values) else ""
            paragraph = cell.paragraphs[0]
            add_bold_text(paragraph, value)
            if row_index == 0:
                for run in paragraph.runs:
                    run.bold = True


def add_code_line(document, text):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(0)
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), CODE_BACKGROUND)
    paragraph._p.get_or_add_pPr().append(shading)
    run = paragraph.add_run(text or " ")
    run.font.name = "Consolas"
    run.font.size = Pt(10)


def add_markdown_file(document, path):
    lines = path.read_text(encoding="utf-8").splitlines()
    in_code_block = False
    code_lines = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if line.strip().startswith("```"):
            if in_code_block:
                for code_line in code_lines:
                    add_code_line(document, code_line)
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            index += 1
            continue

        if in_code_block:
            code_lines.append(line)
            index += 1
            continue

        if line.strip().startswith("|") and index + 1 < len(lines):
            if is_table_separator(lines[index + 1]):
                table_lines = [line, lines[index + 1]]
                index += 2
                while index < len(lines) and lines[index].strip().startswith("|"):
                    table_lines.append(lines[index])
                    index += 1
                add_markdown_table(document, table_lines)
                continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            paragraph = document.add_paragraph(style=f"Heading {level}")
            add_bold_text(paragraph, heading.group(2).strip())
        elif re.match(r"^\s*[-*+]\s+", line):
            paragraph = document.add_paragraph(style="List Bullet")
            add_bold_text(paragraph, re.sub(r"^\s*[-*+]\s+", "", line))
        elif re.match(r"^\s*\d+[.)]\s+", line):
            paragraph = document.add_paragraph(style="List Number")
            add_bold_text(paragraph, re.sub(r"^\s*\d+[.)]\s+", "", line))
        elif line.strip():
            paragraph = document.add_paragraph()
            add_bold_text(paragraph, line.strip())

        index += 1

    if in_code_block:
        for code_line in code_lines:
            add_code_line(document, code_line)


def top_level_title(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^#\s+(.+)$", line)
        if match:
            return match.group(1).strip()
    return path.stem.replace("_", " ")


def build_guide():
    guide_files = sorted(GUIDE_DIR.glob("*.md"), key=lambda path: path.name.lower())
    document = Document()
    normal_style = document.styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = Pt(11)
    for style_name in ("Heading 1", "Heading 2", "Heading 3"):
        document.styles[style_name].font.name = "Calibri"

    section = document.sections[0]
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)

    title = document.add_paragraph(style="Title")
    title.add_run("Churn Intelligence System: Project Guide")
    title.alignment = 1
    document.add_page_break()

    document.add_heading("Table of contents", level=1)
    for path in guide_files:
        paragraph = document.add_paragraph(style="List Bullet")
        paragraph.add_run(top_level_title(path))
    document.add_page_break()

    for file_index, path in enumerate(guide_files):
        if file_index:
            document.add_page_break()
        add_markdown_file(document, path)

    document.save(OUTPUT_FILE)
    print(f"Created {OUTPUT_FILE.name} from {len(guide_files)} guide file(s).")


if __name__ == "__main__":
    build_guide()
