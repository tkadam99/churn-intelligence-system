# How the Word guide is built

## What we built

The project has a Python helper called `tools/build_guide.py`. It reads every Markdown (`.md`) chapter in `docs/guide/` and builds one Word document, `PROJECT_GUIDE.docx`, in the project root.

The builder creates a title page, a contents page listing each chapter title, and the chapter pages. It supports the Markdown formatting used by this project: headings, bullet lists, numbered lists, tables, bold text, inline code, and fenced code blocks.

## Why it matters (and which part of a real ML analyst or ML engineer job it matches)

A written guide helps keep a project understandable and useful for study. Generating the Word document from Markdown gives us one editable source for each chapter. Markdown is easy to review as text and to track as a file in Git. We can rebuild the Word document after an edit instead of manually copying the chapter into Word and maintaining two versions.

The builder is also an example of a small automation task. In a real data or machine-learning job, automation can make a repeated reporting step more consistent and reduce manual formatting work. The guide itself is a project learning tool; it does not analyze customer data or train a model.

## How it works, step by step

1. **Find and order the chapters.** The script looks in `docs/guide/` for files ending in `.md`, then sorts the names without regard to uppercase or lowercase. For example, `00_project_overview.md` comes before `00b_guide_builder.md`.
2. **Prepare the Word document.** The script creates a blank document and sets normal paragraph text to Calibri, size 11. It also sets page margins and the heading font.
3. **Make the title page.** It writes “Churn Intelligence System: Project Guide” in the Word title style, centers it, and adds a page break (a command to start on the next page).
4. **Make the contents page.** It reads the first top-level Markdown heading (`# ...`) in each chapter and lists that heading as a bullet. This simple contents list does not include page numbers.
5. **Convert each chapter.** The script reads a chapter as UTF-8 text and examines it line by line. For example, `## Terms explained` becomes a Word heading, `- SQL` becomes a bullet, and `1. Ask a question` becomes a numbered list item.
6. **Keep list numbering separate.** Consecutive numbered lines share a numbering ID so they appear as one list. When another kind of line ends that list, the next numbered list receives a new ID and starts at 1.
7. **Format special text.** A Markdown table is recognized when a row of cells is followed by a separator row such as `| --- | --- |`. Text wrapped in `**two stars**` becomes bold. Text wrapped in single backticks, such as `sql/`, becomes Consolas code text with gray shading. Text between triple-backtick fences is written one code line per paragraph, also in Consolas with a light-gray background.
8. **Separate and save chapters.** Each chapter begins after a page break. Once all chapters are converted, the script saves the Word file and prints a message with the number of Markdown files it used.

## Code walkthrough (explain the important lines)

The following are real lines from `tools/build_guide.py`. Each example is followed by an explanation.

The script first finds the project folder and the two paths it will use:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUIDE_DIR = PROJECT_ROOT / "docs" / "guide"
OUTPUT_FILE = PROJECT_ROOT / "PROJECT_GUIDE.docx"
```

`__file__` is the path of the Python script that is running. `resolve()` makes it an absolute path, and `parents[1]` moves up from `tools/` to the project root. The other lines build the input folder and output document paths from that root, so the script can find them even if it is run from a different current folder.

The chapters are found and sorted here:

```python
guide_files = sorted(GUIDE_DIR.glob("*.md"), key=lambda path: path.name.lower())
```

`glob("*.md")` finds Markdown files in the guide folder. `sorted(...)` puts them in filename order. Converting each filename to lowercase for sorting makes the order less sensitive to capitalization.

Normal text and heading styles are prepared here:

```python
normal_style = document.styles["Normal"]
normal_style.font.name = "Calibri"
normal_style.font.size = Pt(11)
```

These lines select Word's normal paragraph style and set its font to Calibri at 11 points. A point is a unit used to describe text size.

The title page starts with:

```python
title = document.add_paragraph(style="Title")
title.add_run("Churn Intelligence System: Project Guide")
title.alignment = 1
document.add_page_break()
```

The first line creates a paragraph using Word's title style. The second writes the title into it. `alignment = 1` centers the title. The page break makes the following contents page start on a fresh page.

The contents heading and each chapter title are added with:

```python
document.add_heading("Table of contents", level=1)
paragraph.add_run(top_level_title(path))
```

`add_heading` creates a large heading. `top_level_title(path)` reads the chapter and returns its first `#` heading. The returned title is written as an entry in the contents list. The builder does not calculate page numbers.

Markdown files are read as text, one line at a time:

```python
lines = path.read_text(encoding="utf-8").splitlines()
```

`read_text` opens the file with UTF-8 encoding, a common way of representing text. `splitlines()` gives the script a list of lines to examine.

The heading and list patterns include:

```python
heading = re.match(r"^(#{1,3})\s+(.+)$", line)
```

This regular expression (a text-matching pattern) checks whether a line starts with one, two, or three `#` symbols followed by a title. The number of symbols chooses Word heading level 1, 2, or 3.

```python
elif re.match(r"^\s*[-*+]\s+", line):
```

This recognizes a Markdown bullet that starts with `-`, `*`, or `+`, allowing indentation before the marker.

```python
elif re.match(r"^\s*\d+[.)]\s+", line):
```

This recognizes a numbered item, such as `1. First step` or `1) First step`. The script removes the Markdown marker and asks Word to show the list number.

Each uninterrupted run of numbered items uses a fresh Word numbering ID:

```python
if current_numbering_id is None:
    current_numbering_id = new_numbering_id(document)
set_numbering_id(paragraph, current_numbering_id)
```

When a numbered list begins, `new_numbering_id` creates a new list identity based on Word's numbered-list style. All consecutive items receive that same identity. A paragraph that is not numbered clears the current identity, so a later numbered list gets a different one and restarts from 1.

Inline code and bold text are separated by this pattern:

```python
for part in re.split(r"(`[^`]+`|\*\*.*?\*\*)", text):
```

`re.split` breaks a line into normal text, backtick-wrapped code, and double-asterisk-wrapped bold text. The next lines remove the Markdown marks and format the content:

```python
if part.startswith("`") and part.endswith("`"):
    add_inline_code(paragraph, part[1:-1])
elif part.startswith("**") and part.endswith("**"):
    paragraph.add_run(part[2:-2]).bold = True
```

`part[1:-1]` keeps the text inside the single backticks, without the backticks themselves. The helper sets that text in Consolas and adds gray background shading. For bold text, `part[2:-2]` removes the two asterisks on each side, and `.bold = True` makes the Word text bold.

A table is recognized when the current line looks like a table row and the next line is a separator:

```python
if line.strip().startswith("|") and index + 1 < len(lines):
    if is_table_separator(lines[index + 1]):
```

The first condition checks for a row that starts with `|` and makes sure another line exists. The second checks that the next line is made of Markdown table dashes (and optional alignment colons). The builder then splits the rows into cells and fills a Word table. The first row is made bold to serve as the header.

A fenced code block is collected before it is written:

```python
if line.strip().startswith("```"):
    if in_code_block:
        for code_line in code_lines:
            add_code_line(document, code_line)
```

A line starting with triple backticks marks the start or end of a code block. While inside the block, lines are saved without Markdown interpretation. When the closing fence is reached, each saved line is added using the code-block formatter.

The code-block formatter gives each code line its appearance:

```python
run.font.name = "Consolas"
run.font.size = Pt(10)
```

These real lines set the code font to Consolas at 10 points. The formatter also adds a light-gray background to each paragraph using Word's document formatting information.

Finally, the complete document is saved and a result message is printed:

```python
document.save(OUTPUT_FILE)
print(f"Created {OUTPUT_FILE.name} from {len(guide_files)} guide file(s).")
```

The first line writes the `.docx` file at the project root. The second reports the output filename and the number of chapter files processed. The message confirms what the script did; checking the file itself confirms that the document exists.

## Data flow (what goes in, what happens, what comes out)

| What goes in | What happens | What comes out |
| --- | --- | --- |
| Markdown chapter files in `docs/guide/` | The script sorts the files, reads their lines, and converts recognized formatting into Word paragraphs, lists, and tables | `PROJECT_GUIDE.docx` at the project root |

The guide text is the input. The Python script and `python-docx` do the conversion. The output is a Word file; the script does not add project data or model results.

## Terms explained

- **Markdown:** a simple text format that uses marks such as `#`, `-`, and `**` to describe headings, lists, and emphasis.
- **Parser:** a program that reads formatted text and identifies its parts. This builder uses simple patterns rather than implementing every Markdown feature.
- **Regular expression:** a pattern used to find text with a particular shape, such as a heading or list marker.
- **Word paragraph:** a block of text in a Word document. A heading or each line in a list is stored as a paragraph.
- **Run:** a piece of a paragraph that can have its own formatting, such as bold or a different font.
- **Table of contents:** a list of chapter titles that helps a reader find sections. This builder's contents list is not automatically page-numbered.
- **Numbering ID:** Word's label for one numbered list. Giving separate lists different IDs lets each list start at 1.
- **`python-docx`:** the Python package this script uses to create and save Word documents.
- **Word document formatting data:** structured information stored inside the document to describe details such as shading and numbering.

## What could go wrong

- If `python-docx` is not installed in the active Python environment, the script cannot import its document tools.
- If `docs/guide/` contains no Markdown files, the title and contents pages are still made, but no chapters are added.
- If a Markdown table is malformed or contains an unescaped pipe inside a cell, the simple cell splitter may divide it incorrectly.
- An unmatched backtick or triple-backtick fence may prevent text from being formatted as intended.
- The parser supports the Markdown forms described here, not every feature of the full Markdown specification.
- The contents list shows chapter titles but not page numbers. Adding accurate page numbers requires a Word layout/update step beyond this simple builder.
- A failed save, such as when Word has the output file locked, can prevent the new document from being written.

## Interview questions with model answers

**Question: Why generate the Word guide from Markdown instead of editing the Word file by hand?**  
**Model answer:** Markdown gives the project one simple, reviewable source for its guide chapters. The builder can repeatedly turn those files into the same Word format, so we do not have to copy text and formatting by hand after every edit. This also reduces the chance that the Markdown and Word versions drift apart. We still need to open or inspect the generated Word file to confirm the output.

**Question: How does the builder decide which Markdown line becomes a heading?**  
**Model answer:** It checks the start of each line for one to three `#` symbols followed by spaces and heading text. The number of `#` marks selects Word heading level 1, 2, or 3. A line such as `## Data flow` becomes a level-2 Word heading.

**Question: How does the builder keep two numbered lists from continuing one another?**  
**Model answer:** It gives every uninterrupted group of numbered items a separate Word numbering ID. Consecutive numbered lines share one ID. When another kind of line ends the group, the builder clears the current ID, so the next numbered item creates a fresh list that begins at 1.

**Question: What is the difference between inline code and a fenced code block in this guide?**  
**Model answer:** Inline code is a short piece of code inside a sentence, written between single backticks, such as `sql/`. The builder changes only that piece to Consolas and gives it gray shading. A fenced code block starts and ends with triple backticks; the builder collects all the lines between them and writes each line as a separate shaded Consolas paragraph.

**Question: How does the builder recognize a Markdown table?**  
**Model answer:** It looks for a line beginning with a pipe (`|`) and checks whether the following line has table separator cells made from dashes, with optional colons. Once it recognizes the header and separator, it splits the rows at the pipes and places their contents into a Word table. This is a simple parser, so unusual Markdown tables may need adjustment.

**Question: Why might the contents page not show accurate page numbers?**  
**Model answer:** The builder creates a list of chapter titles, but `python-docx` does not lay out pages like Microsoft Word does. Final page positions can change with fonts, margins, or Word's rendering. Reliable page numbers therefore require an additional Word-side layout or field-update step; this builder deliberately leaves them out.

## Check yourself (2-3 questions)

1. What are the Markdown files going into the builder, and what Word file comes out?
2. What line pattern starts a numbered list item, and how does the builder make a later list restart?
3. How do inline code and fenced code blocks differ in Markdown?
