from __future__ import annotations

import re
from typing import List, Dict
from config import MAX_TELEGRAM_MESSAGE_LEN

# Набор спецсимволов, которые Telegram требует экранировать в MarkdownV2
MD_V2_SPECIAL_CHARS = set("_*[]()~`>#+-=|{}.!")

# Паттерн заголовков вида #..###### Заголовок
HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# Паттерн ссылок [text](url) и ![alt](url
LINK_RE = re.compile(r"!?\[([^\]]+)\]\(([^)]+)\)")

# Паттерны для таблиц
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|?\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$")

# Паттерны для формул LaTeX-подобного вида
FORMULA_INLINE_RE = re.compile(r"\$(.+?)\$")
FORMULA_BLOCK_RE = re.compile(r"^\s*\$(.+)\$\s*$")
FORMULA_DBL_BLOCK_RE = re.compile(r"^\s*\$\$(.+)\$\$\s*$")

# Плейсхолдеры для "защищённых" фрагментов (код, ссылки, готовое форматирование)
_PLACEHOLDER_PREFIX = "\u0000P"
_PLACEHOLDER_SUFFIX = "\u0000"

# Паттерны для сплита
HR_SPLIT_RE = re.compile(
    r"^\s*(\\-\\-\\-|\\\*\\\*\\\*|\\_\\_\\_)\s*$"
)
CODE_FENCE_LINE_RE = re.compile(r"^(`{3,})(.*)$")
BOLD_LINE_RE = re.compile(r"^\s*\*.+\*\s*$")


def _escape_md_v2(text: str) -> str:
    """
    Экранирует спецсимволы MarkdownV2 во всём тексте.
    Используется для "голого" текста без разметки.
    """
    return re.sub(r"([_*[\]()~`>#+\-=|{}.!])", r"\\\1", text)


def _escape_md_v2_code(text: str) -> str:
    """
    Экранирует текст внутри кодовых блоков/инлайн-кода.
    В Telegram внутри кода нужно экранировать только backslash и `.
    """
    return text.replace("\\", "\\\\").replace("`", "\\`")


def _escape_md_v2_link_url(url: str) -> str:
    """
    Экранирует URL внутри () части ссылки.
    """
    return re.sub(r"([()])", r"\\\1", url)




def _new_placeholder(store: Dict[str, str], value: str) -> str:
    """
    Создаёт уникальный плейсхолдер и сохраняет для последующей подстановки.
    """
    key = f"{_PLACEHOLDER_PREFIX}{len(store)}{_PLACEHOLDER_SUFFIX}"
    store[key] = value
    return key


def _process_inline(text: str) -> str:
    """
    Обрабатывает инлайновое форматирование в строке и возвращает корректный MarkdownV2.
    Логика:
      1. Выкусываем "особые" конструкции (код, ссылки, жирный, курсив, подчёркивание,
         зачёркнутый, спойлер) и заменяем их на плейсхолдеры.
      2. Оставшийся plain text экранируем _escape_md_v2.
      3. Подставляем плейсхолдеры обратно.
    """
    placeholders: Dict[str, str] = {}
    processed = text

    # Спец-кейс: ~~`code`~~ -> `code`
    def repl_strike_code(m: re.Match) -> str:
        inner = _escape_md_v2_code(m.group(1))
        return _new_placeholder(placeholders, f"`{inner}`")

    processed = re.sub(r"~~\s*`([^`]+)`\s*~~", repl_strike_code, processed)

    # Инлайн-код `code`
    def repl_code(m: re.Match) -> str:
        inner = _escape_md_v2_code(m.group(1))
        return _new_placeholder(placeholders, f"`{inner}`")

    processed = re.sub(r"`([^`]+)`", repl_code, processed)

    # Уже экранированные пользователем символы markdown: \* \_ \# ...
    def repl_md_escape(m: re.Match) -> str:
        return m.group(1)

    processed = re.sub(r"\\([_*[\]()~`>#+\-=|{}.!])", repl_md_escape, processed)

    # Формулы $...$ -> инлайн-код `...`
    def repl_formula(m: re.Match) -> str:
        inner = _escape_md_v2_code(m.group(1).strip())
        return _new_placeholder(placeholders, f"`{inner}`")

    processed = FORMULA_INLINE_RE.sub(repl_formula, processed)

    # Ссылки [text](url) и ![alt](url
    def repl_link(m: re.Match) -> str:
        text_inner = _escape_md_v2(m.group(1))
        body = m.group(2).strip()

        # Есть ли title?
        m_title = re.match(r'(\S+)\s+"([^"]+)"$', body)
        if m_title:
            url_raw = m_title.group(1)
            title = m_title.group(2)

            title_text = _escape_md_v2(title)
            title_suffix = f" \\({title_text}\\)"
        else:
            url_raw = body
            title_suffix = ""

        url_inner = _escape_md_v2_link_url(url_raw)

        link_md = f"[{text_inner}]({url_inner}){title_suffix}"
        return _new_placeholder(placeholders, link_md)

    processed = LINK_RE.sub(repl_link, processed)

    # Многократные звёздочки ***text***, ****text****, *****text***** ...
    # Чётное количество -> bold (*text*), нечётное -> italic (_text_)
    def repl_multi_stars(m: re.Match) -> str:
        stars = m.group(1)
        inner_raw = m.group(2)
        count = len(stars)
        inner = _escape_md_v2(inner_raw)

        if count % 2 == 0:
            # чётное -> жирный
            return _new_placeholder(placeholders, f"*{inner}*")
        else:
            # нечётное -> курсив
            return _new_placeholder(placeholders, f"_{inner}_")

    # (\*{3,}) - левая группа из 3+ звёздочек, \1 - такая же справа
    processed = re.sub(r"(\*{3,})(.+?)\1", repl_multi_stars, processed)

    # Жирный **text** -> *text*
    def repl_bold(m: re.Match) -> str:
        inner = _escape_md_v2(m.group(1))
        return _new_placeholder(placeholders, f"*{inner}*")

    processed = re.sub(r"\*\*(.+?)\*\*", repl_bold, processed)

    # Подчёркивание __text__ -> __text__
    def repl_underline(m: re.Match) -> str:
        inner = _escape_md_v2(m.group(1))
        return _new_placeholder(placeholders, f"__{inner}__")

    processed = re.sub(r"__(.+?)__", repl_underline, processed)

    # Курсив *text* -> _text_
    # Стараемся не зацепить уже обработанный **bold**
    def repl_italic(m: re.Match) -> str:
        inner = _escape_md_v2(m.group(1))
        return _new_placeholder(placeholders, f"_{inner}_")
    
    # курсив через звёздочки
    processed = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", repl_italic, processed)
    # курсив через подчёркивания
    processed = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", repl_italic, processed)

    # Зачёркнутый ~~text~~ -> ~text~
    def repl_strike(m: re.Match) -> str:
        inner = _escape_md_v2(m.group(1))
        return _new_placeholder(placeholders, f"~{inner}~")

    processed = re.sub(r"~~(.+?)~~", repl_strike, processed)

    # Спойлер ||text|| -> ||text||
    def repl_spoiler(m: re.Match) -> str:
        inner = _escape_md_v2(m.group(1))
        return _new_placeholder(placeholders, f"||{inner}||")

    processed = re.sub(r"\|\|(.+?)\|\|", repl_spoiler, processed)

    # На оставшийся текст навешиваем экранирование
    escaped_plain = _escape_md_v2(processed)

    # Многошаговое раскрытие плейсхолдеров:
    # плейсхолдеры могут быть вложенными (один внутри значения другого),
    # поэтому гоняем замену, пока в тексте остаются маркеры P\d+.
    placeholder_pattern = re.compile(
        re.escape(_PLACEHOLDER_PREFIX) + r"\d+" + re.escape(_PLACEHOLDER_SUFFIX)
    )

    while placeholder_pattern.search(escaped_plain):
        replaced = escaped_plain
        for key, value in placeholders.items():
            replaced = replaced.replace(key, value)
        if replaced == escaped_plain:
            break
        escaped_plain = replaced

    return escaped_plain


def _is_table_header(line: str) -> bool:
    return bool(TABLE_ROW_RE.match(line))


def _is_table_sep(line: str) -> bool:
    return bool(TABLE_SEP_RE.match(line))


def _split_table_row(line: str) -> List[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _normalize_table_cell(cell: str) -> str:
    """
    Внутри таблиц разрешаем пользовательское экранирование Markdown-символов:
    \* -> *, \_ -> _, \# -> #, \[ -> [, \] -> ]
    Но НЕ трогаем такие последовательности внутри inline-code (`...`).
    """
    ESCAPABLE = set("*_#[]")
    result: List[str] = []
    i = 0
    n = len(cell)
    in_code = False

    while i < n:
        ch = cell[i]

        if ch == "`":
            in_code = not in_code
            result.append(ch)
            i += 1
            continue

        if not in_code and ch == "\\" and i + 1 < n and cell[i + 1] in ESCAPABLE:
            # \* -> *
            result.append(cell[i + 1])
            i += 2
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def _render_table_block(lines: List[str]) -> List[str]:
    """
    Превращает таблицу в выровненную псевдотаблицу в кодовом блоке.
    """
    parsed_rows: List[List[str]] = []

    for idx, ln in enumerate(lines):
        # вторую строку с :---|:---|:--- считаем разделителем и пропускаем
        if idx == 1 and _is_table_sep(ln):
            continue
        raw_cells = _split_table_row(ln)
        normalized_cells = [_normalize_table_cell(c) for c in raw_cells]
        parsed_rows.append(normalized_cells)

    if not parsed_rows:
        # fallback: просто отдать как есть, экранировав
        return ["```"] + [_escape_md_v2_code(ln) for ln in lines] + ["```"]

    max_cols = max(len(r) for r in parsed_rows)
    for r in parsed_rows:
        if len(r) < max_cols:
            r.extend([""] * (max_cols - len(r)))

    # ширина колонок по max длине текста в колонке
    widths = [max(len(c) for c in col) for col in zip(*parsed_rows)]

    text_rows: List[str] = []

    # заголовок
    header = parsed_rows[0]
    header_line = " | ".join(c.ljust(widths[i]) for i, c in enumerate(header))
    text_rows.append(header_line)

    # разделитель
    sep_line = "-+-".join("-" * widths[i] for i in range(max_cols))
    text_rows.append(sep_line)

    # данные
    for row in parsed_rows[1:]:
        text_rows.append(" | ".join(c.ljust(widths[i]) for i, c in enumerate(row)))

    # заворачиваем всё в кодовый блок
    out: List[str] = ["```"]
    out.extend(_escape_md_v2_code(r) for r in text_rows)
    out.append("```")
    return out


def convert_to_md_v2(text: str) -> str:
    """
    Конвертирует "обычный" markdown (как его генерирует LLM) в Telegram MarkdownV2.

    Поддерживает:
      - #..###### заголовки -> жирный текст
      - **bold**
      - *italic*
      - _italic_
      - __underline__
      - ~~strike~~
      - ||spoiler||
      - `inline code`
      - ```code blocks``` (включая внешние ````markdown ... ````)
      - $inline-формулы$ -> `inline`
      - блочные формулы \n$...$\n и $$...$$ -> кодовый блок
      - [text](url) и ![alt](url
      - > цитаты
      - таблицы -> псевдотаблица в кодовом блоке
    """
    lines = text.splitlines()
    out_lines: List[str] = []

    in_code_block = False
    code_fence_len = 0

    i = 0
    n = len(lines)

    while i < n:
        raw_line = lines[i]
        line = raw_line
        stripped = line.lstrip()

        # если уже внутри кодового блока
        if in_code_block:
            # попытка закрыть кодовый блок?
            if stripped.startswith("```"):
                m = re.match(r"^(`{3,})(.*)$", stripped)
                if m:
                    fence, _ = m.groups()
                    fence_len = len(fence)
                    if fence_len == code_fence_len:
                        in_code_block = False
                        code_fence_len = 0
                        out_lines.append("```")
                        i += 1
                        continue
            # обычная строка кода
            out_lines.append(_escape_md_v2_code(line))
            i += 1
            continue

        # детект таблицы
        if _is_table_header(line) and i + 1 < n and _is_table_sep(lines[i + 1]):
            table_lines: List[str] = []
            j = i
            while j < n and TABLE_ROW_RE.match(lines[j]):
                table_lines.append(lines[j])
                j += 1

            out_lines.extend(_render_table_block(table_lines))
            i = j
            continue

        # детект формулы
        if stripped == "$$":
            j = i + 1
            formula_lines: List[str] = []

            # собираем всё до следующей строки с $$ или конца текста
            while j < n and lines[j].strip() != "$$":
                formula_lines.append(lines[j])
                j += 1

            # рендерим как кодовый блок
            out_lines.append("```")
            for fl in formula_lines:
                out_lines.append(_escape_md_v2_code(fl))
            out_lines.append("```")

            # пропускаем закрывающую $$, если она есть
            if j < n and lines[j].strip() == "$$":
                i = j + 1
            else:
                i = j
            continue

        # блочная формула: $$ ... $$
        m_dbl_block = FORMULA_DBL_BLOCK_RE.match(line)
        if m_dbl_block:
            inner_raw = m_dbl_block.group(1).strip()
            code_line = _escape_md_v2_code(inner_raw)
            out_lines.append("```")
            out_lines.append(code_line)
            out_lines.append("```")
            i += 1
            continue

        # блочная формула: \n$ ... \n$
        m_formula_block = FORMULA_BLOCK_RE.match(line)
        if m_formula_block:
            inner_raw = m_formula_block.group(1).strip()
            code_line = _escape_md_v2_code(inner_raw)
            out_lines.append("```")
            out_lines.append(code_line)
            out_lines.append("```")
            i += 1
            continue

        # кодовые блоки ```...``` / ````...```
        if stripped.startswith("```"):
            m = re.match(r"^(`{3,})(.*)$", stripped)
            if m:
                fence, rest = m.groups()
                fence_len = len(fence)

                # открываем новый блок кода
                in_code_block = True
                code_fence_len = fence_len

                lang = rest.strip()
                if lang:
                    out_lines.append(f"```{lang}")
                else:
                    out_lines.append("```")

                i += 1
                continue

        # Заголовки #..###### ---
        m = HEADER_RE.match(line)
        if m:
            content = m.group(2).strip()
            inner = _process_inline(content)
            out_lines.append(f"*{inner}*")
            i += 1
            continue

        # Цитаты >
        if stripped.startswith(">"):
            m_quote = re.match(r"^\s*(>+)\s?(.*)$", line)
            if m_quote:
                quote_marks = m_quote.group(1)   # '>', '>>', '>>>'
                content = m_quote.group(2) or "" # текст без '>'
                inner = _process_inline(content) if content else ""
                if inner:
                    out_lines.append(f"{quote_marks} {inner}")
                else:
                    out_lines.append(quote_marks)
                i += 1
                continue

        # Обычная строка с инлайновым форматированием
        out_lines.append(_process_inline(line))
        i += 1

    # если LLM открыл блок кода и не закрыл - закроем за него
    if in_code_block:
        out_lines.append("```")

    return "\n".join(out_lines)


def _split_on_horizontal_rules(text: str) -> List[str]:
    """
    Делит текст на секции по строкам-разделителям:

      ---
      ***
      ___

    Эти строки в вывод не попадают.
    """
    lines = text.splitlines(keepends=True)
    sections: List[str] = []
    buf: List[str] = []

    for line in lines:
        if HR_SPLIT_RE.match(line.strip()):
            if buf:
                sections.append("".join(buf).rstrip())
                buf = []
            # разделитель выкидываем
            continue
        buf.append(line)

    if buf:
        sections.append("".join(buf).rstrip())

    return sections


def _parse_blocks(section: str) -> List[Dict[str, str]]:
    """
    Разбивает секцию на блоки двух типов:
      - {"type": "text", "text": "..."}
      - {"type": "code", "text": "```lang\\n...```"}
    """
    lines = section.splitlines(keepends=True)
    blocks: List[Dict[str, str]] = []

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.lstrip()
        m = CODE_FENCE_LINE_RE.match(stripped)

        if m:
            # Блок кода c ```...```
            opening = lines[i]
            i += 1
            body_lines: List[str] = []

            # Собираем до закрывающего ```
            while i < n:
                l2 = lines[i]
                if l2.lstrip().startswith("```"):
                    closing = l2
                    i += 1
                    break
                body_lines.append(l2)
                i += 1
            else:
                closing = ""

            full_block = opening + "".join(body_lines) + closing
            blocks.append({"type": "code", "text": full_block})
        else:
            # Обычный текстовый блок до ближайшего ``` или конца
            text_lines = [line]
            i += 1
            while i < n and not CODE_FENCE_LINE_RE.match(lines[i].lstrip()):
                text_lines.append(lines[i])
                i += 1
            blocks.append({"type": "text", "text": "".join(text_lines)})

    return blocks


def _smart_split_point(text: str, max_len: int) -> int:
    """
    Ищет "красивую" позицию разрыва в пределах max_len (для обычного текста).

    Приоритет деления:
      - по двойным переводам строк (\n\n)
      - по одиночному переводу строки (\n)
      - по табуляции (\t)
      - по точка+пробел (". ")
      - по пробелу
      - в противном случае - жёстко режем на max_len
    """
    if len(text) <= max_len:
        return len(text)

    window = text[:max_len]

    # \n\n
    idx = window.rfind("\n\n")
    if idx != -1:
        return idx + 2

    # \n
    idx = window.rfind("\n")
    if idx != -1:
        return idx + 1

    # \t
    idx = window.rfind("\t")
    if idx != -1:
        return idx + 1

    # ". "
    idx = window.rfind(". ")
    if idx != -1:
        return idx + 2

    # пробел
    idx = window.rfind(" ")
    if idx != -1:
        return idx + 1

    # жёсткий разрез
    return max_len


def _smart_split_point_code(text: str, max_len: int) -> int:
    """
    "Красивый" разрыв для кода (без точки-пробела).
    Приоритет:
      - \n\n
      - \n
      - \t
      - пробел
      - иначе жёсткий разрез
    """
    if len(text) <= max_len:
        return len(text)

    window = text[:max_len]

    idx = window.rfind("\n\n")
    if idx != -1:
        return idx + 2

    idx = window.rfind("\n")
    if idx != -1:
        return idx + 1

    idx = window.rfind("\t")
    if idx != -1:
        return idx + 1

    idx = window.rfind(" ")
    if idx != -1:
        return idx + 1

    return max_len


def _split_long_code_block(block_text: str, limit: int) -> List[str]:
    """
    Делит один кодовый блок (с уже расставленными ```...```) на несколько
    самостоятельных код-блоков, если он сам по себе превышает limit.

    Каждый кусок будет иметь свои собственные:
      ```lang
      ...часть кода...
      ```
    """
    lines = block_text.splitlines(keepends=True)
    if not lines:
        return []

    # Первая строка - header с ```lang
    header = lines[0]

    # Ищем последнюю строку, начинающуюся с ```
    footer_idx = None
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].lstrip().startswith("```"):
            footer_idx = idx
            break

    if footer_idx is None:
        body_lines = lines[1:]
        footer = ""
    else:
        body_lines = lines[1:footer_idx]
        footer = lines[footer_idx]

    header_len = len(header)
    footer_len = len(footer)
    max_body = max(1, limit - header_len - footer_len)

    body_text = "".join(body_lines)
    chunks: List[str] = []

    rest = body_text
    while rest:
        if len(rest) <= max_body:
            part = rest
            rest = ""
        else:
            split_at = _smart_split_point_code(rest, max_body)
            part = rest[:split_at]
            rest = rest[split_at:]

        chunks.append(header + part + footer)

    return chunks


def _detach_last_bold_line(text: str) -> (str, str):
    """
    Ищет последнюю "жирную" строку вида *Заголовок*,
    возвращает кортеж (prefix, bold_tail), где:

      text = prefix + bold_tail

    Если жирной строки нет - bold_tail == "".
    """
    lines = text.splitlines(keepends=True)
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if BOLD_LINE_RE.match(line.strip()):
            prefix = "".join(lines[:idx])
            bold_tail = "".join(lines[idx:])
            return prefix, bold_tail
    return text, ""


def _append_text_block(
    chunks: List[str],
    current: str,
    block_text: str,
    limit: int,
) -> (List[str], str):
    """
    Добавляет текстовый блок в текущий chunk с учётом лимита.
    Если блок не помещается, режет его по _smart_split_point().
    """
    text = block_text

    while text:
        remaining = limit - len(current)
        if remaining <= 0:
            # Текущий chunk заполнен - отправляем и начинаем новый
            if current.strip():
                chunks.append(current.rstrip())
            current = ""
            remaining = limit

        # всё целиком помещается
        if len(text) <= remaining:
            current += text
            break

        # нужно разделить внутри блока
        split_at = _smart_split_point(text, remaining)
        # защита от зацикливания
        if split_at <= 0:
            # ничего не помещается - вынужденно режем жёстко
            split_at = remaining

        current += text[:split_at]
        if current.strip():
            chunks.append(current.rstrip())
        current = ""
        text = text[split_at:].lstrip()

    return chunks, current


def _append_code_block(
    chunks: List[str],
    current: str,
    block_text: str,
    limit: int,
) -> (List[str], str):
    """
    Добавляет кодовый блок в разбиение с учётом правил:
      - если блок не помещается в остаток текущего сообщения,
        переносим его в новое сообщение;
      - при переносе ищем предшествующую строку вида *Заголовок* и
        переносим её вместе с блоком в новое сообщение;
      - если сам кодовый блок > limit, делим его на несколько самостоятельных
        код-блоков (см. _split_long_code_block).
    """
    # Блок сам по себе длиннее лимита -> делим его на несколько
    if len(block_text) > limit:
        bold_tail = ""
        if current:
            prefix, bold_tail = _detach_last_bold_line(current)
            if prefix.strip():
                chunks.append(prefix.rstrip())
            current = ""  # оставляем только возможный bold_tail для использования ниже

        pieces = _split_long_code_block(block_text, limit)
        first = True
        for piece in pieces:
            piece = piece.rstrip()
            if first and bold_tail:
                merged = bold_tail + piece
                if len(merged) <= limit:
                    chunks.append(merged.rstrip())
                else:
                    # не поместился вместе - отправляем заголовок отдельно
                    chunks.append(bold_tail.rstrip())
                    chunks.append(piece)
                bold_tail = ""
                first = False
            else:
                chunks.append(piece)
        return chunks, current

    # Блок умещается в одно сообщение
    if len(current) + len(block_text) <= limit:
        # просто добавляем к текущему
        current += block_text
        return chunks, current

    # Блок не помещается в остаток текущего сообщения -> новый chunk
    bold_tail = ""
    if current:
        prefix, bold_tail = _detach_last_bold_line(current)
        if prefix.strip():
            chunks.append(prefix.rstrip())
        # bold_tail пойдёт в новое сообщение вместе с кодом
        current = ""

    new_chunk = (bold_tail + block_text).rstrip()
    chunks.append(new_chunk)
    return chunks, ""


def _split_section(section: str, limit: int) -> List[str]:
    """
    Сплитит одну «логическую секцию» (между ---/***/___) на сообщения.
    """
    blocks = _parse_blocks(section)
    chunks: List[str] = []
    current = ""

    for blk in blocks:
        if blk["type"] == "text":
            chunks, current = _append_text_block(chunks, current, blk["text"], limit)
        else:
            chunks, current = _append_code_block(chunks, current, blk["text"], limit)

    if current.strip():
        chunks.append(current.rstrip())

    return chunks


def split_md_v2(text: str, limit: int = MAX_TELEGRAM_MESSAGE_LEN) -> List[str]:
    """
    Разбивает итоговый Telegram.MarkdownV2-текст на части, каждая из которых
    укладывается в лимит Telegram по длине.

    Правила:
      1. Строки из `---`, `***`, `___` (окружённые переводами строк) делят текст
         на независимые секции.
      2. При достижении лимита текста внутри секции:
         - текст режется по "красивым" позициям (_smart_split_point);
         - кодовые блоки либо целиком переносятся в новое сообщение
           вместе с предшествующим жирным заголовком (*Заголовок*),
           либо делятся на несколько самостоятельных код-блоков,
           если они сами превышают лимит.
      3. В конце все сообщения `strip()`-ятся; пустые/пробельные отбрасываются.
    """
    # Сначала делим на секции по горизонтальным разделителям
    sections = _split_on_horizontal_rules(text)

    all_chunks: List[str] = []
    for sec in sections:
        if not sec.strip():
            continue
        all_chunks.extend(_split_section(sec, limit))

    # Финальная очистка: убираем пробелы по краям и пустые сообщения
    result: List[str] = []
    for ch in all_chunks:
        cleaned = ch.strip()
        if cleaned:
            result.append(cleaned)

    return result
