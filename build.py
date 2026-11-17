#!/usr/bin/env python3
"""把 posts/ 里的 Markdown 构建成 _site/ 下的静态网站。

用法：
    python build.py            # 构建到 _site/
    python build.py --serve    # 构建并在 http://localhost:8000 预览，改文件后自动重建
    python build.py --drafts   # 连 draft: true 的草稿一起构建
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import http.server
import math
import re
import shutil
import sys
import threading
import time
import xml.etree.ElementTree as etree
from pathlib import Path

import markdown
import yaml
from markdown.extensions import Extension
from markdown.inlinepatterns import InlineProcessor
from markdown.treeprocessors import Treeprocessor

ROOT = Path(__file__).resolve().parent
POSTS_DIR = ROOT / "posts"
THEME_DIR = ROOT / "theme"
STATIC_DIR = ROOT / "static"
OUT = ROOT / "_site"

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

STRINGS = {
    "en": {
        "blog": "Blog", "posts": "Posts", "series": "Series", "elsewhere": "Elsewhere",
        "recent": "Recent posts", "all_posts": "All posts →",
        "min_read": "{n} min read", "with": "with {names}", "part": "Part {n}",
        "references": "References", "tldr": "TL;DR.", "figure": "Figure {n}.",
        "and": " and ",
        "date": lambda d: f"{MONTHS[d.month - 1]} {d.day}, {d.year}",
    },
    "zh": {
        "blog": "博客", "posts": "文章", "series": "系列", "elsewhere": "其他地方",
        "recent": "最近文章", "all_posts": "全部文章 →",
        "min_read": "阅读约 {n} 分钟", "with": "与 {names} 合写", "part": "第 {n} 篇",
        "references": "参考文献", "tldr": "TL;DR", "figure": "图 {n}.",
        "and": "、",
        "date": lambda d: f"{d.year}年{d.month}月{d.day}日",
    },
}

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"


def esc(s) -> str:
    return html.escape(str(s or ""), quote=True)


# --------------------------------------------------------------------------- 配置与读取

def load_config() -> dict:
    cfg = yaml.safe_load((ROOT / "site.yml").read_text(encoding="utf-8")) or {}
    cfg.setdefault("lang", "en")
    cfg.setdefault("author", "Your Name")
    cfg.setdefault("url", "")
    cfg["url"] = cfg["url"].rstrip("/")
    cfg["series"] = cfg["series"] or {}
    cfg["elsewhere"] = cfg.get("elsewhere") or []
    cfg["links"] = cfg.get("links") or []
    cfg.setdefault("homepage", True)
    return cfg


def to_date(v) -> dt.date:
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    return dt.date.fromisoformat(str(v).strip())


FRONT_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)


def split_front(text: str) -> tuple[dict, str]:
    m = FRONT_RE.match(text)
    if not m:
        return {}, text
    return (yaml.safe_load(m.group(1)) or {}), text[m.end():]


def find_posts() -> list[Path]:
    """posts/xxx.md 或 posts/xxx/index.md 两种写法都支持。"""
    found = []
    if not POSTS_DIR.exists():
        return found
    for p in sorted(POSTS_DIR.iterdir()):
        if p.is_dir() and (p / "index.md").exists():
            found.append(p / "index.md")
        elif p.is_file() and p.suffix == ".md":
            found.append(p)
    return found


def slug_for(path: Path, meta: dict) -> str:
    if meta.get("slug"):
        return str(meta["slug"])
    name = path.parent.name if path.name == "index.md" else path.stem
    return re.sub(r"^\d{4}-\d{2}-\d{2}-", "", name)


# --------------------------------------------------------------------------- Markdown 扩展

# 保护代码块与公式，避免 Markdown 把公式里的 _ 和 \ 当成格式符号
PROTECT_RE = re.compile(
    r"(?P<fence>^[ \t]*(?P<f>`{3,}|~{3,})[^\n]*\n.*?^[ \t]*(?P=f)[ \t]*$)"
    r"|(?P<code>(?P<bt>`+).+?(?P=bt))"
    r"|(?P<math>\$\$.+?\$\$|\\\[.+?\\\]|\\\(.+?\\\)"
    r"|(?<![\\$\w])\$(?=[^\s$])[^$\n]+?(?<=[^\s\\])\$(?![\w$]))",
    re.S | re.M,
)


def protect_math(src: str) -> tuple[str, list[str]]:
    stash: list[str] = []

    def repl(m):
        if m.group("math") is None:
            return m.group(0)
        stash.append(m.group("math"))
        return f"KATEXMATHSLOT{len(stash) - 1}END"

    return PROTECT_RE.sub(repl, src), stash


def restore_math(html_text: str, stash: list[str]) -> str:
    return re.sub(r"KATEXMATHSLOT(\d+)END",
                  lambda m: html.escape(stash[int(m.group(1))], quote=False), html_text)


class CiteBracket(InlineProcessor):
    """[@key] 或 [@a; @b] → (Author, 2026; Other, 2025)"""

    def __init__(self, pattern, md, ctx):
        super().__init__(pattern, md)
        self.ctx = ctx

    def handleMatch(self, m, data):
        keys = re.findall(r"@([\w:-]+)", m.group(1))
        span = etree.Element("span")
        span.text = "("
        last = None
        for i, k in enumerate(keys):
            a = self.ctx.cite_el(k, narrative=False)
            span.append(a)
            a.tail = "; " if i < len(keys) - 1 else ")"
            last = a
        if last is None:
            return None, None, None
        return span, m.start(0), m.end(0)


class CiteNarrative(InlineProcessor):
    """@key → Author (2026)，只在 key 存在于 references 时生效（不会误伤邮箱）"""

    def __init__(self, pattern, md, ctx):
        super().__init__(pattern, md)
        self.ctx = ctx

    def handleMatch(self, m, data):
        if m.group(1) not in self.ctx.refs:
            return None, None, None
        return self.ctx.cite_el(m.group(1), narrative=True), m.start(0), m.end(0)


class Figures(Treeprocessor):
    """单独成段的图片 ![alt](src "说明") → <figure> + 自动编号的 <figcaption>"""

    def __init__(self, md, ctx):
        super().__init__(md)
        self.ctx = ctx

    def run(self, root):
        targets = []
        for parent in root.iter():
            for i, child in enumerate(list(parent)):
                if (child.tag == "p" and len(child) == 1 and child[0].tag == "img"
                        and not (child.text or "").strip() and not (child[0].tail or "").strip()):
                    targets.append((parent, i, child))
        n = 0
        for parent, i, p in targets:
            img = p[0]
            img.tail = None
            img.set("loading", "lazy")
            cap = img.attrib.pop("title", None)
            fig = etree.Element("figure")
            fig.append(img)
            if cap:
                n += 1
                fc = etree.SubElement(fig, "figcaption")
                if self.ctx.cfg.get("figure_numbers", True):
                    b = etree.SubElement(fc, "b")
                    b.text = self.ctx.t["figure"].format(n=n)
                    b.tail = " " + cap
                else:
                    fc.text = cap
            parent.remove(p)
            parent.insert(i, fig)


class BlogExtension(Extension):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx

    def extendMarkdown(self, md):
        md.inlinePatterns.register(
            CiteBracket(r"\[(@[\w:-]+(?:\s*;\s*@[\w:-]+)*)\]", md, self.ctx), "cite_bracket", 175)
        md.inlinePatterns.register(
            CiteNarrative(r"(?<![\w@./])@([A-Za-z][\w:-]*)", md, self.ctx), "cite_narrative", 174)
        md.treeprocessors.register(Figures(md, self.ctx), "figures", 5)


class PostContext:
    def __init__(self, cfg, refs, path):
        self.cfg = cfg
        self.t = STRINGS.get(cfg["lang"], STRINGS["en"])
        self.refs = refs or {}
        self.path = path
        self.cited: list[str] = []

    def cite_el(self, key, narrative):
        a = etree.Element("a")
        a.set("class", "cite")
        a.set("href", f"#ref-{key}")
        ref = self.refs.get(key)
        if ref is None:
            print(f"  ! {self.path.relative_to(ROOT)}: 找不到引用 @{key}", file=sys.stderr)
            a.set("class", "cite missing")
            a.text = f"{key}?"
            return a
        if key not in self.cited:
            self.cited.append(key)
        who, year = ref.get("author", key), ref.get("year", "")
        if narrative:
            a.text = f"{who} ({year})" if year else who
        else:
            a.text = f"{who}, {year}" if year else who
        return a


def md_converter(ctx) -> markdown.Markdown:
    return markdown.Markdown(
        extensions=["extra", "sane_lists", "toc", BlogExtension(ctx)],
        extension_configs={"toc": {"slugify": lambda v, sep: re.sub(r"[^\w一-鿿]+", sep, v).strip(sep).lower()}},
        output_format="html",
    )


def render_md(src: str, ctx) -> tuple[str, bool]:
    protected, stash = protect_math(src)
    out = md_converter(ctx).convert(protected)
    return restore_math(out, stash), bool(stash)


def inline_md(src: str, ctx=None) -> str:
    """把一小段 Markdown（如参考文献条目、TL;DR）转成不带外层 <p> 的 HTML。"""
    out, _ = render_md(str(src), ctx or PostContext({"lang": "en"}, {}, ROOT))
    return re.sub(r"^<p>(.*)</p>$", r"\1", out.strip(), flags=re.S)


# --------------------------------------------------------------------------- 文章

CJK_RE = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]")


def reading_minutes(html_text: str) -> int:
    text = re.sub(r"<[^>]+>", " ", html_text)
    cjk = len(CJK_RE.findall(text))
    words = len(re.findall(r"[A-Za-z0-9][\w'’-]*", CJK_RE.sub(" ", text)))
    return max(1, math.ceil(words / 220 + cjk / 400))


def first_paragraph(html_text: str, limit=180) -> str:
    m = re.search(r"<p>(.*?)</p>", html_text, re.S)
    if not m:
        return ""
    t = html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
    return t if len(t) <= limit else t[:limit].rstrip() + "…"


def load_post(path: Path, cfg: dict) -> dict | None:
    meta, body = split_front(path.read_text(encoding="utf-8"))
    if "title" not in meta or "date" not in meta:
        print(f"  ! 跳过 {path.relative_to(ROOT)}：缺少 title 或 date", file=sys.stderr)
        return None
    ctx = PostContext(cfg, meta.get("references"), path)
    content, has_math = render_md(body, ctx)
    authors = meta.get("authors") or [cfg["author"]]
    if isinstance(authors, str):
        authors = [authors]
    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [s.strip() for s in tags.split(",") if s.strip()]
    return {
        "src": path,
        "dir": path.parent if path.name == "index.md" else None,
        "slug": slug_for(path, meta),
        "title": str(meta["title"]),
        "date": to_date(meta["date"]),
        "summary": meta.get("summary") or first_paragraph(content),
        "tags": tags,
        "authors": [str(a) for a in authors],
        "series": meta.get("series"),
        "part": meta.get("part"),
        "thumb": meta.get("thumb"),
        "tldr": meta.get("tldr"),
        "links": meta.get("links") or [],
        "draft": bool(meta.get("draft")),
        "math": has_math or bool(meta.get("math")),
        "content": content,
        "minutes": reading_minutes(content),
        "ctx": ctx,
        "refs": meta.get("references") or {},
    }


def join_names(names, t):
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    if t["and"] == " and ":
        return (", ".join(names[:-1]) + " and " + names[-1]) if len(names) > 2 else " and ".join(names)
    return t["and"].join(names)


# --------------------------------------------------------------------------- 模板

def page(cfg, *, title, body, css, path, desc="", image="", head_extra="", tail_extra=""):
    url = f"{cfg['url']}{path}" if cfg["url"] else ""
    meta = []
    if desc:
        meta.append(f'<meta name="description" content="{esc(desc)}">')
        meta.append(f'<meta property="og:description" content="{esc(desc)}">')
    meta.append(f'<meta property="og:title" content="{esc(title)}">')
    if url:
        meta.append(f'<meta property="og:url" content="{esc(url)}">')
        meta.append(f'<link rel="canonical" href="{esc(url)}">')
    if image and cfg["url"]:
        meta.append(f'<meta property="og:image" content="{esc(cfg["url"] + image)}">')
        meta.append('<meta name="twitter:card" content="summary_large_image">')
    feed = f'<link rel="alternate" type="application/atom+xml" title="{esc(cfg["author"])}" href="/blog/feed.xml">'
    fav = '<link rel="icon" href="/favicon.ico">' if (STATIC_DIR / "favicon.ico").exists() else ""
    return f"""<!DOCTYPE html>
<html lang="{esc(cfg['lang'])}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
{chr(10).join(meta)}
{feed}{fav}
<style>
{css}
</style>
{head_extra}</head>
<body>
<main>
{body}
</main>
{tail_extra}</body>
</html>
"""


def css(*names) -> str:
    return "\n".join((THEME_DIR / f"{n}.css").read_text(encoding="utf-8").strip() for n in names)


def post_meta_line(p, cfg, t, link_series=True):
    bits = []
    chip = ""
    if p["series"] and p["series"] in cfg["series"]:
        s = cfg["series"][p["series"]]
        label = esc(s.get("short") or s["title"])
        if p.get("part"):
            label += " · " + esc(t["part"].format(n=p["part"]))
        chip = (f'<a class="chip" href="/blog/{esc(p["series"])}/">{label}</a>' if link_series
                else f'<span class="chip">{label}</span>')
    bits.append(esc(t["date"](p["date"])))
    bits.append(esc(t["min_read"].format(n=p["minutes"])))
    return f'<div class="meta">{chip}{" · ".join(bits)}</div>'


def post_item(p, cfg, t, base="/blog/"):
    href = f'{base}{p["slug"]}/'
    others = [a for a in p["authors"] if a != cfg["author"]]
    who = f'<div class="who">{esc(t["with"].format(names=join_names(others, t)))}</div>' if others else ""
    tags = f'<div class="tags">{esc(", ".join(map(str, p["tags"])))}</div>' if p["tags"] else ""
    summary = f'<p>{esc(p["summary"])}</p>' if p["summary"] else ""
    thumb = ""
    if p["thumb"]:
        thumb = f'<a class="thumb" href="{href}"><img src="{href}{esc(p["thumb"])}" alt="" loading="lazy"></a>'
    return (f'<div class="post"><div class="text">{post_meta_line(p, cfg, t)}'
            f'<h3><a href="{href}">{esc(p["title"])}</a></h3>{who}{summary}{tags}</div>{thumb}</div>')


def render_post(p, cfg, t, posts):
    s = cfg["series"].get(p["series"]) if p["series"] else None
    kicker = ""
    if s:
        part = f' · {esc(t["part"].format(n=p["part"]))}' if p.get("part") else ""
        kicker = f'<div class="kicker"><a href="/blog/{esc(p["series"])}/">{esc(s["title"])}</a>{part}</div>'
    byline = [esc(", ".join(p["authors"])), esc(t["date"](p["date"]))]
    for link in p["links"]:
        byline.append(f'<a href="{esc(link["url"])}">{esc(link["title"])}</a>')
    tldr = ""
    if p["tldr"]:
        tldr = f'<div class="tldr"><b>{esc(t["tldr"])}</b> {inline_md(p["tldr"], p["ctx"])}</div>'
    refs_html = ""
    if p["refs"]:
        items = []
        for key, r in sorted(p["refs"].items(), key=lambda kv: str(kv[1].get("text", kv[0])).lower()):
            text = inline_md(r.get("text") or f'{r.get("author", key)} ({r.get("year", "")})')
            url = f' <a href="{esc(r["url"])}">{esc(r["url"])}</a>' if r.get("url") else ""
            items.append(f'<li id="ref-{esc(key)}">{text}{url}</li>')
        refs_html = f'<h2 id="references">{esc(t["references"])}</h2>\n<ul class="refs">\n' + "\n".join(items) + "\n</ul>"
    body = (f'<div class="home"><a href="/blog/">← {esc(t["blog"])}</a></div>\n{kicker}'
            f'<h1>{esc(p["title"])}</h1>\n<div class="byline">{" · ".join(byline)}</div>\n'
            f'{tldr}\n{p["content"]}\n{refs_html}')
    head = tail = ""
    if p["math"]:
        head = f'<link rel="stylesheet" href="{KATEX}/katex.min.css">\n'
        tail = (f'<script defer src="{KATEX}/katex.min.js"></script>\n'
                f'<script defer src="{KATEX}/contrib/auto-render.min.js"></script>\n'
                '<script>window.addEventListener("DOMContentLoaded",function(){renderMathInElement(document.body,'
                '{delimiters:[{left:"$$",right:"$$",display:true},{left:"\\\\[",right:"\\\\]",display:true},'
                '{left:"$",right:"$",display:false},{left:"\\\\(",right:"\\\\)",display:false}],throwOnError:false});});</script>\n')
    path = f'/blog/{p["slug"]}/'
    image = f'{path}{p["thumb"]}' if p["thumb"] else ""
    return page(cfg, title=p["title"], body=body, css=css("common", "post"), path=path,
                desc=p["summary"], image=image, head_extra=head, tail_extra=tail)


def render_blog_index(cfg, t, posts):
    parts = [f'<div class="home"><a href="/">← {esc(cfg["author"])}</a></div>' if cfg["homepage"] else "",
             f'<h1>{esc(t["blog"])}</h1>']
    if cfg.get("blog_intro"):
        parts.append(f'<div class="intro">{inline_md(cfg["blog_intro"])}</div>')
    parts.append(f'<h2>{esc(t["posts"])}</h2>\n<div class="list">')
    parts += [post_item(p, cfg, t) for p in posts]
    parts.append("</div>")
    if cfg["series"]:
        parts.append(f'<h2>{esc(t["series"])}</h2>\n<div class="list series">')
        for sid, s in cfg["series"].items():
            desc = f'<p>{esc(s["description"])}</p>' if s.get("description") else ""
            parts.append(f'<div class="post"><div class="text"><a class="name" href="/blog/{esc(sid)}/">'
                         f'{esc(s["title"])}</a>{desc}</div></div>')
        parts.append("</div>")
    if cfg["elsewhere"]:
        parts.append(f'<h2>{esc(t["elsewhere"])}</h2>\n<div class="list">')
        for e in cfg["elsewhere"]:
            meta = f'<div class="meta">{esc(t["date"](to_date(e["date"])))}</div>' if e.get("date") else ""
            others = [a for a in (e.get("authors") or []) if a != cfg["author"]]
            who = f'<div class="who">{esc(t["with"].format(names=join_names(others, t)))}</div>' if others else ""
            desc = f'<p>{esc(e["description"])}</p>' if e.get("description") else ""
            parts.append(f'<div class="post"><div class="text">{meta}<h3><a href="{esc(e["url"])}">'
                         f'{esc(e["title"])}</a><span class="ext">↗</span></h3>{who}{desc}</div></div>')
        parts.append("</div>")
    return page(cfg, title=f'{t["blog"]} | {cfg["author"]}', body="\n".join(parts),
                css=css("common", "list"), path="/blog/", desc=cfg.get("description", ""))


def render_series(cfg, t, sid, s, posts):
    items = sorted([p for p in posts if p["series"] == sid], key=lambda p: (p.get("part") or 0, p["date"]))
    desc = f'<div class="intro">{inline_md(s["description"])}</div>' if s.get("description") else ""
    body = (f'<div class="home"><a href="/blog/">← {esc(t["blog"])}</a></div>\n'
            f'<h1>{esc(s["title"])}</h1>\n{desc}\n<h2>{esc(t["posts"])}</h2>\n<div class="list">'
            + "".join(post_item(p, cfg, t) for p in items) + "</div>")
    return page(cfg, title=f'{s["title"]} | {cfg["author"]}', body=body, css=css("common", "list"),
                path=f"/blog/{sid}/", desc=s.get("description", ""))


def render_home(cfg, t, posts):
    about = ROOT / "about.md"
    intro = ""
    if about.exists():
        _, src = split_front(about.read_text(encoding="utf-8"))
        intro, _ = render_md(src, PostContext(cfg, {}, about))
    links = "".join(f'<a href="{esc(l["url"])}">{esc(l["title"])}</a>' for l in cfg["links"])
    recent = "".join(post_item(p, cfg, t) for p in posts[:3])
    body = (f'<h1>{esc(cfg["author"])}</h1>\n<div class="intro">{intro}</div>\n'
            + (f'<div class="links">{links}</div>\n' if links else "")
            + f'<h2>{esc(t["recent"])}</h2>\n<div class="list">{recent}</div>\n'
            f'<div class="more"><a href="/blog/">{esc(t["all_posts"])}</a></div>')
    return page(cfg, title=cfg["author"], body=body, css=css("common", "list"), path="/",
                desc=cfg.get("description", ""))


def render_feed(cfg, t, posts):
    base = cfg["url"] or ""
    updated = (posts[0]["date"] if posts else dt.date.today()).isoformat() + "T00:00:00Z"
    entries = []
    for p in posts[:20]:
        link = f'{base}/blog/{p["slug"]}/'
        entries.append(
            f'<entry><title>{esc(p["title"])}</title><link href="{esc(link)}"/><id>{esc(link)}</id>'
            f'<updated>{p["date"].isoformat()}T00:00:00Z</updated><summary>{esc(p["summary"])}</summary></entry>')
    return (f'<?xml version="1.0" encoding="utf-8"?>\n<feed xmlns="http://www.w3.org/2005/Atom">'
            f'<title>{esc(cfg["author"])} · {esc(t["blog"])}</title><link href="{esc(base)}/blog/"/>'
            f'<id>{esc(base)}/blog/</id><updated>{updated}</updated>'
            f'<author><name>{esc(cfg["author"])}</name></author>{"".join(entries)}</feed>\n')


# --------------------------------------------------------------------------- 构建

def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build(drafts=False) -> int:
    cfg = load_config()
    t = STRINGS.get(cfg["lang"], STRINGS["en"])
    posts = [p for p in (load_post(f, cfg) for f in find_posts()) if p]
    posts = [p for p in posts if drafts or not p["draft"]]
    posts.sort(key=lambda p: (p["date"], p.get("part") or 0), reverse=True)

    seen = {}
    for p in posts:
        if p["slug"] in seen:
            sys.exit(f"错误：{p['src']} 和 {seen[p['slug']]} 的网址（slug）重复了")
        if p["slug"] in cfg["series"]:
            sys.exit(f"错误：文章 slug「{p['slug']}」和系列 id 重名了")
        seen[p["slug"]] = p["src"]
        if p["series"] and p["series"] not in cfg["series"]:
            print(f"  ! {p['src'].relative_to(ROOT)}：series「{p['series']}」没有在 site.yml 里定义", file=sys.stderr)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    if STATIC_DIR.exists():
        shutil.copytree(STATIC_DIR, OUT, dirs_exist_ok=True)
    (OUT / ".nojekyll").touch()

    for p in posts:
        dest = OUT / "blog" / p["slug"]
        if p["dir"]:
            shutil.copytree(p["dir"], dest, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("*.md", ".*"))
        write(dest / "index.html", render_post(p, cfg, t, posts))
    write(OUT / "blog" / "index.html", render_blog_index(cfg, t, posts))
    for sid, s in cfg["series"].items():
        write(OUT / "blog" / sid / "index.html", render_series(cfg, t, sid, s, posts))
    write(OUT / "blog" / "feed.xml", render_feed(cfg, t, posts))
    if cfg["homepage"]:
        write(OUT / "index.html", render_home(cfg, t, posts))
    elif not (OUT / "index.html").exists():
        write(OUT / "index.html", '<!DOCTYPE html><meta http-equiv="refresh" content="0; url=/blog/">')
    return len(posts)


def snapshot() -> dict:
    files = {}
    for d in (POSTS_DIR, THEME_DIR, STATIC_DIR):
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file():
                    files[f] = f.stat().st_mtime
    for f in (ROOT / "site.yml", ROOT / "about.md", Path(__file__)):
        if f.exists():
            files[f] = f.stat().st_mtime
    return files


def serve(port: int, drafts: bool):
    def watch():
        last = snapshot()
        while True:
            time.sleep(1)
            now = snapshot()
            if now != last:
                last = now
                try:
                    n = build(drafts)
                    print(f"  ↻ 已重建（{n} 篇文章）")
                except Exception as e:  # 写作过程中出错不要让预览服务器退出
                    print(f"  ! 构建失败：{e}", file=sys.stderr)

    threading.Thread(target=watch, daemon=True).start()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(OUT), **kw)

        def log_message(self, *a):
            pass

    print(f"预览：http://localhost:{port}/   （Ctrl+C 退出）")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", help="构建后启动本地预览，并在文件变化时自动重建")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--drafts", action="store_true", help="包含草稿")
    args = ap.parse_args()
    n = build(args.drafts or args.serve)
    print(f"已生成 {n} 篇文章 → {OUT.relative_to(ROOT)}/")
    if args.serve:
        try:
            serve(args.port, True)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
