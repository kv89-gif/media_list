
import json
import re
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup


# -----------------------------
# Helpers
# -----------------------------
def _safe_strip(x):
    return "" if x is None else str(x).strip()


def normalize_author(author: str) -> str:
    """
    Normalize messy bylines to improve de-duplication.
    Examples:
      - "By John Doe | Editor" -> "John Doe"
      - "John Doe, Staff Writer" -> "John Doe"
    """
    s = _safe_strip(author)
    if not s:
        return ""
    s = re.sub(r"^\s*by\s+", "", s, flags=re.IGNORECASE).strip()
    s = re.split(r"\s*(?:\||,| - )\s*", s, maxsplit=1)[0].strip()
    s = re.sub(r"\s+", " ", s).strip()
    # avoid junk values
    if s.lower() in {"staff", "editorial", "news desk", "associated press", "ap"}:
        return s
    return s


def domain_from_url(u: str) -> str:
    u = _safe_strip(u)
    if not u:
        return ""
    try:
        netloc = urlparse(u).netloc.lower()
    except Exception:
        return ""
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def split_cell(cell: str) -> list[str]:
    s = _safe_strip(cell)
    if not s:
        return []
    return [p.strip() for p in re.split(r"[\n,;]+", s) if p.strip()]


def looks_like_url(s: str) -> bool:
    s = _safe_strip(s)
    return s.startswith("http://") or s.startswith("https://") or ("." in s and "/" in s)


def ensure_url(u: str) -> str:
    u = _safe_strip(u)
    if not u:
        return ""
    if u.startswith("http://") or u.startswith("https://"):
        return u
    if "." in u and " " not in u:
        return "https://" + u
    return ""


def load_any_table(uploaded_file) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = pd.read_excel(uploaded_file)
    df.columns = [c.strip() for c in df.columns]
    return df


# -----------------------------
# Light author extraction from URL (best-effort)
# -----------------------------
def extract_author_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # 1) Common meta tags
    candidates = []
    for sel in [
        ('meta', {'name': 'author'}),
        ('meta', {'property': 'article:author'}),
        ('meta', {'name': 'parsely-author'}),
        ('meta', {'name': 'byl'}),
    ]:
        tag = soup.find(*sel)
        if tag and tag.get("content"):
            candidates.append(tag.get("content"))

    # 2) JSON-LD (Person / author)
    for script in soup.find_all("script", type="application/ld+json"):
        txt = script.string
        if not txt:
            continue
        try:
            data = json.loads(txt.strip())
        except Exception:
            continue

        def pull_author(obj):
            if isinstance(obj, dict):
                if "author" in obj:
                    return obj["author"]
                if obj.get("@type") == "Person" and "name" in obj:
                    return obj.get("name")
                for v in obj.values():
                    res = pull_author(v)
                    if res:
                        return res
            elif isinstance(obj, list):
                for it in obj:
                    res = pull_author(it)
                    if res:
                        return res
            return None

        a = pull_author(data)
        if a:
            if isinstance(a, dict):
                nm = a.get("name")
                if nm:
                    candidates.append(nm)
            elif isinstance(a, list):
                for it in a:
                    if isinstance(it, dict) and it.get("name"):
                        candidates.append(it["name"])
                    elif isinstance(it, str):
                        candidates.append(it)
            elif isinstance(a, str):
                candidates.append(a)

    # 3) Try visible byline patterns (last resort, noisy)
    if not candidates:
        text = soup.get_text(" ", strip=True)
        m = re.search(r"\bBy\s+([A-Z][A-Za-z\.\-]+\s+[A-Z][A-Za-z\.\-]+)\b", text)
        if m:
            candidates.append(m.group(1))

    for c in candidates:
        c = normalize_author(c)
        if c:
            return c
    return ""


@st.cache_data(show_spinner=False)
def fetch_author_from_url(url: str) -> str:
    url = ensure_url(url)
    if not url:
        return ""
    try:
        r = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; OutreachBot/1.0)"},
            allow_redirects=True,
        )
        if r.status_code >= 400:
            return ""
        return extract_author_from_html(r.text)
    except Exception:
        return ""


# -----------------------------
# NewsAPI
# -----------------------------
def get_news(api_key: str, query: str, page_size: int = 100) -> list[dict]:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": min(max(int(page_size), 1), 100),
    }
    headers = {"X-Api-Key": api_key}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("articles", [])


def newsapi_to_rows(articles: list[dict], niches: list[str]) -> list[dict]:
    # de-dupe articles by URL first
    seen = set()
    rows = []
    for a in articles:
        u = _safe_strip(a.get("url"))
        if not u or u in seen:
            continue
        seen.add(u)
        # best-effort niche inference using URL + title + source
        blob = " ".join(
            [
                _safe_strip(a.get("title")),
                _safe_strip(a.get("description")),
                _safe_strip((a.get("source") or {}).get("name")),
                domain_from_url(u),
                u,
            ]
        )
        matched = ", ".join(infer_matched_niches(blob, niches)) if niches else ""
        rows.append(
            {
                "Author": normalize_author(a.get("author")),
                "Article URL": u,
                "Title": _safe_strip(a.get("title")),
                "Source": _safe_strip((a.get("source") or {}).get("name")),
                "Matched Niches": matched,
                "Origin": "NewsAPI",
            }
        )
    # keep only rows with an author
    return [r for r in rows if r["Author"]]


# -----------------------------
# CSV/XLSX → rows (names and/or URLs)
# -----------------------------
def detect_columns(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}

    # IMPORTANT:
    # Contact lists often include columns like "Domain Authority" / "Domain Authority Tier".
    # Naive substring matching for "author" would incorrectly treat "authority" as an author column.
    # We therefore use word-boundary regex matching + explicit exclusions.

    name_cols: list[str] = []
    url_cols: list[str] = []
    industry_col: str | None = None
    pub_col: str | None = None
    email_col: str | None = None
    notes_col: str | None = None

    # Prefer exact known headers when present
    if "name" in cols:
        name_cols = [cols["name"]]

    # Regex patterns (word-boundary)
    author_pat = re.compile(r"\b(author|writer|journalist|byline)\b", re.IGNORECASE)
    name_pat = re.compile(r"\bname\b", re.IGNORECASE)

    for c in df.columns:
        lc = c.lower().strip()

        # ---- name/author columns ----
        # If we already found an explicit "Name" column, don't add other weak matches
        if not name_cols:
            # Exclude DA/metrics columns that can contain the substring "author" via "authority"
            is_metric = any(k in lc for k in ["authority", "domain authority", "da", "dr", "metric", "score", "tier"])
            if (author_pat.search(lc) or name_pat.search(lc)) and not is_metric:
                name_cols.append(c)

        # ---- url columns ----
        if "url" in lc or "link" in lc:
            url_cols.append(c)

        # ---- industry column ----
        if industry_col is None and lc in {"industry", "niche", "category", "categories"}:
            industry_col = c

        # ---- publication / website column ----
        if pub_col is None and any(k in lc for k in ["publication", "website"]):
            pub_col = c


        # ---- email column ----
        if email_col is None and "email" in lc:
            email_col = c

        # ---- notes column ----
        if notes_col is None and lc in {"notes", "note", "comment", "comments"}:
            notes_col = c

    return {
        "name_cols": name_cols,
        "url_cols": url_cols,
        "industry_col": industry_col,
        "pub_col": pub_col,
        "email_col": email_col,
        "notes_col": notes_col,
    }


def row_relevant_to_niches(row: pd.Series, niches: list[str], industry_col: str | None) -> bool:
    if not niches:
        return True
    keys = [k.lower() for k in niches if k.strip()]
    if not keys:
        return True

    # Prefer industry/category field if present
    if industry_col:
        s = _safe_strip(row.get(industry_col)).lower()
        if s == "all":
            return True
        if any(k in s for k in keys):
            return True

    # fallback: match against whole row text (incl. urls)
    blob = " ".join(_safe_strip(v).lower() for v in row.values.tolist())
    return any(k in blob for k in keys)


# -----------------------------
# Niche inference (keyword-based)
# -----------------------------
def infer_matched_niches(text: str, niches: list[str]) -> list[str]:
    """Return niches that appear in text (case-insensitive substring match)."""
    if not niches:
        return []
    t = _safe_strip(text).lower()
    if not t:
        return []
    out: list[str] = []
    for n in niches:
        nn = _safe_strip(n)
        if not nn:
            continue
        if nn.lower() in t and nn not in out:
            out.append(nn)
    return out


@st.cache_data(show_spinner=False)
def fetch_page_text(url: str) -> str:
    """Best-effort fetch page text (used for niche inference when CSV Industry is sparse)."""
    url = ensure_url(url)
    if not url:
        return ""
    try:
        r = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; OutreachBot/1.0)"},
            allow_redirects=True,
        )
        if r.status_code >= 400:
            return ""
        soup = BeautifulSoup(r.text, "lxml")
        # keep it light: title + meta description + visible text
        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        desc = ""
        md = soup.find("meta", attrs={"name": "description"})
        if md and md.get("content"):
            desc = md.get("content")
        body_txt = soup.get_text(" ", strip=True)
        blob = f"{title} {desc} {body_txt}"
        # avoid huge cached blobs
        return blob[:20000]
    except Exception:
        return ""


def contacts_to_rows(
    df: pd.DataFrame,
    niches: list[str],
    enrich_urls: bool,
    infer_niche_from_website: bool,
) -> list[dict]:
    if df is None or df.empty:
        return []

    cols = detect_columns(df)
    name_cols = cols["name_cols"]
    url_cols = cols["url_cols"]
    industry_col = cols["industry_col"]
    pub_col = cols["pub_col"]
    email_col = cols["email_col"]
    notes_col = cols.get("notes_col")

    rows = []

    for _, r in df.iterrows():
        # CSV filtering can be overly strict if Industry is sparse.
        # We first try Industry/row-text matching. If it fails and the user enabled
        # website-based inference, we fetch the publication/website and look for niche keywords.
        relevant = row_relevant_to_niches(r, niches, industry_col)

        pub = _safe_strip(r.get(pub_col)) if pub_col else ""
        pub_url = ensure_url(pub)

        if (not relevant) and infer_niche_from_website and pub_url and niches:
            page_blob = fetch_page_text(pub_url)
            matched = infer_matched_niches(page_blob, niches)
            relevant = bool(matched)

        if not relevant:
            continue
        email = _safe_strip(r.get(email_col)) if email_col else ""
        notes = _safe_strip(r.get(notes_col)) if notes_col else ""

        # For CSV contact rows, user requested that "Article URL" should be the Source itself.
        # So we treat the publication/website as the URL for outreach.
        csv_article_url = pub_url or ""

        # Matched niches (best-effort)
        matched_niches = ""
        if niches:
            # primary signals from the row itself + website (if enabled)
            row_blob = " ".join(_safe_strip(v) for v in r.values.tolist())
            m1 = infer_matched_niches(row_blob, niches)
            if infer_niche_from_website and pub_url:
                m2 = infer_matched_niches(fetch_page_text(pub_url), niches)
            else:
                m2 = []
            matched_niches = ", ".join(sorted(set(m1 + m2)))

        # 1) Names
        for c in name_cols:
            nm = normalize_author(r.get(c))
            if nm:
                rows.append(
                    {
                        "Author": nm,
                        "Article URL": csv_article_url,
                        "Title": "",
                        "Source": pub,
                        "Email": email,
                        "Notes": notes,
                        "Matched Niches": matched_niches,
                        "Origin": "CSV",
                    }
                )

        # 2) URLs (try to extract author)
        for c in url_cols:
            for maybe in split_cell(r.get(c)):
                u = ensure_url(maybe)
                if not u:
                    continue
                author = fetch_author_from_url(u) if enrich_urls else ""
                rows.append(
                    {
                        "Author": author,
                        "Article URL": u,
                        "Title": "",
                        "Source": pub or domain_from_url(u),
                        "Email": email,
                        "Notes": notes,
                        "Matched Niches": matched_niches,
                        "Origin": "CSV",
                    }
                )

    # keep only with author (URL-only rows might fail extraction)
   # CHANGE: keep CSV rows even if author not found; author can never be numeric
for x in rows:
    a = _safe_strip(x.get("Author"))
    if not a:
        x["Author"] = ""
        continue
    # numeric check (including decimals like 3.0)
    if re.fullmatch(r"\d+(\.\d+)?", a):
        x["Author"] = ""
    else:
        x["Author"] = a

return rows



# -----------------------------
# Combine → UNIQUE journalists
# -----------------------------
def combined_unique_journalists(rows: list[dict]) -> pd.DataFrame:
    """
    Input: rows with Author, Article URL, Title, Source, Email(optional), Origin
    Output: ONE ROW PER UNIQUE AUTHOR
    """
    buckets = {}
    for r in rows:
        author = normalize_author(r.get("Author"))
        if not author:
            continue
        key = author.lower()

        b = buckets.setdefault(
            key,
            {
                "Author": author,
                "Article Count": 0,
                "Sources": set(),
                "Origins": set(),
                "Emails": set(),
                "Notes": set(),
                "Matched Niches": set(),
                "Sample Titles": [],
                "Article URLs": [],
            },
        )
        if r.get("Article URL"):
            b["Article Count"] += 1
            if r["Article URL"] not in b["Article URLs"]:
                b["Article URLs"].append(r["Article URL"])
        if r.get("Source"):
            b["Sources"].add(_safe_strip(r.get("Source")))
        if r.get("Origin"):
            b["Origins"].add(_safe_strip(r.get("Origin")))
        if r.get("Email"):
            if _safe_strip(r.get("Email")):
                b["Emails"].add(_safe_strip(r.get("Email")))
        if r.get("Notes"):
            if _safe_strip(r.get("Notes")):
                b["Notes"].add(_safe_strip(r.get("Notes")))
        if r.get("Matched Niches"):
            for n in split_cell(r.get("Matched Niches")):
                if n:
                    b["Matched Niches"].add(n)
        if r.get("Title"):
            t = _safe_strip(r.get("Title"))
            if t and len(b["Sample Titles"]) < 3 and t not in b["Sample Titles"]:
                b["Sample Titles"].append(t)

    out_rows = []
    for b in buckets.values():
        out_rows.append(
            {
                "Author": b["Author"],
                "Origins": ", ".join(sorted(x for x in b["Origins"] if x)),
                "Article Count": b["Article Count"],
                "Sources": ", ".join(sorted(x for x in b["Sources"] if x)),
                "Emails": "\n".join(sorted(b["Emails"])) if b["Emails"] else "",
                "Notes": " | ".join(sorted(b["Notes"])) if b["Notes"] else "",
                "Matched Niches": ", ".join(sorted(b["Matched Niches"])) if b["Matched Niches"] else "",
                "Sample Titles": " | ".join(b["Sample Titles"]),
                "Article URLs": "\n".join(b["Article URLs"][:15]),
            }
        )
    df = pd.DataFrame(out_rows)
    if not df.empty:
        df = df.sort_values(["Article Count", "Author"], ascending=[False, True]).reset_index(drop=True)
    return df


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("📰 Journalist Finder (NewsAPI + CSV Combined)")
st.caption("Combines journalists from NewsAPI and your uploaded CSV/XLSX, then de-duplicates to unique authors.")

st.markdown(
    """
### 🧭 How to Use This Tool

1. **Enter your NewsAPI Key** (get it from https://newsapi.org → *Get API Key*)
2. **Enter the NewsAPI Search Query** *(only if you want to pull writers from external news sources)*
3. **Upload the contacts list (CSV file)**
4. **Enter the niches** (example: Food, Legal, Real estate, ...)
5. **Click Run**
"""
)

st.markdown("---")

st.sidebar.header("⚙️ Inputs")

api_key = st.sidebar.text_input("NewsAPI Key (optional)", type="password")
query = st.sidebar.text_input("NewsAPI search query (optional)", "real estate")
page_size = st.sidebar.slider("NewsAPI max articles (pageSize)", 10, 100, 100, 10)

uploaded_contacts = st.sidebar.file_uploader(
    "Upload your contacts CSV/XLSX (Name, Industry, Website/Publication, Email, Notes…)",
    type=["csv", "xlsx"],
)

niches_raw = st.sidebar.text_input("Your niches (comma-separated)", "real estate")
niches = [n.strip() for n in niches_raw.split(",") if n.strip()]

enrich_urls = st.sidebar.checkbox("If CSV has URLs, try to extract author from the page", value=False)
infer_niche_from_website = st.sidebar.checkbox(
    "If CSV Industry is sparse, infer niche by scanning the Website/Publication page",
    value=True,
)
run_btn = st.sidebar.button("Run")

if run_btn:
    combined_rows = []

    # 1) CSV rows
    if uploaded_contacts is not None:
        contacts_df = load_any_table(uploaded_contacts)
        csv_rows = contacts_to_rows(
            contacts_df,
            niches=niches,
            enrich_urls=enrich_urls,
            infer_niche_from_website=infer_niche_from_website,
        )
        combined_rows.extend(csv_rows)
        st.sidebar.success(f"CSV: added {len(csv_rows)} rows (with authors).")
    else:
        st.sidebar.info("No CSV uploaded (optional).")

    # 2) NewsAPI rows
    if api_key and query:
        try:
            with st.spinner(f"Fetching NewsAPI for '{query}'..."):
                articles = get_news(api_key, query, page_size=page_size)
            news_rows = newsapi_to_rows(articles, niches=niches)
            combined_rows.extend(news_rows)
            st.sidebar.success(f"NewsAPI: added {len(news_rows)} rows (with authors).")
        except requests.exceptions.HTTPError as err:
            try:
                msg = err.response.json().get("message")
            except Exception:
                msg = str(err)
            st.error(f"NewsAPI error: {msg}")
        except Exception as e:
            st.error(f"NewsAPI failed: {e}")
    else:
        st.sidebar.info("NewsAPI not used (key or query missing).")

    # 3) Unique journalists
    out_df = combined_unique_journalists(combined_rows)

    if out_df.empty:
        st.warning("No authors found after combining inputs. Try enabling URL author extraction or adjust niches/query.")
    else:
        st.success(f"Unique journalists found: {len(out_df):,}")
        st.dataframe(out_df, use_container_width=True, height=600)

        @st.cache_data
        def to_csv_bytes(df: pd.DataFrame) -> bytes:
            return df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download combined unique journalists (CSV)",
            data=to_csv_bytes(out_df),
            file_name="combined_unique_journalists.csv",
            mime="text/csv",
        )

        with st.expander("Show raw combined rows (debug)"):
            st.dataframe(pd.DataFrame(combined_rows), use_container_width=True, height=400)
else:
    st.info("Fill inputs in the sidebar and click **Run**.")
