import re
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st


# -----------------------------
# Helpers
# -----------------------------
def _safe_strip(x):
    return "" if x is None else str(x).strip()


def normalize_author(author: str) -> str:
    """
    NewsAPI 'author' is messy (e.g., 'By John Doe | Editor', 'John Doe, Staff Writer').
    We normalize to improve de-duplication.
    """
    s = _safe_strip(author)
    if not s:
        return ""
    s = re.sub(r"^\s*by\s+", "", s, flags=re.IGNORECASE).strip()
    # Keep the most likely name portion
    s = re.split(r"\s*(?:\||,| - )\s*", s, maxsplit=1)[0].strip()
    # Collapse repeated whitespace
    s = re.sub(r"\s+", " ", s).strip()
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


def split_urls_cell(cell: str) -> list[str]:
    """
    Contacts sheet has Website/Publication cells that can contain multiple URLs.
    """
    s = _safe_strip(cell)
    if not s:
        return []
    parts = [p.strip() for p in re.split(r"[\n,]+", s) if p.strip()]
    # Keep only items that look like URLs/domains
    out = []
    for p in parts:
        if p.startswith("http://") or p.startswith("https://"):
            out.append(p)
        else:
            # try to coerce to url for parsing (e.g., "example.com")
            if "." in p and " " not in p:
                out.append("https://" + p)
    return out


def load_contacts(uploaded_file) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = pd.read_excel(uploaded_file)
    # standardize column names (don't break if user edits headers)
    df.columns = [c.strip() for c in df.columns]
    return df


def build_contacts_index(contacts_df: pd.DataFrame):
    """
    Creates quick lookup indices:
      - name -> row
      - publication domain -> best matching row
    """
    name_col = None
    for c in contacts_df.columns:
        if c.lower() in {"name", "author", "writer", "journalist"}:
            name_col = c
            break
    if name_col is None:
        name_col = contacts_df.columns[0]

    # Email column (best effort)
    email_col = None
    for c in contacts_df.columns:
        if "email" in c.lower():
            email_col = c
            break

    pub_col = None
    for c in contacts_df.columns:
        if c.lower() in {"website/publication", "publication", "website", "site"}:
            pub_col = c
            break

    name_index = {}
    domain_index = {}

    for _, row in contacts_df.iterrows():
        nm = _safe_strip(row.get(name_col))
        if nm:
            name_index[nm.lower()] = row

        if pub_col:
            for u in split_urls_cell(row.get(pub_col)):
                d = domain_from_url(u)
                if d and d not in domain_index:
                    domain_index[d] = row

    return {
        "name_col": name_col,
        "email_col": email_col,
        "pub_col": pub_col,
        "name_index": name_index,
        "domain_index": domain_index,
    }


# -----------------------------
# NewsAPI
# -----------------------------
def get_news(api_key: str, query: str, page_size: int = 100) -> list[dict]:
    """
    Fetches news from NewsAPI based on a query.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": min(max(int(page_size), 1), 100),
    }
    headers = {"X-Api-Key": api_key}

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get("articles", [])


def articles_to_unique_journalists(articles: list[dict], contacts_index=None) -> pd.DataFrame:
    """
    1) Remove duplicate articles (by URL)
    2) Aggregate into UNIQUE journalists (by normalized author)
       - For each journalist: article count, sources, sample titles, urls
    3) Optional: enrich with email/contact info from contacts list (match by name or domain)
    """
    # 1) De-dupe articles by URL
    seen_urls = set()
    cleaned = []
    for a in articles:
        url = _safe_strip(a.get("url"))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        cleaned.append(a)

    buckets = {}  # key: author_norm -> aggregator
    for a in cleaned:
        author_raw = a.get("author")
        author = normalize_author(author_raw)
        if not author:
            continue

        src = _safe_strip((a.get("source") or {}).get("name"))
        title = _safe_strip(a.get("title"))
        url = _safe_strip(a.get("url"))
        dom = domain_from_url(url)

        b = buckets.setdefault(
            author.lower(),
            {
                "Author": author,
                "Sources": set(),
                "Article Count": 0,
                "Sample Titles": [],
                "Article URLs": [],
                "Article Domains": set(),
            },
        )
        if src:
            b["Sources"].add(src)
        b["Article Count"] += 1
        if title and len(b["Sample Titles"]) < 3 and title not in b["Sample Titles"]:
            b["Sample Titles"].append(title)
        if url:
            b["Article URLs"].append(url)
        if dom:
            b["Article Domains"].add(dom)

    rows = []
    for b in buckets.values():
        row = {
            "Author": b["Author"],
            "Article Count": b["Article Count"],
            "Sources": ", ".join(sorted(b["Sources"])) if b["Sources"] else "",
            "Sample Titles": " | ".join(b["Sample Titles"]),
            "Article URLs": "\n".join(b["Article URLs"][:10]),  # keep table readable
        }

        # 3) Enrich with contacts (best effort)
        if contacts_index:
            match = None

            # Match by name
            match = contacts_index["name_index"].get(b["Author"].lower())

            # If not found, try publication domain match
            if match is None:
                for d in b["Article Domains"]:
                    match = contacts_index["domain_index"].get(d)
                    if match is not None:
                        break

            if match is not None:
                email_col = contacts_index["email_col"]
                pub_col = contacts_index["pub_col"]
                row["Email (from contacts)"] = _safe_strip(match.get(email_col)) if email_col else ""
                row["Publication (from contacts)"] = _safe_strip(match.get(pub_col)) if pub_col else ""
            else:
                row["Email (from contacts)"] = ""
                row["Publication (from contacts)"] = ""

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["Article Count", "Author"], ascending=[False, True]).reset_index(drop=True)
    return out


def filter_contacts_by_niches(contacts_df: pd.DataFrame, niches: list[str]) -> pd.DataFrame:
    """
    Pull "relevant ones that can be reached out to" from the contacts sheet.
    Best-effort logic:
      - If Industry contains any niche keyword (case-insensitive) OR Industry == 'All' -> keep
    """
    if contacts_df is None or contacts_df.empty:
        return contacts_df

    # find an Industry column if present
    industry_col = None
    for c in contacts_df.columns:
        if c.lower() in {"industry", "niche", "category", "categories"}:
            industry_col = c
            break
    if industry_col is None:
        # No industry column -> return as-is (can't filter reliably)
        return contacts_df

    keys = [k.strip().lower() for k in niches if k.strip()]
    if not keys:
        return contacts_df

    def is_relevant(val):
        s = _safe_strip(val).lower()
        if not s:
            return False
        if s == "all":
            return True
        return any(k in s for k in keys)

    return contacts_df[contacts_df[industry_col].apply(is_relevant)].copy()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("📰 Journalist & Outreach Contact Finder")
st.markdown("Search NewsAPI and/or filter your contacts list to get a clean, de-duplicated outreach list.")

st.sidebar.header("⚙️ Settings")
st.sidebar.markdown(
    """
**1) NewsAPI key**
Register to generate your key:
https://newsapi.org/register
"""
)

api_key = st.sidebar.text_input("Enter YOUR NewsAPI Key", type="password")
query = st.sidebar.text_input("News search query (e.g., 'real estate')", "real estate")
page_size = st.sidebar.slider("Max articles to fetch (NewsAPI pageSize)", min_value=10, max_value=100, value=100, step=10)

st.sidebar.markdown("---")
st.sidebar.subheader("📎 Optional: Upload contacts list")
uploaded_contacts = st.sidebar.file_uploader(
    "Upload CSV/XLSX with contacts (Name OR Publication URL, Email, Industry, etc.)",
    type=["csv", "xlsx"],
)
niches_raw = st.sidebar.text_input("Your niches (comma-separated)", "real estate")
niches = [n.strip() for n in niches_raw.split(",") if n.strip()]

only_matched_contacts = st.sidebar.checkbox("Show only journalists that match my contacts list", value=False)

search_button = st.sidebar.button("Search / Refresh")

tab1, tab2 = st.tabs(["NewsAPI → Unique journalists", "Contacts list → Filter by niche"])

contacts_df = None
contacts_index = None
if uploaded_contacts is not None:
    contacts_df = load_contacts(uploaded_contacts)
    contacts_index = build_contacts_index(contacts_df)

with tab1:
    st.subheader("Unique journalists from NewsAPI (duplicates removed)")
    if search_button:
        if not api_key:
            st.error("Please enter your NewsAPI Key in the sidebar.")
        elif not query:
            st.warning("Please enter a search query.")
        else:
            try:
                with st.spinner(f"Searching NewsAPI for '{query}'..."):
                    articles = get_news(api_key, query, page_size=page_size)
                st.success(f"Fetched {len(articles)} articles (before de-dup).")

                journalists_df = articles_to_unique_journalists(articles, contacts_index=contacts_index)

                if contacts_index and only_matched_contacts and not journalists_df.empty:
                    journalists_df = journalists_df[
                        (journalists_df["Email (from contacts)"] != "") | (journalists_df["Publication (from contacts)"] != "")
                    ].copy()

                if journalists_df.empty:
                    st.warning("No usable authors found (NewsAPI often omits authors). Try a different query.")
                else:
                    st.info(f"Showing {len(journalists_df)} UNIQUE journalists.")
                    st.dataframe(journalists_df, use_container_width=True, height=520)

                    @st.cache_data
                    def convert_df(df: pd.DataFrame) -> bytes:
                        return df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download unique journalists as CSV",
                        data=convert_df(journalists_df),
                        file_name=f"{query}_unique_journalists.csv",
                        mime="text/csv",
                    )

            except requests.exceptions.HTTPError as err:
                try:
                    msg = err.response.json().get("message")
                except Exception:
                    msg = str(err)
                st.error(f"NewsAPI error: {msg}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    else:
        st.caption("Use the sidebar and click **Search / Refresh**.")

with tab2:
    st.subheader("Filter your contacts list by niche (no NewsAPI needed)")
    if uploaded_contacts is None:
        st.info("Upload your contacts list from the sidebar to use this tab.")
    else:
        filtered = filter_contacts_by_niches(contacts_df, niches)
        st.write(f"Contacts uploaded: **{len(contacts_df):,}**  •  Matching niches: **{len(filtered):,}**")
        st.dataframe(filtered, use_container_width=True, height=520)

        @st.cache_data
        def convert_df2(df: pd.DataFrame) -> bytes:
            return df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download filtered contacts as CSV",
            data=convert_df2(filtered),
            file_name="filtered_contacts.csv",
            mime="text/csv",
        )
