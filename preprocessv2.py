# ============================================================
# ENTITY RESOLUTION — PREPROCESSING PIPELINE
# (cleaning + missing-value flags + duplicate-cluster tagging only;
#  blocking/candidate generation is a separate, later stage)
# ============================================================
#
# WHY THIS PIPELINE IS STRUCTURED THIS WAY (read this before running)
# ------------------------------------------------------------
# Your data (from EDA):
#   - S1 ~2.2M rows, S2/S3 ~5M rows each -> must be vectorized, not row-by-row.
#   - business_name, business_address have missing values (~0-3%).
#   - country is clean (US/India in train, +France in test) -> use as a
#     HARD blocking key, never fuzzy-match across countries.
#   - Multi-script data: Devanagari, Kannada, French-accented Latin all
#     appear in business_name/address -> need a transliteration branch.
#   - No exact duplicate ROWS, but business_name/address uniqueness is
#     only 70-96% -> some of that is coincidental name collisions
#     (different shops, same name), some is genuine duplicate-entity
#     cases (same shop, multiple source records) -> must NOT be
#     conflated. This is handled at the clustering step (Section E),
#     not by blindly grouping on name alone.
#   - ~5.6% of ground_truth rows have NO match at all -> pipeline must
#     support "no match" as a valid outcome, never force a match.
#
# CORE PRINCIPLE carried through every step:
#   Missing data is NOT the same as "no evidence of mismatch" and NOT
#   the same as "matches anything else that's also missing". A missing
#   field is a third state ("unknown"), and comparisons must SKIP a
#   field when either side is missing rather than silently treating
#   two blanks as equal (that creates false matches) or a blank as
#   different from a real value (that creates false mismatches).
#   This is implemented via *_is_missing boolean flags kept alongside
#   every cleaned field, never by filling missing with "" and comparing.
#
# Libraries used are all on the ALLOWED list from your rules doc:
#   pandas, re, unicodedata (stdlib), jellyfish (phonetic), rapidfuzz
#   (optional, for later matching stage) -- all pure algorithms, no
#   external business/geographic reference data.
# ============================================================

import re
import unicodedata
import pandas as pd
import numpy as np

# jellyfish: phonetic codes (metaphone/nysiis) -- allowed, pure algorithm
try:
    import jellyfish
    HAVE_JELLYFISH = True
except ImportError:
    HAVE_JELLYFISH = False
    print("WARNING: jellyfish not installed -> phonetic codes will be skipped. "
          "pip install jellyfish --break-system-packages")

# unidecode: offline Latin-transliteration fallback -- pure algorithm,
# no external gazetteer/business data, so it stays within the rules.
try:
    from unidecode import unidecode
    HAVE_UNIDECODE = True
except ImportError:
    HAVE_UNIDECODE = False
    print("WARNING: unidecode not installed -> transliteration will be skipped. "
          "pip install unidecode --break-system-packages")


# ------------------------------------------------------------
# 0. NORMALIZATION DICTIONARIES (hand-written, allowed per rules doc)
# ------------------------------------------------------------

# Business legal-suffix normalization.
# WHY: sources record the same legal form inconsistently
# ("Pvt Ltd" / "P.V.T. L.T.D." / "pvt.ltd."), and your rules doc
# explicitly allows small hand-written dictionaries for this.
# Business legal-suffix normalization.
# WHY: sources record the same legal form inconsistently
# ("Pvt Ltd" / "P.V.T. L.T.D." / "pvt.ltd."), and your rules doc
# explicitly allows small hand-written dictionaries for this.
#
# BUILT FROM REAL FREQUENCY DATA (discover_dictionaries.py output on
# all 24.2M rows across train+test), not guessed. Only tokens that are
# genuine LEGAL-FORM markers are included here. Deliberately EXCLUDED,
# with reasons:
#   - "com"        : part of domain-style business names (e.g.
#                    "wilfordhancock.com" seen in your EDA sample),
#                    not a legal suffix -- stripping it would corrupt
#                    real brand names.
#   - "center", "services", "partners", "group", "holdings", "club",
#     "fils", "groupe", "développement", "france" : generic business-
#     descriptor words that are often part of the actual distinguishing
#     name (e.g. "XYZ Group" != "XYZ"), not a legal-entity marker --
#     stripping these risks merging genuinely different businesses.
#   - "c" / "s" / single letters : too ambiguous/high-collision risk
#     to map confidently without more context.
SUFFIX_MAP = {
    # multi-word patterns FIRST -- apply_dictionary() applies patterns
    # in dict order, so single-word replacements (pvt->private,
    # ltd->limited) must not run before their multi-word combinations
    # are matched, or "pvt ltd" would already be "private limited"
    # tokens by the time this pattern runs and never match as a unit.
    r"\bpvt\.?\s+ltd\.?\b": "private limited",
    r"\bprivate\s+ltd\.?\b": "private limited",
    r"\bpublic\s+limited\b": "public limited",
    r"\bl\s+c\b": "llc",          # "l c" variant seen in data (212k)
    r"\bs\s+a\s+s\b": "sas",      # "S.A.S" -> "s a s" after punct strip
    r"\ba\s+s\b": "sas",          # partial "a s" fallback (25k)
    r"\br\s+l\b": "sarl",         # "r l" -- S.à r.l.-style variant
    r"प्राइवेट\s+लिमिटेड": "private limited",
    r"प्रा\s*लि": "private limited",

    # single-word patterns after
    r"\bpvt\.?\b": "private",
    r"\bltd\.?\b": "limited",
    r"\bllc\b": "llc",
    r"\binc\.?\b": "inc",
    r"\bcorp\.?\b": "corporation",
    r"\bco\.?\b": "company",
    r"\bllp\b": "llp",
    r"\blp\b": "lp",
    r"\bgmbh\b": "gmbh",
    r"\bsarl\b": "sarl",
    r"\bsas\b": "sas",
    r"\bs\.a\.s\.?\b": "sas",
    r"\bsasu\b": "sasu",
    r"\beurl\b": "eurl",
    r"\bsci\b": "sci",
    r"\bsa\b": "sa",
    r"लिमिटेड": "limited",
    r"\bलि\b": "limited",
}

# All suffix tokens (canonical form), used to build a suffix-STRIPPED
# version of the name separately from the suffix-NORMALIZED version.
SUFFIX_TOKENS = set(SUFFIX_MAP.values())

# Street-type normalization for addresses.
# BUILT FROM REAL FREQUENCY DATA (token-after-first-number counts,
# global + per-country). Only genuine STREET-TYPE words included.
# Deliberately EXCLUDED, with reasons:
#   - "floor", "flat", "plot", "sector", "ground", "c/o", "1/2", "bis"
#     : these are Indian/French address-STRUCTURE markers (unit/plot/
#     floor descriptors), not street types like Road/Avenue -- they
#     belong with unit extraction (UNIT_PATTERN), not street renaming.
#   - "1st"/"2nd"/"3rd"/"4th"/"first"/"second"/"new"/"old"/"main"/
#     "south" : ordinals/directions/generic descriptors, often part
#     of the actual street NAME (e.g. "Old Mill Road", "2nd Street"
#     is itself a name, not a suffix to normalize away).
#   - "mumbai", "bangalore", "hyderabad", "washington", "france" :
#     place names, not street types -- normalizing these would
#     conflate location identity with street-type cleanup.
#   - "a", "b", "s", "r" (bare single letters) : too ambiguous on
#     their own (could be unit letters, initials) without more
#     context to map safely.
STREET_MAP = {
    # English / US
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bdr\.?\b": "drive",
    r"\bln\.?\b": "lane",
    r"\bct\.?\b": "court",
    r"\bpl\.?\b": "place",
    r"\bapt\.?\b": "apartment",
    r"\bbldg\.?\b": "building",
    r"\bhwy\.?\b": "highway",

    # French (from France-specific frequency breakdown: rue, avenue,
    # bd, allée/all, impasse, route, av all appear at real volume)
    r"\brue\b": "rue",
    r"\bav\.?\b": "avenue",
    r"\bbd\.?\b": "boulevard",
    r"\ballée\b": "allee",
    r"\ballee\b": "allee",
    r"\ball\.?\b": "allee",
    r"\bimpasse\b": "impasse",
    r"\broute\b": "route",
}

# Regex to find a unit/suite/apartment fragment inside an address string.
# WHY: this is the single most important disambiguator for the
# "same building, different shop" vs "same shop" problem you flagged.
# We EXTRACT it into its own field rather than deleting it.
UNIT_PATTERN = re.compile(
    r"\b(?:(?:unit|apt|apartment|suite|ste|#)\s*[:\-]?\s*)+([a-z0-9\-]+)",
    flags=re.IGNORECASE,
)
# NOTE: deliberately does NOT include "no"/"number" as a trigger.
# Real sample data (e.g. "KH NO. -570/13", "H.NO 204") showed "No."
# almost always means HOUSE/PLOT/KHASRA number in Indian addresses --
# a core, high-value part of the address itself -- not a "different
# unit in the same building" marker like "Unit 5" or "Suite 200".
# Including it caused the house number to be wrongly stripped out of
# address_normalized and mislabeled as unit_value (confirmed on a
# real S2 sample run). If you later want the house number as its own
# feature, extract it separately with its own pattern/field -- don't
# conflate it with unit_value.

LEADING_JUNK_PATTERN = re.compile(r"^[^a-zA-Z0-9\u0900-\u097F\u0C80-\u0CFF]+")
PUNCT_PATTERN = re.compile(r"[.,;:()\[\]{}\"'`]")
AMP_PATTERN = re.compile(r"&")
WHITESPACE_PATTERN = re.compile(r"\s+")


# ------------------------------------------------------------
# 1. CORE TEXT CLEANUP (shared by name and address)
# ------------------------------------------------------------

def unicode_normalize(series: pd.Series) -> pd.Series:
    """NFKC normalization. WHY: makes visually-identical characters
    (e.g. accented letters encoded differently) byte-identical, so
    downstream string comparisons/n-grams aren't silently broken by
    encoding differences rather than real content differences."""
    return series.astype(str).map(
        lambda x: unicodedata.normalize("NFKC", x) if pd.notna(x) else x
    )


def casefold_series(series: pd.Series) -> pd.Series:
    """Casefold (not just .lower()) -- handles multi-language case
    rules correctly (e.g. German sharp s), important since data spans
    US/India/France scripts and locales."""
    return series.astype(str).map(lambda x: x.casefold() if pd.notna(x) else x)


def strip_leading_junk(series: pd.Series) -> pd.Series:
    """WHY: EDA showed raw junk like '-- Holloway Peak Inc Seafood' and
    '<< Team Ecole' -- these leading symbols would otherwise corrupt
    n-gram/token-based blocking."""
    return series.str.replace(LEADING_JUNK_PATTERN, "", regex=True)


def normalize_whitespace(series: pd.Series) -> pd.Series:
    """Collapse double/irregular spaces (EDA showed
    'Private  (Limited)' with a double space) and trim edges."""
    return series.str.replace(WHITESPACE_PATTERN, " ", regex=True).str.strip()


def normalize_punctuation(series: pd.Series) -> pd.Series:
    """Strip most punctuation; convert & -> 'and' rather than deleting
    it, so 'Smith & Sons' and 'Smith and Sons' converge instead of
    'Smith & Sons' becoming 'Smith  Sons' (broken) vs 'Smith and Sons'."""
    series = series.str.replace(AMP_PATTERN, " and ", regex=True)
    series = series.str.replace(PUNCT_PATTERN, " ", regex=True)
    return normalize_whitespace(series)


def apply_dictionary(series: pd.Series, mapping: dict) -> pd.Series:
    """Apply a hand-written regex->canonical-token dictionary.
    Order matters: run this AFTER whitespace/punctuation cleanup so
    suffix regexes don't need to handle every punctuation variant
    separately (e.g. 'Pvt.' and 'Pvt' are already the same by here)."""
    for pattern, replacement in mapping.items():
        series = series.str.replace(pattern, replacement, regex=True)
    return normalize_whitespace(series)


def basic_clean(series: pd.Series) -> pd.Series:
    """Pipeline stages 1-4 (Unicode -> casefold -> junk -> whitespace ->
    punctuation), shared by name and address before they diverge into
    field-specific handling (suffix dict for name, street dict for
    address)."""
    s = unicode_normalize(series)
    s = casefold_series(s)
    s = strip_leading_junk(s)
    s = normalize_whitespace(s)
    s = normalize_punctuation(s)
    return s


# ------------------------------------------------------------
# 2. TRANSLITERATION BRANCH
# ------------------------------------------------------------

def transliterate_series(series: pd.Series) -> pd.Series:
    """Offline Latin transliteration for Devanagari/Kannada names
    (EDA showed e.g. 'राम मार्केटिंग प्राइवेट लिमिटेड').
    WHY: a business may appear in native script in one source and
    Latin script in another -- without this branch those records can
    never be blocked/matched together. unidecode is a pure character-
    mapping algorithm, not an external business/geo database, so it
    stays within the rules."""
    # if not HAVE_UNIDECODE:
    #     return series
    return series.map(lambda x: unidecode(x) if pd.notna(x) else x)


# ------------------------------------------------------------
# 3. NAME-SPECIFIC DERIVED FIELDS
# ------------------------------------------------------------

def strip_suffixes(series: pd.Series) -> pd.Series:
    """Remove suffix tokens entirely (separate from suffix
    NORMALIZATION above). WHY keep both: normalized-with-suffix
    catches exact business-form matches; suffix-stripped catches
    cases where one source recorded the legal suffix and another
    didn't (e.g. 'Prime Money' vs 'Prime Money Inc')."""
    pattern = r"\b(" + "|".join(SUFFIX_TOKENS) + r")\b"
    return series.str.replace(pattern, "", regex=True).pipe(normalize_whitespace)


def sorted_tokens(series: pd.Series) -> pd.Series:
    """Alphabetically sort words in the name. WHY: catches word-order
    swaps across sources, e.g. 'Star Bakery' vs 'Bakery Star'."""
    return series.map(
        lambda x: " ".join(sorted(x.split())) if pd.notna(x) and x else x
    )


def phonetic_code(series: pd.Series) -> pd.Series:
    """NYSIIS phonetic code. WHY: catches typos/minor spelling
    variants that pure string similarity may miss. Only meaningful
    for Latin-script text, so run this on the TRANSLITERATED name,
    not the native-script one."""
    if not HAVE_JELLYFISH:
        return pd.Series([np.nan] * len(series), index=series.index)
    return series.map(
        lambda x: jellyfish.nysiis(x) if pd.notna(x) and x else np.nan
    )


# ------------------------------------------------------------
# 4. ADDRESS-SPECIFIC DERIVED FIELDS
# ------------------------------------------------------------

def extract_unit(series: pd.Series) -> pd.Series:
    """Pull out unit/suite/apartment number into its own field
    INSTEAD of deleting it. WHY: this is the key signal for
    'same building, different shop' (different unit -> do NOT match)
    vs a real duplicate (same/missing unit -> supports a match)."""
    return series.map(
        lambda x: (m.group(1) if (m := UNIT_PATTERN.search(x)) else np.nan)
        if pd.notna(x) else np.nan
    )


def remove_unit_fragment(series: pd.Series) -> pd.Series:
    """Strip the matched unit fragment out of the main address string
    so the 'street part' of the address can be compared cleanly on
    its own, separate from the unit signal."""
    return series.str.replace(UNIT_PATTERN, " ", regex=True).pipe(normalize_whitespace)


# ------------------------------------------------------------
# 5. MAIN PER-SOURCE PREPROCESSING FUNCTION
# ------------------------------------------------------------

def preprocess_source(df: pd.DataFrame, id_col: str = "entity_id") -> pd.DataFrame:
    """
    Runs the full pipeline on one source dataframe
    (columns expected: id_col, business_name, business_address, country).

    Returns the ORIGINAL dataframe with extra columns appended --
    raw fields are never overwritten, only added to, since raw values
    are still needed later as a high-precision exact-match feature.
    """
    out = df.copy()

    # --- missing flags FIRST, before any filling/cleaning ---
    # WHY: must be computed on the raw column, before anything
    # (including basic_clean) could turn a blank into a non-null
    # empty string that would defeat the flag.
    out["name_is_missing"] = out["business_name"].isna()
    out["address_is_missing"] = out["business_address"].isna()

    # --- NAME pipeline ---
    name_clean = basic_clean(out["business_name"].fillna(""))
    out["name_normalized"] = apply_dictionary(name_clean, SUFFIX_MAP)
    out["name_suffix_stripped"] = strip_suffixes(out["name_normalized"])
    out["name_sorted_tokens"] = sorted_tokens(out["name_suffix_stripped"])
    out["name_transliterated"] = transliterate_series(out["name_normalized"])
    out["name_phonetic"] = phonetic_code(out["name_transliterated"])

    # re-apply missing flag: after fillna(""), an originally-missing
    # name is now an empty string post-cleaning -- blank out the
    # derived fields too so they can't accidentally string-match
    # another blank (see Section D discussion: never let two missing
    # values look like a match).
    out.loc[out["name_is_missing"], [
        "name_normalized", "name_suffix_stripped",
        "name_sorted_tokens", "name_transliterated", "name_phonetic"
    ]] = np.nan

    # --- ADDRESS pipeline ---
    addr_clean = basic_clean(out["business_address"].fillna(""))
    addr_clean = apply_dictionary(addr_clean, STREET_MAP)
    out["unit_value"] = extract_unit(addr_clean)
    out["address_normalized"] = remove_unit_fragment(addr_clean)

    out.loc[out["address_is_missing"], [
        "unit_value", "address_normalized"
    ]] = np.nan

    # --- COUNTRY: light touch only, never map to a fixed vocabulary ---
    out["country_normalized"] = (
        out["country"].astype(str).str.strip().str.casefold()
    )

    return out


# ------------------------------------------------------------
# 6. DUPLICATE-CLUSTER TAGGING (within S2 or within S3)
# ------------------------------------------------------------

def tag_duplicate_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups rows that look like the SAME business recorded more than
    once within one source, WITHOUT deleting or merging any row.

    WHY these exact conditions (per your rules doc section 3 + the
    unit-number discussion):
      - name must match (on the suffix-stripped normalized field, so
        'Prime Money' and 'Prime Money Inc' can still cluster)
      - street address must match exactly (on address_normalized,
        i.e. WITHOUT the unit fragment)
      - unit must match OR both be missing -- a DIFFERENT unit at an
        otherwise-identical address means DIFFERENT businesses
        (same building, different shop) and must NOT be clustered
      - country must match
      - rows where name or address is missing are excluded from
        clustering entirely -- you cannot safely cluster on unknown
        data (this is the same "missing != match" principle as
        Section D).
    """
    out = df.copy()
    out["dup_cluster_id"] = np.nan

    clusterable = out[
        ~out["name_is_missing"] & ~out["address_is_missing"]
    ].copy()

    # unit_key: use the real unit value, or a shared sentinel for
    # "no unit at all" so that two unit-less addresses at the same
    # street can still cluster together.
    clusterable["unit_key"] = clusterable["unit_value"].fillna("__NO_UNIT__")

    group_cols = [
        "name_suffix_stripped", "address_normalized",
        "unit_key", "country_normalized",
    ]
    clusterable["dup_cluster_id"] = clusterable.groupby(group_cols).ngroup()

    out.loc[clusterable.index, "dup_cluster_id"] = clusterable["dup_cluster_id"]
    return out


# ------------------------------------------------------------
# 7. RUN IT -- S2 ONLY, 500 ROWS (quick test run)
# ------------------------------------------------------------
#
# WHY nrows=500 at read time (not .head(500) after loading): with
# millions of rows in the real file, reading only the first 500 via
# pd.read_csv(nrows=...) avoids loading the full multi-GB file into
# memory just to test the pipeline logic.
#
# Change SPLIT below to "test" if you want test_source2.tsv instead.

if __name__ == "__main__":
    BASE = "/kaggle/input/datasets/ishanabharathi/datasetml/student_resource"
    SPLIT = "train"   # or "test"
    N_ROWS = 500

    path = f"{BASE}/dataset/{SPLIT}/{SPLIT}_source2.tsv"
    print(f"Reading first {N_ROWS} rows from {path} ...")
    df = pd.read_csv(path, sep="\t", nrows=N_ROWS, low_memory=False)
    print(f"Loaded shape: {df.shape}")

    df = preprocess_source(df)
    df = tag_duplicate_clusters(df)   # S2/S3 only, not S1

    out_path = f"{SPLIT}_source2_preprocessed_sample500.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved -> {out_path}, shape={df.shape}")

    # quick sanity peek
    cols_to_show = [
        "business_name", "name_normalized", "name_suffix_stripped",
        "business_address", "address_normalized", "unit_value",
        "dup_cluster_id",
    ]
    print("\nSample output:")
    display(df.head(15))
