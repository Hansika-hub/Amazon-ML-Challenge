#!/usr/bin/env python3
"""
Canonical preprocessing + feature extraction — Amazon ML Challenge 2026
(Business Entity Resolution).  ONE module, used for S1/S2/S3 (train + test).

Purpose: turn raw (business_name, business_address, country) into the fields
blocking and the matcher need. Fixes the 7 bugs found in final_preprocess.py:
  1. Accents now folded on ALL Latin text (was Tamil-branch only) -> café==cafe.
  2. Indic digits (०१२३…) normalized to ASCII BEFORE numeric extraction, so
     house numbers / PIN codes in Hindi-script addresses aren't lost.
  3. Transliteration goes via IAST then accent-fold -> no ITRANS ~ ^ junk chars.
  4. No aksharamukha anywhere (GPL-3.0). All Indic scripts via sanscript (MIT).
  5. Dedicated postal_code + house_number fields (not buried in a token list).
  6. Output is PARQUET -> list columns (tokens) round-trip as real lists.
  7. Importable: everything defined before use; run via preprocess_file().

Licensing (all permissive, safe to ship):
  indic_transliteration/sanscript = MIT, jellyfish = BSD, anyascii = ISC,
  pandas/numpy/pyarrow = BSD/Apache.

Blocking INTERFACE this produces (per record):
  entity_id, country,
  business_name, business_address        (raw, kept for matcher fuzzy features)
  name_norm, addr_norm                   (normalized latin text, for features)
  block_name, block_addr                 (aggressive canonical string for keys)
  name_tokens, addr_tokens               (list<str>, for inverted-index blocking)
  postal_code, house_number              (script-independent exact-key parts)
  name_phonetic                          (NYSIIS, for phonetic bridge)
  script_name, script_addr               (dominant script label; feature+routing)

Usage:
  python3 preprocess.py --in dataset/train/source2.tsv \
                        --out dataset/train/source2_prep.parquet
  # or import: from preprocess import preprocess_file
"""

import argparse, csv, re, unicodedata
from functools import lru_cache

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAVE_ARROW = True
except Exception:
    HAVE_ARROW = False

from indic_transliteration import sanscript
try:
    from anyascii import anyascii
except Exception:
    def anyascii(s): return s
try:
    import jellyfish
    def nysiis(s): return jellyfish.nysiis(s) if s else ""
except Exception:
    def nysiis(s): return ""


# =====================================================================
# 1. SCRIPT DETECTION  (Unicode code-point ranges)
# =====================================================================
# Ordered so the first-matching wins; Latin handled separately as fallback.
_SCRIPT_RANGES = [
    ("devanagari", 0x0900, 0x097F),
    ("bengali",    0x0980, 0x09FF),
    ("gurmukhi",   0x0A00, 0x0A7F),
    ("gujarati",   0x0A80, 0x0AFF),
    ("oriya",      0x0B00, 0x0B7F),
    ("tamil",      0x0B80, 0x0BFF),
    ("telugu",     0x0C00, 0x0C7F),
    ("kannada",    0x0C80, 0x0CFF),
    ("malayalam",  0x0D00, 0x0D7F),
    ("arabic",     0x0600, 0x06FF),
    ("cyrillic",   0x0400, 0x04FF),
    ("greek",      0x0370, 0x03FF),
    ("cjk",        0x4E00, 0x9FFF),
]
# Indo-Aryan scripts -> sanscript IAST + schwa-strip (good quality, MIT).
# Dravidian scripts (Tamil/Telugu/Kannada/Malayalam) -> anyascii: sanscript's
# Sanskrit-oriented IAST invents aspiration (dh/gh) these languages don't have
# (மோட்டார்ஸ்->modhdhars vs anyascii mottars). Measured; anyascii is closer.
_SANSCRIPT_SCHEME = {
    "devanagari": sanscript.DEVANAGARI,
    "bengali":    sanscript.BENGALI,
    "gurmukhi":   sanscript.GURMUKHI,
    "gujarati":   sanscript.GUJARATI,
    "oriya":      sanscript.ORIYA,
}

def _char_script(o):
    for name, lo, hi in _SCRIPT_RANGES:
        if lo <= o <= hi:
            return name
    return None

def detect_script(text):
    """Dominant non-Latin script in `text`, else 'latin' (covers ASCII+accents)."""
    if not text:
        return "latin"
    counts = {}
    for ch in text:
        o = ord(ch)
        if o < 0x0080 or (0x00C0 <= o <= 0x024F):   # ASCII or Latin-accent
            continue
        s = _char_script(o)
        if s:
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return "latin"
    return max(counts, key=counts.get)


# =====================================================================
# 2. DIGIT NORMALIZATION  (bug #2)  — map Indic/Arabic digits -> ASCII
# =====================================================================
# Built from Unicode Nd category so we don't hand-maintain per-script tables.
_DIGIT_MAP = {}
for _cp in range(0x0660, 0x0FFF + 1):          # Arabic + Indic digit blocks
    ch = chr(_cp)
    if unicodedata.category(ch) == "Nd":
        try:
            _DIGIT_MAP[ch] = str(unicodedata.digit(ch))
        except (ValueError, TypeError):
            pass
_DIGIT_TABLE = {ord(k): v for k, v in _DIGIT_MAP.items()}

def normalize_digits(text):
    return text.translate(_DIGIT_TABLE) if text else text


# =====================================================================
# 3. TRANSLITERATION  (bugs #1,#3,#4)
#    Indic script -> IAST -> accent-fold -> clean latin (no ~ ^ junk).
#    Non-Indic non-ASCII (Arabic/Cyrillic/CJK/…) -> anyascii.
# =====================================================================
def _fold_accents(text):
    """NFKD then drop combining marks: café->cafe, IAST diacritics->plain."""
    return "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c))

def _strip_schwa(s):
    """Approximate Devanagari schwa deletion: drop a trailing inherent 'a'
    (rama->ram, limiteda->limited). Only for longer tokens, to avoid mangling."""
    return s[:-1] if len(s) > 3 and s.endswith("a") else s

@lru_cache(maxsize=200_000)
def _translit_token(tok):
    """Transliterate ONE token that contains >=1 non-ASCII char (ASCII tokens
    are fast-pathed in transliterate() and never reach here)."""
    scr = detect_script(tok)
    if scr == "latin":                       # accented Latin (café -> cafe)
        return _fold_accents(tok)
    scheme = _SANSCRIPT_SCHEME.get(scr)       # Indo-Aryan only
    if scheme is not None:
        try:
            iast = sanscript.transliterate(tok, scheme, sanscript.IAST)
            return _strip_schwa(_fold_accents(iast))
        except Exception:
            pass
    # Dravidian (Tamil/Telugu/Kannada/Malayalam) + Arabic/Cyrillic/CJK/etc.
    return anyascii(tok)

def transliterate(text):
    """ASCII fast-path (points 6/7): a pure-ASCII string skips all script work;
    otherwise only the tokens that actually contain a non-ASCII char are
    transliterated, ASCII tokens pass through untouched."""
    if not text:
        return text
    if text.isascii():                        # all US rows + ASCII India rows
        return text
    return " ".join(t if t.isascii() else _translit_token(t) for t in text.split())


# =====================================================================
# 4. SUFFIX / STREET NORMALIZATION
# =====================================================================
# Company suffixes -> drop (they add noise to name blocking).
# NOTE: "and"/"the" are NOT dropped (point 3) — "&" is normalized to "and" up
# front so "Lee & Lawson" and "Lee and Lawson" converge instead of diverging.
_SUFFIX = {
    "pvt", "private", "ltd", "limited", "llp", "inc", "incorporated", "corp",
    "corporation", "co", "company", "plc", "gmbh", "sarl", "sa", "sas", "srl",
    "bv", "ag", "kg", "pte",
}
# Web-junk tokens to drop from NAMES (point 1: some names are URLs).
_WEB = {"www", "http", "https", "com", "net", "org", "io", "biz", "info"}
# Street-type variants -> canonical token (address blocking).
_STREET = {
    "street": "st", "st": "st", "road": "rd", "rd": "rd", "avenue": "ave",
    "ave": "ave", "av": "ave", "lane": "ln", "ln": "ln", "boulevard": "blvd",
    "blvd": "blvd", "drive": "dr", "dr": "dr", "nagar": "nagar", "marg": "marg",
    "phase": "phase", "sector": "sector", "block": "block", "floor": "fl",
    "flr": "fl", "fl": "fl", "building": "bldg", "bldg": "bldg", "opposite": "opp",
    "opp": "opp", "near": "near",
}

_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")

def _clean_latin(text):
    """casefold -> keep [a-z0-9 ] -> collapse whitespace."""
    text = text.casefold()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()

def _dedup(seq):
    """Order-preserving unique (point 2: 'delhi delhi delhi' -> one 'delhi')."""
    seen = set(); out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


# =====================================================================
# 5. POSTAL CODE + HOUSE NUMBER  (bug #5)
# =====================================================================
# Postal is country-specific: India PIN = 6 digits, US ZIP / France = 5 digits.
# Measured on real data: postal is SPARSE (~6.5% overall, US ~10%, India ~1%),
# so this is a high-precision BONUS key, not the blocking backbone.
_RE_6 = re.compile(r"\b\d{6}\b")
_RE_5 = re.compile(r"\b\d{5}\b")
_RE_46 = re.compile(r"\b\d{4,6}\b")
_RE_3_3 = re.compile(r"\b(\d{3})\s+(\d{3})\b")   # Indian PIN written "110 001"
_RE_HOUSE = re.compile(r"\b\d{1,4}\b")

def extract_postal_house(addr_digits_normalized, country=""):
    """Input: address AFTER digit-normalization + its country code.
    Postal is country-specific (IN=6-digit PIN, US/FR=5-digit); prefer the LAST
    matching run (postal usually sits at the end of an address). House = first
    short (1-4 digit) run that isn't the postal. Returns (postal, house)."""
    addr = addr_digits_normalized
    c = (country or "").strip().upper()
    postal = ""
    if c.startswith("IN"):
        six = _RE_6.findall(addr)
        if six:
            postal = six[-1]
        else:                                    # "110 001" spaced form
            spaced = _RE_3_3.findall(addr)
            if spaced:
                postal = spaced[-1][0] + spaced[-1][1]
    elif c.startswith("US") or c.startswith("FR"):
        five = _RE_5.findall(addr)
        if five:
            postal = five[-1]
    else:
        g = _RE_46.findall(addr)
        if g:
            postal = g[-1]
    house = ""
    for r in _RE_HOUSE.findall(addr):
        if r != postal:
            house = r
            break
    return postal, house


# =====================================================================
# 6. PER-FIELD PIPELINES
# =====================================================================
def process_name(raw):
    scr = detect_script(raw)
    s = normalize_digits(raw).replace("&", " and ")   # point 3
    clean = _clean_latin(transliterate(s))
    toks = [t for t in clean.split()
            if t not in _SUFFIX and t not in _WEB and len(t) > 1]
    toks = _dedup(toks)                                # point 2
    block = " ".join(toks)
    phon = nysiis(re.sub(r"[^a-z]", "", block)) if block else ""
    return scr, clean, block, toks, phon

def process_addr(raw, country=""):
    scr = detect_script(raw)
    digits = normalize_digits(raw)
    postal, house = extract_postal_house(digits, country)
    clean = _clean_latin(transliterate(digits.replace("&", " and ")))
    toks = [_STREET.get(t, t) for t in clean.split() if len(t) > 1]
    toks = _dedup(toks)                                # point 2
    block = " ".join(toks)
    return scr, clean, block, toks, postal, house


# =====================================================================
# 7. STREAMING DRIVER  (bug #6: parquet; bug #7: importable)
# =====================================================================
SRC_COLS = ["entity_id", "business_name", "business_address", "country"]

OUT_SCHEMA = [
    "entity_id", "country",
    "business_name", "business_address",
    "name_norm", "addr_norm",
    "block_name", "block_addr",
    "name_tokens", "addr_tokens",
    "postal_code", "house_number",
    "name_phonetic", "script_name", "script_addr",
]

def _process_chunk(df):
    df = df.reindex(columns=SRC_COLS, fill_value="")
    rows = {c: [] for c in OUT_SCHEMA}
    for eid, nm, ad, ctry in zip(df["entity_id"], df["business_name"],
                                 df["business_address"], df["country"]):
        n_scr, n_norm, n_block, n_toks, n_phon = process_name(nm or "")
        a_scr, a_norm, a_block, a_toks, postal, house = process_addr(ad or "", ctry or "")
        rows["entity_id"].append(eid)
        rows["country"].append((ctry or "").strip())
        rows["business_name"].append(nm or "")
        rows["business_address"].append(ad or "")
        rows["name_norm"].append(n_norm)
        rows["addr_norm"].append(a_norm)
        rows["block_name"].append(n_block)
        rows["block_addr"].append(a_block)
        rows["name_tokens"].append(n_toks)
        rows["addr_tokens"].append(a_toks)
        rows["postal_code"].append(postal)
        rows["house_number"].append(house)
        rows["name_phonetic"].append(n_phon)
        rows["script_name"].append(n_scr)
        rows["script_addr"].append(a_scr)
    return pd.DataFrame(rows, columns=OUT_SCHEMA)

def preprocess_file(in_path, out_path, chunksize=200_000, limit=None, verbose=True):
    """Stream a source TSV -> preprocessed parquet. Memory-flat (chunked)."""
    reader = pd.read_csv(in_path, sep="\t", dtype=str, keep_default_na=False,
                         na_values=[], quoting=csv.QUOTE_NONE, on_bad_lines="warn",
                         engine="c", chunksize=chunksize)
    writer = None
    total = 0
    fallback_frames = []       # used only if pyarrow missing
    for i, chunk in enumerate(reader):
        out_df = _process_chunk(chunk)
        total += len(out_df)
        if HAVE_ARROW:
            table = pa.Table.from_pandas(out_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
        else:
            fallback_frames.append(out_df)
        if verbose:
            print(f"  chunk {i}: {total:,} rows processed")
        if limit and total >= limit:
            break
    if HAVE_ARROW:
        if writer is not None:
            writer.close()
    else:
        big = pd.concat(fallback_frames, ignore_index=True)
        # parquet still preferred; needs pyarrow OR fastparquet. Warn + TSV.
        try:
            big.to_parquet(out_path)
        except Exception:
            tsv = out_path.rsplit(".", 1)[0] + ".tsv"
            print(f"!! no parquet engine -> writing TSV (list cols stringified): {tsv}")
            big.to_csv(tsv, sep="\t", index=False)
            out_path = tsv
    if verbose:
        print(f"done: {total:,} rows -> {out_path}")
    return out_path


def parquet_to_tsv(parquet_path, tsv_path=None):
    """Convenience: read a *_prep.parquet and write a TSV for eyeballing in Excel.
    List columns (name_tokens/addr_tokens) are joined with '|' so they read
    cleanly in a cell. The parquet stays the canonical artifact for downstream."""
    if tsv_path is None:
        tsv_path = parquet_path.rsplit(".", 1)[0] + ".tsv"
    df = pd.read_parquet(parquet_path)
    for col in ("name_tokens", "addr_tokens"):
        if col in df.columns:
            df[col] = df[col].apply(lambda xs: "|".join(xs) if xs is not None else "")
    df.to_csv(tsv_path, sep="\t", index=False)
    print(f"wrote {len(df):,} rows -> {tsv_path}")
    return tsv_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N rows (for a quick sample run)")
    a = ap.parse_args()
    preprocess_file(a.in_path, a.out_path, a.chunksize, a.limit)


if __name__ == "__main__":
    main()
