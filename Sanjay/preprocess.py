#!/usr/bin/env python3
"""Canonical preprocessing v3 — Amazon ML Challenge 2026."""
import argparse, csv, re, unicodedata
from functools import lru_cache
import pandas as pd
try:
    import pyarrow as pa, pyarrow.parquet as pq
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

_SCRIPT_RANGES = [("devanagari",0x0900,0x097F),("bengali",0x0980,0x09FF),("gurmukhi",0x0A00,0x0A7F),
    ("gujarati",0x0A80,0x0AFF),("oriya",0x0B00,0x0B7F),("tamil",0x0B80,0x0BFF),("telugu",0x0C00,0x0C7F),
    ("kannada",0x0C80,0x0CFF),("malayalam",0x0D00,0x0D7F),("arabic",0x0600,0x06FF),
    ("cyrillic",0x0400,0x04FF),("greek",0x0370,0x03FF),("cjk",0x4E00,0x9FFF)]
# Indo-Aryan -> sanscript IAST + schwa-strip; Dravidian -> anyascii (better)
_SANSCRIPT_SCHEME = {"devanagari":sanscript.DEVANAGARI,"bengali":sanscript.BENGALI,
    "gurmukhi":sanscript.GURMUKHI,"gujarati":sanscript.GUJARATI,"oriya":sanscript.ORIYA}
def _char_script(o):
    for name,lo,hi in _SCRIPT_RANGES:
        if lo<=o<=hi: return name
    return None
def detect_script(text):
    if not text: return "latin"
    counts={}
    for ch in text:
        o=ord(ch)
        if o<0x0080 or (0x00C0<=o<=0x024F): continue
        s=_char_script(o)
        if s: counts[s]=counts.get(s,0)+1
    return max(counts,key=counts.get) if counts else "latin"

_DIGIT_TABLE={}
for _cp in range(0x0660,0x0FFF+1):
    _ch=chr(_cp)
    if unicodedata.category(_ch)=="Nd":
        try: _DIGIT_TABLE[ord(_ch)]=str(unicodedata.digit(_ch))
        except (ValueError,TypeError): pass
def normalize_digits(text): return text.translate(_DIGIT_TABLE) if text else text

def _fold_accents(text):
    return "".join(c for c in unicodedata.normalize("NFKD",text) if not unicodedata.combining(c))
def _strip_schwa(s):
    return s[:-1] if len(s)>3 and s.endswith("a") else s
@lru_cache(maxsize=200_000)
def _translit_token(tok):
    scr=detect_script(tok)
    if scr=="latin": return _fold_accents(tok)
    scheme=_SANSCRIPT_SCHEME.get(scr)
    if scheme is not None:
        try: return _strip_schwa(_fold_accents(sanscript.transliterate(tok,scheme,sanscript.IAST)))
        except Exception: pass
    return anyascii(tok)
def transliterate(text):
    if not text: return text
    if text.isascii(): return text
    return " ".join(t if t.isascii() else _translit_token(t) for t in text.split())

_SUFFIX={"pvt","private","ltd","limited","llp","inc","incorporated","corp","corporation","co",
    "company","plc","gmbh","sarl","sa","sas","srl","bv","ag","kg","pte"}
_WEB={"www","http","https","com","net","org","io","biz","info"}
_STREET={"street":"st","st":"st","road":"rd","rd":"rd","avenue":"ave","ave":"ave","av":"ave",
    "lane":"ln","ln":"ln","boulevard":"blvd","blvd":"blvd","drive":"dr","dr":"dr","nagar":"nagar",
    "marg":"marg","phase":"phase","sector":"sector","block":"block","floor":"fl","flr":"fl",
    "fl":"fl","building":"bldg","bldg":"bldg","opposite":"opp","opp":"opp","near":"near"}
_PUNCT_RE=re.compile(r"[^a-z0-9 ]+"); _WS_RE=re.compile(r"\s+")
def _clean_latin(text):
    text=text.casefold(); text=_PUNCT_RE.sub(" ",text)
    return _WS_RE.sub(" ",text).strip()
def _dedup(seq):
    seen=set(); out=[]
    for x in seq:
        if x not in seen: seen.add(x); out.append(x)
    return out

_RE_6=re.compile(r"\b\d{6}\b"); _RE_5=re.compile(r"\b\d{5}\b"); _RE_46=re.compile(r"\b\d{4,6}\b")
_RE_3_3=re.compile(r"\b(\d{3})\s+(\d{3})\b"); _RE_HOUSE=re.compile(r"\b\d{1,4}\b")
def extract_postal_house(addr, country=""):
    c=(country or "").strip().upper(); postal=""
    if c.startswith("IN"):
        six=_RE_6.findall(addr)
        if six: postal=six[-1]
        else:
            sp=_RE_3_3.findall(addr)
            if sp: postal=sp[-1][0]+sp[-1][1]
    elif c.startswith("US") or c.startswith("FR"):
        f=_RE_5.findall(addr)
        if f: postal=f[-1]
    else:
        g=_RE_46.findall(addr)
        if g: postal=g[-1]
    house=""
    for r in _RE_HOUSE.findall(addr):
        if r!=postal: house=r; break
    return postal, house

def process_name(raw):
    scr=detect_script(raw); s=normalize_digits(raw).replace("&"," and ")
    clean=_clean_latin(transliterate(s))
    toks=_dedup([t for t in clean.split() if t not in _SUFFIX and t not in _WEB and len(t)>1])
    block=" ".join(toks); phon=nysiis(re.sub(r"[^a-z]","",block)) if block else ""
    return scr,clean,block,toks,phon
def process_addr(raw, country=""):
    scr=detect_script(raw); digits=normalize_digits(raw)
    postal,house=extract_postal_house(digits,country)
    clean=_clean_latin(transliterate(digits.replace("&"," and ")))
    toks=_dedup([_STREET.get(t,t) for t in clean.split() if len(t)>1])
    return scr,clean," ".join(toks),toks,postal,house

SRC_COLS=["entity_id","business_name","business_address","country"]
OUT_SCHEMA=["entity_id","country","business_name","business_address","name_norm","addr_norm",
    "block_name","block_addr","name_tokens","addr_tokens","postal_code","house_number",
    "name_phonetic","script_name","script_addr"]
def _process_chunk(df):
    df=df.reindex(columns=SRC_COLS,fill_value=""); rows={c:[] for c in OUT_SCHEMA}
    for eid,nm,ad,ctry in zip(df["entity_id"],df["business_name"],df["business_address"],df["country"]):
        n_scr,n_norm,n_block,n_toks,n_phon=process_name(nm or "")
        a_scr,a_norm,a_block,a_toks,postal,house=process_addr(ad or "",ctry or "")
        rows["entity_id"].append(eid); rows["country"].append((ctry or "").strip())
        rows["business_name"].append(nm or ""); rows["business_address"].append(ad or "")
        rows["name_norm"].append(n_norm); rows["addr_norm"].append(a_norm)
        rows["block_name"].append(n_block); rows["block_addr"].append(a_block)
        rows["name_tokens"].append(n_toks); rows["addr_tokens"].append(a_toks)
        rows["postal_code"].append(postal); rows["house_number"].append(house)
        rows["name_phonetic"].append(n_phon); rows["script_name"].append(n_scr); rows["script_addr"].append(a_scr)
    return pd.DataFrame(rows,columns=OUT_SCHEMA)
def preprocess_file(in_path,out_path,chunksize=200_000,limit=None,verbose=True):
    reader=pd.read_csv(in_path,sep="\t",dtype=str,keep_default_na=False,na_values=[],
        quoting=csv.QUOTE_NONE,on_bad_lines="warn",engine="c",chunksize=chunksize)
    writer=None; total=0; fb=[]
    for i,chunk in enumerate(reader):
        out_df=_process_chunk(chunk); total+=len(out_df)
        if HAVE_ARROW:
            table=pa.Table.from_pandas(out_df,preserve_index=False)
            if writer is None: writer=pq.ParquetWriter(out_path,table.schema)
            writer.write_table(table)
        else: fb.append(out_df)
        if verbose: print(f"  chunk {i}: {total:,} rows")
        if limit and total>=limit: break
    if HAVE_ARROW:
        if writer is not None: writer.close()
    else:
        big=pd.concat(fb,ignore_index=True)
        try: big.to_parquet(out_path)
        except Exception:
            tsv=out_path.rsplit(".",1)[0]+".tsv"; big.to_csv(tsv,sep="\t",index=False); out_path=tsv
    if verbose: print(f"done: {total:,} rows -> {out_path}")
    return out_path
def parquet_to_tsv(parquet_path, tsv_path=None):
    if tsv_path is None: tsv_path=parquet_path.rsplit(".",1)[0]+".tsv"
    df=pd.read_parquet(parquet_path)
    for col in ("name_tokens","addr_tokens"):
        if col in df.columns:
            df[col]=df[col].apply(lambda xs: "|".join(xs) if xs is not None else "")
    df.to_csv(tsv_path,sep="\t",index=False); print(f"wrote {len(df):,} rows -> {tsv_path}"); return tsv_path
