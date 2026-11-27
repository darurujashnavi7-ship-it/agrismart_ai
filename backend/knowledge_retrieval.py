import json
import math
import re
from collections import Counter
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
CHUNKS_PATH=ROOT/"knowledge_base"/"chunks.json"
MANIFEST_PATH=ROOT/"knowledge_base"/"source_manifest.json"
MAX_CONTEXT_CHARACTERS=5000
STOP_WORDS={
    "a","about","after","an","and","are","as","at","be","can","could","do",
    "does","for","from","give","how","i","in","is","it","me","my","of",
    "on","or","please","should","tell","that","the","their","this","to",
    "use","what","when","where","which","who","why","with",
}
QUERY_EXPANSIONS={
    "agromet":{"agricultural","meteorological","advisory"},
    "agriculture":{"agricultural"},
    "agricultural":{"agriculture"},
    "advice":{"advisory","advisories"},
    "advisory":{"advice","advisories"},
    "forecast":{"weather","meteorological"},
    "farmer":{"farming","farmers"},
    "farming":{"farmer","farmers"},
    "imd":{"meteorological","agriculture"},
    "weather":{"forecast","meteorological"},
}
HIGH_SIGNAL_TERMS={
    "agromet","agriculture","agricultural","advisory","advisories","farming",
    "farm","farmer","forecast","imd","meteorological","weather",
}


class KnowledgeBaseError(RuntimeError):
    pass


def load_knowledge_chunks():
    try:
        chunks=json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        manifest=json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as error:
        raise KnowledgeBaseError(
            f"Could not load AgriBot's local knowledge base: {error}"
        ) from error

    if not isinstance(chunks,list) or not chunks:
        raise KnowledgeBaseError("AgriBot's chunks.json must contain a non-empty list.")
    if not isinstance(manifest,list):
        raise KnowledgeBaseError("AgriBot's source_manifest.json must contain a list.")

    sources={}
    for source in manifest:
        if (
            not isinstance(source,dict)
            or not isinstance(source.get("title"),str)
            or not source["title"].strip()
            or not isinstance(source.get("url"),str)
            or not source["url"].startswith(("https://","http://"))
        ):
            raise KnowledgeBaseError("AgriBot's source manifest contains an invalid entry.")
        sources[source["url"]]=source["title"]

    validated=[]
    for chunk in chunks:
        if (
            not isinstance(chunk,dict)
            or not isinstance(chunk.get("title"),str)
            or not isinstance(chunk.get("url"),str)
            or not isinstance(chunk.get("text"),str)
            or not chunk["text"].strip()
            or chunk["url"] not in sources
        ):
            raise KnowledgeBaseError("AgriBot's chunks.json contains an invalid or unsourced chunk.")
        validated.append({
            "title":sources[chunk["url"]],
            "url":chunk["url"],
            "text":chunk["text"].strip(),
        })
    return validated


def _normalize_token(token):
    if len(token)>5 and token.endswith("ies"):
        return token[:-3]+"y"
    if len(token)>6 and token.endswith("ing"):
        return token[:-3]
    if len(token)>5 and token.endswith("ers"):
        return token[:-3]
    if len(token)>4 and token.endswith("s"):
        return token[:-1]
    return token


def _tokenize(text):
    return [
        _normalize_token(token)
        for token in re.findall(r"[a-z0-9]+",text.casefold())
        if token not in STOP_WORDS and len(token)>1
    ]


def _expand_query_terms(query_tokens):
    expanded=set(query_tokens)
    for token in query_tokens:
        expanded.update(
            _normalize_token(expansion)
            for expansion in QUERY_EXPANSIONS.get(token,())
        )
    return expanded


def retrieve_knowledge(question,limit=3):
    if not isinstance(question,str) or not question.strip():
        return []
    chunks=load_knowledge_chunks()
    query_tokens=set(_tokenize(question))
    if not query_tokens:
        return []
    query_terms=_expand_query_terms(query_tokens)
    documents=[
        Counter(_tokenize(chunk["title"]+" "+chunk["text"]))
        for chunk in chunks
    ]
    document_frequencies=Counter(
        term
        for document in documents
        for term in document
    )
    ranked=[]
    for index,(chunk,document) in enumerate(zip(chunks,documents)):
        matches=query_terms.intersection(document)
        if not matches:
            continue
        if len(matches)==1 and not matches.intersection(HIGH_SIGNAL_TERMS):
            continue
        score=sum(
            math.log(1+(len(documents)-document_frequencies[term]+0.5)/
                     (document_frequencies[term]+0.5))
            *(1+math.log(document[term]))
            for term in matches
        )
        ranked.append((score,index,chunk))
    ranked.sort(key=lambda result:(-result[0],result[1]))
    return [chunk for _,_,chunk in ranked[:max(1,limit)]]


def format_knowledge_context(chunks,max_characters=MAX_CONTEXT_CHARACTERS):
    passages=[]
    remaining=max_characters
    for chunk in chunks:
        header=f"Source: {chunk['title']} ({chunk['url']})\n"
        available=remaining-len(header)
        if available<=0:
            break
        passage=chunk["text"][:available].strip()
        passages.append(header+passage)
        remaining-=len(header)+len(passage)+2
    return "\n\n".join(passages)


def format_knowledge_fallback(chunks):
    context=format_knowledge_context(chunks)
    return (
        "Relevant excerpts from AgriBot's knowledge base:\n\n"+
        context+
        "\n\nThese excerpts may not cover every part of your question."
    )
