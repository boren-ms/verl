"""Portuguese orthographic modernizer.

Normalizes pre-1911 (etymological) and pre-Acordo-Ortográfico-de-1990
spellings to modern post-Acordo Portuguese.  Designed to reduce spurious
WER differences when references come from 19th-century public-domain
sources (e.g. MLS / LibriVox) while hypotheses are produced by modern
models that emit post-reform orthography (or vice versa).

The transformation is purely lexical and applied per whitespace-separated
token, after the OpenASR ``BasicMultilingualTextNormalizer`` pipeline has
lowercased / stripped punctuation / kept diacritics.

The function is intentionally conservative:

* Regex rules cover digraph and double-consonant patterns that do **not**
  exist in modern Portuguese (``ph``, ``th``, ``rh``, ``ll``, ``mm``,
  ``nn``, ``pp``, ``tt``, ``ff``, ``cc``, ``dd``, ``gg``, ``bb``).
* An explicit lexicon handles 1911 reform (``ch``/``y``/silent ``h``) and
  1990 reform (silent ``c``/``p`` before another consonant) cases that
  would be unsafe to express as a generic rule.
* Accent-only differences that the Acordo removed (``idéia``->``ideia``,
  ``vôo``->``voo``, differential ``pêlo``->``pelo``, adverb ``sómente``->
  ``somente``) are also covered by the lexicon.
"""

from __future__ import annotations

import re
from functools import lru_cache


# ---------------------------------------------------------------------------
# Lexicon: pre-reform -> modern.  Keys are lower-cased, accent-preserving.
# ---------------------------------------------------------------------------

# Acordo Ortográfico de 1990 (PT-PT silent c/p drops + accent removals).
_ACORDO_1990 = {
    # cc / ct / pç / pt where the first consonant is silent in PT-PT
    "acção": "ação",
    "acções": "ações",
    "acto": "ato",
    "actos": "atos",
    "actor": "ator",
    "actores": "atores",
    "actriz": "atriz",
    "actrizes": "atrizes",
    "activo": "ativo",
    "activos": "ativos",
    "activa": "ativa",
    "activas": "ativas",
    "actividade": "atividade",
    "actividades": "atividades",
    "actual": "atual",
    "actuais": "atuais",
    "actualmente": "atualmente",
    "actuar": "atuar",
    "actuou": "atuou",
    "actuação": "atuação",
    "afectar": "afetar",
    "afecto": "afeto",
    "afectos": "afetos",
    "afectivo": "afetivo",
    "arquitecto": "arquiteto",
    "arquitectos": "arquitetos",
    "arquitectura": "arquitetura",
    "colectivo": "coletivo",
    "colectivos": "coletivos",
    "colecção": "coleção",
    "colecções": "coleções",
    "correcto": "correto",
    "correcta": "correta",
    "correctos": "corretos",
    "correctas": "corretas",
    "correctamente": "corretamente",
    "contacto": "contato",
    "contactos": "contatos",
    "contactar": "contatar",
    "convicção": "convicção",  # double consonant kept (still pronounced)
    "detectar": "detetar",
    "detective": "detetive",
    "detectives": "detetives",
    "directo": "direto",
    "directa": "direta",
    "directos": "diretos",
    "directas": "diretas",
    "directamente": "diretamente",
    "director": "diretor",
    "directores": "diretores",
    "directora": "diretora",
    "directoras": "diretoras",
    "direcção": "direção",
    "direcções": "direções",
    "eficaz": "eficaz",
    "electricidade": "eletricidade",
    "electrico": "elétrico",
    "electricos": "elétricos",
    "eléctrico": "elétrico",
    "eléctricos": "elétricos",
    "eléctrica": "elétrica",
    "eléctricas": "elétricas",
    "electrónica": "eletrónica",
    "electrónico": "eletrónico",
    "espectáculo": "espetáculo",
    "espectáculos": "espetáculos",
    "exacto": "exato",
    "exacta": "exata",
    "exactos": "exatos",
    "exactas": "exatas",
    "exactamente": "exatamente",
    "excepção": "exceção",
    "excepções": "exceções",
    "excepcional": "excecional",
    "facto": "fato",
    "factos": "fatos",
    "fracção": "fração",
    "fracções": "frações",
    "infeccioso": "infecioso",
    "infecções": "infeções",
    "infecção": "infeção",
    "lectivo": "letivo",
    "lectiva": "letiva",
    "leccionar": "lecionar",
    "leccionado": "lecionado",
    "objectivo": "objetivo",
    "objectivos": "objetivos",
    "objectiva": "objetiva",
    "objectivas": "objetivas",
    "objecto": "objeto",
    "objectos": "objetos",
    "óptimo": "ótimo",
    "óptimos": "ótimos",
    "óptima": "ótima",
    "óptimas": "ótimas",
    "optimismo": "otimismo",
    "optimista": "otimista",
    "optimistas": "otimistas",
    "perfectivo": "perfetivo",
    "perspectiva": "perspetiva",
    "perspectivas": "perspetivas",
    "projecto": "projeto",
    "projectos": "projetos",
    "protecção": "proteção",
    "protecções": "proteções",
    "reflectir": "refletir",
    "reflectido": "refletido",
    "reflector": "refletor",
    "respectivo": "respetivo",
    "respectiva": "respetiva",
    "respectivamente": "respetivamente",
    "selecção": "seleção",
    "selecções": "seleções",
    "seleccionar": "selecionar",
    "subjectivo": "subjetivo",
    "subjectiva": "subjetiva",
    "sumptuoso": "suntuoso",
    "sumptuária": "suntuária",
    "adopção": "adoção",
    "adopções": "adoções",
    "adoptar": "adotar",
    "adoptado": "adotado",
    "adoptada": "adotada",
    "adopta": "adota",
    "baptismo": "batismo",
    "baptizar": "batizar",
    "baptizado": "batizado",
    "egipto": "egito",
    "egípcio": "egípcio",
    # accent-removal cases
    "idéia": "ideia",
    "idéias": "ideias",
    "geléia": "geleia",
    "geléias": "geleias",
    "vôo": "voo",
    "vôos": "voos",
    "enjôo": "enjoo",
    "enjôos": "enjoos",
    "abençôo": "abençoo",
    "côa": "coa",
    "côas": "coas",
    "pêlo": "pelo",
    "pêlos": "pelos",  # differential accent (Acordo 1990)
    "pélo": "pelo",  # archaic
    "pára": "para",
    "pólo": "polo",
    "pólos": "polos",
    "péla": "pela",  # differential
    "sómente": "somente",  # 1971 reform removed sólo-class accents
    "fôra": "fora",  # differential (often kept; safe to collapse for WER)
    "fôrma": "forma",  # differential
}

# Pre-1911 etymological forms (the bulk of MLS Portuguese references).
_PRE_1911 = {
    # silent / etymological "h"
    "hontem": "ontem",
    "hum": "um",
    "huma": "uma",
    "huns": "uns",
    "humas": "umas",
    "hi": "i",
    # "ch" with /k/ value (Greek origin)
    "christão": "cristão",
    "christãos": "cristãos",
    "christã": "cristã",
    "christãs": "cristãs",
    "christo": "cristo",
    "christianismo": "cristianismo",
    "chronica": "crônica",
    "chronicas": "crônicas",
    "chronista": "cronista",
    "chronologia": "cronologia",
    "echo": "eco",
    "echos": "ecos",
    "monarchia": "monarquia",
    "monarchias": "monarquias",
    "anarchia": "anarquia",
    "patriarcha": "patriarca",
    # "sc" (Latin scientia) -> "c" in some cases
    "sciencia": "ciência",
    "sciencias": "ciências",
    "scientifico": "científico",
    # "y" used between consonants (mostly handled by regex rule below;
    # explicit entries help proper nouns and accented derivatives)
    "estylo": "estilo",
    "estylos": "estilos",
    "rhythmo": "ritmo",
    "rhythmos": "ritmos",
    # "ph" examples (regex also catches these; lexicon preserves accents)
    "pharmácia": "farmácia",
    "pharmacia": "farmácia",
    "philosóphico": "filosófico",
    "philosophía": "filosofia",
    # explicit double-consonant words common in the MLS corpus
    "cabellos": "cabelos",
    "cabello": "cabelo",
    "belleza": "beleza",
    "bellezas": "belezas",
    "bello": "belo",
    "bella": "bela",
    "bellos": "belos",
    "bellas": "belas",
    "elle": "ele",
    "elles": "eles",
    "ella": "ela",
    "ellas": "elas",
    "aquelle": "aquele",
    "aquelles": "aqueles",
    "aquella": "aquela",
    "aquellas": "aquelas",
    "aquillo": "aquilo",
    "collo": "colo",
    "collos": "colos",
    "anno": "ano",
    "annos": "anos",
    "annaes": "anais",
    "annual": "anual",
    "somma": "soma",
    "sommas": "somas",
    "summa": "suma",
    "commum": "comum",
    "communs": "comuns",
    "communicar": "comunicar",
    "commercio": "comércio",
    "commércio": "comércio",
    "connosco": "conosco",  # PT-PT modern is "connosco", BR is "conosco"; pick BR
    "innumero": "inúmero",
    "innumeros": "inúmeros",
    "diffícil": "difícil",
    "diffíceis": "difíceis",
    "difficil": "difícil",
    "official": "oficial",
    "officiaes": "oficiais",
    "officio": "ofício",
    "affecto": "afeto",
    "affectar": "afetar",
    "attenção": "atenção",
    "attento": "atento",
    "attentos": "atentos",
    "attingir": "atingir",
    "occasião": "ocasião",
    "occasiões": "ocasiões",
    "occorrer": "ocorrer",
    "occorreu": "ocorreu",
    "ennobrecer": "enobrecer",
    "assignar": "assinar",
    "assignado": "assinado",
    "assignatura": "assinatura",
    # archaic plurals / endings
    "taes": "tais",
    "quaes": "quais",
    "mortaes": "mortais",
    "espirituaes": "espirituais",
    "ideaes": "ideais",
    # other commonly seen
    "recto": "reto",
    "rectos": "retos",
    "recta": "reta",
    "rectas": "retas",
    "fructo": "fruto",
    "fructos": "frutos",
    "fructa": "fruta",
    "instructo": "instruto",
    "instructor": "instrutor",
    "instrucção": "instrução",
    "destructo": "destruto",
    "destrucção": "destruição",
    "vencido": "vencido",
    # proper-name-ish (lowercase forms after normalizer)
    "raphael": "rafael",
    "philippe": "filipe",
    "phelipe": "filipe",
    "stephania": "estefânia",
    "theresa": "teresa",
    "theodoro": "teodoro",
    "phenomeno": "fenômeno",
    "phenomenos": "fenômenos",
    "physica": "física",
    "physico": "físico",
}


_LEXICON: dict[str, str] = {}
_LEXICON.update(_ACORDO_1990)
_LEXICON.update(_PRE_1911)


# ---------------------------------------------------------------------------
# Generic regex rules: applied per-token, in order.  Each rule is safe in the
# sense that the resulting pattern does not occur in modern Portuguese
# vocabulary (so applying it to an already-modern token is a no-op).
# ---------------------------------------------------------------------------

# (pattern, replacement) pairs.  Patterns are compiled lazily.
_RULES: list[tuple[re.Pattern[str], str]] = [
    # Greek-origin digraphs (1911 reform)
    (re.compile(r"ph"), "f"),
    (re.compile(r"th"), "t"),
    (re.compile(r"rh"), "r"),
    # "y" between consonants or word-internal -> "i" (1911).  Preserves
    # word-initial "y" (foreign words like "yoga") and "ay/oy/ey" diphthongs.
    (re.compile(r"(?<=[bcdfghjklmnpqrstvwxz])y(?=[bcdfghjklmnpqrstvwxz])"), "i"),
    # double consonants that do not occur in modern Portuguese
    (re.compile(r"ll"), "l"),
    (re.compile(r"mm"), "m"),
    (re.compile(r"nn"), "n"),
    (re.compile(r"pp"), "p"),
    (re.compile(r"tt"), "t"),
    (re.compile(r"ff"), "f"),
    (re.compile(r"cc(?=[aouáâãóôõú]|$)"), "c"),  # cc + back vowel/end (keep cc+e/i)
    (re.compile(r"dd"), "d"),
    (re.compile(r"gg"), "g"),
    (re.compile(r"bb"), "b"),
]


@lru_cache(maxsize=131072)
def modernize_token(token: str) -> str:
    """Return the modern Portuguese form of ``token`` (already lowercased)."""

    if not token:
        return token

    # 1. lexicon overrides (preserve accents, special cases)
    if token in _LEXICON:
        return _LEXICON[token]

    # 2. generic regex rules
    out = token
    for pattern, repl in _RULES:
        out = pattern.sub(repl, out)
    return out


def modernize_text(text: str) -> str:
    """Apply :func:`modernize_token` to every whitespace-separated token."""

    if not text:
        return text
    return " ".join(modernize_token(t) for t in text.split())
