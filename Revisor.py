# -*- coding: utf-8 -*-
# revisor_autor.py
# Streamlit: Revisão de arquivos do autor (métricas + IA opcional)
#
# Inclui:
# - Regras de páginas (2+) para "Vamos Começar" e "Siga em Frente" ignorando imagens/legendas
# - Ponto de Partida ↔ Vamos Exercitar (cenário e retomada)
# - Heurísticas de títulos por tema e referências ABNT-like
# - IA opcional com prompt estruturado
# - Confiabilidade: self-test de API, timeout maior, retry com backoff e fallback p/ Chat

import os
import re
import io
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd

# Parsing
from docx import Document as DocxDocument
import pdfplumber
from pypdf import PdfReader

# env
from dotenv import load_dotenv
load_dotenv()

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass


def load_api_key() -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None


# IA opcional
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ===================== Config =====================
st.set_page_config(page_title="Revisão de Conteúdos (Autor)", layout="wide")
st.title(" 🕵️‍♂️ Revisão de Arquivos - Aula texto")

# ===================== Helpers =====================
def load_api_key() -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["OPENAI_API_KEY"]  # não explode se não existir
    except Exception:
        return None

def quick_api_selftest() -> Tuple[bool, str]:
    key = load_api_key()
    if not key or OpenAI is None:
        return False, "Sem OPENAI_API_KEY (ou SDK indisponível)."
    try:
        client = OpenAI(api_key=key, timeout=40)
        _ = client.models.list()  # chamada leve
        return True, "OK"
    except Exception as e:
        return False, f"{e.__class__.__name__}: {e}"

def _responses_call_with_retry(client, prompt: str, retries: int = 4, base_sleep: float = 1.5):
    last_err = None
    for i in range(retries):
        try:
            return client.responses.create(
                model="gpt-4o-mini",
                input=prompt,
                temperature=0.2,
            )
        except Exception as e:
            last_err = e
            if e.__class__.__name__ not in {"APIConnectionError", "APITimeoutError"}:
                break
            time.sleep(base_sleep * (2 ** i))
    raise last_err

def sanitize_text(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s, flags=re.UNICODE))

def approx_pages_from_words(n_words: int, wpp: int = 350) -> float:
    return n_words / float(wpp or 350)

def contains_theme(text: str, theme: str) -> bool:
    if not theme:
        return True
    pat = re.escape(theme).replace(r"\ ", r"[\s\-]+")
    return bool(re.search(pat, text, flags=re.IGNORECASE))

def has_tag_in_videoaula(s: str) -> bool:
    return bool(re.search(r"(?:\bTAG\b|TAG\s*:|\[TAG:|#tag)", s, flags=re.IGNORECASE))

# ABNT-like (heurística leve)
def abnt_like_line(line: str) -> bool:
    l = sanitize_text(line)
    has_author = bool(re.search(r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ\-]{2,},\s*[A-Za-zÀ-ÿ'\-]+", l))
    has_year = bool(re.search(r"\b(19|20)\d{2}\b", l))
    has_vehicle = bool(re.search(r"(Disponível em:|http[s]?://|DOI:)", l, flags=re.IGNORECASE))
    has_access = bool(re.search(r"Acesso\s+em", l, flags=re.IGNORECASE))
    return has_author and has_year and (has_vehicle or has_access)

def link_requires_acesso_em(line: str) -> bool:
    l = sanitize_text(line)
    has_link = bool(re.search(r"http[s]?://|DOI:", l, flags=re.IGNORECASE))
    has_access = bool(re.search(r"Acesso\s+em", l, flags=re.IGNORECASE))
    return (not has_link) or (has_link and has_access)

def count_questions(s: str) -> int:
    return s.count("?")

# ===================== Remoção de imagens/legendas =====================
_IMG_PAT = re.compile(
    r"^(?:Figura|Imagem|Ilustração|Gráfico|Tabela)\s*\d+.*$|"
    r"!\[.*?\]\(.*?\)|"
    r"<img\b[^>]*>|<figure\b[^>]*>.*?</figure>|<figcaption\b[^>]*>.*?</figcaption>|"
    r"\b\w+\.(?:png|jpe?g|gif|svg)\b",
    flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
)

def text_without_images(text: str) -> Tuple[str, List[str]]:
    lines = text.splitlines()
    kept, ignored = [], []
    for ln in lines:
        if _IMG_PAT.search(ln.strip()):
            if len(ignored) < 3:
                ignored.append(ln.strip()[:200])
            continue
        kept.append(ln)
    return ("\n".join(kept).strip(), ignored)

# ===================== Seções =====================
SECTION_NAMES = {
    "videoaula": r"(?:^|\n)\s*Texto\s*:\s*V[ií]deo(?:\s|-)?aula\b",
    "ponto": r"(?:^|\n)\s*Ponto\s+de\s+Partida\b",
    "comecar": r"(?:^|\n)\s*Vamos\s+Começar\b",
    "siga": r"(?:^|\n)\s*Siga\s+em\s+Frente\b",
    "exercitar": r"(?:^|\n)\s*Vamos\s+Exercitar\b",
    "saibamais": r"(?:^|\n)\s*Saiba\s+Mais\b",
    "referencias": r"(?:^|\n)\s*Refer[eê]ncias\b",
}

def split_sections(text: str) -> Dict[str, str]:
    title_patterns = "|".join(SECTION_NAMES.values())
    matches = list(re.finditer(title_patterns, text, flags=re.IGNORECASE | re.MULTILINE | re.UNICODE))
    result = {k: "" for k in SECTION_NAMES.keys()}
    if not matches:
        return result

    spans = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        block = text[start:end]
        spans.append(block)

    for block in spans:
        for key, pat in SECTION_NAMES.items():
            if re.search(pat, block, flags=re.IGNORECASE):
                body = re.sub(pat, "", block, flags=re.IGNORECASE).strip()
                result[key] = body
                break
    return result

# ===================== Parsing =====================
def extract_text_from_docx(f: io.BytesIO) -> str:
    doc = DocxDocument(f)
    texts = [p.text for p in doc.paragraphs]
    return sanitize_text("\n".join(texts))

def extract_text_from_pdf(f: io.BytesIO) -> str:
    try:
        with pdfplumber.open(f) as pdf:
            pages_text = [p.extract_text() or "" for p in pdf.pages]
        return sanitize_text("\n".join(pages_text))
    except Exception:
        f.seek(0)
        reader = PdfReader(f)
        pages_text = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                pages_text.append("")
        return sanitize_text("\n".join(pages_text))

# ===================== IA: prompt estruturado =====================
PROMPT_TEMPLATE = """
Você é um revisor pedagógico e técnico. Analise as seções abaixo de um conteúdo didático e verifique os requisitos.

================= CONTEXTO =================
Tema da aula: {TEMA}

Assuntos granulares (lista curta, termos-chave):
{ASSUNTOS}

Parâmetros:
- Palavras por página (heurística): {WPP}
- Idioma do relatório: pt-BR

================= TEXTO DO ARQUIVO =================
[Texto: Vídeoaula]
{SEC_VIDEAULA}

[Ponto de Partida]
{SEC_PONTO}

[Vamos Começar]
{SEC_COMECAR}

[Siga em Frente]
{SEC_SIGA}

[Vamos Exercitar]
{SEC_EXERCITAR}

[Saiba Mais]
{SEC_SAIBAMAIS}

[Referências]
{SEC_REFERENCIAS}

================= REGRAS DE AVALIAÇÃO =================
A) Páginas em “Vamos Começar” e “Siga em Frente”
- Conte apenas texto, desconsidere imagens/legendas. Considere imagem quando:
  * linha com “Figura/Imagem/Ilustração/Gráfico/Tabela” + número/legenda;
  * markdown ![...](...);
  * tags HTML <img>, <figure>, <figcaption>;
  * nomes com .png .jpg .jpeg .gif .svg.
- Remova também legendas/chamadas curtas de figura.
- Depois de remover, conte palavras do texto restante e estime páginas:
  páginas ≈ palavras_sem_imagem / {WPP}.
- Critério: pelo menos 2,0 páginas (≈ ≥ 700 palavras) **em cada** seção.

B) Ponto de Partida – correlação e cenário
- Deve conter correlação explícita entre os assuntos granulares.
- Deve conter um cenário a ser retomado no “Vamos Exercitar”.
- Extraia:
  * resumo do cenário (2–3 frases),
  * lista de relações entre assuntos (pares X→Y e como se relacionam).

C) Vamos Exercitar – retomar o cenário
- Verifique se as atividades retomam/endereçam o mesmo cenário do Ponto de Partida.
- Valide por sobreposição semântica: entidades/locais/atores/termos técnicos do cenário reaparecem? 

D) Títulos com tema (checagem leve)
- “Vamos Começar”, “Siga em Frente” e “Vamos Exercitar” devem estar alinhados ao tema
  (uma ocorrência do tema ou variação próxima no primeiro parágrafo/título).
- Não reprovar somente por isso; apenas aponte.

================= SAÍDA JSON (OBRIGATÓRIA) =================
Devolva apenas um JSON:

{{
  "paginas": {{
    "comecar": {{
      "palavras_sem_imagem": <int>,
      "paginas_est": <float>,
      "ok_duas_paginas": <true|false>,
      "amostras_ignoradas": ["..."]
    }},
    "siga_em_frente": {{
      "palavras_sem_imagem": <int>,
      "paginas_est": <float>,
      "ok_duas_paginas": <true|false>,
      "amostras_ignoradas": ["..."]
    }}
  }},
  "ponto_de_partida": {{
    "tem_correlacao_assuntos": <true|false>,
    "relacoes": [
      {{"de":"<assunto_A>","para":"<assunto_B>","como":"<descrição>"}}
    ],
    "tem_cenario": <true|false>,
    "cenario_resumo": "<2-3 frases>"
  }},
  "vamos_exercitar": {{
    "retoma_cenario": <true|false>,
    "entidades_cenario": ["..."],
    "entidades_exercitar": ["..."],
    "sobreposicao_scores": {{
      "jaccard_termos": <float 0-1>,
      "avaliacao_semantica_geral": "<alta|média|baixa>"
    }},
    "exemplos_questoes_relacionadas": ["..."]
  }},
  "alinhamento_titulos_com_tema": {{
    "comecar": "<alinhado|não claramente>",
    "siga_em_frente": "<alinhado|não claramente>",
    "vamos_exercitar": "<alinhado|não claramente>"
  }},
  "conclusao": {{
    "status_geral": "<OK|AJUSTAR>",
    "motivos": ["..."],
    "acoes_recomendadas": ["..."]
  }},
  "citações": {{
    "ponto_de_partida": ["..."],
    "vamos_exercitar": ["..."]
  }}
}}

Regras:
- Apenas JSON.
- Use floats com 2 casas em "paginas_est" e "jaccard_termos".
- Não invente dados: se faltar, use false/"" e explique nos motivos/ações.
"""

def _build_prompt(tema: str, assuntos: List[str], wpp: int, secs: Dict[str, str]) -> str:
    def block(name: str) -> str:
        return secs.get(name, "") or ""
    return PROMPT_TEMPLATE.format(
        TEMA=tema,
        ASSUNTOS=json.dumps(assuntos, ensure_ascii=False),
        WPP=wpp,
        SEC_VIDEAULA=block("videoaula")[:12000],
        SEC_PONTO=block("ponto")[:12000],
        SEC_COMECAR=block("comecar")[:12000],
        SEC_SIGA=block("siga")[:12000],
        SEC_EXERCITAR=block("exercitar")[:12000],
        SEC_SAIBAMAIS=block("saibamais")[:12000],
        SEC_REFERENCIAS=block("referencias")[:12000],
    )

def call_ia_sections(tema: str, assuntos: List[str], wpp: int, secs: Dict[str, str]) -> Optional[dict]:
    api_key = load_api_key()
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key, timeout=60)

    prompt = _build_prompt(tema, assuntos, wpp, secs)

    # 1) tenta Responses com retry
    try:
        resp = _responses_call_with_retry(client, prompt)
        text = resp.output_text
    except Exception as e1:
        # 2) fallback: Chat Completions
        try:
            chat = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você é um revisor pedagógico. Responda apenas JSON válido."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                timeout=60,
            )
            text = chat.choices[0].message.content or ""
        except Exception:
            return None

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

# ===================== Inserção (não-invasiva) dos PROMPTS FIXOS + helper JSON =====================
# Este bloco adiciona prompts fixos e a função call_fixed_prompt sem alterar
# as funcionalidades existentes. Use onde preferir no fluxo UI para checagens
# específicas solicitadas pelo usuário.

# Extrator robusto de JSON em string (pega o bloco JSON mais provável)
def extract_json_from_text(text: str) -> Optional[dict]:
    """
    Tenta localizar o bloco JSON final no texto retornado por um LLM.
    Retorna dict ou None.
    """
    if not text:
        return None
    # procurar por últimos '{' e tentar decodificar blocos candidatos
    opens = [i for i, ch in enumerate(text) if ch == '{']
    if not opens:
        return None
    for start in reversed(opens):
        candidate = text[start:]
        end_idx = candidate.rfind('}')
        if end_idx == -1:
            continue
        jtxt = candidate[:end_idx+1]
        try:
            return json.loads(jtxt)
        except json.JSONDecodeError:
            # tentar pequenas correções/ignorar e continuar
            continue
    return None

# Prompts fixos solicitados (em pt-BR) — retorno em JSON esperado
PROMPT_FIXOS = {
    "ponto_exercitar": """
RESPOSTA EM JSON:
Pergunta: O Ponto de Partida sugere um cenário que seja resolvido no bloco 'Vamos Exercitar'?
Instruções: Analise o texto fornecido nas seções [Ponto de Partida] e [Vamos Exercitar].
Retorne JSON com as chaves:
- ok: true|false (se o cenário do Ponto de Partida é claramente retomado em Vamos Exercitar)
- evidence: string curta com trechos/indicação das seções (ex.: "[Ponto de Partida] menciona X; [Vamos Exercitar] ...")
- acoes_recomendadas: array de strings (o que ajustar se não retomar)
Exemplo de retorno:
{"ok": false, "evidence": "...", "acoes_recomendadas": ["..."]}
""",
    "vc_encadeia": """
RESPOSTA EM JSON:
Pergunta: A sessão "Vamos Começar" e "Siga em Frente" encadeia e aborda os conteúdos granulares?
Nota: Se os 'assuntos granulares' não foram fornecidos, solicite ao usuário a inserção desses assuntos.
Instruções: Analise [Vamos Começar] e [Siga em Frente] em relação à lista de assuntos granulares.
Retorne JSON:
- ok: true|false (se há encadeamento e cobertura dos assuntos)
- evidence: breve explicação / exemplos de sobreposição
- solicitar_assuntos: true|false (true se os assuntos não foram fornecidos e o modelo está solicitando que o usuário os insira)
- acoes_recomendadas: []
""",
    "exercitar_respond": """
RESPOSTA EM JSON:
Pergunta: O 'Vamos Exercitar' responde o que foi proposto em 'Ponto de Partida'?
Instruções: Verifique correspondência entre entidades do cenário e as atividades/questões do 'Vamos Exercitar'.
Retorne JSON:
- ok: true|false
- evidence: texto curto apontando provas (ex.: termos/atores/locais)
- acoes_recomendadas: lista de ajustes (se falso)
- exemplos_questoes_faltantes: lista (se aplicável)
""",
    "saiba_mais_completa": """
RESPOSTA EM JSON:
Pergunta: O 'Saiba Mais' entra como uma complementação de estudo ao aluno, trazendo referências?
Instruções: Avalie a seção [Saiba Mais] quanto a:
  - presença de resumo/expansão de estudo (texto)
  - indicação bibliográfica (links, DOI, 'Acesso em', anos)
Retorne JSON:
- ok: true|false
- evidence: breve
- acoes_recomendadas: [...]
- referencias_detectadas: array de strings (linhas de referência detectadas, se houver)
"""
}

def call_fixed_prompt(prompt_key: str, sections: Dict[str, str], tema: str = "",
                      assuntos: List[str] = None, max_chars: int = 12000) -> Optional[dict]:
    """
    Chama o prompt fixo identificado por prompt_key usando a mesma estratégia IA do app:
    - tenta Responses via _responses_call_with_retry
    - fallback para chat completions
    Retorna dict decodificado do JSON de resposta, ou None se falhar.
    """
    if prompt_key not in PROMPT_FIXOS:
        return None
    prompt_base = PROMPT_FIXOS[prompt_key]

    # montar contexto resumido (cortar se muito grande)
    def get_block(k):
        return (sections.get(k, "") or "")[:max_chars]

    contexto = (
        f"Tema: {tema}\n"
        f"Assuntos granulares: {json.dumps(assuntos, ensure_ascii=False) if assuntos else '[]'}\n\n"
        f"[Ponto de Partida]\n{get_block('ponto')}\n\n"
        f"[Vamos Começar]\n{get_block('comecar')}\n\n"
        f"[Siga em Frente]\n{get_block('siga')}\n\n"
        f"[Vamos Exercitar]\n{get_block('exercitar')}\n\n"
        f"[Saiba Mais]\n{get_block('saibamais')}\n\n"
        f"[Referências]\n{get_block('referencias')}\n\n"
    )

    # se assuntos vazios e prompt pede solicitação, anotamos para o prompt
    if not assuntos:
        contexto += "\nObservação: não há 'assuntos granulares' fornecidos. Se a análise depender deles, indique 'solicitar_assuntos': true.\n"

    full_prompt = f"{contexto}\n{prompt_base}\n\nResposta esperada: apenas JSON."

    # preparar cliente IA
    api_key = load_api_key()
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key, timeout=60)

    # 1) tentar Responses com retry
    try:
        resp = _responses_call_with_retry(client, full_prompt)
        # atenção: estrutura pode variar; tentar extrair texto
        text = getattr(resp, "output_text", None) or getattr(resp, "text", None) or str(resp)
    except Exception:
        # fallback para chat completions (como você já usa)
        try:
            chat = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Você é um revisor pedagógico. Responda APENAS com JSON válido."},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.0,
                timeout=60
            )
            text = chat.choices[0].message.content or ""
        except Exception:
            return None

    # extrair JSON do texto retornado
    parsed = extract_json_from_text(text)
    return parsed

# ===================== Métricas locais =====================
@dataclass
class MetricResult:
    item: str
    status: str  # "OK" | "AJUSTAR"
    detalhes: str
    valor_medido: str

def checklist_from_results(results: List[MetricResult]) -> List[str]:
    items = [f"[ ] {r.item} — {r.detalhes}" for r in results if r.status != "OK"]
    return items or ["✅ Sem pendências — tudo OK!"]

def eval_metrics(sections: Dict[str, str], tema: str, wpp: int, min_refs: int,
                 assuntos: List[str]) -> Tuple[List[MetricResult], Optional[dict]]:
    out: List[MetricResult] = []

    # 1) Vídeoaula
    vid = sections.get("videoaula", "")
    vid_wc = word_count(vid)
    vid_ok_words = vid_wc >= 105
    vid_has_tag = has_tag_in_videoaula(vid)
    out.append(MetricResult(
        "Texto: Vídeoaula (>=105 palavras + TAG)",
        "OK" if (vid_ok_words and vid_has_tag) else "AJUSTAR",
        f"Palavras: {vid_wc}; TAG: {'sim' if vid_has_tag else 'não'}",
        f"{vid_wc} palavras / TAG: {vid_has_tag}"
    ))

    # 2) Ponto de Partida (~450)
    pp = sections.get("ponto", "")
    pp_wc = word_count(pp)
    pp_ok_range = (400 <= pp_wc <= 520)
    out.append(MetricResult(
        "Ponto de Partida (~450 palavras)",
        "OK" if pp_ok_range else "AJUSTAR",
        f"Palavras: {pp_wc} (faixa aceitável 400–520)",
        f"{pp_wc} palavras"
    ))

    # 3) Vamos Começar (2 páginas, ignorando imagens)
    vc_raw = sections.get("comecar", "")
    vc_noimg, vc_ignored = text_without_images(vc_raw)
    vc_wc = word_count(vc_noimg)
    vc_pages = approx_pages_from_words(vc_wc, wpp=wpp)
    vc_ok = vc_wc >= 700
    vc_title_ok = contains_theme(vc_raw[:160], tema)
    msg_title = "título contém o tema" if vc_title_ok else "título NÃO contém o tema"
    out.append(MetricResult(
        "Vamos Começar (>= 2 páginas, sem imagens/legendas, + título com tema)",
        "OK" if (vc_ok and vc_title_ok) else "AJUSTAR",
        f"Palavras (sem imagem): {vc_wc} (~{vc_pages:.1f} pág). {msg_title}. Ignorados: {len(vc_ignored)}",
        f"{vc_wc} palavras (~{vc_pages:.1f} pág) / título ok: {vc_title_ok}"
    ))

    # 4) Siga em Frente (2 páginas, ignorando imagens)
    sf_raw = sections.get("siga", "")
    sf_noimg, sf_ignored = text_without_images(sf_raw)
    sf_wc = word_count(sf_noimg)
    sf_pages = approx_pages_from_words(sf_wc, wpp=wpp)
    sf_ok = sf_wc >= 700
    sf_title_ok = contains_theme(sf_raw[:160], tema)
    msg_title2 = "título contém o tema" if sf_title_ok else "título NÃO contém o tema"
    out.append(MetricResult(
        "Siga em Frente (>= 2 páginas, sem imagens/legendas, + título com tema)",
        "OK" if (sf_ok and sf_title_ok) else "AJUSTAR",
        f"Palavras (sem imagem): {sf_wc} (~{sf_pages:.1f} pág). {msg_title2}. Ignorados: {len(sf_ignored)}",
        f"{sf_wc} palavras (~{sf_pages:.1f} pág) / título ok: {sf_title_ok}"
    ))

    # 5) Vamos Exercitar (>=600 + ≥3 perguntas + título com tema)
    ve = sections.get("exercitar", "")
    ve_wc = word_count(ve)
    ve_qs = count_questions(ve)
    ve_title_ok = contains_theme(ve[:160], tema)
    ve_ok = (ve_wc >= 600) and (ve_qs >= 3) and ve_title_ok
    out.append(MetricResult(
        "Vamos Exercitar (>=600 palavras + ≥3 perguntas + título com tema)",
        "OK" if ve_ok else "AJUSTAR",
        f"Palavras: {ve_wc}; Perguntas: {ve_qs}; título ok: {'sim' if ve_title_ok else 'não'}",
        f"{ve_wc} palavras / {ve_qs} perguntas / título ok: {ve_title_ok}"
    ))

    # 6) Saiba Mais
    sm = sections.get("saibamais", "")
    sm_wc = word_count(sm)
    sm_has_biblio = bool(re.search(r"(DOI:|http[s]?://|Acesso\s+em|\b(19|20)\d{2}\b)", sm, flags=re.IGNORECASE))
    sm_ok = (sm_wc >= 80) and sm_has_biblio
    out.append(MetricResult(
        "Saiba Mais (expansão + indicação bibliográfica)",
        "OK" if sm_ok else "AJUSTAR",
        f"Palavras: {sm_wc}; Indicadores biblio: {'sim' if sm_has_biblio else 'não'}",
        f"{sm_wc} palavras / biblio: {sm_has_biblio}"
    ))

    # 7) Referências
    ref = sections.get("referencias", "")
    ref_lines = [l for l in ref.splitlines() if l.strip()]
    n_refs = len(ref_lines)
    abnt_ok_lines = sum(1 for l in ref_lines if abnt_like_line(l))
    links_ok = sum(1 for l in ref_lines if link_requires_acesso_em(l))
    refs_ok_minimo = n_refs >= min_refs
    refs_ok_formato = abnt_ok_lines >= max(1, int(0.6 * n_refs)) if n_refs else False
    refs_ok_acesso = (links_ok == n_refs) if n_refs else False
    refs_all_ok = refs_ok_minimo and refs_ok_formato and refs_ok_acesso

    detalhes_ref = (
    f"Total: {n_refs} (mínimo {min_refs}). "
    f"ABNT-like aprox.: {abnt_ok_lines}/{n_refs}. "
    f"'Acesso em' quando há link: {links_ok}/{n_refs} ok."
    )

    out.append(MetricResult(
    "Referências (mínimo, ABNT-like, 'Acesso em' quando link)",
    "OK" if refs_all_ok else "AJUSTAR",
    detalhes_ref,
    f"{n_refs} refs / ABNT-like: {abnt_ok_lines} / Acesso-em: {links_ok}"
    ))

    # 8) IA estruturada
    ia_json = call_ia_sections(tema, assuntos, wpp, sections)
    if ia_json:
        concl = ia_json.get("conclusao", {}).get("status_geral", "AJUSTAR")
        motivos = ia_json.get("conclusao", {}).get("motivos", [])
        out.append(MetricResult(
            "IA: Ponto de Partida (correlação/cenário) e retomada em Vamos Exercitar + checagem páginas",
            "OK" if concl == "OK" else "AJUSTAR",
            " | ".join(motivos) if motivos else "Sem detalhes.",
            f"Conclusão IA: {concl}"
        ))
    else:
        out.append(MetricResult(
            "IA: Ponto de Partida (correlação/cenário) e retomada em Vamos Exercitar + checagem páginas",
            "AJUSTAR",
            "IA desativada ou falha ao obter JSON; usando apenas heurísticas locais.",
            "—"
        ))

    return out, ia_json

# ===================== UI =====================
st.sidebar.header("Configurações")
tema = st.sidebar.text_input("Tema da Aula (para checagem de títulos)", value="")
assuntos_raw = st.sidebar.text_input("Assuntos granulares (separe por ; )", value="")
assuntos = [s.strip() for s in assuntos_raw.split(";") if s.strip()]
wpp = st.sidebar.number_input("Palavras por página (heurística)", min_value=200, max_value=600, value=350, step=10)
min_refs = st.sidebar.number_input("Mínimo de referências", min_value=1, max_value=20, value=3, step=1)

ok_api, msg_api = quick_api_selftest()
st.sidebar.markdown("---")
st.sidebar.write("**Status da IA:**", "🟢" if ok_api else "🔴", msg_api)

uploaded = st.file_uploader(
    "Envie arquivos DOCX ou PDF (pode ser múltiplo)",
    type=["docx", "pdf"],
    accept_multiple_files=True
)

# -------------------- Chat IA sobre o arquivo --------------------
def have_api() -> bool:
    try:
        return load_api_key() is not None and OpenAI is not None
    except Exception:
        return False

def build_context_for_chat(sections: Dict[str, str], max_chars: int = 16000) -> str:
    # Junta seções com rótulos, corta se ficar grande
    order = ["videoaula","ponto","comecar","siga","exercitar","saibamais","referencias"]
    parts = []
    total = 0
    for k in order:
        label = {
            "videoaula":"Texto: Vídeoaula", "ponto":"Ponto de Partida",
            "comecar":"Vamos Começar", "siga":"Siga em Frente",
            "exercitar":"Vamos Exercitar", "saibamais":"Saiba Mais",
            "referencias":"Referências",
        }[k]
        txt = sections.get(k, "") or ""
        frag = f"\n[{label}]\n{txt}\n"
        if total + len(frag) > max_chars:
            frag = frag[: max(0, max_chars - total)]
        parts.append(frag)
        total += len(frag)
        if total >= max_chars:
            break
    return "".join(parts).strip()

def chat_answer_on_file(question: str, tema: str, assuntos: List[str], sections: Dict[str, str]) -> str:
    """
    Responde com base SOMENTE no conteúdo do arquivo (sections).
    Se a evidência não estiver explícita, devolve isso claramente.
    """
    if not have_api():
        return "IA desativada (sem OPENAI_API_KEY ou biblioteca indisponível)."

    client = OpenAI(api_key=load_api_key())
    context = build_context_for_chat(sections)

    system_msg = (
        "Você é um revisor pedagógico. Responda APENAS com base no conteúdo fornecido.\n"
        "Se a resposta não estiver explicitamente no texto, diga: 'Não está explícito no arquivo'. "
        "Sempre aponte a(s) seção(ões) onde encontrou a evidência."
    )
    user_msg = f"""
Tema: {tema or '(não informado)'}
Assuntos granulares: {", ".join(assuntos) if assuntos else '(não informados)'}

=== CONTEÚDO DO ARQUIVO ===
{context}

=== PERGUNTA ===
{question}

Responda em português, curto e direto, incluindo referência às seções (ex.: [Ponto de Partida], [Vamos Exercitar]).
"""

    # usar /responses com fallback para chat completions
    try:
        resp = client.responses.create(model="gpt-4o-mini", input=[{"role":"system","content":system_msg},
                                                                   {"role":"user","content":user_msg}],
                                       temperature=0.0)
        return resp.output_text.strip()
    except Exception:
        try:
            chat = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":system_msg},
                          {"role":"user","content":user_msg}],
                temperature=0.0
            )
            return (chat.choices[0].message.content or "").strip()
        except Exception as e:
            return f"Falha ao consultar a IA: {e}"


if uploaded:
    resultados = []
    for i, up in enumerate(uploaded):
        st.markdown(f"### 📄 {up.name}")
        data = up.read()
        bio = io.BytesIO(data)

        # extrai texto
        try:
            if up.name.lower().endswith(".docx"):
                fulltext = extract_text_from_docx(bio)
            else:
                fulltext = extract_text_from_pdf(bio)
        except Exception as e:
            st.error(f"Falha ao extrair texto: {e}")
            continue

        # divide por seção
        sections = split_sections(fulltext)

        # sumário rápido
        cols = st.columns(3)
        cols[0].write(f"**Vídeoaula**: {word_count(sections.get('videoaula',''))} palavras")
        cols[1].write(f"**Ponto de Partida**: {word_count(sections.get('ponto',''))} palavras")
        cols[2].write(f"**Exercitar**: {word_count(sections.get('exercitar',''))} palavras")

        # avalia (inclui IA estruturada)
        res, ia_json = eval_metrics(sections, tema=tema, wpp=wpp, min_refs=min_refs, assuntos=assuntos)
        df_res = pd.DataFrame([asdict(r) for r in res])
        st.dataframe(df_res, use_container_width=True, hide_index=True)

        # IA (detalhes)
        if ia_json:
            with st.expander("🔎 Detalhes da IA (JSON)"):
                st.json(ia_json)

        # Adição: uso das checagens fixas por prompt (não-invasivo)
        # (chama os prompts fixos adicionados; se IA não disponível, retorna None)
        with st.expander("🧰 Checagens rápidas da IA (prompts fixos)"):
            try:
                res_ponto = call_fixed_prompt("ponto_exercitar", sections, tema=tema, assuntos=assuntos)
                res_vc = call_fixed_prompt("vc_encadeia", sections, tema=tema, assuntos=assuntos)
                res_ex = call_fixed_prompt("exercitar_respond", sections, tema=tema, assuntos=assuntos)
                res_sm = call_fixed_prompt("saiba_mais_completa", sections, tema=tema, assuntos=assuntos)
                st.write({
                    "ponto_exercitar": res_ponto,
                    "vc_encadeia": res_vc,
                    "vamos_exercitar_responde": res_ex,
                    "saiba_mais": res_sm
                })
            except Exception as e:
                st.write("Falha ao executar checagens fixas da IA:", e)

        # 💬 Chat com a IA sobre este arquivo
        q = st.text_input(f"Pergunta sobre: {up.name}", key=f"q_{i}_{Path(up.name).stem}")
        if q:
            st.write(chat_answer_on_file(q, tema=tema, assuntos=assuntos, sections=sections))

        # checklist de ações
        st.markdown("#### ✅ Checklist de Ações")
        for item in checklist_from_results(res):
            st.write(item)

        # downloads individuais
        out_xlsx = io.BytesIO()
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as wr:
            df_res.to_excel(wr, index=False, sheet_name="Relatório")
        st.download_button(
            label="⬇️ Baixar relatório (XLSX)",
            data=out_xlsx.getvalue(),
            file_name=f"relatorio_{Path(up.name).stem}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("---")

        # acumulado
        df_res.insert(0, "arquivo", up.name)
        resultados.append(df_res)

    # consolidado
    if resultados:
        final = pd.concat(resultados, ignore_index=True)
        st.subheader("Consolidado")
        st.dataframe(final, use_container_width=True, hide_index=True)

        out_all = io.BytesIO()
        with pd.ExcelWriter(out_all, engine="openpyxl") as wr:
            final.to_excel(wr, index=False, sheet_name="Relatório Consolidado")
        st.download_button(
            label="⬇️ Baixar relatório consolidado (XLSX)",
            data=out_all.getvalue(),
            file_name="relatorio_consolidado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Envie um ou mais arquivos para iniciar a revisão.")
