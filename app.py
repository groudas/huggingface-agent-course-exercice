import os
import gradio as gr
import requests
import inspect
import pandas as pd
import yaml
import json
import re
from pocketflow import Node, Flow
from ddgs import DDGS

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
# --- Complex Agent System: Orchestrator Flow ---
class OrchestratorNode(Node):
    def prep(self, shared):
        # Use the latest question and reasoning
        question = shared.get("current_question", shared.get("question", ""))
        reasoning = shared.get("reasoning", "")
        return question, reasoning
    def exec(self, inputs):
        question, reasoning = inputs
        system_prompt = (
            "You are an agentic orchestrator. Given the current question and reasoning, decide if you need to search the web, do more reasoning, or if you are ready to give a final answer. Respond in YAML: action: search|reason|final_answer, reason: <your reasoning>."
        )
        prompt = f"<question>{question}</question>\nCurrent reasoning: {reasoning}"
        try:
            response = call_perplexity_api([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ])
            print(f"[OrchestratorNode] LLM response: {response}")
            parsed_response = extract_json_from_text(response)
            if parsed_response:
                return parsed_response
            # Fallback to raw response if JSON parsing fails
            return response
        except Exception as e:
            print(f"[OrchestratorNode] LLM or YAML parse error: {e}")
            return {"action": "parse_error", "reason": str(e)}
    def post(self, shared, prep_res, exec_res):
        if isinstance(exec_res, dict) and "action" in exec_res:
            shared["orchestrator_action"] = exec_res["action"]
            shared["reasoning"] = exec_res.get("reason", "")
        else:
            # Handle raw string fallback
            shared["orchestrator_action"] = "parse_error"
            shared["reasoning"] = exec_res  # Log raw response for debugging
        return shared["orchestrator_action"]

class ReasoningNode(Node):
    def prep(self, shared):
        # Use the latest question and reasoning
        return shared.get("current_question", shared.get("question", "")), shared.get("reasoning", "")
    def exec(self, inputs):
        question, reasoning = inputs
        prompt = f"<question>{question}</question>\nCurrent reasoning: {reasoning}\n Continue reasoning. Detect if there are flaws or gaps."
        try:
            response = call_perplexity_api([
                {"role": "user", "content": prompt}
            ])
            print(f"[ReasoningNode] LLM response: {response}")
            return response
        except Exception as e:
            print(f"[ReasoningNode] LLM error: {e}")
            return f"Reasoning error: {e}"
    def post(self, shared, prep_res, exec_res):
        # Append reasoning
        prev = shared.get("reasoning", "")
        shared["reasoning"] = prev + "\n" + exec_res
        return "orchestrate"

class WebSearchToolNode(Node):
    def prep(self, shared):
        # Use the latest question
        return shared.get("current_question", shared.get("question", ""))
    def exec(self, question):
        return search_web_ddgs(question)
    def post(self, shared, prep_res, exec_res):
        # Add web search result to reasoning
        prev = shared.get("reasoning", "")
        shared["reasoning"] = prev + f"\n[WebSearch]: {exec_res}"
        return "orchestrate"


class EvidenceExtractionNode(Node):
    """Extract short evidence snippets from structured web search results.
    Produces `shared['evidence']` as a list of {'href','snippet','source'}.
    """
    def prep(self, shared):
        # Try multiple places where search results might be stored
        return shared.get('answer') or shared.get('search_results') or shared.get('reasoning') or []

    def exec(self, search_results):
        import re
        structured = []
        # If reasoning is a string (contains the websearch dump), we cannot easily parse hrefs.
        if isinstance(search_results, str):
            # attempt to pull hrefs and short surrounding contexts
            hrefs = re.findall(r'https?://[^\s)]+', search_results)
            for h in hrefs[:5]:
                structured.append({'href': h, 'snippet': h, 'source': h})
            return structured

        if not isinstance(search_results, list):
            return []

        for r in search_results[:8]:
            href = r.get('href', '') if isinstance(r, dict) else ''
            # prefer wiki_summary, then body, then title
            text = ''
            if isinstance(r, dict):
                text = r.get('wiki_summary') or r.get('body') or r.get('title') or ''
            else:
                text = str(r)

            # pick sentence with digits if present, else first sentence
            sentences = re.split(r'(?<=[.!?])\s+', text)
            chosen = ''
            for s in sentences:
                if re.search(r'\d', s):
                    chosen = s.strip()
                    break
            if not chosen and sentences:
                chosen = sentences[0].strip()

            source = ''
            try:
                if href and '//' in href:
                    source = href.split('/')[2]
            except Exception:
                source = href

            structured.append({'href': href, 'snippet': chosen, 'source': source})

        return structured

    def post(self, shared, prep_res, exec_res):
        shared['evidence'] = exec_res or []
        # keep flow going to orchestrator (we assume evidence was extracted mid-flow)
        return 'orchestrate'

class FinalAnswerNode(Node):
    def prep(self, shared):
        # Use the question and all reasoning
        return shared.get("current_question", shared.get("question", "")), shared.get("reasoning", "")
    def exec(self, inputs):
        question, reasoning = inputs
        system_prompt = (
            "You are a final-answer agent. STRICTLY follow the output format below.\n"
            "Return ONLY a fenced YAML block and nothing else. The YAML must contain a single key `final_answer`. Example exactly:\n"
            "```yaml\nfinal_answer: 3\n```\n"
            "If you cannot determine an answer, return exactly:\n"
            "```yaml\nfinal_answer: null\n```\n"
            "Do NOT include any extra commentary, explanation, or text outside the fenced YAML."
        )
        prompt = f"<question>{question}</question>\nAll reasoning: {reasoning}\n\nNow produce the final answer in YAML as described above."
        try:
            response = call_perplexity_api([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ], max_tokens=200)
            print(f"[FinalAnswerNode] LLM response: {response}")
            return response
        except Exception as e:
            print(f"[FinalAnswerNode] LLM error: {e}")
            return f"Final answer error: {e}"
    def post(self, shared, prep_res, exec_res):
        # exec_res is the raw LLM response (string). Try to extract YAML/JSON structured answer.
        raw = exec_res if isinstance(exec_res, str) else str(exec_res)
        parsed_answer = None

        # 1) Try to find a fenced YAML block
        try:
            if "```yaml" in raw:
                yaml_str = raw.split("```yaml")[1].split("```", 1)[0].strip()
                data = yaml.safe_load(yaml_str)
                if isinstance(data, dict) and "final_answer" in data:
                    parsed_answer = data["final_answer"]
            elif "```json" in raw:
                json_str = raw.split("```json")[1].split("```", 1)[0].strip()
                import json
                data = json.loads(json_str)
                if isinstance(data, dict) and "final_answer" in data:
                    parsed_answer = data["final_answer"]
        except Exception as e:
            print(f"[FinalAnswerNode] Structured parse error: {e}")

        # 2) Try to parse whole text as YAML/JSON
        if parsed_answer is None:
            try:
                data = yaml.safe_load(raw)
                if isinstance(data, dict) and "final_answer" in data:
                    parsed_answer = data["final_answer"]
            except Exception:
                pass

        # 3) Look for 'FINAL ANSWER:' marker
        if parsed_answer is None:
            marker = "FINAL ANSWER:"
            if marker in raw:
                parsed_answer = raw.split(marker, 1)[1].strip()

        # 4) Fallback: take last non-empty line
        if parsed_answer is None:
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            if lines:
                parsed_answer = lines[-1]
            else:
                parsed_answer = ""

        shared["final_answer"] = parsed_answer
        shared["final_answer_raw"] = raw
        return "done"


class AnswerNode(Node):
    """Unified Answer node: request a strict JSON object containing `final_answer`.
    Exec produces raw LLM string; post parses JSON-first (fenced or inline) and writes
    `shared['final_answer']` and `shared['final_answer_raw']`.
    """
    def prep(self, shared):
        return {
            'question': shared.get('current_question', shared.get('question', '')),
            'reasoning': shared.get('reasoning', ''),
            'search_results': shared.get('search_results', shared.get('answer', []))
        }

    def exec(self, inputs):
        q = inputs['question']
        reasoning = inputs['reasoning']
        search_results = inputs['search_results']

        system_prompt = (
            "You are an assistant that MUST respond with a single JSON object and nothing else.\n"
            "Return exactly one JSON object with a single key `final_answer`. Example: {\"final_answer\": 3}.\n"
            "If you cannot determine an answer, set the value to null: {\"final_answer\": null}.\n"
            "Do NOT include any surrounding text, explanation, or markdown fences.\n\n"
            "Examples:\n"
            "### Example 1\n"
            "Question: How many planets are in the Solar System?\n"
            "Reasoning: Based on the search results, the Solar System has eight planets.\n"
            "Search Results: ['Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune']\n"
            "JSON Response: {\"final_answer\": 8}\n\n"
            "### Example 2\n"
            "Question: What is the capital of France?\n"
            "Reasoning: The search results clearly indicate that the capital of France is Paris.\n"
            "Search Results: ['Paris is the capital city of France.']\n"
            "JSON Response: {\"final_answer\": \"Paris\"}\n\n"
            "### Example 3\n"
            "Question: What are the primary colors?\n"
            "Reasoning: The search results list red, blue, and yellow as the primary colors.\n"
            "Search Results: ['Primary colors are red, blue, and yellow.']\n"
            "JSON Response: {\"final_answer\": [\"red\", \"blue\", \"yellow\"]}\n\n"
            "### Example 4\n"
            "Question: What is the range of temperatures on Mars?\n"
            "Reasoning: The search results indicate that temperatures on Mars range from -125°C to 20°C.\n"
            "Search Results: ['Mars temperatures range from -125°C to 20°C.']\n"
            "JSON Response: {\"final_answer\": \"-125°C to 20°C\"}\n\n"
            "### Example 5\n"
            "Question: What is the population of Atlantis?\n"
            "Reasoning: The search results indicate that Atlantis is a fictional place and has no population.\n"
            "Search Results: ['Atlantis is a fictional place.']\n"
            "JSON Response: {\"final_answer\": null}\n"
        )

        user_prompt = f"<question>{q}</question>\nSearch results: {search_results}\n\nChain-of-thought:\n{reasoning}\n\nProduce the final answer as a JSON object with key `final_answer`."

        try:
            resp = call_perplexity_api([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ], max_tokens=300)
            print(f"[AnswerNode] LLM response: {resp}")
            return resp
        except Exception as e:
            print(f"[AnswerNode] LLM error: {e}")
            return f"AnswerNode error: {e}"

    def post(self, shared, prep_res, exec_res):
        parsed_response = extract_json_from_text(exec_res)
        if isinstance(parsed_response, dict) and "final_answer" in parsed_response:
            shared["final_answer"] = parsed_response["final_answer"]
            shared["final_answer_raw"] = exec_res
        else:
            # Fallback to raw response
            shared["final_answer"] = None
            shared["final_answer_raw"] = exec_res
        return "done"


class ValidatorNode(Node):
    """Validate that `shared['final_answer_raw']` contains a fenced YAML with `final_answer`.
    If invalid, first try a small repair prompt (ask LLM to return ONLY a fenced YAML),
    then fall back to re-running the SynthesizerNode. The node sets
    `shared['final_answer_valid']` True/False and stores `validation_error` on failure.
    """
    def __init__(self, max_retries=3):
        super().__init__()
        self.max_retries = max_retries

    def prep(self, shared):
        return {
            'raw': shared.get('final_answer_raw', ''),
            'parsed': shared.get('final_answer', None),
            'question': shared.get('current_question', shared.get('question','')),
            'evidence': shared.get('evidence', [])
        }

    def exec(self, inputs):
        raw = inputs['raw']
        parsed = inputs['parsed']
        evidence = inputs.get('evidence', [])
        import yaml

        # If already parsed and scalar, accept
        if parsed is not None and isinstance(parsed, (str, int, float)):
            return {'status': 'valid', 'value': parsed}

        # 1) fenced YAML
        try:
            if '```yaml' in raw:
                yaml_str = raw.split('```yaml',1)[1].split('```',1)[0].strip()
                obj = yaml.safe_load(yaml_str)
                if isinstance(obj, dict) and 'final_answer' in obj:
                    return {'status':'valid','value':obj['final_answer']}
        except Exception as e:
            print(f"[ValidatorNode] fenced YAML parse error: {e}")

        # 2) whole-text YAML/JSON
        try:
            obj = yaml.safe_load(raw)
            if isinstance(obj, dict) and 'final_answer' in obj:
                return {'status':'valid','value':obj['final_answer']}
        except Exception:
            pass

        return {'status':'invalid', 'reason':'no parseable final_answer found', 'evidence': evidence}

    def post(self, shared, prep_res, exec_res):
        # exec_res contains status + optional evidence
        ev = exec_res.get('evidence', []) if isinstance(exec_res, dict) else shared.get('evidence', [])
        if exec_res['status'] == 'valid':
            shared['final_answer_valid'] = True
            shared['final_answer'] = exec_res['value']
            shared['validation_trace'] = {'status': 'valid', 'reason': 'parsed_structured', 'evidence': ev}
            return 'valid'

        # invalid -> check evidence rules before repair
        # Rule: if parsed is scalar and an evidence snippet matches, accept; otherwise attempt repair
        parsed = prep_res.get('parsed') if isinstance(prep_res, dict) else None
        evidence = prep_res.get('evidence', []) if isinstance(prep_res, dict) else []

        def normalize_number(s):
            try:
                if isinstance(s, (int, float)):
                    return float(s)
                ss = str(s).replace(',', '').strip()
                if ss.endswith('%'):
                    return float(ss[:-1])
                return float(ss)
            except Exception:
                return None

        # quick evidence match checks
        if parsed is not None and evidence:
            # numeric case
            num = normalize_number(parsed)
            if num is not None:
                matches = 0
                for e in evidence[:5]:
                    snip = e.get('snippet','')
                    if snip:
                        v = normalize_number(snip)
                        if v is not None and abs(v - num) < 1e-6:
                            matches += 1
                if matches >= 1:
                    shared['final_answer_valid'] = True
                    shared['final_answer'] = parsed
                    shared['validation_trace'] = {'status':'valid','reason':'numeric_match','matches':matches,'evidence': evidence[:5]}
                    return 'valid'

            # string/text case: substring match in evidence snippets
            try:
                pstr = str(parsed).lower()
                matches = 0
                for e in evidence[:5]:
                    snip = (e.get('snippet','') or '').lower()
                    if pstr and snip and pstr in snip:
                        matches += 1
                if matches >= 1:
                    shared['final_answer_valid'] = True
                    shared['final_answer'] = parsed
                    shared['validation_trace'] = {'status':'valid','reason':'text_match','matches':matches,'evidence': evidence[:5]}
                    return 'valid'
            except Exception:
                pass

        # attempt repair via LLM AnswerNode, feeding top evidence
        retries = shared.get('_validator_retries', 0)
        if retries < self.max_retries:
            shared['_validator_retries'] = retries + 1
            try:
                ans_node = AnswerNode()
                tmp = dict(shared)
                ev = shared.get('evidence', [])
                ev_summary = '\n'.join([f"- {e.get('source','')} {e.get('snippet','')} ({e.get('href','')})" for e in ev[:5]])
                tmp['reasoning'] = (tmp.get('reasoning','') or '') + "\n\nEVIDENCE:\n" + ev_summary
                ans_node.run(tmp)
                shared['final_answer_raw'] = tmp.get('final_answer_raw', tmp.get('reasoning',''))
                shared['final_answer'] = tmp.get('final_answer', None)
                shared['final_answer_valid'] = False
                shared['validation_trace'] = {'status':'repair_invoked','evidence_used': ev[:5]}
                return 'retry'
            except Exception as e:
                print(f"[ValidatorNode] AnswerNode retry failed: {e}")

        # give up: mark invalid and record trace
        shared['final_answer_valid'] = False
        shared['validation_error'] = exec_res.get('reason','invalid format')
        shared['validation_trace'] = {'status':'failed','reason': shared.get('validation_error'), 'evidence': exec_res.get('evidence', [])}
        return 'invalid'


class FinishNode(Node):
    def prep(self, shared):
        return shared.get('final_answer', None), shared.get('final_answer_valid', False)
    def exec(self, inputs):
        final, valid = inputs
        # Nothing to call LLM for; just pass through
        return {'final': final, 'valid': valid}
    def post(self, shared, prep_res, exec_res):
        shared['final_answer'] = exec_res['final']
        shared['final_answer_valid'] = exec_res['valid']
        return None


class SynthesizerNode(Node):
    """A focused synthesizer that extracts concise, machine-parseable facts
    (especially numeric answers) from the accumulated reasoning and search results.
    It writes `shared['final_answer']` as a single concise value (string or number).
    """
    def prep(self, shared):
        # Provide the question, reasoning and any structured web search results
        return {
            'question': shared.get('current_question', shared.get('question', '')),
            'reasoning': shared.get('reasoning', ''),
            'search_results': shared.get('search_results', shared.get('answer', []))
        }

    def exec(self, inputs):
        q = inputs['question']
        reasoning = inputs['reasoning']
        search_results = inputs['search_results']

        system_prompt = (
            "You are a concise synthesizer. Given a question, LLM chain-of-thought reasoning, and optional structured web search results (title, body, href, wiki_summary, extracted_numbers),\n"
            "extract the single best final answer. If the question asks for a numeric fact (e.g., how many, count, number of), return only the integer or number.\n"
            "Return the answer inside a fenced YAML block with a single key `final_answer`. Example:\n```yaml\nfinal_answer: 3\n```\n"
            "If there is uncertainty, give your best concise answer."
        )

        # Build a compact user prompt
        prompt = f"<question>{q}</question>\n\nSearch results:\n{search_results}\n\nChain-of-thought:\n{reasoning}\n\nProvide the final answer as YAML as instructed."

        try:
            response = call_perplexity_api([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ], max_tokens=300)
            print(f"[SynthesizerNode] LLM response: {response}")
            return response
        except Exception as e:
            print(f"[SynthesizerNode] LLM error: {e}")
            return f"Synthesizer error: {e}"

    def post(self, shared, prep_res, exec_res):
        # Reuse FinalAnswerNode parsing logic by attempting to parse YAML/JSON
        raw = exec_res if isinstance(exec_res, str) else str(exec_res)
        parsed = None
        try:
            if "```yaml" in raw:
                yaml_str = raw.split("```yaml")[1].split("```", 1)[0].strip()
                obj = yaml.safe_load(yaml_str)
                if isinstance(obj, dict) and 'final_answer' in obj:
                    parsed = obj['final_answer']
        except Exception as e:
            print(f"[SynthesizerNode] structured parse error: {e}")

        if parsed is None:
            try:
                obj = yaml.safe_load(raw)
                if isinstance(obj, dict) and 'final_answer' in obj:
                    parsed = obj['final_answer']
            except Exception:
                pass

        if parsed is None:
            # Fallback: last non-empty line
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            parsed = lines[-1] if lines else ''

        shared['final_answer'] = parsed
        shared['final_answer_raw'] = raw
        return 'done'

# --- Orchestrator Flow Setup ---
def create_orchestrator_flow():
    orchestrator = OrchestratorNode()
    websearch = WebSearchToolNode()
    evidence_extraction = EvidenceExtractionNode()
    answer_node = AnswerNode()
    validator = ValidatorNode()

    # Define transitions
    orchestrator - "search" >> websearch
    websearch - "default" >> evidence_extraction
    evidence_extraction - "orchestrate" >> orchestrator
    orchestrator - "final_answer" >> answer_node
    orchestrator - "parse_error" >> answer_node  # Handle parse errors
    answer_node - "done" >> validator
    validator - "retry" >> answer_node

    return Flow(start=orchestrator)
class FetchQuestionNode(Node):
    def prep(self, shared):
        api_url = DEFAULT_API_URL + "/questions"
        print(f"[FetchQuestionNode] Fetching all questions from: {api_url}")
        try:
            resp = requests.get(api_url, timeout=15)
            resp.raise_for_status()
            questions = resp.json()
            if questions and isinstance(questions, list):
                # Store the entire list in shared so callers can iterate
                shared["questions_list"] = questions
                print(f"[FetchQuestionNode] Fetched {len(questions)} questions.")
                return questions
            else:
                print("[FetchQuestionNode] No questions found.")
                shared["questions_list"] = []
                return []
        except Exception as e:
            print(f"[FetchQuestionNode] Exception: {e}")
            shared["questions_list"] = []
            return []

    def exec(self, question):
        # This node's exec simply passes through the fetched questions list.
        # When run via Node.run(), `question` will actually be the questions list
        # returned by prep(). We simply return it unchanged.
        return question

    def post(self, shared, prep_res, exec_res):
        # Store action and reason in shared, return action for flow branching
        # Exec_res here is the full questions list (or []). Keep it in shared.
        shared["questions_list"] = exec_res
        return None

def search_web_ddgs(query):
    print(f"[DDGS] Searching for: {query}")
    try:
        results = DDGS().text(query, max_results=10)
        results = list(results)
        if not results:
            print("[DDGS] No results found.")
            return []

        # Prefer Wikipedia result if available
        wiki = next((r for r in results if r.get('href') and 'wikipedia.org' in r.get('href')), None)
        ordered = []
        if wiki:
            ordered.append(wiki)
        # Append other results excluding the chosen wiki
        ordered.extend([r for r in results if r is not wiki])

        structured = []
        for r in ordered:
            item = {
                'title': r.get('title', '') or r.get('text', ''),
                'body': r.get('body', '') or r.get('text', ''),
                'href': r.get('href', '')
            }
            # If this looks like a Wikipedia URL, try to fetch the page intro
            href = item.get('href') or ''
            if 'wikipedia.org' in href:
                try:
                    wiki_summary = fetch_wikipedia_summary(href)
                    item['wiki_summary'] = wiki_summary
                    nums = extract_numbers_from_text(wiki_summary)
                    item['extracted_numbers'] = nums
                except Exception as e:
                    print(f"[DDGS] Wikipedia parse error for {href}: {e}")
            structured.append(item)
        print(f"[DDGS] Found {len(structured)} results (wiki preferred if present)")
        return structured
    except Exception as e:
        print(f"[DDGS] Exception: {e}")
        return f"Web search error: {e}"


def fetch_wikipedia_summary(url: str) -> str:
    """Fetch the wikipedia page and return the introductory paragraphs as plain text.
    This function uses a simple requests.get and attempts to extract the lead paragraph(s) by
    stripping HTML tags between the first <p> tags. It's a best-effort helper and may fail
    if Wikipedia changes layout or if the page is not accessible."""
    print(f"[WIKI] Fetching summary for: {url}")
    try:
        headers = {"User-Agent": "PocketFlowBot/1.0 (+https://example.com)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        text = resp.text
        # Minimal HTML parsing to extract the first paragraph(s)
        # Find the first '<p>' occurrence after the content div
        import re
        # Remove newlines to simplify regex
        cleaned = re.sub(r"\n+", " ", text)
        # Try to find the first two paragraphs
        paras = re.findall(r"<p>(.*?)</p>", cleaned, flags=re.IGNORECASE)
        if not paras:
            return ''
        # Join first two paragraphs for more context
        lead = ' '.join(paras[:2])
        # Strip tags and unescape HTML entities
        lead_text = re.sub(r'<.*?>', '', lead)
        import html
        lead_text = html.unescape(lead_text).strip()
        return lead_text
    except Exception as e:
        print(f"[WIKI] Error fetching wiki summary: {e}")
        return ''


def extract_numbers_from_text(text: str):
    """Return a list of numeric mentions found in the text (integers, years, percentages).
    Returns strings as they appear (no normalization)."""
    import re
    if not text:
        return []
    # capture things like 1999, 3, 3.5, 20%, 2,000
    pattern = r"\b\d{1,3}(?:[,\d]{0,})(?:\.\d+)?%?\b"
    found = re.findall(pattern, text)
    # Also capture year ranges like 2000–2009 or 2000-2009
    ranges = re.findall(r"\b\d{4}[–-]\d{4}\b", text)
    return list(dict.fromkeys(found + ranges))

def extract_json_from_text(text: str):
    """Try to extract a JSON object from arbitrary LLM text.
    Returns a dict if successful, otherwise None.
    Order: fenced ```json```, full-text JSON, first {...} object via regex.
    """
    if not text or not isinstance(text, str):
        return None
    # 1) fenced json
    try:
        if '```json' in text:
            body = text.split('```json', 1)[1].split('```', 1)[0].strip()
            return json.loads(body)
    except Exception:
        pass

    # 2) try parse whole text
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    # 3) find first JSON object with regex
    m = re.search(r"\{[\s\S]*?\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

def call_perplexity_api(messages, model="sonar-medium-online", max_tokens=300):
    """Call Perplexity chat completions API and return the assistant content string.
    Expects messages as a list of {'role':..,'content':..} dicts.
    The API key must be provided in the PERPLEXITY_API_KEY environment variable.
    """
    api_key = os.getenv("PERPLEXITY_API_KEY", "")
    # Trim whitespace/newlines that can accidentally be included when copying keys
    api_key = api_key.strip()
    if not api_key:
        raise RuntimeError("PERPLEXITY_API_KEY environment variable not set or empty after trimming whitespace")

    # Allow overriding the default model via env var
    env_model = os.getenv("PERPLEXITY_MODEL")
    if env_model is not None:
        env_model = env_model.strip()
    model = env_model or model
    print(f"[Perplexity API] Using model: {model}")
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,  # Deterministic output
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            # Try to surface response body for debugging
            body = None
            try:
                body = resp.json()
            except Exception:
                body = resp.text[:1000]
            print(f"[Perplexity API] HTTP {resp.status_code} Error: {http_err}; body={body}")
            # Helpful hint for the common invalid_model error
            try:
                if isinstance(body, dict) and body.get('error', {}).get('code') == 400 and 'invalid_model' in str(body.get('error', {})):
                    print("[Perplexity API] The model you requested appears invalid. Set a valid model using the PERPLEXITY_MODEL environment variable or consult https://docs.perplexity.ai/getting-started/models for permitted models.")
            except Exception:
                pass
            raise
        data = resp.json()
        # Follow the sample structure: choices[0].message.content
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[Perplexity API] Error: {e}")
        raise

class WebSearchNode(Node):
    def prep(self, shared):
        return shared["question"]
    def exec(self, question):
        return search_web_ddgs(question)
    def post(self, shared, prep_res, exec_res):
        shared["answer"] = exec_res
        return "default"

class PocketFlowWebSearchAgent:
    def __init__(self):
        self.node = WebSearchNode()
        self.flow = Flow(start=self.node)
    def __call__(self, question: str) -> str:
        shared = {"question": question}
        self.flow.run(shared)
        return shared.get("answer", "No answer produced.")

def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Orchestrator Flow
    try:
        orchestrator_flow = create_orchestrator_flow()
    except Exception as e:
        print(f"Error creating orchestrator flow: {e}")
        return f"Error initializing orchestrator: {e}", None
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions via FetchQuestionNode (so the logic is centralized)
    print("Fetching questions via FetchQuestionNode...")
    try:
        fetch_node = FetchQuestionNode()
        fetch_flow = Flow(start=fetch_node)
        temp_shared = {}
        fetch_flow.run(temp_shared)
        questions_data = temp_shared.get("questions_list", [])
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions via FetchQuestionNode.")
    except Exception as e:
        print(f"Error fetching questions via FetchQuestionNode: {e}")
        return f"Error fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running orchestrator flow on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            shared = {"question": question_text, "current_question": question_text, "reasoning": ""}
            orchestrator_flow.run(shared)
            final = shared.get("final_answer", "")
            # Ensure validator has been run and set final_answer_valid
            if 'final_answer_valid' not in shared:
                try:
                    validator = ValidatorNode()
                    validator.run(shared)
                except Exception as e:
                    print(f"Validator node failed for task {task_id}: {e}")
            is_valid = shared.get('final_answer_valid', False)
            # If the flow didn't produce a final answer, run the AnswerNode (JSON-first)
            if not final:
                try:
                    ans = AnswerNode()
                    ans_shared = {**shared}
                    ans.run(ans_shared)
                    final = ans_shared.get("final_answer", "") or ans_shared.get("reasoning", "").strip()
                    shared["final_answer"] = final
                    shared["final_answer_raw"] = ans_shared.get("final_answer_raw", shared.get("final_answer_raw"))
                except Exception as e:
                    print(f"AnswerNode failed for task {task_id}: {e}")
                    final = shared.get("reasoning", "").strip() or ""
            # Use null when validation failed
            if not shared.get('final_answer_valid', False):
                submitted = None
            else:
                submitted = final
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": final})
        except Exception as e:
            print(f"Error running orchestrator on task {task_id}: {e}")
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
def launch_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("# Basic Agent Evaluation Runner")
        gr.Markdown(
            """
            **Instructions:**

            1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
            2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
            3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

            ---
            **Disclaimers:**
            Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
            This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
            """
        )

        gr.LoginButton()

        run_button = gr.Button("Run Evaluation & Submit All Answers")

        status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
        # Removed max_rows=10 from DataFrame constructor
        results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

        run_button.click(
            fn=run_and_submit_all,
            outputs=[status_output, results_table]
        )

    demo.launch(debug=True, share=False)

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    launch_gradio()