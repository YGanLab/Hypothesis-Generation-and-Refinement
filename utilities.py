import re
import json
import yaml
import pandas as pd
import itertools
import requests
from xml.etree import ElementTree as ET
import time
from typing import List, Tuple
from retrieval import GetPmids
from rag import RAG
config_path = "config.yaml"
llm_api = RAG(config_path)
    
def read_config(config_path: str):
    """
    Reads the YAML configuration file.
    :param config_path: Path to the YAML file.
    :return: Dictionary with configuration data.
    """
    try:
        with open(config_path, 'r', encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        return {}

def load_excel_strip_columns(file_path: str) -> pd.DataFrame:
    """
    Loads an Excel file into a DataFrame and strips whitespace from all column names.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: DataFrame with stripped column names.
    """
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    return df

def parse_json_string(input_string):
    """
    Parses a JSON string safely, cleaning common formatting issues without corrupting valid JSON escape sequences.

    Parameters:
        input_string (str): The input string containing a JSON object.

    Returns:
        dict or list: The parsed JSON object.

    Raises:
        ValueError: If no valid JSON is found or if the JSON is invalid.
    """

    # Extract JSON content from surrounding text
    match = re.search(r'(\{.*\}|\[.*\])', input_string, re.DOTALL)
    if not match:
        print(input_string)
        raise ValueError("No valid JSON found in the input string.")

    cleaned_string = match.group(1)  # Extract the JSON part

    try:
        # Replace curly quotes with standard quotes
        cleaned_string = cleaned_string.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")

        # Remove hidden control characters
        cleaned_string = re.sub(r'[\x00-\x1F\x7F]', '', cleaned_string)

        # Fix trailing commas before closing braces or brackets
        cleaned_string = re.sub(r',\s*([\]}])', r'\1', cleaned_string)

        # Ensure that only truly unescaped double quotes inside JSON values are escaped
        def escape_unescaped_quotes(text):
            """
            Finds unescaped double quotes inside JSON values and properly escapes them.
            It avoids touching already escaped quotes (e.g., `\"` remains `\"`).
            """
            fixed_text = ""
            in_string = False
            escape_next = False

            for i, char in enumerate(text):
                if char == "\\" and not escape_next:
                    escape_next = True
                    fixed_text += char
                elif char == "\"" and not escape_next:
                    if in_string:
                        in_string = False
                    else:
                        in_string = True
                    fixed_text += char
                elif char == "\"" and escape_next:
                    fixed_text += char  # Preserve escaped quotes
                    escape_next = False
                elif char == "\\" and escape_next:
                    fixed_text += char  # Preserve existing escapes
                    escape_next = False
                else:
                    fixed_text += char
                    escape_next = False

            return fixed_text

        cleaned_string = escape_unescaped_quotes(cleaned_string)

        # Parse JSON safely
        return json.loads(cleaned_string)

    except json.JSONDecodeError as e:
        print(input_string)
        raise ValueError(f"Invalid JSON format after cleaning: {e}")

def read_pmids_from_excel(file_path: str, sheet_name: str = 0, column_name: str = "PMID"):
    """
    Reads PMIDs from an Excel file.
    
    :param file_path: Path to the Excel file.
    :param sheet_name: Sheet name or index (default is first sheet).
    :param column_name: Column name containing PMIDs (default is 'PMID').
    :return: List of PMIDs.
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        if column_name in df.columns:
            pmids = df[column_name].dropna().astype(str).tolist()
            return pmids
        else:
            raise ValueError(f"Column '{column_name}' not found in the Excel file.")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return []

def construct_pubmed_queries(keywords_reference):
    query_list = []
    
    for field, keywords in keywords_reference.items():
        if keywords:  # Ensure the list is not empty
            field_query = " AND ".join(keywords)  # Combine keywords from the same field with AND
            query_list.append(f"({field_query})")  # Wrap the field query in parentheses
    
    return query_list  # Return a list of query strings

def construct_pubmed_query_variants(keywords_reference, min_terms=2, max_terms=None, quote=True):
    """
    Generate all keyword combinations (AND-based) for PubMed queries.
    
    Parameters:
    - keywords_reference (dict): Keyword lists by field.
    - min_terms (int): Minimum number of terms per query.
    - max_terms (int or None): Max number of terms (defaults to all).
    - quote (bool): Whether to wrap multi-word terms in quotes.
    
    Returns:
    - List[str]: List of PubMed-compatible query strings.
    """
    # Flatten all keywords from all fields
    all_keywords = []
    for kw_list in keywords_reference.values():
        all_keywords.extend(kw_list)
    
    # Clean + quote if needed
    all_keywords = list(set(kw.strip() for kw in all_keywords if kw.strip()))
    if quote:
        all_keywords = [f'"{kw}"' if " " in kw else kw for kw in all_keywords]

    if not max_terms:
        max_terms = len(all_keywords)

    query_list = []
    
    # Generate combinations from min_terms to max_terms
    for r in range(min_terms, max_terms + 1):
        for combo in itertools.combinations(all_keywords, r):
            query = " AND ".join(combo)
            query_list.append(f"({query})")

    return query_list


def get_similar_articles(pmid, api_key=None):
    """
    Fetch PMIDs of similar articles for a given PMID using the elink API.
    
    Args:
        pmid (str): The PubMed ID to find similar articles for.
        api_key (str, optional): NCBI API key for higher rate limits.
    
    Returns:
        list: List of PMIDs for similar articles.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    params = {
        "dbfrom": "pubmed",
        "db": "pubmed",
        "id": pmid,
        "cmd": "neighbor",
        "retmode": "xml"
    }
    if api_key:
        params["api_key"] = api_key
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise an error for bad responses
        root = ET.fromstring(response.content)
        
        # Extract related PMIDs
        similar_pmids = [link.text for link in root.findall(".//LinkSetDb/Link/Id")]
        return similar_pmids
    except requests.RequestException as e:
        print(f"Error fetching similar articles: {e}")
        return []
    finally:
        time.sleep(0.1)  # Respect NCBI rate limits (0.1s delay)
        
        
def extract_numbered_hypotheses(raw_text):
    """
    Extracts numbered hypotheses from LLM output and returns as a list.
    """
    raw_text = raw_text.replace("\\n", "\n").strip()
    pattern = r"(?:\n|\A)\d+\.\s*(.*?)(?=\n\d+\.|\Z)"
    matches = re.findall(pattern, raw_text, re.DOTALL)
    return [match.strip() for match in matches if match.strip()]

def evaluate_hypothesis_pair(gpt, generated: str, groundtruth: str, keypoints: str = "") -> Tuple[int, str]:
    prompt = f"""
    "You are an expert scientific evaluator. Rate how well the GENERATED hypothesis matches the MAIN IDEA of the GROUNDTRUTH hypothesis in chemistry.\n"
    "\n"
    "Focus ONLY on conceptual alignment of the primary claim and mechanism (the central research idea, goal, and high-level mechanistic rationale),\n"
    "not on writing quality, length, citations, or minor experimental details. Paraphrases are acceptable. Penalize contradictions or shifts in scope.\n"
    "\n"
    "Scoring rubric (MIOS, 1.0–5.0; decimals allowed):\n"
    "5.0  = Nearly the same main idea: same objective and high-level mechanism/logic; only minor phrasing/detail differences.\n"
    "4.x  = Strongly aligned main idea with small deviations in mechanism or scope.\n"
    "3.x  = Partially aligned: overlaps on objective or mechanism but not both; noticeable gaps or additions.\n"
    "2.x  = Weak alignment: only tangentially related theme; major differences in objective and mechanism.\n"
    "1.x  = Unrelated or contradicts the groundtruth’s main idea.\n"
    "\n"
    "GENERATED Hypothesis:\n{generated}\n\n"
    "GROUNDTRUTH Hypothesis:\n{groundtruth}\n\n"
    "Return ONLY the following two lines:\n"
    "Reason: <one concise sentence explaining your score>\n"
    "Matched score: <score between 1.0 and 5.0>\n"
"""


    response = gpt.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content.strip()
    score = 0
    reason = ""
    for line in content.split("\n"):
        if "Matched score" in line:
            score = float(line.split(":")[1].strip())
        elif "Reason" in line:
            reason = line.split(":", 1)[1].strip()
    return score, reason

def evaluate_hypothesis_pair_2(gpt, generated: str, groundtruth: str, keypoints: str = "") -> Tuple[int, str]:
    prompt = f"""
You are helping to evaluate the quality of a proposed research hypothesis in Chemistry by a phd student. The groundtruth
hypothesis will also be provided to compare. Here we mainly focus on whether the proposed hypothesis has covered the key
points in terms of the methodology in the groundtruth hypothesis. You will also be given a summary of the key points in
the methodology of the groundtruth hypothesis for reference. Please note that for the proposed hypothesis to cover one key
point, it is not necessary to explicitly mention the name of the key point, but might also can integrate the key point implicitly
in the proposed method. The evaluation criteria is called ’Matched score’, which is in a 6-point Likert scale (from 5 to 0).
Particularly, 5 points mean that the proposed hypothesis (1) covers all the key points and leverage them similarly as in the
methodology of the groundtruth hypothesis, and (2) does not contain any extra key point that has apparent flaws; 4 points mean
that the proposed hypothesis (1) covers all the key points (or at least three key points) and leverage them similarly as in the
methodology of the groundtruth hypothesis, (2) but also with extra key points that have apparent flaws; 3 points mean that the
proposed hypothesis (1) covers at least two key points and leverage them similarly as in the methodology of the groundtruth
hypothesis, (2) but does not cover all key points in the groundtruth hypothesis, (3) might or might not contain extra key
points; 2 points mean that the proposed hypothesis (1) covers at least one key point in the methodology of the groundtruth
hypothesis, and leverage it similarly as in the methodology of groundtruth hypothesis, (2) but does not cover all key points in
the groundtruth hypothesis, and (3) might or might not contain extra key points; 1 point means that the proposed hypothesis (1)
covers at least one key point in the methodology of the groundtruth hypothesis, (2) but is used differently as in the methodology
of groundtruth hypothesis, and (3) might or might not contain extra key points; 0 point means that the proposed hypothesis
does not cover any key point in the methodology of the groundtruth hypothesis at all. Please note that the total number of key
points in the groundtruth hypothesis might be less than three, so that multiple points can be given. E.g., there’s only one key
point in the groundtruth hypothesis, and the proposed hypothesis covers the one key point, it’s possible to give 2 points, 4
points, and 5 points. In this case, we should choose score from 4 points and 5 points, depending on the existence and quality
of extra key points. ’Leveraging a key point similarly as in the methodology of the groundtruth hypothesis’ means that in the
proposed hypothesis, the same (or very related) concept (key point) is used in a similar way with a similar goal compared to
the groundtruth hypothesis (not necessarily for the proposed hypothesis to be exactly the same with the groudtruth hypothesis
to be classified as ’similar’). When judging whether an extra key point has apparent flaws, you should use your own knowledge
to judge, but rather than to rely on the count number of pieces of extra key point to judge.
Please evaluate the proposed hypothesis based on the groundtruth hypothesis.
---

**Groundtruth Hypothesis**:
{groundtruth}

**Proposed Hypothesis**:
{generated}

**The key points in the groundtruth hypothesis are:**
{keypoints}
---

Please evaluate the proposed hypothesis based on the groundtruth hypothesis, and give a score. (response format: 'Reason: 
Matched score: <score>
')
"""
    # print("evaluation prompt:  ",prompt)
    response = gpt.chat.completions.create(
        model="gpt-4o",
        
        messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                    ],
        temperature=0,
    )
    content = response.choices[0].message.content.strip()
    score = 0
    reason = ""
    for line in content.split("\n"):
        if "Matched score" in line:
            match = re.search(r"Matched score:\s*([0-5](?:\.\d+)?)", line)
            if match:
                score = float(match.group(1))
            else:
                print("⚠️ Unable to extract score from line:", line)
                score = -1  # or raise an error
        elif "Reason" in line:
            reason = line.split(":", 1)[1].strip()
    # print(f"Score: {score}, Reason: {reason}")
    return score, reason

def evaluate_all(gpt, generated_hyps: List[str], groundtruth_hyp: str, keypoints: str = "") -> List[dict]:
    results = []
    for hyp in generated_hyps:
        score, reason = evaluate_hypothesis_pair_2(gpt, hyp, groundtruth_hyp, keypoints)
        results.append({
            "generated_hypothesis": hyp,
            "groundtruth_hypothesis": groundtruth_hyp,
            "score": score,
            "reason": reason
        })
    return results

def evaluate_all_normal(gpt, generated_hyps: List[str], groundtruth_hyp: str, keypoints: str = "") -> List[dict]:
    results = []
    for hyp in generated_hyps:
        score, reason = evaluate_hypothesis_pair(gpt, hyp, groundtruth_hyp, keypoints)
        results.append({
            "generated_hypothesis": hyp,
            "groundtruth_hypothesis": groundtruth_hyp,
            "score": score,
            "reason": reason
        })
    return results

def evaluate_batch_hypotheses(gpt, generated_hyps: List[str], groundtruth_hyp: str, keypoints: str = "") -> List[dict]:
    hypotheses_block = "\n".join([f"{i+1}. {hyp}" for i, hyp in enumerate(generated_hyps)])
    
    prompt = f"""
You are helping to evaluate the quality of a proposed research hypothesis in Chemistry by a phd student. The groundtruth
hypothesis will also be provided to compare. Here we mainly focus on whether the proposed hypothesis has covered the key
points in terms of the methodology in the groundtruth hypothesis. You will also be given a summary of the key points in
the methodology of the groundtruth hypothesis for reference. Please note that for the proposed hypothesis to cover one key
point, it is not necessary to explicitly mention the name of the key point, but might also can integrate the key point implicitly
in the proposed method. The evaluation criteria is called ’Matched score’, which is in a 6-point Likert scale (from 5 to 0).
Particularly, 5 points mean that the proposed hypothesis (1) covers all the key points and leverage them similarly as in the
methodology of the groundtruth hypothesis, and (2) does not contain any extra key point that has apparent flaws; 4 points mean
that the proposed hypothesis (1) covers all the key points (or at least three key points) and leverage them similarly as in the
methodology of the groundtruth hypothesis, (2) but also with extra key points that have apparent flaws; 3 points mean that the
proposed hypothesis (1) covers at least two key points and leverage them similarly as in the methodology of the groundtruth
hypothesis, (2) but does not cover all key points in the groundtruth hypothesis, (3) might or might not contain extra key
points; 2 points mean that the proposed hypothesis (1) covers at least one key point in the methodology of the groundtruth
hypothesis, and leverage it similarly as in the methodology of groundtruth hypothesis, (2) but does not cover all key points in
the groundtruth hypothesis, and (3) might or might not contain extra key points; 1 point means that the proposed hypothesis (1)
covers at least one key point in the methodology of the groundtruth hypothesis, (2) but is used differently as in the methodology
of groundtruth hypothesis, and (3) might or might not contain extra key points; 0 point means that the proposed hypothesis
does not cover any key point in the methodology of the groundtruth hypothesis at all. Please note that the total number of key
points in the groundtruth hypothesis might be less than three, so that multiple points can be given. E.g., there’s only one key
point in the groundtruth hypothesis, and the proposed hypothesis covers the one key point, it’s possible to give 2 points, 4
points, and 5 points. In this case, we should choose score from 4 points and 5 points, depending on the existence and quality
of extra key points. ’Leveraging a key point similarly as in the methodology of the groundtruth hypothesis’ means that in the
proposed hypothesis, the same (or very related) concept (key point) is used in a similar way with a similar goal compared to
the groundtruth hypothesis (not necessarily for the proposed hypothesis to be exactly the same with the groudtruth hypothesis
to be classified as ’similar’). When judging whether an extra key point has apparent flaws, you should use your own knowledge
to judge, but rather than to rely on the count number of pieces of extra key point to judge.
Please evaluate the proposed hypothesis based on the groundtruth hypothesis.
---

**Groundtruth Hypothesis**:
{groundtruth_hyp}

**Key Methodological Points**:
{keypoints}

---

**Proposed Hypotheses to Evaluate:**
{hypotheses_block}

---

For each hypothesis, return a structured response in the format:

[Number].  
Reason: <your explanation>  
Matched score: <score>

Only return numbered items from 1 to {len(generated_hyps)}. Do not include any extra text or commentary.
"""
    
    response = gpt.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    content = response.choices[0].message.content.strip()

    # Parse results
    results = []
    pattern = r"(\d+)\.\s*Reason:\s*(.*?)\s*Matched score:\s*([0-5](?:\.\d+)?)"
    matches = re.findall(pattern, content, re.DOTALL)

    for idx, reason, score in matches:
        idx = int(idx) - 1
        results.append({
            "generated_hypothesis": generated_hyps[idx],
            "groundtruth_hypothesis": groundtruth_hyp,
            "score": float(score),
            "reason": reason.strip()
        })

    return results

def normalize_title(title):
    return re.sub(r'[^a-z0-9]+', '', title.lower().strip())


def extract_inspiration_keywords(text: str) -> list:
    parts = text.split(";")
    keywords = []
    for part in parts:
        # Match 'inspX:', 'inspX/Y/Z:', etc.
        match = re.search(r"insp[\d/]+\s*:\s*(.*)", part.strip(), re.IGNORECASE)
        if match:
            raw_keyword = match.group(1).strip()

            # Remove known ref ID or note patterns at end
            cleaned_keyword = re.sub(
                r"\s*\((ref\s*(id)?[:]?|fine online|find online).*?\)\s*$",
                "",
                raw_keyword,
                flags=re.IGNORECASE
            ).strip()

            if cleaned_keyword:
                keywords.append(cleaned_keyword)
    return keywords

def search_query(query, title_abs_reference):
    
    fetcher = GetPmids(config_path)
    fetched_pmids = fetcher.search_articles_pubmed(query)
    if fetched_pmids is not None:
        title_abs_reference.extend(fetcher.fetch_article_content(fetched_pmids))
    
    fetched_semantic = fetcher.search_semantic_scholar(query)
    response_json = fetched_semantic.json()
    top_papers = response_json.get("data", [])
    # print(f"Found {len(top_papers)} articles in Semantic Scholar for query: {query}")
    for item in top_papers:
        time.sleep(0.5)
        try:
            doi = item.get("externalIds", {}).get("DOI", None)
            response = requests.get(f"https://api.crossref.org/works/{doi}")
            if response.status_code == 200:
                message = response.json().get("message", {})
                published = message.get("published-print") or message.get("published-online") or {}
                date_parts = published.get("date-parts", [])
                if date_parts and len(date_parts[0]) > 0:
                    pub_year = date_parts[0][0]
                    if item["abstract"]:
                        title_abs_reference.append({
                            "title": item["title"],
                            "abstract": item["abstract"],
                            "doi": item["externalIds"]["DOI"],
                            "pub_year": pub_year
                        })
        except Exception as e:
            print(f"❌ Error fetching year from CrossRef for DOI {doi}: {e}")
                    
    # time.sleep(1.5)
    crossref_url = "https://api.crossref.org/works"
    crossref_params = {"query": query, "rows": 3}
    try:
        crossref_response = requests.get(crossref_url, params=crossref_params)
        if crossref_response.status_code == 200:
            items = crossref_response.json().get("message", {}).get("items", [])
            # print(f"Found {len(items)} articles in CrossRef for query: {query}")
            for item in items:
                title = item.get("title", ["No Title"])[0]
                # journal = item.get("container-title", ["No Journal"])[0]
                doi = item.get("DOI", "No DOI")
                doi = re.sub(r'\.s\d{3}$', '', doi)
                
                date_info = (
                    item.get("published-print") or 
                    item.get("published-online") or 
                    item.get("created") or {}
                )
                date_parts = date_info.get("date-parts", [])
                pub_year = date_parts[0][0] if date_parts and date_parts[0] else "Unknown Year"
                
                url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,abstract"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    abstract = data.get("abstract")
                    if abstract:  # Only append if abstract is present
                        title_abs_reference.append({
                                    "title": title,
                                    "abstract": abstract,
                                    "doi": doi,
                                    "pub_year": pub_year
                                })
                        continue  # Done for this item, no need to fallback
                    else:
                        url = f"https://api.crossref.org/works/{doi}"
                        response = requests.get(url)
                        if response.status_code == 200:
                            metadata = response.json()["message"]
                            abstract = metadata.get("abstract", None)
                            if abstract:
                                title_abs_reference.append({
                                    "title": title,
                                    "abstract": abstract,
                                    "doi": doi,
                                    "pub_year": pub_year
                                })
                            else: 
                                abstract = llm_api.get_abstract_from_title_via_llm(title)
                                if (abstract and isinstance(abstract, str) and len(abstract) > 200  ):
                                    title_abs_reference.append({
                                        "title": title,
                                        "abstract": abstract,
                                        "doi": doi,
                                        "pub_year": pub_year
                                    })        
    except Exception as e:
        print("❌ Error querying CrossRef:", e)
    return title_abs_reference
    
def search_title(title, title_abs_reference):
    crossref_url = "https://api.crossref.org/works"
    crossref_params = {"query.title": title, "rows": 3}
    resp = requests.get(crossref_url, params=crossref_params)
    
    if resp.status_code == 200:
        items = resp.json()["message"].get("items", [])
        if items:
            item = items[0]
            found_title = item.get("title", ["No Title"])[0]
            doi = item.get("DOI", "No DOI")
            doi = re.sub(r'\.s\d{3}$', '', doi)
            
            date_info = (
                item.get("published-print") or 
                item.get("published-online") or 
                item.get("created") or {}
            )
            date_parts = date_info.get("date-parts", [])
            pub_year = date_parts[0][0] if date_parts and date_parts[0] else "Unknown Year"
            
            # Try to get abstract from Semantic Scholar
            semantic_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,abstract"
            paper_resp = requests.get(semantic_url)
            abstract = paper_resp.json().get("abstract", None)
            
            if abstract:
                title_abs_reference.append({
                    "title": found_title,
                    "abstract": abstract,
                    "doi": doi,
                    "pub_year": pub_year
                })
                
            else:
                abstract = llm_api.get_abstract_from_title_via_llm(title)
                title_abs_reference.append({
                    "title": found_title,
                    "abstract": abstract,
                    "doi": doi,
                    "pub_year": pub_year
                })
                # print(f"⚠️ No abstract found for DOI: {doi}")
        else:
            print(f"No results found in CrossRef for title: {title}")
    else:
        print(f"CrossRef query failed for title: {title}")
    
    return title_abs_reference