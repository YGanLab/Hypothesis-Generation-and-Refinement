import yaml
import os
import re
import time

# from openai import OpenAI
import openai
from anthropic import Anthropic
from typing import List, Tuple
import json


class RAG:
    def __init__(self, config_file):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' not found.")
        try:
            with open(config_file, 'r', encoding="utf-8") as file:
                config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
            
        if 'keyword_extraction' not in config:
            raise KeyError("Missing required key 'rag' in config file.")

        self.params = config.get('rag', {})            
        self.api_to_use = config.get('use_api', "")
        
        # Initialize OpenAI instance
        if self.api_to_use == 'openai':
            openai_api_key = config.get('openai_api_key', None)
            if openai_api_key:
                self.llm = openai.OpenAI(api_key = openai_api_key)
                print('Connected to OpenAI API for RAG...')
            else:
                print('Warning: OpenAI API key is not provided!')

        elif self.api_to_use == 'claude':
            # Initialize Claude (Anthropic) instance
            claude_api_key = config.get('claude_api_key', None)
            if claude_api_key:
                self.llm = Anthropic(api_key=claude_api_key)
                print('Connected to Claude API for RAG...')
            else:
                print('Warning: Claude API key is not provided!')
                
        else:
            print(f'The api {self.api_to_use} is not supported!')
            
    def llm_summarize(self, documents, temp):
        model_name = self.params.get('model_name', None)
        # temperature = self.params.get('temperature_summarize', 1)
        temperature = temp
        
        prompt_summarize = self.params.get('prompt_summarize', None)
        prompt_summarize = prompt_summarize.format(documents=documents)
        
        if not prompt_summarize:
            print('The prompt_summarize is not provided!')
            return
        
        return self._get_llm_response_backup(model_name, prompt_summarize, temperature)

    def llm_keyowrds(self, documents, temp):
        model_name = self.params.get('model_name', None)
        # temperature = self.params.get('temperature_keywords', 1)
        temperature = temp
        
        prompt_keywords = self.params.get('prompt_keywords', None)
        prompt_keywords = prompt_keywords.format(documents=documents)

        if not prompt_keywords:
            print('The prompt_keywords is not provided!')
            return
        
        return self._get_llm_response_backup(model_name, prompt_keywords, temperature)
        
            
    def _get_llm_response(self, model_name, context, user_prompt, temperature=1):
        """
        Identifies the model provider, determines model type, and makes the API call.
        """
        # Identify providers and supported models
        providers = set(self.llm.show_provider())
        supported_models = {pvd: set(self.llm.show_models(pvd)) for pvd in providers}

        # Determine model type
        model_type = next((k for k, v in supported_models.items() if model_name in v), None)

        if not model_type:
            print(f'Model {model_name} not found in available providers.')
            return

        if model_type not in {'groq', 'openai', 'claude'}:
            print(f'Model type {model_name} not tested for keyword extraction!')
            return 

        # Prepare request parameters and call the API
        if model_type in {'groq', 'openai'}:
            param = {
                "model": model_name,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "system",
                        "content": 'You are given the following context. ' + 'Context: ' + context
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            }
            res = self.llm.completion(model_type, **param)
            return res['choices'][0]['message']['content']
        
        elif model_type == 'claude':
            param = {
                "model": model_name,
                "max_tokens": 1024,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": 'You are given the following context: ' + context + user_prompt
                    }
                ]
            }
            res = self.llm.completion(model_type, **param)
            return res['content'][0]['text']
        
        else:
            print(f'Something is wrong with model {model_name}')
            return
        

    def _get_llm_response_backup(self, model_name, user_prompt, temperature=1):
        MAX_RETRIES = 5
        """
        Identifies the model provider, determines model type, and makes the API call with retry mechanism.
        """
        if not user_prompt:
            print("Error: The user prompt is not provided!")
            return None

        if self.api_to_use not in ["openai", "claude"]:
            print("Error: Unsupported API type. Choose 'openai' or 'claude'")
            return None

        for attempt in range(MAX_RETRIES):
            try:
                if self.api_to_use == "openai":
                    model_name = 'gpt-4o'
                    param = {
                        "model": model_name,
                        "temperature": temperature,
                        "messages": [
                            {"role": "system", "content": f'You must follow the user instructions.'},
                            {"role": "user", "content": f'{user_prompt}'}
                        ]
                    }
                    response = self.llm.chat.completions.create(**param)
                    return response.choices[0].message.content

                elif self.api_to_use == "claude":
                    model_name = 'claude-3-5-sonnet-20241022'
                    param = {
                        "model": model_name,
                        "max_tokens": 1024,
                        "temperature": temperature,
                        "messages": [
                            {"role": "user", "content": f'You must follow the user instructions. {user_prompt}.'}
                        ]
                    }
                    response = self.llm.messages.create(**param)
                    return response.content[0].text

                else:
                    print(f'The API {self.api_to_use} is not supported!!')

            except Exception as e:
                print(f"Attempt {attempt + 1} failed in _get_llm_response_backup: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff before retrying

        print("Max retries reached. Returning None.")
        return None  # Graceful failure after max retries
    
    def get_abstract_from_title_via_llm(self, title: str) -> str:
        prompt = f"""You are a scientific assistant tasked with inferring the likely abstract of a scientific paper based solely on its title.

Title:
"{title}"

Please provide a plausible abstract that matches the style and content of real scientific abstracts. Keep it concise and structured in scientific language. Avoid guessing beyond the scope of the title.
"""
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    
    
    def llm_generate_pubmed_query(self, bg, keywords_reference: dict, temperature: float = 0.5, num_queries: int = 5) -> List[str]:
        """
        Generate multiple flexible and effective PubMed queries using GPT based on extracted keywords.
        """
        keyword_str = ""
        for category, keywords in keywords_reference.items():
            keyword_str += f"- {category}: {', '.join(keywords)}\n" # Using bullet points for readability

        prompt = f"""
        """

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )

        raw_text = response.choices[0].message.content.strip()
        pattern = r"\d+\.\s*(.*)"
        matches = re.findall(pattern, raw_text)
        return [match.strip() for match in matches if match.strip()]


    def evaluate_hypothesis_llm(self, hypo, max_retries=3, delay=2):
        """Use LLM to evaluate a hypothesis in terms of novelty, specificity, and plausibility.
        Returns: avg_score (float), details (dict), raw_content (str)
        """
        prompt = f"""
        You are an expert scientific peer reviewer. Your task is to critically evaluate a given scientific hypothesis based on three key dimensions: **novelty**, **specificity**, and **plausibility**.

        **Hypothesis to Evaluate:**
        "{hypo}"

        **Evaluation Criteria:**
        * **Novelty**: How new, original, or surprising is the hypothesis? Does it propose a genuinely new idea, or is it a rehash/obvious extension of existing work?
        * **Specificity**: How precisely is the hypothesis stated? Is it clear, unambiguous, and testable? Does it avoid vague language?
        * **Plausibility**: How scientifically sound and believable is the hypothesis? Is it consistent with current scientific understanding and evidence? Is it conceptually logical?

        **Instructions:**
        1.  For each dimension, assign a score from **1 (very poor)** to **10 (excellent)**.
        2.  Provide a **concise, one-sentence justification** for each score.

        **Output Format (Strict JSON ONLY):**
        ```json
        {{
            "novelty": {{
                "score": <integer from 1 to 10>,
                "reason": "<your concise one-sentence justification for novelty>"
            }},
            "specificity": {{
                "score": <integer from 1 to 10>,
                "reason": "<your concise one-sentence justification for specificity>"
            }},
            "plausibility": {{
                "score": <integer from 1 to 10>,
                "reason": "<your concise one-sentence justification for plausibility>"
            }}
        }}
        ```
        **IMPORTANT:** Your response must be **ONLY** the valid JSON object. Do not include any introductory or concluding text, additional explanations, or markdown outside the JSON block.
        """

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()

        # Remove ```json ... ``` if present
        match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, re.DOTALL)
        cleaned = match.group(1).strip() if match else raw

        parsed = json.loads(cleaned)
        avg = sum(parsed[k]["score"] for k in ["novelty", "specificity", "plausibility"]) / 3.0

        return avg, parsed, raw

    
    def generate_next_round_prompt_via_llm(
        self,
        current_hypothesis: str,
        past_revisions: list,
        llm_score: float,
        similarity_texts: list,
        round_num: int,
        background_question: str
    ) -> str:
        system_prompt = """
    You are a scientific assistant that helps write prompt instructions to improve research hypotheses in a multi-round refinement system.
    You will receive:
    - Background question
    - Previous versions of the hypothesis and scores
    - Current hypothesis and similarity warnings
    - Ground-truth-inspired hypothesis (hidden from refinement but useful for direction)

    Your goal is to generate a new **prompt** that will be sent to another LLM to revise the hypothesis further.
    This new prompt must:
    - Encourage novelty, plausibility, and specificity
    - Avoid repeating past content
    - Reference scores/similarity issues to steer change
    - Avoid generic suggestions
    Return only the generated prompt.
    """

        similarity_block = "\n".join([f"- {text} (score={score:.2f})" for text, score in similarity_texts[:3]]) if similarity_texts else "None"
        revision_block = "\n".join(
            f"Round {i}: \"{r['text']}\" | Score: {r['score']} "
            for i, r in enumerate(past_revisions)
        )

        user_prompt = f"""
    Background question:
    {background_question}

    Current hypothesis (Round {round_num}):
    \"{current_hypothesis}\"

    Similarity to corpus:
    {similarity_block}

    Previous revisions and scores:
    {revision_block}
    
    LLM average scored based on novelty, specificity, and plausibility (from 1 (very poor) to 10 (excellent)):
    \"{llm_score}\"

    Write a prompt to refine the current hypothesis for the next round.
    """
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        base_prompt = response.choices[0].message.content.strip()

        # Build a prefix that includes hypothesis history
        revision_block = "\n".join([
            f"Round {i} Hypothesis:\n\"{r['text']}\"\n→ Score: {r['score']}"
            for i, r in enumerate(past_revisions)
        ])

        # Inject history into the generated prompt
        full_prompt = f"""
        You are improving a scientific hypothesis based on previous rounds.

        Background question:
        {background_question}

        Prior hypothesis revisions:
        {revision_block}

        Instruction:
        {base_prompt}

        Please return only the revised hypothesis.
        """

        return full_prompt.strip()


    def llm_generate_pubmed_query(self, bg_question_survey, keywords_reference: dict, temperature: float = 0.5, num_queries: int = 8) -> List[str]:
        """
        Generate multiple flexible and effective PubMed queries using GPT based on extracted keywords.
        """
        # keyword_str = ""
        # for category, keywords in keywords_reference.items():
        #     keyword_str += f"- {category}: {', '.join(keywords)}\n" # Using bullet points for readability

        prompt = f"""
        You are a creative Principal Scientist AND a meticulous PubMed search expert. Your goal is to generate concrete, syntax-correct search queries that execute the creative search strategies below, based *only* on the provided background.

        **Background Context:**
        "{bg_question_survey}"

        **Your Task:**
        Generate a total of **8** distinct, PubMed-compatible search queries — **exactly 2 per strategy (A through D)**. 
        Each query must clearly align with one of the strategies and cover a unique concept from the background.


        ---
        ### **Mandatory Search Strategies**

        * **Strategy A: Find the Fundamental METHOD:**
            * **Instruction:** Create at least two distinct queries focused ONLY on **methods** for making materials with desired properties (like strength, stability, or conductivity) that are relevant to the background. Identify these methods from the background survey or from general scientific knowledge about the problems described.

        * **Strategy B: Find the Fundamental CONCEPT:**
            * **Instruction:** Create at least two distinct queries focused ONLY on the underlying **scientific principles** (like specific chemical interactions or physical phenomena) mentioned or implied in the background.

        * **Strategy C: Find ANALOGOUS Problems:**
            * **Instruction:** Create at least two distinct queries that look for solutions to the core trade-offs described in the background, but in **different scientific fields**.

        * **Strategy D: Retrieve FOUNDATIONAL or CLASSICAL Work:**
            * **Instruction:** Create at least two distinct queries that aim to retrieve **early**, **foundational**, or **classic studies or methods** relevant to the background problem. These queries should focus on retrieving **pioneering work**, **seminal discoveries**, or **widely cited frameworks** that are considered the backbone of the scientific domain mentioned in the background.
        ---
        ### **Mandatory PubMed Syntax Rules**
        (This section remains the same)
        1.  **Use Boolean Operators:** ...
        2.  **Use Parentheses:** ...
        3.  **Combine Synonyms:** ...
        4.  **Decouple Concepts:** ...

        ---
        Your goal is to **maximize mechanistic and material diversity**. Each query must focus on a **different core mechanism**, material system, or problem aspect described in the background. Avoid repeating chemical names or structural elements unless justified by different goals.
        ⚠️ Do NOT use generic or overused terms like “hydrogel”, “PAM-PVA”, or “vanadium” unless clearly justified by the background. Your queries must explore **new angles**.

        ---
        ### **Example of How to Apply the Strategies**

        To be perfectly clear, here is a complete example for a HYPOTHETICAL background about hydrogel strength and thermoelectricity. Use this as a guide for your reasoning process, but **DO NOT use these specific keywords unless the new background context supports it.**

        * **Example Background Problem:** Designing a hydrogel that is mechanically robust but also has high thermoelectric efficiency.
        * **Applying Strategy A (Method):** The background mentions "double-network designs" and "ice-templating." Good queries would be: `(("double network hydrogel" OR "DN hydrogel") AND ("mechanical propert*" OR toughness))` AND `(("freeze casting" OR "ice templating") AND ("anisotropic" OR "aligned structure"))`.
        * **Applying Strategy B (Concept):** The background mentions the "Hofmeister series" and "salting-in." Good queries would be: `("Hofmeister series" AND ("ion-polymer interaction" OR stability))` AND `(("salting out" OR "salting in") AND (solubility OR "phase separation"))`.
        * **Applying Strategy C (Analogy):** The core problem is strength + flexibility. Other fields with this problem are soft robotics and tissue engineering. Good queries would be: `(("soft robotics" OR "actuator") AND ("mechanical robustness" OR elasticity))` AND `(("tissue engineering" OR "biomaterial scaffold") AND ("mechanical properties" OR durability))`.
        * **Applying Strategy D (Foundational or Classical Work):** You want to retrieve early or seminal studies relevant to thermoelectric hydrogels.  Good queries would be: `(("thermoelectric hydrogel") AND ("seminal" OR "classical" OR "foundational"))`  `(("thermogalvanic" OR "ionic thermocell") AND ("first report" OR "early study"))`  Or using date filters: `("thermoelectric materials") AND ("mechanical properties") AND ("1970"[Date - Publication] : "2000"[Date - Publication])`
        ---
        Generate only the numbered list of queries. Do not include any introductory or concluding text, explanations, or additional conversational remarks.

        1. <Query 1>
        2. <Query 2>
        3. <Query 3>
        ... (up to {num_queries} queries)
        """
        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )

        raw_text = response.choices[0].message.content.strip()
        pattern = r"\d+\.\s*(.*)"
        matches = re.findall(pattern, raw_text)
        return [match.strip() for match in matches if match.strip()]