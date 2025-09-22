import json
import openai
import os
import re
import json
import numpy as np
from collections import defaultdict
import utilities
from rag import RAG
import retrieval
import requests
from sentence_transformers import SentenceTransformer, util
import time
import pandas as pd

# 1. Read the config file.
config_path = "config.yaml"
config = utilities.read_config(config_path)
openai_api_key = config["openai_api_key"]
gpt = openai.OpenAI(api_key=openai_api_key)
print(f"Config file at {config_path} loaded...")
# Initialize API tools
llm_api = RAG(config_path)
llm_temputure = config.get("llm_temperature", 0.7)
fetcher = retrieval.GetPmids(config_path)
model = SentenceTransformer('all-MiniLM-L6-v2')
chem_research = utilities.load_excel_strip_columns(config["dataset_path"])

top_k_retrived = config.get("top_k_retrived", 5)
score_dict = {f"score_round_{i}": [] for i in range(config.get("number_of_rounds", 5))}
score_dict["scores_golden"] = []

prompt_final_hypo_selection = config.get("rag", {}).get("prompt_final_hypo_selection", "")
print(prompt_final_hypo_selection)

for idx, row in chem_research.iterrows():
    mean_before = []
    mean_after = []
    # Extract background info
    bg_question: str = row["Background Question"]
    bg_survey: str = row["Background Little Survey"]
    bg_knowledge: str = f"Research Question:\n{bg_question}\n\nBackground Survey:\n{bg_survey}"

    # Extract ground truth info
    groundtruth_hypo: str = row["Main hypothesis"]
    groundtruth_title: str = row["Title"]
    keypoints = utilities.extract_inspiration_keywords(row["Note"])

    # LLM: Extract and parse keywords for PubMed query
    bg_keywords_summary = llm_api.llm_keyowrds(bg_knowledge, temp=1)
    bg_keywords_summary_parsed = utilities.parse_json_string(bg_keywords_summary)
    bg_keywords = bg_keywords_summary_parsed['Extracted Keywords']

    # # Optionally, build PubMed query
    # queries = utilities.construct_pubmed_queries(bg_keywords)
    # Use LLM to generate query
    # queries: str = llm_api.llm_generate_pubmed_query(bg_knowledge, bg_keywords)

    # Initialize inspiration containers
    title_abs_inspiration = []
    # Search PubMed, Semantic Scholar, and CrossRef for articles related to the background question
    for query in queries:
        fetched_pmids = fetcher.search_articles_pubmed(query)
        if fetched_pmids is not None:
            title_abs_inspiration.extend(fetcher.fetch_article_content(fetched_pmids))
        fetched_semantic = fetcher.search_semantic_scholar(query, limit=top_k_retrived)
        if fetched_semantic is not None:
            title_abs_inspiration.extend(retrieval.fetch_semantic_scholar_articles(fetched_semantic))
        title_abs_inspiration.extend(retrieval.fetch_crossref_articles(query, limit=top_k_retrived, row=row))

    # Remove ground-truth title from the list of inspiration titles for reliability
    normalized_groundtruth_title = utilities.normalize_title(groundtruth_title)
    unique_refs = []
    seen_titles = set()
    for entry in title_abs_inspiration:
        # Clean and normalize the current entry's title
        title = entry["title"].strip()
        normalized_title = utilities.normalize_title(title)
        # Filter: unique, not ground-truth, and reasonable title length
        if (normalized_title not in seen_titles and
            normalized_title != normalized_groundtruth_title and
            len(normalized_title) > 10):
            seen_titles.add(normalized_title)
            unique_refs.append(entry)
    
    num_rounds = config.get("number_of_rounds", 3)
    year_start, year_end = 2015, 2023
    split_bins = num_rounds - 1
    main_bins = np.linspace(year_start, year_end + 1, split_bins + 1, dtype=int)
    round_refs = [[] for _ in range(num_rounds)]
    for entry in unique_refs:
        year = entry.get("pub_year", None)
        try:
            year = int(year)
        except (TypeError, ValueError):
            continue
        if year < year_start:
            round_refs[0].append(entry)
        else:
            for i in range(split_bins):
                if main_bins[i] <= year < main_bins[i + 1]:
                    round_refs[i + 1].append(entry)
                    break

    # Print the rest, with correct intervals from main_bins
    for i in range(split_bins):
        lower = main_bins[i]
        upper = main_bins[i + 1] - 1
    first_round = True
    # print("list of retrived articles:")
    prev_inspiration_texts = []
    previous_hypos = []
    feedbacks = []
    memory_summary = ""
    for round_num, round_data in enumerate(round_refs):
        if len(round_data) == 0:
            print(f"No articles found for round {round_num}. Skipping to next round.")
            continue
        if round_num == 0:
            memory_summary = ""  # No prior memory in the first round
        else:
            # Summarize prior inspiration, hypotheses, and feedbacks into a contextual memory block
            memory_prompt = f"""
            Summarize the following prior inspiration materials, hypotheses, and feedback into a structured and concise paragraph.
            Focus on recurring scientific mechanisms, materials, and core themes relevant to the research question.
            Previous memory:
            {memory_summary}
            Previous hypotheses:
            {previous_hypos}   
            Feedback received:
            {feedbacks}
            New inspirations:
            {prev_inspiration_texts} 
            """
            memory_summary = gpt.chat.completions.create(
                model="gpt-4o",
                temperature=0.5,
                messages=[{"role": "user", "content": memory_prompt}]
            ).choices[0].message.content.strip()
        inspiration_texts = [] 
        scores = []
        scores_prev = []
        new_hypos = []
        idx_retrived = 0
        feedbacks_this_round = []
        for title_obj in round_data: 
            curr_title = title_obj['title']
            # print(curr_title)
            curr_abs = title_obj['abstract']

            inspiration = (
                f"Next we will introduce inspiration candidate {idx_retrived}. "
                f"Title: {curr_title}; "
                f"Abstract: {curr_abs} "
                f"The introduction of inspiration candidate {idx_retrived} has come to an end.\n"
            )
            idx_retrived += 1
            inspiration_texts.append(inspiration)
        if first_round:
            first_round = False 
            # First round: generate hypotheses from background question, survey
            prompt_round_0 = f"""
You are an expert scientific research strategist specializing in generating testable, high-impact hypotheses grounded in mechanistic reasoning.

Your task is to formulate a set of original and technically precise research hypotheses based on three sources of input:
1. A clearly defined background research question.
2. A brief survey of the current scientific landscape relevant to that question.
3. A set of structured summaries of related research papers, provided as inspiration.

You are now in **Stage 3** of the hypothesis generation process:
1. Define the problem → 2. Gather inspiration → **3. Formulate new hypotheses** → 4. Design experiments.

---

### INPUT MATERIAL

**1. Research Question**
{bg_question}

**2. Background Survey**
{bg_survey}

**3. Inspiration Materials**
These are summaries of relevant scientific papers retrieved from multiple time periods. They may contain experimental results, methods, theories, or limitations that can inform your hypothesis.

{inspiration_texts}

---

### YOUR TASK

Generate **8 to 10 novel, technically detailed, and mechanistically grounded research hypotheses** that directly address the research question by **combining or reinterpreting the information above**. Each hypothesis should propose a new mechanism, method, or framework that plausibly advances the field.

---

### CRITICAL GUIDELINES

** Scientific Novelty**
- Do not paraphrase or repackage content from the inspiration sources.
- Instead, combine ideas across papers, reinterpret results in new contexts, or introduce bridging mechanisms between unrelated phenomena.
- Hypotheses that simply extend an existing method by "adding X" or "optimizing Y" are too weak unless justified by a novel mechanism.

** Mechanistic Specificity**
- Explicitly describe the **underlying mechanism** or **scientific principle** by which the hypothesis would work.
- You must explain *how* the proposed method changes the system behavior (e.g., via ion transport enhancement, interfacial stabilization, entropy reduction).

** Technical Testability**
- Each hypothesis should be experimentally testable with available techniques or setups.
- Whenever appropriate, include details like material systems, experimental conditions, or expected intermediate outcomes.

** Format & Precision**
- Use concise, numbered format (1 to 10).
- Each hypothesis should be **2–4 sentences**, avoiding generalities or filler.

---


**Output Format (Strict):**
Provide only a numbered list of the generated hypotheses. Each hypothesis should be concise, scientifically precise, and clearly worded.

1.  <Hypothesis 1>
2.  <Hypothesis 2>
3.  <Hypothesis 3>
... (up to 10 hypotheses)
                """
            
            response = gpt.chat.completions.create(
                model="gpt-4o",
                temperature=llm_temputure,
                messages=[{"role": "user", "content": prompt_round_0}]
            )
            generated_hypos = utilities.extract_numbered_hypotheses(response.choices[0].message.content.strip())
            hypos = generated_hypos
        else:
            # Subsequent rounds: refine hypotheses based on feedback and new inspirations
            for hypo in hypos:
                scoring_prompt = f"""
You are known as a diligent and super harsh reviewer in Chemistry and Material Science that will spend much time to find flaws when reviewing and therefore usually gives a relatively much lower score than other reviewers. But when you meet with a hypothesis you truly appreciate, you don't mind to give it good scores. Given a not yet peer reviewed research hypothesis in the Chemistry and Material Science domain, try to evaluate the research hypothesis from four research aspects and give score according to evaluation guidelines provided below. All four aspects should be evaluated in a 5 point scale." + f"\nAspect 1: Validness. \n5 points: The hypothesis is a logical next step from current research, strongly supported by theory, perhaps with some indirect experimental evidence or highly predictive computational results. The experimental verification seems straightforward with a high probability of confirming the hypothesis; 4 points: Here, the hypothesis is well-rooted in existing theory with some preliminary data or computational models supporting it. It extends known science into new but logically consistent areas, where experiments are feasible with current technology, and there's a reasonable expectation of positive results; 3 points: This hypothesis is within the realm of theoretical possibility but stretches the boundaries of what's known. It might combine existing knowledge in very novel ways or predict outcomes for which there's no direct evidence yet. There's a conceptual framework for testing, but success is uncertain; 2 points: While the hypothesis might be grounded in some theoretical aspects, it significantly deviates from current understanding or requires conditions or materials that are currently impossible or highly improbable to achieve or synthesize; 1 point: The hypothesis proposes concepts or outcomes that are not only unsupported by current theory but also contradict well-established principles or data. There's no clear path to experimental testing due to fundamental theoretical or practical barriers. " + f"\nAspect 2: Novelty. \n5 points: This level of novelty could fundamentally alter our understanding of Chemistry and Material Science or create entirely new fields. It often involves predictions or discoveries that, if proven, would require a significant overhaul of existing Chemistry and Material Science theories; 4 points: The hypothesis significantly departs from established norms, potentially redefining how certain Chemistry and Material Science phenomena are understood or applied. It might involve entirely new materials or theoretical frameworks; 3 points: This level involves a hypothesis that could potentially lead to new insights or applications. It might challenge minor aspects of current theories or introduce new methodologies or materials; 2 points: The hypothesis introduces a new angle or method within an established framework. It might involve known compounds or reactions but in contexts or combinations not previously explored; 1 point: The hypothesis involves minor tweaks or applications of well-known principles or techniques. It might slightly extend existing knowledge but doesn't introduce fundamentally new concepts. " + f"\nAspect 3: Significance. \n5 points: This hypothesis could fundamentally change one or more branches of Chemistry and Material Science. It might introduce entirely new principles, theories, or methodologies that redefine the boundaries of Chemistry and Material Science; 4 points: This hypothesis challenges current understanding or introduces a concept that could lead to substantial changes in how a particular area of Chemistry and Material Science is viewed or applied. It might lead to new technologies or significant theoretical advancements; 3 points: this hypothesis proposes something new or an innovative approach that could lead to noticeable advancements in a specific area of Chemistry and Material Science. It might open new avenues for research or application but doesn't revolutionize the field; 2 points: This hypothesis might offer a small variation or incremental improvement on existing knowledge. It could potentially refine a known concept but doesn't significantly alter the field; 1 point: The hypothesis addresses a very narrow or already well-established aspect of Chemistry. It might confirm what is already known without adding much new insight." + f"\nAspect 4: Potential. \n5 points: The hypothesis, while potentially intriguing now, holds the promise of being revolutionary with the addition of a key methodological component. This could introduce entirely new concepts or fields, fundamentally changing our understanding or capabilities in Chemistry and Material Science; 4 points: The hypothesis, though promising, could be transformative with the right methodological enhancement. This enhancement might lead to groundbreaking discoveries or applications, significantly advancing the field; 3 points: The hypothesis, while interesting in its current form, could be significantly elevated with the right methodological addition. This might lead to new insights or applications that go beyond the initial scope; 2 points: The hypothesis currently offers some value but has the potential for more substantial contributions if enhanced with a new methodological approach. This could lead to incremental advancements in understanding or application; 1 point: The hypothesis, as it stands, might be straightforward or well-trodden. Even with methodological enhancements, it's unlikely to significantly expand current knowledge or applications beyond minor improvements. \
\nThe hypothesis is:\n", "\nPlease give a response to the initial question on scoring the hypothesis from four aspects. Remember that you are a diligent and harsh reviewer. (response format: 'Concise reason for validness score: \nValidness score: \nConcise reason for novelty score: \nNovelty score: \nConcise reason for significance score: \nSignificance score: \nConcise reason for potential score: \nPotential score: \n').
The hypothesis is:
"{hypo}"
                """                
                response = gpt.chat.completions.create(
                    model="gpt-4o",
                    temperature=llm_temputure,
                    messages=[{"role": "user", "content": scoring_prompt}]
                )
                feedback = response.choices[0].message.content.strip()
                feedbacks_this_round.append(feedback)
                # print(f"Feedback for hypothesis: {feedback}"

                if len(inspiration_texts) > 2:
                    prompt_inspiratiion_selection = f"""
You are assisting in a scientific hypothesis refinement task.

A research hypothesis was generated from background context and prior literature. It has received critical feedback, and your job is to identify new papers that can help **revise or strengthen** it.

---

## TASK CONTEXT

**Research Question:**  
{bg_question}

**Background Summary:**  
{bg_survey}

**Current Hypothesis:**  
{hypo}

**Feedback on Hypothesis:**  
{feedback}

**Candidate Inspiration Papers:**  
{inspiration_texts}

**Accumulated Memory Summary**
{memory_summary}
---

## YOUR TASK

Select the **two best inspiration papers** that could help address the issues identified in the feedback or improve the current hypothesis. Each selected paper should:

- Introduce a **mechanism**, method, or concept** that directly relates to fixing one or more weaknesses in the hypothesis.
- Offer **novel direction**, clarify a **missing mechanism**, or support a more **specific or plausible design**.

---

## OUTPUT FORMAT (STRICT)

Respond in the format below with exactly two selections:
(response format: 'Title: 
Reason: 
Title: 
Reason: 
')
    """
                    response = gpt.chat.completions.create(
                        model="gpt-4o",
                        temperature=llm_temputure,
                        messages=[{"role": "user", "content": prompt_inspiratiion_selection}]
                    )        
                    response_text = response.choices[0].message.content.strip()
                    pattern = r"Title:\s*(.*?)\s*Reason:\s*(.*?)(?=\nTitle:|\Z)"
                    matches = re.findall(pattern, response_text, re.DOTALL)
                    inspiration_texts = [{"title": title.strip(), "reason": reason.strip()} for title, reason in matches]
                else:
                    
                    inspiration_texts = inspiration_texts
                prev_inspiration_texts.append(inspiration_texts)

                prompt_new_round = f"""
You are a scientific reasoning assistant specializing in hypothesis refinement.

Your task is to update a previously proposed research hypothesis using new feedback and inspirations. If the original hypothesis already meets high standards of novelty, specificity, validity, and significance (per feedback), you may return it unchanged. Otherwise, improve it by addressing weaknesses and integrating new inspiration materials.

---

### INPUTS

**1. Background Research Question**  
{bg_question}

**2. Background Survey**  
{bg_survey}

**3. Hypothesis from Round {round_num - 1}**  
{hypo}

**4. Feedback on Hypothesis (Round {round_num})**  
The hypothesis was evaluated based on: **Specificity**, **Novelty**, **Validity**, and **Significance**. Below is the structured feedback:
{feedback}

**5. New Inspiration Articles (Round {round_num})**  
This paper was selected specifically to help improve the current hypothesis based on feedback.
{inspiration_texts}

**6. Accumulated Memory Summary**
{memory_summary}
---

### YOUR TASK

Based on the inputs above, generate **one revised research hypothesis** that:
- Clearly addresses the core research question
- Fixes issues identified in the feedback (e.g., vagueness, lack of novelty or mechanism)
- Integrates at least one idea, mechanism, or method from the new inspiration articles
- Uses clear, mechanistic reasoning to describe **how** and **why** the hypothesis works
- Is concise (2–5 sentences) and testable with plausible experimental design

Avoid simply rephrasing the original hypothesis. If reusing content, justify it by alignment with positive feedback. If rewriting, make the improvements clearly traceable to the feedback and inspiration.

---

### OUTPUT FORMAT (STRICT)

Respond with a **single, numbered hypothesis** like this:

1. <Your revised hypothesis here>
"""
                # Aggregate data into a single prompt block
                response = gpt.chat.completions.create(
                    model="gpt-4o",
                    temperature=llm_temputure,
                    messages=[{"role": "user", "content": prompt_new_round}]
                )
                new_hypo = utilities.extract_numbered_hypotheses(response.choices[0].message.content.strip())
                try:
                    new_hypos.append(new_hypo[0])
                except Exception as e:
                    if isinstance(new_hypo, str):
                        new_hypos.append(new_hypo.strip())
        
            previous_hypos += new_hypos
            hypos = new_hypos
            feedbacks += feedbacks_this_round
        
        for hypo in hypos: 
            result = utilities.evaluate_all(gpt, [hypo], groundtruth_hypo, keypoints)
            if result and isinstance(result[0].get('score'), (int, float)):
                scores.append(result[0]['score'])
            else:
                continue 
        score_dict[f"score_round_{round_num}"].append(np.mean(scores))

    # Final selection of the best hypothesis
    selection_prompt = prompt_final_hypo_selection.format(
        bg_question=bg_question,
        bg_survey=bg_survey,
        hypos=hypos,
        memory_summary=memory_summary
    )
    response = gpt.chat.completions.create(
        model="gpt-4o",
        temperature=llm_temputure,
        messages=[{"role": "user", "content": selection_prompt}]
    )
    selected_hypo = utilities.extract_numbered_hypotheses(response.choices[0].message.content.strip())
    selected_hypo_score = utilities.evaluate_all(gpt, [selected_hypo], groundtruth_hypo, keypoints)[0].get('score')
    score_dict["scores_golden"].append(selected_hypo_score)
    validation_prompt = f"""
You are assisting Chemistry scientists by providing detailed feedback on their newly proposed research hypotheses. The goal is to help them refine these hypotheses for potential publication in a top Chemistry venue such as *Nature Chemistry* or *Science*.

As you know, to meet the standards of such venues, a strong research hypothesis should satisfy the following four criteria:

1. **Specificity**: The hypothesis should provide sufficient methodological detail so that other researchers can clearly understand what the proposed method is and how it will be carried out in practice, leaving no room for confusion or misinterpretation.

In particular, if the hypothesis involves introducing a new concept or component into an existing method, it should not stop at describing what the new concept is — it must also explain how the new concept will be concretely integrated, applied, or operationalized within the method.

Whenever possible, please also suggest specific parameters, conditions, or operational procedures (e.g., algorithm settings, material properties, experimental setups) that would enable researchers to directly test or implement the hypothesis in a laboratory or experimental environment.

2. **Novelty**: The hypothesis should propose a new idea, mechanism, or approach that has not been reported or established in existing literature.

Please carefully assess whether the core idea of the hypothesis — including its key concepts, methods, or combinations of techniques — has already been proposed or widely studied. If any part of the hypothesis appears similar to prior work, please point it out and explain why it may not be sufficiently novel.

Conversely, if the hypothesis is novel, please briefly explain what makes it distinct from existing approaches, such as introducing a new principle, a previously unexplored mechanism, or a new combination of known techniques in an original way.

3. **Validity / Effectiveness**: The hypothesis should be testable, verifiable, and practically feasible within real-world Chemistry experimental settings.

Please evaluate whether the hypothesis can, in principle, be validated through Chemistry experiments — assuming sufficient experimental resources. Consider whether the proposed method relies on reasonable assumptions, whether each step is technically executable, and whether the expected outcomes are measurable in a real-world setting.

Although we currently do not have access to lab experiments, please assess the validity based on your knowledge and understanding of Chemistry, and highlight any potential challenges, limitations, or conditions that may affect the experimental verification of the hypothesis.

4. **Significance**: If possible, the hypothesis should have the potential for meaningful impact in the research community, such as advancing scientific understanding or opening new research directions. It is not necessary for the hypothesis to be groundbreaking, but it should ideally contribute to the field in a way that is recognized as significant by peers.

Please provide constructive feedback on whether the given hypothesis meets these four criteria. If any aspect is lacking, please explain why, and suggest concrete ways to improve it.

**Important**: Your feedback should focus on improving the *methodological content* of the hypothesis — that is, how to make the hypothesis itself more specific, novel, valid, and significant — rather than suggesting ways to improve the writing or description of these qualities.

The hypothesis is:
"{selected_hypo}"

Please give a response to the initial question on determining whether the research hypothesis needs to be more specific, novel, valid, and significant. If so, what are your advice to be more specific, novel, valid, and significant?
"""
    response = gpt.chat.completions.create(
        model="gpt-4o",
        temperature=llm_temputure,
        messages=[{"role": "user", "content": scoring_prompt}]
    )
    # Prompt optimization based on internal iteration
    feedback = response.choices[0].message.content.strip()
    refine_prompt = f"""
    You are a prompt engineer tasked with improving a prompt for selecting the best research hypothesis from a set of generated hypotheses.

    **Previous Prompt:**
    {prompt_final_hypo_selection}

    **Selection Performance:**
    - Selected Hypothesis: {selected_hypo}
    - Feedback on the Selected Hypothesis: {feedback}


    **Task:**
    Revise the prompt to improve its ability to select the hypothesis that:
    - Maximizes addressing issues mentioned feedback of the selected hypothesis in terms of mechanistic detail, novelty, validity, and significance (** Note that the groundtruth hypothesis will not be shown in prompt**).
    - Avoids vague or generic selection criteria.
    - Emphasizes clear, specific reasoning for the selection.

    **Output:**
    Return the improved prompt template without explanation.
    """
    response = gpt.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[{"role": "user", "content": refine_prompt}]
    )
    prompt_final_hypo_selection = response.choices[0].message.content.strip()
            
print("\n======================== Summary of Average Hypothesis Scores per Round ========================")
for i in range(config.get("number_of_rounds", 5)):
    scores = score_dict[f"score_round_{i}"]
    round_mean = np.mean(scores) if scores else 0
    print(f"Round {i} - Mean Score across all background questions: {round_mean:.2f} (from {len(scores)} questions)")
# Print golden scores
golden_scores = score_dict["scores_golden"]
golden_mean = np.mean(golden_scores) if golden_scores else 0
print(f"Golden - Mean Score: {golden_mean:.2f} (from {len(golden_scores)} questions)")
print("===============================================================================================")
