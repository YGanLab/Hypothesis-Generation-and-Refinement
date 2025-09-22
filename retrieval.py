import requests
import time
import os
import yaml
import xml.etree.ElementTree as ET
import re
from lxml import etree

class GetPmids:
    def __init__(self, config_file):
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' not found.")
        try:
            with open(config_file, 'r', encoding="utf-8") as file:
                config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
        
        if 'document_retrieval' not in config:
            raise KeyError("Missing required key 'document_retrieval' in config file.")

        self.params = config.get('document_retrieval', {})
        
    def search_articles(self, keywords):
        search_mode = self.params.get('search_mode')

        if search_mode not in {'isearch', 'pubmed'}:
            print("Error: The search mode can only be either 'isearch' or 'pubmed'.")
            return None

        print(f'Use {search_mode} for searching articles...')

        if search_mode == 'pubmed':
            return self.search_articles_pubmed(keywords)

        else:
            print(f'Search mode {search_mode} not supported!')
            return

        print("Error: Something went wrong with the searching mode!")
        return None
    
    def search_articles_pubmed(self, keywords):
        def reduce_keywords(keywords):
            """Reduce keywords by removing the last word iteratively."""
            keyword_list = keywords.split()
            if len(keyword_list) > 1:
                return ' '.join(keyword_list[:-1])
            return ""

        params = {
            "db": "pubmed",
            "term": keywords,
            "retmax": self.params.get('max_results_pubmed', 5),
            "retmode": "json",
            "sort": self.params.get('sort_by_pubmed', 'relevance'),
            "mindate": self.params.get('min_year', 1900), # optional
            "maxdate": self.params.get('max_year', 2025) # optional
        }

        if self.params.get('pubmed_api_key', None):
            params["api_key"] = self.params.get('pubmed_api_key')

        try:
            while keywords:
                response = requests.get(self.params.get('search_url'), params=params)
                #print(response)
                if self.params.get('pumbed_api_request_interval', False):
                    time.sleep(self.params.get('pumbed_api_request_interval'))  # Ensure we don't exceed rate limits

                if response.status_code == 200:
                    data = response.json()
                    pmid_list = data.get("esearchresult", {}).get("idlist", [])
                    total_count = int(data.get("esearchresult", {}).get("count", 0))

                    # if self.params.get('count_articles_pubmed', False):
                        # print(f"Total articles available: {total_count}")
                        # print(f"Articles retrieved: {len(pmid_list)}")

                    if pmid_list:
                        return pmid_list  # Return results if articles are found

                    if self.params.get('pumbed_keyword_reduce', False):
                        if len(keywords.split()) > 1:
                            print(f"No results found. Reducing keywords: '{keywords}'")
                            keywords = reduce_keywords(keywords)
                            params["term"] = keywords
                            
                    else:
                        # print("No results found, and keywords cannot be reduced further.")
                        break
                else:
                    # print(f"Error: {response.status_code}, {response.text}")
                    return None

            # print("No articles found for the given keywords.")
            return None

        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    
    # Search for articles in Semantic_Scholar based on keywords.
    def search_semantic_scholar(self, query, limit=5, fields=["title", "abstract", "paperId", "externalIds", "year"]):
        url = ""
        headers = {""}
        
        if not isinstance(query, str):
            query = str(query)
        query = re.sub(r'["()]+', '', query)
        # Replace AND/OR with space
        query = re.sub(r'\bAND\b|\bOR\b', ' ', query, flags=re.IGNORECASE)
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        params = {
            "query": query,
            "limit": limit,
            "fields": ",".join(fields)
        }
        resp = requests.get(url, headers=headers, params=params)

        response_json = resp.json()
        top_papers = response_json.get("data", [])
        return top_papers

    def fetch_article_content(self, pmids):
        if isinstance(pmids, str):
            pmids = [pmids]
        elif not isinstance(pmids, list):
            raise ValueError("pmids must be a string or list of strings")

        # Validate that all elements are strings
        pmids = [str(pmid) for pmid in pmids if pmid]

        if not pmids:
            return []
        
        # print(pmids)
        # Prepare the parameters
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),  # Join PMIDs with commas
            "rettype": "abstract",
            "retmode": "xml",  # Request XML format
        }

        if self.params.get('pubmed_api_key', None):
            params["api_key"] = self.params.get('pubmed_api_key')

        try:
            # Send the request
            response = requests.get(self.params.get('fetch_url'), params=params)
            
            if self.params.get('pumbed_api_request_interval', False):
                time.sleep(self.params.get('pumbed_api_request_interval'))  # Ensure we don't exceed rate limits
            
            if response.status_code == 200:
                # Parse the XML response using lxml
                root = etree.fromstring(response.content)
                
                # Extract title and abstract for each article
                articles = []
                for article in root.xpath(".//PubmedArticle"):
                    title = article.xpath("string(.//ArticleTitle)").strip()
                    abstract = "".join(
                        abstract_text.strip()
                        for abstract_text in article.xpath(".//AbstractText/text()")
                    ).strip()
                    
                    doi = None
                    for id_node in article.xpath(".//ArticleIdList/ArticleId"):
                        if id_node.get("IdType") == "doi":
                            doi = id_node.text.strip()
                            break
                    pub_year = article.xpath("string(.//PubDate/Year)")
                    if not pub_year:
                        pub_year = article.xpath("string(.//PubDate/MedlineDate)")
                        if pub_year:
                            pub_year = pub_year.split(" ")[0]
                    articles.append({
                        "title": title,
                        "abstract": abstract,
                        "doi": doi,
                        "pub_year": pub_year
                    })
                return articles
            else:
                print(f"Error: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def display_articles(self, pmids):
        """
        Display article details including title and abstract for a list of PMIDs.

        :param pmids: List of PMIDs
        """
        if not pmids:
            print("No PMIDs provided.")
            return

        for pmid in pmids:
            article = self.fetch_article_content(pmid)
            if article:
                print(f"Title: {article['title']}")
                print(f"Abstract:\n{article['abstract']}\n")
            else:
                print(f"Details for PMID {pmid} could not be retrieved.")


    def get_titles_and_abstracts(self, pmids):
        fetch_mode = self.params.get('fetch_mode')

        if fetch_mode not in {'mongodb', 'pubmed'}:
            print("Error: The search mode can only be either 'mongodb' or 'pubmed'.")
            return None

        if fetch_mode == 'mongodb':
            return self.get_titles_abstracts_from_MongoDB(pmids)
        elif fetch_mode == 'pubmed':
            return self.get_titles_and_abstracts_pubmed(pmids)

        print("Error: Something went wrong with the fetching mode!")
        return None

    def get_titles_and_abstracts_pubmed(self, pmids):
        if not pmids:
            print("No PMIDs provided.")
            return {}
        
        # Ensure chunk size is valid
        chunk_size = self.params.get('chunk_size_pubmed', 300)
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than 0.")
        
        all_results = {}
        retrieved_count = 0  # Counter for successfully retrieved articles
        
        # Split PMIDs into chunks and load. Much faster than fetch them one-by-one
        for i in range(0, len(pmids), chunk_size):
            chunk = pmids[i:i + chunk_size]
            results = self.fetch_article_content(chunk)
            if results:
                for j, extracted_contents in enumerate(results):
                    all_results[chunk[j]] = extracted_contents
            retrieved_count += len(results)
            
            if self.params.get('count_articles_pubmed', False):
                print(f"{retrieved_count} articles retrieved so far.")
            
        return all_results
    
    def get_pubmed_years_and_citations(self, query, max_results=5, api_key=None):
        base_url = ""
        api_key = self.params.get('pubmed_api_key', '')
        # Step 1: Search for PMIDs using esearch
        esearch_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&retmode=json"
        if api_key:
            esearch_url += f"&api_key={api_key}"  # Add API key if provided

        esearch_response = requests.get(esearch_url)
        if esearch_response.status_code != 200:
            raise Exception("Error fetching data from PubMed esearch API")
        esearch_data = esearch_response.json()
        pmids = esearch_data.get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return {}

        # Step 2: Fetch article details using efetch to get years
        pmid_str = ",".join(pmids)
        efetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={pmid_str}&retmode=xml"
        if api_key:
            efetch_url += f"&api_key={api_key}"

        efetch_response = requests.get(efetch_url)
        if efetch_response.status_code != 200:
            raise Exception("Error fetching data from PubMed efetch API")
        efetch_xml = efetch_response.text

        # Step 3: Parse XML to extract publication years
        root = ET.fromstring(efetch_xml)
        pub_data = {}
        for article in root.findall(".//PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            if pmid_elem is None:
                continue
            pmid = pmid_elem.text
            year_elem = article.find(".//PubDate/Year")
            
            # Use MedlineDate as a fallback
            if year_elem is None:
                medline_date = article.find(".//PubDate/MedlineDate")
                if medline_date is not None and medline_date.text[:4].isdigit():
                    pub_year = int(medline_date.text[:4])
                else:
                    pub_year = 0
            else:
                pub_year = int(year_elem.text) if year_elem.text.isdigit() else 0
            
            pub_data[pmid] = {"years": pub_year, "count_ref": 0}  # Default citation count to 0

        # Step 4: Fetch citation counts from iCite API
        icite_url = "https://icite.od.nih.gov/api/pubs"
        icite_response = requests.post(icite_url, json={"pmids": pmids})
        if icite_response.status_code == 200:
            icite_data = icite_response.json()
            for entry in icite_data:
                pmid = str(entry.get("pmid"))
                if pmid in pub_data:
                    pub_data[pmid]["count_ref"] = entry.get("citation_count", 0)  # Default to 0 if missing

        return pub_data


def fetch_semantic_scholar_articles(top_papers):
    title_abs_reference = []
    for item in top_papers:
        time.sleep(0.5)
        doi = item.get("externalIds", {}).get("DOI", None)
        if not doi:
            continue
        try:
            response = requests.get(f"")
            if response.status_code == 200:
                message = response.json().get("message", {})
                published = (
                    message.get("published-print") or
                    message.get("published-online") or
                    {}
                )
                date_parts = published.get("date-parts", [])
                pub_year = date_parts[0][0] if date_parts and date_parts[0] else None
                if item.get("abstract"):
                    title_abs_reference.append({
                        "title": item.get("title"),
                        "abstract": item.get("abstract"),
                        "doi": doi,
                        "pub_year": pub_year
                    })
        except Exception as e:
            print(f"❌ Error fetching year from CrossRef for DOI {doi}: {e}")

    return title_abs_reference

def fetch_crossref_articles(query: str, limit: int = 5, llm_api=None, row=None):
    title_abs_reference = []
    crossref_url = ""
    crossref_params = {"query": query, "rows": limit}
    try:
        crossref_response = requests.get(crossref_url, params=crossref_params)
        if crossref_response.status_code == 200:
            items = crossref_response.json().get("message", {}).get("items", [])
            for item in items:
                title = item.get("title", ["No Title"])[0]
                doi = item.get("DOI", "No DOI")
                doi = re.sub(r'\.s\d{3}$', '', doi)
                date_info = (
                    item.get("published-print") or
                    item.get("published-online") or
                    item.get("created") or
                    {}
                )
                date_parts = date_info.get("date-parts", [])
                pub_year = date_parts[0][0] if date_parts and date_parts[0] else "Unknown Year"

                # Try Semantic Scholar API first
                url = f""
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    abstract = data.get("abstract")
                    if abstract:
                        title_abs_reference.append({
                            "title": title,
                            "abstract": abstract,
                            "doi": doi,
                            "pub_year": pub_year
                        })
                        continue
                # Fallback to CrossRef API for abstract
                url = f""
                response = requests.get(url)
                if response.status_code == 200:
                    metadata = response.json().get("message", {})
                    abstract = metadata.get("abstract", None)
                    if abstract:
                        title_abs_reference.append({
                            "title": title,
                            "abstract": abstract,
                            "doi": doi,
                            "pub_year": pub_year
                        })
                        continue
                # Fallback to LLM for abstract if available
                if llm_api is not None:
                    abstract = llm_api.get_abstract_from_title_via_llm(title)
                    if abstract and isinstance(abstract, str) and len(abstract) > 200:
                        title_abs_reference.append({
                            "title": title,
                            "abstract": abstract,
                            "doi": doi,
                            "pub_year": pub_year
                        })
        else:
            print(f"❌ CrossRef API error: {crossref_response.status_code}")
    except Exception as e:
        print("❌ Error querying CrossRef:", e)
    paper_titles = [row["Inspiration paper 1 title"], row["Inspiration paper 2 title"], row["Inspiration paper 3 title"]]
    title_abs_reference.extend(fetch_paper_details_by_title(paper_titles, llm_api=llm_api))
    return title_abs_reference
    
def fetch_paper_details_by_title(titles, llm_api=None):
    
    title_abs_reference = []

    for title in titles:
        try:
            # Search CrossRef by title
            crossref_url = ""
            params = {"query.title": title, "rows": 1}
            resp = requests.get(crossref_url, params=params, timeout=10)
            if resp.status_code != 200:
                print(f"CrossRef query failed for title: {title}")
                continue

            items = resp.json()["message"].get("items", [])
            if not items:
                print(f"No results found in CrossRef for title: {title}")
                continue

            item = items[0]
            found_title = item.get("title", ["No Title"])[0]
            doi = item.get("DOI", "No DOI")
            doi = re.sub(r'\.s\d{3}$', '', doi)

            # Get publication year
            date_info = (
                item.get("published-print")
                or item.get("published-online")
                or item.get("created")
                or {}
            )
            date_parts = date_info.get("date-parts", [])
            pub_year = date_parts[0][0] if date_parts and date_parts[0] else "Unknown Year"

            # Try to get abstract from Semantic Scholar
            abstract = None
            if doi and doi != "No DOI":
                semantic_url = f""
                paper_resp = requests.get(semantic_url, timeout=10)
                if paper_resp.status_code == 200:
                    abstract = paper_resp.json().get("abstract")

            # Fallback to LLM if abstract still not found
            if not abstract and llm_api is not None:
                abstract = llm_api.get_abstract_from_title_via_llm(found_title)

            title_abs_reference.append({
                "title": found_title,
                "abstract": abstract,
                "doi": doi,
                "pub_year": pub_year
            })

        except Exception as e:
            print(f"❌ Error fetching details for title '{title}': {e}")

    return title_abs_reference