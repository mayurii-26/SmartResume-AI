"""
SmartResume AI - Topic Modeling Module
Discovers latent career and technological themes across resumes using
Latent Dirichlet Allocation (LDA) with Scikit-learn.
Includes adaptive vectorization, topic keyword extraction, document assignment,
and corpus-level distributions with small-corpus resilience.
"""

import logging
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logger = logging.getLogger(__name__)

# Industry domain seed labels for semantic mapping
DOMAIN_KEYWORDS = {
    "Software & Web Engineering": ["software", "web", "application", "developer", "backend", "frontend", "javascript", "react", "api", "code"],
    "Data Science & AI / ML": ["data", "model", "machine", "learning", "python", "neural", "analytics", "deep", "nlp", "prediction"],
    "Cloud & DevOps Architecture": ["cloud", "aws", "azure", "ci", "cd", "docker", "kubernetes", "infrastructure", "devops", "pipeline"],
    "Database & Big Data Engineering": ["database", "sql", "spark", "kafka", "etl", "warehouse", "postgres", "queries", "hadoop", "migration"],
    "Enterprise & Project Management": ["management", "project", "business", "agile", "scrum", "lead", "team", "client", "strategy", "delivery"],
    "Cybersecurity & Infrastructure": ["security", "network", "firewall", "vulnerability", "risk", "audit", "compliance", "threat", "protocols"],
    "Mobile App Development": ["android", "ios", "swift", "kotlin", "flutter", "mobile", "react", "native", "apps", "store"]
}


class ResumeTopicModeler:
    """Discovers career clusters and technology domains using Latent Dirichlet Allocation."""

    def __init__(self, n_topics: int = 5, max_iter: int = 25, random_state: int = 42):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.random_state = random_state
        self.vectorizer = None
        self.lda_model = None
        self.feature_names = []

    def fit_transform(self, documents: List[str], n_words_per_topic: int = 10) -> Dict[str, Any]:
        """
        Fits LDA topic model across a corpus of resumes.
        Handles small corpora, duplicate documents, and sparse vocabulary gracefully.
        """
        if not documents:
            return {
                "status": "insufficient_data",
                "message": "No resume documents were provided for topic modeling.",
                "topics": [],
                "document_topics": []
            }

        valid_docs = [d.strip() for d in documents if d and len(d.strip()) > 20]
        n_docs = len(valid_docs)

        if n_docs < 2:
            return {
                "status": "insufficient_data",
                "message": f"Topic modeling requires at least 2 non-empty resume documents (Found: {n_docs}).",
                "topics": [],
                "document_topics": []
            }

        # Vectorization with adaptive fallback to avoid pruning all terms
        dtm = None
        vectorizer_configs = [
            # Config 1: standard English stop words with max_df=1.0 (never prunes words in small corpora)
            {
                "stop_words": "english",
                "min_df": 1,
                "max_df": 1.0,
                "ngram_range": (1, 2),
                "token_pattern": r"\b[A-Za-z][A-Za-z0-9+#]{1,25}\b"
            },
            # Config 2: unigrams with max_df=1.0 if bigrams/token pattern failed
            {
                "stop_words": "english",
                "min_df": 1,
                "max_df": 1.0,
                "ngram_range": (1, 1),
                "token_pattern": r"\b[A-Za-z0-9+#]{2,25}\b"
            },
            # Config 3: no stop words fallback if documents consist primarily of domain stopwords
            {
                "stop_words": None,
                "min_df": 1,
                "max_df": 1.0,
                "ngram_range": (1, 1),
                "token_pattern": r"\b[A-Za-z0-9+#]{2,25}\b"
            }
        ]

        for cfg in vectorizer_configs:
            try:
                vec = CountVectorizer(**cfg)
                transformed = vec.fit_transform(valid_docs)
                features = vec.get_feature_names_out()
                if transformed.shape[1] > 0 and len(features) > 0:
                    self.vectorizer = vec
                    self.feature_names = features
                    dtm = transformed
                    break
            except (ValueError, Exception) as err:
                logger.debug(f"CountVectorizer attempt with cfg failed: {err}")
                continue

        if dtm is None or dtm.shape[1] == 0:
            return {
                "status": "insufficient_data",
                "message": "After text processing, no valid terms remain across the documents. Please provide more detailed resume texts.",
                "topics": [],
                "document_topics": []
            }

        n_features = dtm.shape[1]

        # Dynamically determine effective topic count based on documents and vocabulary size
        if n_features < 2:
            effective_n_topics = 1
        else:
            effective_n_topics = min(self.n_topics, n_docs, n_features)
            effective_n_topics = max(2, effective_n_topics) if (n_docs >= 2 and n_features >= 2) else 1

        try:
            self.lda_model = LatentDirichletAllocation(
                n_components=effective_n_topics,
                max_iter=self.max_iter,
                learning_method="online",
                random_state=self.random_state,
                doc_topic_prior=0.1,
                topic_word_prior=0.01
            )
            doc_topic_matrix = self.lda_model.fit_transform(dtm)
        except Exception as e:
            logger.warning(f"LDA fitting encountered an error: {e}")
            return {
                "status": "error",
                "message": f"LDA model fitting could not be completed: {str(e)}",
                "topics": [],
                "document_topics": []
            }

        # Extract top keywords and identify domain label
        topics_info = []
        actual_k = min(n_words_per_topic, n_features)

        for topic_idx, topic in enumerate(self.lda_model.components_):
            # argsort descending
            top_word_indices = topic.argsort()[::-1][:actual_k]
            top_keywords = [self.feature_names[i] for i in top_word_indices]
            weights = [round(float(topic[i]), 2) for i in top_word_indices]

            # Label mapping based on overlap with domain keywords
            assigned_label = f"Topic {topic_idx + 1}"
            best_overlap = 0
            for domain, kws in DOMAIN_KEYWORDS.items():
                overlap = len(set(top_keywords).intersection(set(kws)))
                if overlap > best_overlap:
                    best_overlap = overlap
                    assigned_label = domain

            top_weight = float(topic[top_word_indices[0]]) if len(top_word_indices) > 0 else 0.0
            total_weight = max(1.0, float(topic.sum()))

            topics_info.append({
                "topic_id": topic_idx,
                "label": assigned_label,
                "top_keywords": top_keywords,
                "weights": weights,
                "coherence_indicator": round(float(top_weight / total_weight), 3)
            })

        # Document topic distributions
        doc_results = []
        for doc_idx, distribution in enumerate(doc_topic_matrix):
            dominant_id = int(np.argmax(distribution))
            dominant_prob = float(distribution[dominant_id])
            dominant_label = topics_info[dominant_id]["label"]

            dist_dict = {
                f"Topic {i+1} ({topics_info[i]['label']})": round(float(p), 4)
                for i, p in enumerate(distribution)
            }

            doc_results.append({
                "document_index": doc_idx,
                "dominant_topic_id": dominant_id,
                "dominant_topic_label": dominant_label,
                "confidence": round(dominant_prob, 3),
                "distribution": dist_dict
            })

        # Calculate resume count per topic
        counts = [0] * effective_n_topics
        for r in doc_results:
            counts[r["dominant_topic_id"]] += 1

        for i, t in enumerate(topics_info):
            t["assigned_document_count"] = counts[i]

        perplexity_val = "N/A"
        try:
            perplexity_val = round(float(self.lda_model.perplexity(dtm)), 2)
        except Exception:
            pass

        return {
            "status": "success",
            "n_topics": effective_n_topics,
            "total_documents": n_docs,
            "topics": topics_info,
            "document_topics": doc_results,
            "perplexity": perplexity_val
        }
