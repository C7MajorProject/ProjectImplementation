Introduction

Navigating large volumes of academic literature to synthesize findings and uncover unexplored research areas is often inefficient and labor-intensive. This project provides an automated, end-to-end natural language processing pipeline designed to ingest scientific papers, generate concise abstractive summaries, and identify emerging research gaps.

By leveraging three specialized, pre-trained Transformer models—SBERT for dense sentence embeddings, BART for sequence-to-sequence abstractive summarization, and BERTopic for semantic clustering—the framework converts unstructured academic text into structured insights, visual topic landscapes, and exportable research reports.

Pipeline Architecture and Workflow

Data Ingestion: Accepts PDF documents via direct uploads or queries research APIs (arXiv and Semantic Scholar) using metadata and user-specified search topics.

Preprocessing: Parses incoming PDF files, extracts raw text, normalizes content, and segments documents into clean sentence units.

Embedding Generation: Employs the pre-trained all-MiniLM-L6-v2 (SBERT) model via Hugging Face to generate high-dimensional semantic vector representations for document sentences.

Abstractive Summarisation: Uses the facebook/bart-large-cnn Seq2Seq architecture to read normalized text sections and synthesize core contributions into concise summaries.

Gap Identification & Clustering: Applies the BERTopic model to cluster semantically related documents, uncover overarching themes, and highlight sparse or underexplored areas across the domain.

Deliverables & Outputs: Generates structured paper summaries, key contribution lists, identified research gaps, visual gap distribution charts, and downloadable reports in PDF/DOCX format.

Core Tech Stack

Core Language & Frameworks: Python, PyTorch, Hugging Face Transformers

NLP & Topic Modeling: Sentence-Transformers (SBERT), BART, BERTopic

Vector Indexing & Machine Learning: FAISS, scikit-learn, NumPy, Pandas

Interface & Visualization: Streamlit, Matplotlib

```
===================================================================================================
                                      SYSTEM ARCHITECTURE
===================================================================================================

[ INPUTS ]
   ├── PDF Files (Research Papers)
   ├── Paper Metadata (Title, Authors, Year, Venue, Abstract)
   ├── User Query / Research Topic
   └── Upload Papers (PDF Format)
            │
            ▼
[ SYSTEM PIPELINE ]
   ┌────────────────────────┐
   │ 1. Data Collection     │ ──► arXiv API | Semantic Scholar API | PDF Upload
   └──────────┬─────────────┘
              ▼
   ┌────────────────────────┐
   │ 2. Preprocessing       │ ──► PDF Parsing ──► Text Extraction ──► Cleaning ──► Segmentation
   └──────────┬─────────────┘
              ▼
   ┌────────────────────────┐
   │ 3. Embedding Gen.      │ ──► SBERT (all-MiniLM-L6-v2) ──► Sentence Embeddings
   └──────────┬─────────────┘
              ▼
   ┌────────────────────────┐
   │ 4. Summarisation       │ ──► BART (facebook/bart-large-cnn) ──► Abstractive Summaries
   └──────────┬─────────────┘
              ▼
   ┌────────────────────────┐
   │ 5. Gap Identification  │ ──► BERTopic Model ──► Clustering & Topic Modeling
   └──────────┬─────────────┘
              │
              ▼
[ OUTPUTS ]
   ├── Paper Summary
   ├── Key Contributions
   ├── Research Gaps
   ├── Gap Visualization
   └── Download Report (PDF / DOCX)

===================================================================================================
[ TOOLS & LIBRARIES ]: Python | PyTorch | Transformers | scikit-learn | Streamlit | FAISS | Pandas
===================================================================================================
```
