"""Document loaders for AgingResearchAI RAG system.

Provides loaders for:
- PubMed papers (via NCBI E-utilities)
- Patents (via Google Patents, Lens.org)
- PDFs (local files)
- Database entries (DrugBank, KEGG, etc.)
"""

import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator

import httpx
from pydantic import BaseModel

from .knowledge_base import Document, DocumentMetadata


# =============================================================================
# Base Loader
# =============================================================================

class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self) -> list[Document]:
        """Load documents.

        Returns:
            List of Document objects
        """
        pass

    @abstractmethod
    def lazy_load(self) -> Generator[Document, None, None]:
        """Lazily load documents one at a time.

        Yields:
            Document objects
        """
        pass


# =============================================================================
# PubMed Loader
# =============================================================================

class PubMedLoader(BaseDocumentLoader):
    """Load papers from PubMed via NCBI E-utilities.

    Requires NCBI API key for better rate limits.
    Set NCBI_API_KEY environment variable.

    Usage:
        loader = PubMedLoader(query="diabetes AMPK", max_results=100)
        documents = loader.load()
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        query: str | None = None,
        pmids: list[str] | None = None,
        max_results: int = 100,
        api_key: str | None = None,
        include_abstract: bool = True,
        chunk_size: int = 1000,
    ):
        """Initialize PubMed loader.

        Args:
            query: PubMed search query
            pmids: List of specific PMIDs to fetch
            max_results: Maximum results for search
            api_key: NCBI API key (uses env var if None)
            include_abstract: Include abstract in content
            chunk_size: Characters per chunk for long abstracts
        """
        self.query = query
        self.pmids = pmids or []
        self.max_results = max_results
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self.include_abstract = include_abstract
        self.chunk_size = chunk_size

        self.client = httpx.Client(timeout=30)

    def _get_params(self, **kwargs) -> dict:
        """Get base parameters with API key."""
        params = {"retmode": "json", **kwargs}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _search(self, query: str, max_results: int) -> list[str]:
        """Search PubMed and return PMIDs."""
        params = self._get_params(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance",
        )

        response = self.client.get(f"{self.BASE_URL}/esearch.fcgi", params=params)
        response.raise_for_status()

        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def _fetch_details(self, pmids: list[str]) -> list[dict]:
        """Fetch paper details for PMIDs."""
        if not pmids:
            return []

        # Fetch in batches of 200
        all_papers = []
        for i in range(0, len(pmids), 200):
            batch = pmids[i:i + 200]

            params = self._get_params(
                db="pubmed",
                id=",".join(batch),
                rettype="abstract",
            )

            response = self.client.get(
                f"{self.BASE_URL}/efetch.fcgi",
                params={**params, "retmode": "xml"},
            )
            response.raise_for_status()

            # Parse XML response
            papers = self._parse_pubmed_xml(response.text)
            all_papers.extend(papers)

            # Rate limiting
            time.sleep(0.34 if self.api_key else 1.0)

        return all_papers

    def _parse_pubmed_xml(self, xml_text: str) -> list[dict]:
        """Parse PubMed XML response."""
        # Simple regex-based parsing (for production, use proper XML parser)
        papers = []

        # Find all articles
        article_pattern = r"<PubmedArticle>(.*?)</PubmedArticle>"
        articles = re.findall(article_pattern, xml_text, re.DOTALL)

        for article in articles:
            paper = {}

            # PMID
            pmid_match = re.search(r"<PMID[^>]*>(\d+)</PMID>", article)
            if pmid_match:
                paper["pmid"] = pmid_match.group(1)

            # Title
            title_match = re.search(r"<ArticleTitle>(.+?)</ArticleTitle>", article, re.DOTALL)
            if title_match:
                paper["title"] = self._clean_xml_text(title_match.group(1))

            # Abstract
            abstract_match = re.search(r"<AbstractText[^>]*>(.+?)</AbstractText>", article, re.DOTALL)
            if abstract_match:
                paper["abstract"] = self._clean_xml_text(abstract_match.group(1))

            # Authors
            authors = []
            author_pattern = r"<Author[^>]*>.*?<LastName>(.+?)</LastName>.*?<ForeName>(.+?)</ForeName>.*?</Author>"
            for match in re.finditer(author_pattern, article, re.DOTALL):
                authors.append(f"{match.group(2)} {match.group(1)}")
            paper["authors"] = authors

            # Journal
            journal_match = re.search(r"<Title>(.+?)</Title>", article)
            if journal_match:
                paper["journal"] = self._clean_xml_text(journal_match.group(1))

            # Date
            year_match = re.search(r"<PubDate>.*?<Year>(\d+)</Year>", article, re.DOTALL)
            if year_match:
                paper["year"] = year_match.group(1)

            # DOI
            doi_match = re.search(r'<ArticleId IdType="doi">(.+?)</ArticleId>', article)
            if doi_match:
                paper["doi"] = doi_match.group(1)

            if paper.get("pmid"):
                papers.append(paper)

        return papers

    def _clean_xml_text(self, text: str) -> str:
        """Clean XML text content."""
        # Remove XML tags
        text = re.sub(r"<[^>]+>", "", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _paper_to_document(self, paper: dict) -> Document:
        """Convert paper dict to Document."""
        # Build content
        content_parts = []
        if paper.get("title"):
            content_parts.append(f"Title: {paper['title']}")
        if self.include_abstract and paper.get("abstract"):
            content_parts.append(f"\nAbstract: {paper['abstract']}")

        content = "\n".join(content_parts)

        # Build metadata
        metadata = DocumentMetadata(
            source="pubmed",
            source_id=paper.get("pmid"),
            title=paper.get("title"),
            authors=paper.get("authors", []),
            publication_date=paper.get("year"),
            journal=paper.get("journal"),
            doi=paper.get("doi"),
            url=f"https://pubmed.ncbi.nlm.nih.gov/{paper.get('pmid')}/" if paper.get("pmid") else None,
        )

        return Document(
            content=content,
            metadata=metadata,
        )

    def load(self) -> list[Document]:
        """Load all documents."""
        return list(self.lazy_load())

    def lazy_load(self) -> Generator[Document, None, None]:
        """Lazily load documents."""
        # Get PMIDs
        pmids = list(self.pmids)
        if self.query:
            searched = self._search(self.query, self.max_results)
            pmids.extend(searched)

        # Remove duplicates
        pmids = list(dict.fromkeys(pmids))

        # Fetch and yield documents
        papers = self._fetch_details(pmids)
        for paper in papers:
            yield self._paper_to_document(paper)

    def search_and_load(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[Document]:
        """Convenience method to search and load.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of documents
        """
        self.query = query
        if max_results:
            self.max_results = max_results
        return self.load()


# =============================================================================
# Patent Loader
# =============================================================================

class PatentLoader(BaseDocumentLoader):
    """Load patent documents.

    Supports:
    - Google Patents (via web scraping)
    - Lens.org (via API with key)

    Usage:
        loader = PatentLoader(query="AMPK inhibitor", source="lens")
        documents = loader.load()
    """

    def __init__(
        self,
        query: str | None = None,
        patent_numbers: list[str] | None = None,
        source: str = "lens",  # "lens" or "google"
        max_results: int = 50,
        api_key: str | None = None,
    ):
        """Initialize patent loader.

        Args:
            query: Patent search query
            patent_numbers: Specific patent numbers
            source: Data source ("lens" or "google")
            max_results: Maximum results
            api_key: API key for Lens.org
        """
        self.query = query
        self.patent_numbers = patent_numbers or []
        self.source = source
        self.max_results = max_results
        self.api_key = api_key or os.environ.get("LENS_API_KEY")

        self.client = httpx.Client(timeout=30)

    def _search_lens(self, query: str) -> list[dict]:
        """Search Lens.org for patents."""
        if not self.api_key:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "query": {
                "match": {
                    "text": query
                }
            },
            "size": self.max_results,
            "include": [
                "lens_id", "title", "abstract", "date_published",
                "inventor", "assignee", "classification_cpc"
            ]
        }

        try:
            response = self.client.post(
                "https://api.lens.org/patent/search",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except Exception as e:
            print(f"Lens.org search failed: {e}")
            return []

    def _patent_to_document(self, patent: dict) -> Document:
        """Convert patent dict to Document."""
        # Build content
        content_parts = []
        if patent.get("title"):
            content_parts.append(f"Title: {patent['title']}")
        if patent.get("abstract"):
            content_parts.append(f"\nAbstract: {patent['abstract']}")

        content = "\n".join(content_parts)

        # Get inventors
        inventors = []
        for inv in patent.get("inventor", []):
            if isinstance(inv, dict):
                inventors.append(inv.get("name", ""))
            else:
                inventors.append(str(inv))

        # Build metadata
        metadata = DocumentMetadata(
            source="patent",
            source_id=patent.get("lens_id") or patent.get("patent_number"),
            title=patent.get("title"),
            authors=inventors,
            publication_date=patent.get("date_published"),
            url=f"https://www.lens.org/lens/patent/{patent.get('lens_id')}" if patent.get("lens_id") else None,
        )

        return Document(
            content=content,
            metadata=metadata,
        )

    def load(self) -> list[Document]:
        """Load all documents."""
        return list(self.lazy_load())

    def lazy_load(self) -> Generator[Document, None, None]:
        """Lazily load documents."""
        patents = []

        if self.query and self.source == "lens":
            patents = self._search_lens(self.query)

        for patent in patents:
            yield self._patent_to_document(patent)


# =============================================================================
# PDF Loader
# =============================================================================

class PDFLoader(BaseDocumentLoader):
    """Load documents from PDF files.

    Uses PyPDF2 or pdfplumber for text extraction.

    Usage:
        loader = PDFLoader(file_path="paper.pdf")
        documents = loader.load()
    """

    def __init__(
        self,
        file_path: str | Path | None = None,
        directory: str | Path | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """Initialize PDF loader.

        Args:
            file_path: Single PDF file
            directory: Directory of PDFs
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.file_path = Path(file_path) if file_path else None
        self.directory = Path(directory) if directory else None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _extract_text(self, file_path: Path) -> str:
        """Extract text from PDF."""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except ImportError:
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += (page.extract_text() or "") + "\n"
                    return text
            except ImportError:
                raise ImportError(
                    "PDF extraction requires pypdf or pdfplumber. "
                    "Run: pip install pypdf"
                )

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size

            # Try to find a good break point
            if end < len(text):
                # Look for paragraph break
                break_point = text.rfind("\n\n", start, end)
                if break_point == -1:
                    # Look for sentence break
                    break_point = text.rfind(". ", start, end)
                if break_point != -1:
                    end = break_point + 1

            chunks.append(text[start:end].strip())
            start = end - self.chunk_overlap

        return [c for c in chunks if c]

    def _pdf_to_documents(self, file_path: Path) -> list[Document]:
        """Convert PDF to documents (one per chunk)."""
        text = self._extract_text(file_path)
        chunks = self._chunk_text(text)

        documents = []
        for i, chunk in enumerate(chunks):
            metadata = DocumentMetadata(
                source="pdf",
                source_id=file_path.stem,
                title=file_path.name,
                chunk_index=i,
                total_chunks=len(chunks),
            )
            documents.append(Document(
                content=chunk,
                metadata=metadata,
            ))

        return documents

    def load(self) -> list[Document]:
        """Load all documents."""
        return list(self.lazy_load())

    def lazy_load(self) -> Generator[Document, None, None]:
        """Lazily load documents."""
        files = []

        if self.file_path:
            files.append(self.file_path)

        if self.directory:
            files.extend(self.directory.glob("*.pdf"))

        for file_path in files:
            for doc in self._pdf_to_documents(file_path):
                yield doc


# =============================================================================
# Utility Functions
# =============================================================================

def load_documents_from_directory(
    directory: str | Path,
    file_types: list[str] | None = None,
    recursive: bool = True,
) -> list[Document]:
    """Load documents from a directory.

    Args:
        directory: Directory path
        file_types: File extensions to load (default: pdf, txt)
        recursive: Search subdirectories

    Returns:
        List of documents
    """
    directory = Path(directory)
    file_types = file_types or ["pdf", "txt"]

    documents = []
    pattern = "**/*" if recursive else "*"

    for file_type in file_types:
        for file_path in directory.glob(f"{pattern}.{file_type}"):
            if file_type == "pdf":
                loader = PDFLoader(file_path=file_path)
                documents.extend(loader.load())
            elif file_type == "txt":
                content = file_path.read_text(encoding="utf-8")
                documents.append(Document(
                    content=content,
                    metadata=DocumentMetadata(
                        source="file",
                        source_id=file_path.stem,
                        title=file_path.name,
                    ),
                ))

    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Chunk documents into smaller pieces.

    Args:
        documents: Documents to chunk
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        Chunked documents
    """
    chunked = []

    for doc in documents:
        if len(doc.content) <= chunk_size:
            chunked.append(doc)
            continue

        # Chunk the content
        text = doc.content
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Find good break point
            if end < len(text):
                for sep in ["\n\n", "\n", ". ", " "]:
                    break_point = text.rfind(sep, start, end)
                    if break_point > start:
                        end = break_point + len(sep)
                        break

            chunks.append(text[start:end].strip())
            start = end - chunk_overlap

        # Create documents for each chunk
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue

            new_metadata = doc.metadata.model_copy()
            new_metadata.chunk_index = i
            new_metadata.total_chunks = len(chunks)

            chunked.append(Document(
                content=chunk,
                metadata=new_metadata,
            ))

    return chunked
