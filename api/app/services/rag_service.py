from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import os
import logging
from dotenv import load_dotenv
import asyncio

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


@dataclass
class RAGConfig:
    """Configuration class for RAG service parameters."""
    chroma_path: str = os.getenv("CHROMA_PATH", os.path.join(os.getcwd(), "data", "chroma_data"))
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gemini-2.5-flash"
    google_api_key: Optional[str] = None
    max_retrieval_docs: int = 10
    primary_retrieval_k: int = 8
    section_specific_k: int = 10
    
    def __post_init__(self):
        if not self.google_api_key:
            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            if not self.google_api_key:
                raise ValueError("Google API key must be provided either via parameter or environment variable")


@dataclass
class RAGResponse:
    """Response object for RAG operations."""
    ticker: str
    question: str
    answer: str
    retrieval_count: int
    sources: List[Dict[str, Any]]


class DocumentFormatter:
    """Utility class for formatting documents with source citations."""
    
    @staticmethod
    def format_with_sources(docs: List[Document]) -> str:
        """
        Format documents with source citations.
        
        Args:
            docs: List of documents to format
            
        Returns:
            Formatted string with citations
        """
        formatted_docs = []
        
        for i, doc in enumerate(docs):
            source_id = f"[Source {i+1}]"
            source_name = doc.metadata.get("source", "N/A")
            page_num = doc.metadata.get("page", "N/A")
            
            formatted_doc = (
                f"{source_id}\n"
                f"Source: {source_name}, Page: {page_num}\n"
                f"Content: {doc.page_content}"
            )
            formatted_docs.append(formatted_doc)
            
        return "\n\n".join(formatted_docs)


class RAGRetriever:
    """Handles document retrieval with section-aware strategies."""
    
    def __init__(self, vector_db: Chroma, config: RAGConfig):
        self.vector_db = vector_db
        self.config = config
        
    def _get_query_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from query for section matching."""
        query_lower = query.lower()
        return [
            keyword for keyword in [
                "risk", "risks", "uncertainties",
                "management discussion", "md&a", "operations", "results of operations", "financial condition",
                "financial statement", "balance sheet", "income statement", "cash flow", "earnings", "revenue", "profit", "loss"
            ]
            if keyword in query_lower
        ]
    
    def _create_retriever(self, ticker: str, section_filter: Optional[Dict] = None) -> Any:
        """Create a retriever with appropriate filters."""
        base_filter = {"ticker": ticker}
        
        if section_filter:
            # Use proper ChromaDB $and syntax
            search_kwargs = {
                "k": self.config.section_specific_k,
                "filter": {"$and": [base_filter, section_filter]}
            }
        else:
            search_kwargs = {
                "k": self.config.primary_retrieval_k,
                "filter": base_filter
            }
            
        return self.vector_db.as_retriever(search_kwargs=search_kwargs)
    
    def retrieve_documents(self, query: str, ticker: str) -> List[Document]:
        """
        Retrieve documents using multi-stage strategy.
        
        Args:
            query: The search query
            ticker: Stock ticker symbol
            
        Returns:
            List of unique documents
        """
        all_docs = []
        seen_content = set()
        keywords = self._get_query_keywords(query)
        
        # Primary retrieval
        primary_retriever = self._create_retriever(ticker)
        primary_docs = primary_retriever.invoke(query)
        all_docs.extend(primary_docs)
        
        # Section-specific retrieval based on query keywords
        if any(keyword in keywords for keyword in ["risk", "risks", "uncertainties"]):
            risk_retriever = self._create_retriever(ticker, {"section": "risk_factors"})
            risk_docs = risk_retriever.invoke(query)
            all_docs.extend(risk_docs)
        
        if any(keyword in keywords for keyword in ["management discussion", "md&a", "operations", "results of operations", "financial condition"]):
            mdna_retriever = self._create_retriever(ticker, {"section": "mdna"})
            mdna_docs = mdna_retriever.invoke(query)
            all_docs.extend(mdna_docs)
        
        if any(keyword in keywords for keyword in ["financial statement", "balance sheet", "income statement", "cash flow", "earnings", "revenue", "profit", "loss"]):
            financial_retriever = self._create_retriever(ticker, {"section": "financial_statements"})
            financial_docs = financial_retriever.invoke(query)
            all_docs.extend(financial_docs)
        
        # Deduplicate and limit results
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        
        return unique_docs[:self.config.max_retrieval_docs]


class RAGPrompt:
    """Handles prompt template creation and management."""
    
    @staticmethod
    def create_system_prompt() -> str:
        """Create the system prompt for financial analysis."""
        return (
            "You are a financial analyst. Based EXCLUSIVELY on the provided context, "
            "provide specific and detailed answers to the user's question. "
            "**After each specific piece of information or claim, you MUST cite the source using the [Source N] tag it came from.**\n\n"
            "Context: {context}"
        )
    
    @staticmethod
    def create_prompt_template() -> ChatPromptTemplate:
        """Create the chat prompt template."""
        return ChatPromptTemplate.from_messages([
            ("system", RAGPrompt.create_system_prompt()),
            ("user", "{input}")
        ])


class RAGService:
    """Main RAG service class for document-based question answering."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG service.
        
        Args:
            config: Optional configuration object. If not provided, default config will be used.
        """
        self.config = config or RAGConfig()
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vector_db: Optional[Chroma] = None
        self.llm: Optional[ChatGoogleGenerativeAI] = None
        self.retriever: Optional[RAGRetriever] = None
        
    async def initialize(self) -> None:
        """Initialize all RAG service components."""
        try:
            logger.info("Initializing RAG service components...")
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model
            )
            logger.info(f"Initialized embeddings with model: {self.config.embedding_model}")
            
            # Initialize vector database
            self.vector_db = Chroma(
                persist_directory=self.config.chroma_path,
                embedding_function=self.embeddings
            )
            logger.info(f"Initialized vector database at: {self.config.chroma_path}")
            
            # Initialize LLM
            if not self.config.google_api_key:
                raise ValueError("Google API key is required for LLM initialization")
            
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.llm_model,
                google_api_key=self.config.google_api_key
            )
            logger.info(f"Initialized LLM with model: {self.config.llm_model}")
            
            # Initialize retriever
            self.retriever = RAGRetriever(self.vector_db, self.config)
            logger.info("Initialized RAG retriever")
            
            logger.info("RAG service initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown RAG service and cleanup resources."""
        try:
            logger.info("Shutting down RAG service...")
            
            # Clear references to allow garbage collection
            self.retriever = None
            self.vector_db = None
            self.llm = None
            self.embeddings = None
            
            logger.info("RAG service shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during RAG service shutdown: {e}")
            # Don't raise exception on shutdown
    
    def _validate_initialization(self) -> None:
        """Validate that service components are properly initialized."""
        missing_components = []
        if not self.vector_db:
            missing_components.append("vector_db")
        if not self.llm:
            missing_components.append("llm")
        if not self.embeddings:
            missing_components.append("embeddings")
        if not self.retriever:
            missing_components.append("retriever")
            
        if missing_components:
            raise RuntimeError(f"RAG service not properly initialized. Missing components: {missing_components}")
    
    async def answer_question(self, ticker: str, question: str) -> RAGResponse:
        """
        Answer a question using RAG methodology.
        
        Args:
            ticker: Stock ticker symbol
            question: Question to answer
            
        Returns:
            RAGResponse with answer and metadata
            
        Raises:
            RuntimeError: If service is not properly initialized
        """
        self._validate_initialization()
        
        try:
            logger.info(f"Processing question for ticker {ticker}: {question[:50]}...")
            
            # Retrieve relevant documents
            docs = self.retriever.retrieve_documents(question, ticker)
            logger.info(f"Retrieved {len(docs)} documents")
            
            # Format documents
            formatted_context = DocumentFormatter.format_with_sources(docs)
            
            # Create prompt
            prompt = RAGPrompt.create_prompt_template()
            
            # Build RAG chain
            if self.llm is None:
                raise RuntimeError("LLM is not initialized.")
            chain = (
                {
                    "context": RunnableLambda(lambda x: formatted_context),
                    "input": RunnablePassthrough(),
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Generate answer
            answer_text = chain.invoke(question)
            
            # Prepare response
            response = RAGResponse(
                ticker=ticker,
                question=question,
                answer=answer_text,
                retrieval_count=len(docs),
                sources=[{
                    "source": doc.metadata.get("source", "N/A"),
                    "page": doc.metadata.get("page", "N/A")
                } for doc in docs]
            )
            
            logger.info(f"Generated answer with {len(docs)} sources")
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise


# Convenience functions for backward compatibility
async def initialize_rag_service(config: Optional[RAGConfig] = None) -> RAGService:
    """
    Initialize RAG service with backward compatibility.
    
    Args:
        config: Optional configuration object
        
    Returns:
        Initialized RAGService instance
    """
    service = RAGService(config)
    await service.initialize()
    return service


async def shutdown_rag_service(service: RAGService) -> None:
    """
    Shutdown RAG service with backward compatibility.
    
    Args:
        service: RAGService instance to shutdown
    """
    await service.shutdown()


async def answer_question(ticker: str, question: str) -> Dict[str, Any]:
    """
    Answer a question with backward compatibility.
    
    Args:
        ticker: Stock ticker symbol
        question: Question to answer
        
    Returns:
        Dictionary with answer and metadata (legacy format)
    """
    service = await initialize_rag_service()
    try:
        response = await service.answer_question(ticker, question)
        return {
            "ticker": response.ticker,
            "question": response.question,
            "answer": response.answer,
            "retrieval_count": response.retrieval_count
        }
    finally:
        await shutdown_rag_service(service)

