from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv
from sec_api import PdfGeneratorApi, QueryApi

# Load environment variables from .env file
load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(os.getcwd(), "data", "chroma_data"))

def enhanced_ingest_10k_pdf(ticker: str, pdf_files: list):
    """
    Enhanced ingestion for PDF files with better content extraction and section-aware chunking.
    """
    print(f"Enhanced ingesting PDF files for {ticker}...")
    all_chunks = []
    
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path}...")
        
        try:
            # Use PyPDFLoader for PDF processing
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # Combine all pages into one text
            full_text = ""
            for doc in docs:
                full_text += doc.page_content + "\n"
            
            if not full_text.strip():
                print(f"No text content extracted from {pdf_path}")
                continue
            
            print(f"Text length: {len(full_text)} characters")
            
            # Enhanced chunking strategy for PDF content
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # Larger chunks for PDF content
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = splitter.split_text(full_text)
            
            # Add metadata and create documents
            for i, chunk_text in enumerate(chunks):
                # Enhanced metadata for PDF
                chunk_metadata = {
                    "ticker": ticker,
                    "source": pdf_path,
                    "chunk_id": f"{ticker}_pdf_{i}",
                    "chunk_size": len(chunk_text),
                    "position": i,
                    "source_type": "pdf"
                }
                
                # Add section detection
                if "risk factor" in chunk_text.lower():
                    chunk_metadata["section"] = "risk_factors"
                elif "management discussion" in chunk_text.lower() or "md&a" in chunk_text.lower():
                    chunk_metadata["section"] = "mdna"
                elif "financial statement" in chunk_text.lower():
                    chunk_metadata["section"] = "financial_statements"
                else:
                    chunk_metadata["section"] = "general"
                
                all_chunks.append(
                    Document(page_content=chunk_text, metadata=chunk_metadata)
                )
            
            print(f"Created {len(chunks)} chunks from {pdf_path}")
            
        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")
            continue
    
    if all_chunks:
        print(f"Adding {len(all_chunks)} enhanced PDF chunks for {ticker} to ChromaDB...")
        
        # Initialize embeddings and vector database
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Clear existing data for this ticker to avoid conflicts
        try:
            existing_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
            # Delete existing documents for this ticker
            existing_db.delete(where={"ticker": ticker})
            print(f"Cleared existing data for {ticker}")
        except Exception as e:
            print(f"Error clearing existing data for {ticker}: {e}")
            pass
        
        # Add new documents
        Chroma.from_documents(
            all_chunks,
            embeddings,
            persist_directory=CHROMA_PATH
        )
        
        print(f"Enhanced {len(all_chunks)} PDF chunks for {ticker} added to ChromaDB.")
        
           
    else:
        print("No content found for PDF ingestion.")

def download_and_enhanced_ingest(ticker: str):
    """
    Download 10-K reports and ingest with enhanced PDF processing.
    """
    pdf_files = download_10k_reports(ticker)
    if pdf_files:
        enhanced_ingest_10k_pdf(ticker, pdf_files)
    else:
        print(f"No PDF files downloaded for {ticker}")

def download_10k_reports(ticker: str):
    """Download function that gets HTML URLs and converts them to PDFs using sec_api"""
   
    print(f"Searching for 10-K filings for {ticker}...")
    
    # Initialize SEC API clients
    pdfGeneratorApi = PdfGeneratorApi(api_key=os.getenv("SEC_API_KEY"))
    queryApi = QueryApi(api_key=os.getenv("SEC_API_KEY"))
    
    try:
        # Search for 10-K filings
        query = {
            "query": {
                "query_string": {
                    "query": f"formType:\"10-K\" AND ticker:\"{ticker}\" AND filedAt:[2020-01-01 TO 2025-12-31]"
                }
            },
            "from": "0",
            "size": "100" # Adjust size as needed
        }

        response = queryApi.get_filings(query)
        
        if not response or "filings" not in response or not response["filings"]:
            print(f"No filings found for {ticker} or invalid response structure.")
            return []

        # Extract filing URLs from the response
        html_urls = [filing["linkToFilingDetails"] for filing in response["filings"]]
        
        if not html_urls:
            print(f"No 10-K filings found for {ticker}")
            return []
        
        # Convert HTML URLs to PDFs and save them
        pdf_files = []
        save_path = os.path.join(os.getcwd(), "data", "pdfs", ticker)
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Converting {len(html_urls)} HTML filings to PDFs...")
        
        for i, html_url in enumerate(html_urls):
            try:
                print(f"Converting {html_url} to PDF...")
                pdf_content = pdfGeneratorApi.get_pdf(html_url)
                
                # Handle None case
                if pdf_content is None:
                    print(f"PDF content is None for {html_url}")
                    continue
                
                # Save PDF to file
                pdf_filename = f"{ticker}_10K_{i+1}.pdf"
                pdf_path = os.path.join(save_path, pdf_filename)
                
                with open(pdf_path, "wb") as pdf_file:
                    pdf_file.write(pdf_content)
                
                pdf_files.append(pdf_path)
                print(f"Saved PDF: {pdf_path}")
                
            except Exception as e:
                print(f"Error converting {html_url} to PDF: {e}")
                continue
        
        print(f"Successfully converted {len(pdf_files)} filings to PDFs for {ticker}")
        return pdf_files
        
    except Exception as e:
        print(f"Search or conversion error: {e}")
        return []

if __name__ == "__main__":
  
    ticker = "AAPL"
    #pdf_files = download_10k_reports(ticker)
    download_and_enhanced_ingest(ticker)
