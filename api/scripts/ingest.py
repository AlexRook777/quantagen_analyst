from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sec_edgar_downloader._Downloader import Downloader
import os
import glob

CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(os.getcwd(), "data", "chroma_data"))

def ingest_10k_html(ticker: str, html_files: list):
    """
    Ingests HTML files for a given ticker into ChromaDB.

    Parameters
    ----------
    ticker : str
        The ticker symbol for the company.
    html_files : list
        A list of HTML file paths to ingest.

    Notes
    -----
    This function assumes that the HTML files contain 10-K reports for the given ticker.
    It will load the HTML files, split them into chunks, add metadata to each chunk, and then
    add the chunks to ChromaDB. If no HTML files are found, it will print a message
    indicating that no files were found for ingestion.
    """
    print(f"Ingesting HTML files for {ticker}...")
    all_chunks = []
    for html_path in html_files:
        print(f"Loading {html_path}...")
        loader = UnstructuredHTMLLoader(html_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata = {"ticker": ticker, "source": html_path}
        all_chunks.extend(chunks)
        print(f"Ingested {len(chunks)} chunks from {html_path}")
    if all_chunks:
        print(f"Adding {len(all_chunks)} chunks for {ticker} to ChromaDB...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma.from_documents(all_chunks, embeddings, persist_directory=CHROMA_PATH)
        print(f"Total {len(all_chunks)} chunks for {ticker} added to ChromaDB.")
    else:
        print("No HTML files found for ingestion.")

def download_10k_reports(ticker: str):
    """
    Downloads 10-K reports for a given ticker from the SEC EDGAR database.

    Parameters
    ----------
    ticker : str
        The ticker symbol for the company.

    Returns
    -------
    list
        A list of HTML file paths downloaded from the SEC EDGAR database.

    Notes
    -----
    This function assumes that an SEC EDGAR API key has been set as an environment variable.
    It will create a directory in the current working directory to save the downloaded files.
    If an error occurs while downloading, it will print an error message and return an empty list.
    """
    save_path = os.path.join(os.getcwd(), "data")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(f"Downloading 10-K reports for {ticker} to {save_path}...")
    dl = Downloader("MyProject", "my.email@example.com", download_folder=save_path)

    try:
        print(f"Downloading 10-K reports for {ticker} to {save_path}...")
        dl.get("10-K", ticker, limit=5, download_details=True)
        #remove file "full-submission.txt" in each created folder
        for folder in glob.glob(os.path.join(dl.download_folder, "sec-edgar-filings", ticker, "10-K", "*")):
            full_submission_path = os.path.join(folder, "full-submission.txt")
            if os.path.exists(full_submission_path):
                os.remove(full_submission_path)

        html_files = glob.glob(os.path.join(dl.download_folder, "sec-edgar-filings", ticker, "10-K", "*", "primary-document.html"))

        print("Downloaded files:", len(html_files))
        for f in html_files:
            print(f)
        return html_files

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    # Example usage
    ticker: str = "AAPL"
    html_files = download_10k_reports(ticker)
    ingest_10k_html(ticker, html_files)

    