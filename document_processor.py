import requests, PyPDF2, io
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk, os, warnings
from PyPDF2.errors import PdfReadWarning

# Suppress PyPDF2 warnings to avoid cluttering output
warnings.filterwarnings("ignore", category=PdfReadWarning)

# Ensure NLTK data is accessible and download required tokenizer if missing
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True) # 'punkt' is not work

class DocumentProcessor:
    """
    Class to handle document processing for Retrieval-Augmented Generation (RAG) mode.
    It supports loading documents from local files or URLs, splitting them into chunks,
    and finding relevant context using TF-IDF similarity.
    """

    def __init__(self):
        # Initialize empty document storage and TF-IDF vectorizer
        self.documents = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None

    def load_documents(self, paths):
        """
        Load documents from the specified paths (local or URLs) and process them into chunks.

        Args:
            paths (list): List of file paths or URLs to process.
        """
        self.documents = []  # Clear existing documents
        for path in paths:
            try:
                if os.path.isfile(path):  # Check if it's a local file
                    if path.endswith('.pdf'):
                        content = self._process_local_pdf(path)
                    else:
                        raise ValueError(f"Unsupported file format: {path}")
                else:  # Assume it's a URL
                    if path.endswith('.pdf'):
                        content = self._process_pdf_from_url(path)
                    else:
                        content = self._process_webpage(path)

                # Split the content into manageable chunks
                chunks = self._split_into_chunks(content)
                self.documents.extend(chunks)
                print(f"Processed {path} into {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")

        # Fit the TF-IDF vectorizer on the loaded documents
        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def _process_local_pdf(self, file_path):
        """
        Extract text content from a local PDF file.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            str: Extracted text content.
        """
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            return ' '.join(page.extract_text() for page in pdf_reader.pages)

    def _process_pdf_from_url(self, url):
        """
        Extract text content from a PDF file available at a URL.

        Args:
            url (str): URL pointing to the PDF file.

        Returns:
            str: Extracted text content.
        """
        response = requests.get(url)
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return ' '.join(page.extract_text() for page in pdf_reader.pages)

    def _process_webpage(self, url):
        """
        Extract text content from a webpage.

        Args:
            url (str): URL of the webpage.

        Returns:
            str: Extracted text content.
        """
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()

    def _split_into_chunks(self, text, chunk_size=512):
        """
        Split text into smaller chunks of a specified size.

        Args:
            text (str): Text to split.
            chunk_size (int): Maximum number of words per chunk.

        Returns:
            list: List of text chunks.
        """
        sentences = nltk.sent_tokenize(text)  # Split text into sentences
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > chunk_size:
                # Add the current chunk to the list and start a new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                # Add sentence to the current chunk
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:  # Add any remaining text as a chunk
            chunks.append(' '.join(current_chunk))
        return chunks

    def get_relevant_context(self, query, k=3):
        """
        Retrieve the most relevant chunks of text for a given query using TF-IDF similarity.

        Args:
            query (str): User query to match against the documents.
            k (int): Number of top relevant chunks to return.

        Returns:
            str: Relevant text chunks concatenated as a single string.
        """
        if not self.documents or self.tfidf_matrix is None or self.tfidf_matrix.shape[0] == 0:
            return ""

        # Transform the query into the TF-IDF vector space
        query_vector = self.vectorizer.transform([query])

        # Compute similarity scores between the query and document chunks
        similarities = (self.tfidf_matrix @ query_vector.T).toarray().flatten()

        # Get indices of the top-k most similar chunks
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        # Return the top-k relevant chunks as a single string
        return "\n".join([self.documents[i] for i in top_k_indices])