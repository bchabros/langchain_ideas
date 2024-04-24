from langchain.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv


load_dotenv()

embeddings = OpenAIEmbeddings()

# Chunking text
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

# Loading files with document loaders
loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
)

# for doc in docs:
#     print(doc.page_content)
#     print("\n")

# create emb file with vector store
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)

# results = db.similarity_search_with_score("What is an interesting fact about English language?")

# for result in results:
#     print("\n")
#     print(result[1])
#     print(result[0].page_content)

results = db.similarity_search("What is an interesting fact about English language?")

for result in results:
    print("\n")
    print(result.page_content)
