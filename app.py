from flask import Flask, render_template, request, jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from dotenv import load_dotenv
load_dotenv()
import magic
import nltk
# ... other imports

app = Flask(__name__)

# Your existing code
loader = DirectoryLoader('../text', glob='**/*.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore = docsearch)
# ... other code

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answer = qa.run(question)
    return jsonify(answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
