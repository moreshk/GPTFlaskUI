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
import os

app = Flask(__name__)

# Your existing code
def reload_documents():
    loader = DirectoryLoader('./text1', glob='**/*.txt')
    return loader.load()

def reload_text_splitter():
    return CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 0)

documents = reload_documents()
text_splitter = reload_text_splitter()
embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(text_splitter.split_documents(documents), embeddings)
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore = docsearch)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answer = qa.run(question)
    return jsonify(answer=answer)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = file.filename
    file.save(os.path.join('./text1', filename))
    global documents, text_splitter, docsearch, qa
    documents = reload_documents()
    text_splitter = reload_text_splitter()
    docsearch = Chroma.from_documents(text_splitter.split_documents(documents), embeddings)
    qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore = docsearch)
    return 'File uploaded successfully!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
