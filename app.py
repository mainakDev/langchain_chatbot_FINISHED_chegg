from flask import Flask, render_template, request, jsonify
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
import chromadb

import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Load the vector database and QA chain

def load_db():
    try:
        
        print("Initializing embeddings...")
        embeddings = CohereEmbeddings(
            cohere_api_key=os.environ["COHERE_API_KEY"], 
            model="embed-english-v3.0"
            )
        
        print("Loading Chroma vector store...")
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)

        print("Number of documents in Chroma DB:", vectordb._collection.count())

        print("Initializing ChatCohere...")
        llm = ChatCohere(cohere_api_key=os.environ["COHERE_API_KEY"])

        print("Building RetrievalQA chain...")
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )

        print("QA system initialized successfully.")
        return qa
    
    except Exception as e:
        print("Error initializing QA system:", e)
        return None

qa = load_db()

def answer_from_knowledgebase(message):
    try:
        res = qa.invoke({"query": message})
        source_docs = res.get('source_documents', [])
        
        if not source_docs:
            return "No relevant knowledge found in the database."

        return res['result']
    except Exception as e:
        print("Error during QA invocation:", e)
        return "Sorry, I couldn't retrieve an answer."

# Search knowledgebase and return sources
def search_knowledgebase(message):
    try:
        res = qa.invoke({"query": message})
        source_docs = res.get('source_documents', [])
        if not source_docs:
            return "No sources found for your query."
        sources = ""
        for count, source in enumerate(source_docs, 1):
            sources += f"Source {count}\n{source.page_content}\n"
        return sources
    except Exception as e:
        print("Error during source retrieval:", e)
        return "Error retrieving sources."

# Chatbot response as expert Python developer
def answer_as_chatbot(message):
    template = """Question: {question}

Answer as if you are an expert Python developer."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = ChatCohere(cohere_api_key=os.environ["COHERE_API_KEY"])
    chain = RunnableSequence(prompt | llm)
    res = chain.invoke({"question": message})
    return res.content  # Extract plain text from AIMessage

@app.route('/kbanswer', methods=['POST'])
def kbanswer():
    message = request.json['message']
    if qa is None:
        return jsonify({'error': 'Knowledgebase not initialized'}), 500
    response_message = answer_from_knowledgebase(message)
    return jsonify({'message': response_message}), 200

@app.route('/search', methods=['POST'])
def search():
    message = request.json['message']
    if qa is None:
        return jsonify({'error': 'Knowledgebase not initialized'}), 500
    response_message = search_knowledgebase(message)
    return jsonify({'message': response_message}), 200

@app.route('/answer', methods=['POST'])
def answer():
    message = request.json['message']
    response_message = answer_as_chatbot(message)
    return jsonify({'message': response_message}), 200

@app.route("/")
def index():
    return render_template("index.html", title="")

#Route used for debugging, can be removed
@app.route('/debug-docs', methods=['GET'])
def debug_docs():
    docs = qa.retriever.get_relevant_documents("mosquito")
    return jsonify({
        f"Document {i+1}": doc.page_content
        for i, doc in enumerate(docs)
    })


if __name__ == "__main__":
    app.run()