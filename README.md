step1:git clone https://github.com/rash-aad/smart_lawyer_assistant_Rag_Based.git
cd smart_lawyer_assistant_Rag_Based

step2:python -m venv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate


step3:pip install -r requirements.txt


step4:ollama pull llama3.2

step5:streamlit run app.py
