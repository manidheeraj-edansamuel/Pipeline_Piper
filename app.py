import streamlit as st
from transformers import pipeline
import PyPDF2
import docx

# Set page configuration
st.set_page_config(
    page_title="Pipeline Piper",
    page_icon="🪈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title & Intro
st.title("🪈 Pipeline Piper")
st.subheader("A Mini AI Toolkit powered by Hugging Face (Custom Models)")
st.markdown("---")

# Helper function to read uploaded documents
def read_document(uploaded_file):
    if uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    elif uploaded_file.name.endswith('.docx'):
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    return ""

# Sidebar Navigation
st.sidebar.header("Navigation")
pipeline_choice = st.sidebar.radio(
    "Choose an AI Pipeline:",
    [
        "1. Sentiment Analysis",
        "2. Text Generation",
        "3. Summarization",
        "4. Question Answering",
        "5. Zero-Shot Classification",
        "6. NER (Named Entity Recognition)",
        "7. Translation",
        "8. Document QA (Advanced QA)",
        "9. Text-to-Text Gen (Paraphrasing)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Model Notice:** The assigned specialized model will download on its first run. Larger models like BART or Flan-T5 might take a minute.")

# -------------------------------------------------------------------------
# 1. SENTIMENT ANALYSIS
# -------------------------------------------------------------------------
if pipeline_choice.startswith("1."):
    st.header("🎭 Sentiment Analysis")
    st.caption("Model: `cardiffnlp/twitter-roberta-base-sentiment`")
    user_input = st.text_area("Enter text to analyze sentiment:", "This toolkit works perfectly!")
    
    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing..."):
            # Map RoBERTa's native labels to human-readable strings
            label_mapping = {"LABEL_0": "Negative 🔴", "LABEL_1": "Neutral 🟡", "LABEL_2": "Positive 🟢"}
            classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
            result = classifier(user_input)[0]
            
            readable_label = label_mapping.get(result['label'], result['label'])
            st.success("Analysis Complete!")
            st.metric(label="Sentiment Prediction", value=readable_label)
            st.metric(label="Confidence Score", value=f"{result['score']:.4f}")

# -------------------------------------------------------------------------
# 2. TEXT GENERATION
# -------------------------------------------------------------------------
elif pipeline_choice.startswith("2."):
    st.header("✍️ Text Generation")
    st.caption("Model: `gpt2`")
    prompt = st.text_area("Enter a prompt to start the text:", "The future of automation relies on...")
    max_length = st.slider("Max Length", 20, 300, 100)
    
    if st.button("Generate Text"):
        with st.spinner("Generating..."):
            generator = pipeline("text-generation", model="gpt2")
            result = generator(prompt, max_length=max_length, num_return_sequences=1)
            
            st.success("Generation Complete!")
            st.write(result[0]['generated_text'])

# -------------------------------------------------------------------------
# 3. SUMMARIZATION
# -------------------------------------------------------------------------
elif pipeline_choice.startswith("3."):
    st.header("📝 Summarization")
    st.caption("Model: `facebook/bart-large-cnn`")
    default_text = (
        "Artificial intelligence text generation systems have evolved rapidly over the last several years. "
        "By analyzing billions of pages of text available on the open web, these deep learning networks "
        "learn the statistical properties of human language. This allows them to predict successive words "
        "in a sentence with stunning accuracy, mimicking human style, tone, and logical flow. While highly "
        "capable, developers must implement structural guardrails to prevent hallucination, bias, or the "
        "output of incorrect facts during complex reasoning tasks."
    )
    text_to_sum = st.text_area("Enter text to summarize:", default_text, height=150)
    
    if st.button("Summarize"):
        with st.spinner("Summarizing..."):
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            result = summarizer(text_to_sum, max_length=60, min_length=20, do_sample=False)
            
            st.success("Summary Generated!")
            st.write(result[0]['summary_text'])

# -------------------------------------------------------------------------
# 4. QUESTION ANSWERING
# -------------------------------------------------------------------------
elif pipeline_choice.startswith("4."):
    st.header("❓ Question Answering")
    st.caption("Model: `deepset/roberta-base-squad2`")
    context = st.text_area("Context:", "Deep learning models excel at answering structural questions based strictly on given context paragraphs.")
    question = st.text_input("Question:", "What do deep learning models excel at?")
    
    if st.button("Find Answer"):
        if context and question:
            with st.spinner("Searching context..."):
                qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
                result = qa_pipeline(question=question, context=context)
                
                st.success("Answer Found!")
                st.markdown(f"**Answer:** {result['answer']}")
                st.caption(f"Score: {result['score']:.4f}")
        else:
            st.warning("Please provide both context and a question.")

# -------------------------------------------------------------------------
# 5. ZERO-SHOT CLASSIFICATION
# -------------------------------------------------------------------------
elif pipeline_choice.startswith("5."):
    st.header("🎯 Zero-Shot Classification")
    st.caption("Model: `facebook/bart-large-mnli`")
    text_to_classify = st.text_area("Text to classify:", "We are launching a new rocket to Mars next Thursday evening.")
    candidate_labels_input = st.text_input("Candidate Labels (comma-separated):", "space, cooking, real estate, education")
    
    if st.button("Classify"):
        labels = [label.strip() for label in candidate_labels_input.split(",")]
        if text_to_classify and labels:
            with st.spinner("Classifying..."):
                zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
                result = zero_shot(text_to_classify, candidate_labels=labels)
                
                st.success("Classification Results:")
                for score, label in zip(result['scores'], result['labels']):
                    st.write(f"• **{label}**: {score*100:.2f}%")
        else:
            st.warning("Please provide both text and candidate labels.")

# -------------------------------------------------------------------------
# 6. NER (NAMED ENTITY RECOGNITION)
# -------------------------------------------------------------------------
elif pipeline_choice.startswith("6."):
    st.header("🏷️ Named Entity Recognition (NER)")
    st.caption("Model: `dslim/bert-base-NER`")
    ner_input = st.text_area("Enter text to extract entities:", "Alice works at Google in London.")
    
    if st.button("Extract Entities"):
        with st.spinner("Extracting..."):
            ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
            results = ner(ner_input)
            
            st.success("Entities Extracted:")
            if results:
                for entity in results:
                    st.markdown(f"- **{entity['word']}** → Category: `{entity['entity_group']}` (Confidence: {entity['score']:.2f})")
            else:
                st.info("No distinct entities found.")

# -------------------------------------------------------------------------
# 7. TRANSLATION
# -------------------------------------------------------------------------
elif pipeline_choice.startswith("7."):
    st.header("🌐 Translation")
    st.caption("Model: `Helsinki-NLP/opus-mt-en-fr`")
    trans_text = st.text_area("Enter English text to translate to French:", "This toolkit makes NLP pipelines incredibly accessible.")
    
    if st.button("Translate"):
        with st.spinner("Translating..."):
            translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
            result = translator(trans_text)
            
            st.success("Translated Text (French):")
            st.write(result[0]['translation_text'])

# -------------------------------------------------------------------------
# 8. DOCUMENT QA (ADVANCED QA)
# -------------------------------------------------------------------------
elif pipeline_choice.startswith("8."):
    st.header("📂 Document QA (Advanced QA)")
    st.caption("Model: `impira/layoutlm-document-qa`")
    uploaded_file = st.file_uploader("Upload a document (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
    doc_question = st.text_input("Ask a question about the document:")
    
    if st.button("Query Document"):
        if uploaded_file and doc_question:
            with st.spinner("Reading document and running LayoutLM..."):
                document_text = read_document(uploaded_file)
                
                if document_text.strip():
                    # LayoutLM Document QA pipeline natively prefers an image or document path context, 
                    # but gracefully processes string contexts for textual files when setup as general QA.
                    doc_qa = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
                    
                    # Truncating content to stay inside model context constraints
                    result = doc_qa(question=doc_question, context=document_text[:2000])
                    
                    st.success("Answer Found from Document:")
                    if isinstance(result, list) and len(result) > 0:
                        st.markdown(f"**Answer:** {result[0]['answer']}")
                    elif isinstance(result, dict):
                        st.markdown(f"**Answer:** {result['answer']}")
                else:
                    st.error("Could not extract any readable text from the file.")
        else:
            st.warning("Please upload a file and type a question.")

# -------------------------------------------------------------------------
# 9. TEXT-TO-TEXT GENERATION (PARAPHRASING / ETC.)
# -------------------------------------------------------------------------
elif pipeline_choice.startswith("9."):
    st.header("🔄 Text to Text Generation (Paraphrasing & More)")
    st.caption("Model: `google/flan-t5-base`")
    
    task_mode = st.selectbox("Choose Task Type:", ["Paraphrase", "Custom Prompt"])
    
    if task_mode == "Paraphrase":
        user_input = st.text_area("Enter text to paraphrase:", "The implementation of AI applications can drastically optimize modern workflows.")
        prompt_text = f"Paraphrase this sentence: {user_input}"
    else:
        prompt_text = st.text_area("Enter a custom text-to-text prompt:", "Answer the following question: What is the capital of France?")
    
    if st.button("Generate"):
        with st.spinner("Processing via Flan-T5..."):
            t2t_gen = pipeline("text2text-generation", model="google/flan-t5-base")
            result = t2t_gen(prompt_text, max_length=128)
            
            st.success("Output Result:")
            st.write(result[0]['generated_text'])