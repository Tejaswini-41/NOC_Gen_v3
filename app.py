import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


def extract_text_and_images_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    all_text = []
    all_images_ocr = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)

        # Extract text
        text = page.get_text()
        all_text.append(text)

        # Extract images and perform OCR
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            # Perform OCR on the image
            ocr_text = pytesseract.image_to_string(image)
            all_images_ocr.append(ocr_text)

    return "\n".join(all_text), "\n".join(all_images_ocr)


def extract_text_from_word(docx_path):
    doc = docx.Document(docx_path)
    all_text = []

    for paragraph in doc.paragraphs:
        all_text.append(paragraph.text)

    return "\n".join(all_text)


def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def extract_content(file):
    file_type = file.type
    if file_type == 'application/pdf':
        return extract_text_and_images_from_pdf(file)
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return extract_text_from_word(file), ""
    elif file_type == 'text/plain':
        return extract_text_from_txt(file), ""
    else:
        raise ValueError("Unsupported file format")


def vectorize_texts(texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer


def compare_texts(tfidf_matrix):
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    return cosine_sim_matrix


def compare_documents(documents):
    all_texts = []
    all_ocr_texts = []

    for document in documents:
        text, ocr_text = extract_content(document)
        all_texts.append(text)
        all_ocr_texts.append(ocr_text)

    # Combine text and OCR results
    combined_texts = [text + " " + ocr_text for text, ocr_text in zip(all_texts, all_ocr_texts)]

    # Vectorize and compare texts
    tfidf_matrix, vectorizer = vectorize_texts(combined_texts)
    cosine_sim_matrix = compare_texts(tfidf_matrix)

    # Calculate similarity ratio
    similarity_ratios = []
    for i in range(len(cosine_sim_matrix)):
        for j in range(i + 1, len(cosine_sim_matrix)):
            ratio = cosine_sim_matrix[i][j]
            similarity_ratios.append((documents[i].name, documents[j].name, ratio))

    return similarity_ratios


def main():
    st.title("Document Similarity Detection")

    uploaded_files = st.file_uploader("Choose documents", accept_multiple_files=True)

    if uploaded_files:
        with st.spinner('Processing...'):
            similarity_ratios = compare_documents(uploaded_files)

            if similarity_ratios:
                st.write("Document Similarity Results:")
                for doc1, doc2, ratio in similarity_ratios:
                    st.write(f"Similarity between {doc1} and {doc2}: {ratio * 100:.2f}%")
            else:
                st.write("No documents to compare.")


if __name__ == "__main__":
    main()
