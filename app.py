import streamlit as st
from docx import Document
from PyPDF2 import PdfFileReader
import requests
from dotenv import load_dotenv
import os
import datetime
from typing import Optional, Dict, List
from PIL import Image
from io import BytesIO

# Get the public URL of the API from the environment variable
api_url =  r"https://1c27-35-237-199-106.ngrok-free.app"

# Global list to store results with timestamps
history: List[Dict[str, Optional[str]]] = []

# Global variable to store API status
api_status: str = "Unknown"

def check_api_status() -> str:
    """
    Checks the status of the API.

    Returns:
        str: The status of the API ("Running", "Not Available", etc.).
    """
    try:
        response = requests.get(f"{api_url}/status")
        response.raise_for_status()
        if response.json().get("status") == "API is running":
            return "Running"
        else:
            return "Unknown Status"
    except requests.exceptions.RequestException:
        return "Not Available"

def read_text_file(file) -> str:
    """
    Reads a plain text file.

    Args:
        file: A file-like object containing the text file.

    Returns:
        str: The content of the text file as a string.
    """
    try:
        return file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading text file: {e}")
        return ""

def read_docx_file(file) -> str:
    """
    Reads a DOCX file.

    Args:
        file: A file-like object containing the DOCX file.

    Returns:
        str: The content of the DOCX file as a string.
    """
    try:
        doc = Document(file)
        full_text = [paragraph.text for paragraph in doc.paragraphs]
        return '\n'.join(full_text)
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return ""

def read_pdf_file(file) -> str:
    """
    Reads a PDF file.

    Args:
        file: A file-like object containing the PDF file.

    Returns:
        str: The content of the PDF file as a string.
    """
    try:
        pdf_reader = PdfFileReader(file)
        text = [pdf_reader.getPage(page).extract_text() for page in range(pdf_reader.getNumPages())]
        return '\n'.join(text)
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def process_file(uploaded_file) -> Optional[str]:
    """
    Processes the uploaded file based on its type.

    Args:
        uploaded_file: A file-like object uploaded by the user.

    Returns:
        Optional[str]: The extracted text from the file, or None if an unsupported file type.
    """
    try:
        if uploaded_file.type == "text/plain":
            return read_text_file(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return read_docx_file(uploaded_file)
        elif uploaded_file.type == "application/pdf":
            return read_pdf_file(uploaded_file)
        else:
            st.error("Unsupported file type")
            return None
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

def get_image_recommendation(query: str , use_ai: bool = False) -> Optional[Image.Image]:
    """
    Fetches an image recommendation from the API.

    Args:
        query (str): The query string to send to the API.

    Returns:
        Optional[Image.Image]: The recommended image or None if there was an error.
    """
    try:
        response = requests.get(f"{api_url}/recommend_images", params={"query": query,"use_ai":use_ai})
        print(response)
        return response
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        st.error(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        st.error(f"Request error occurred: {req_err}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    
    return None

# def get_video_recommendation(query: str) -> Optional[str]:
#     """
#     Fetches a video recommendation from the API.

#     Args:
#         query (str): The query string to send to the API.

#     Returns:
#         Optional[str]: The path to the recommended video or None if there was an error.
#     """
#     try:
#         response = requests.get(f"{api_url}/recommend_video", params={"query": query})
#         response.raise_for_status()
#         # Save video to a temporary file
#         video_path = f"video_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
#         with open(video_path, 'wb') as video_file:
#             video_file.write(response.content)
#         return video_path
#     except requests.exceptions.HTTPError as http_err:
#         st.error(f"HTTP error occurred: {http_err}")
#     except requests.exceptions.ConnectionError as conn_err:
#         st.error(f"Connection error occurred: {conn_err}")
#     except requests.exceptions.Timeout as timeout_err:
#         st.error(f"Timeout error occurred: {timeout_err}")
#     except requests.exceptions.RequestException as req_err:
#         st.error(f"Request error occurred: {req_err}")
#     except Exception as e:
#         st.error(f"An unexpected error occurred: {e}")
    
    return None

def main() -> None:
    global api_status
    api_status = check_api_status()

    st.title("Text-based Image/Video Recommendation App")

    # Create tabs for navigation
    tabs = st.tabs(["Introduction", "Demo", "History"])

    with tabs[0]:
        st.header("Introduction")
        st.write("""
            This application allows you to input text or upload text files (txt, docx, pdf) and receive 
            recommendations for images or videos that match the content of the text. The app sends your 
            text data to an external API and displays the recommended images or videos directly within the app.
            
            ### How to Use:
            1. Go to the Demo tab.
            2. Enter text or upload a file.
            3. Choose whether you want photos, videos, or both.
            4. Submit to see recommended images/videos.
            5. Check the History tab to view past recommendations based on timestamps.
        """)

    with tabs[1]:
        st.header("Demo: Get Image/Video Recommendations")

        # Display API status
        st.write(f"**API Status:** {api_status}")
        
        # User input options
        user_text: str = st.text_area("Enter text here:")
        uploaded_file = st.file_uploader("Or upload a text file (txt, docx, pdf):", type=['txt', 'docx', 'pdf'])

        # Media type selection
        media_type = st.radio(
            "Select the type of recommendations you want:",
            ("AI","Stock Images")
        )

        # Process uploaded file
        if uploaded_file:
            user_text = process_file(uploaded_file)

        if st.button("Submit"):
            if user_text and api_status == "Running":
                results = {"text": user_text}
                if media_type in ["AI"]:
                    # Fetch image recommendation
                    st.write("### Recommended AI Image")
                    image = get_image_recommendation(user_text , use_ai=True)
                    print(image.json())
                    st.image(image.json(), use_column_width=True)
                    # if image:
                    #     st.image(image, use_column_width=True)
                    #     results["AI Image"] = image

                if media_type in ["Stock Images"]:
                    # Fetch video recommendation
                    st.write("### Recommended Stock Images")
                    img_path = get_image_recommendation(user_text , use_ai=False)
                    print(img_path)
                    st.image(img_path.json(), use_column_width=True)
                    # if video_path:
                    #     st.video(video_path)
                    #     results["Stock Image"] = video_path

                # Store results in history with timestamp
                timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                results["timestamp"] = timestamp
                history.append(results)

            elif not user_text:
                st.error("Please enter text or upload a file to get recommendations.")
            elif api_status != "Running":
                st.error("Cannot perform recommendations as the API is not running.")
    
    with tabs[2]:
        st.header("History of Recommendations")
        st.write("Here are your previous recommendations based on the input text:")

        if history:
            for result in history:
                st.write(f"**Timestamp:** {result['timestamp']}")
                st.write(f"**Text:** {result['text']}")
                
                if "AI Image" in result:
                    st.write("**Image:**")
                    st.image(result["image"], use_column_width=True)
                
                if "Stock Image" in result:
                    st.write("**Video:**")
                    st.video(result["video"])
                st.write("---")
        else:
            st.write("No history available yet.")

if __name__ == "__main__":
    main()