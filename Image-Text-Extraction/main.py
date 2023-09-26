import streamlit as st
#to convert image into text
import easyocr
from PIL import Image
import io

# Function to extract and format text from an image using easyocr
def extract_and_format_text(image):
    reader = easyocr.Reader(['en'])
    # Convert PIL Image to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()
    
    # Extract text and coordinates from the image
    results = reader.readtext(img_bytes)

    # Extract text content from the results
    extracted_text = [result[1] for result in results]

    return extracted_text

# Main Streamlit app
def main():
    st.title("Image Text Extraction")

    # File upload widget
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Button to extract text
        if st.button("Extract Text"):
            # Extract and format text from the image
            extracted_text = extract_and_format_text(image)
            
            # Display extracted text
            st.subheader("Extracted Text:")
            for text in extracted_text:
                st.write(text)

if __name__ == "__main__":
    main()
