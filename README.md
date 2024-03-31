# PDF Converter

This Python application allows you to convert image-based PDF files into structured data by extracting text from the images. It provides a user interface built with Streamlit for easy interaction.

## Dependencies

Before running the application, make sure you have the following dependencies installed:

- `streamlit`
- `pandas`
- `pdf2image`
- `easyocr`
- `numpy`
- `opencv-python`
- `pillow`
- `matplotlib`

You can install these dependencies using pip:

```
pip install streamlit pandas pdf2image easyocr numpy opencv-python pillow matplotlib
```

## Usage

1. Clone the repository or download the source code.
2. Install the dependencies mentioned above.
3. Extract Poppler into the same folder as img_pdf_worker.py
4. Run the `img_pdf_worker.py.py` file using Streamlit:

```
streamlit run img_pdf_worker.py
```

4. Upload a PDF file containing images.
5. Configure the settings in the sidebar, such as language, rotation degrees, scale, etc.
6. Navigate through the pages of the PDF file and create bounding boxes around the text areas.
7. Define templates by assigning variable names to the bounding boxes.
8. Apply templates to extract text from the images and structure it into a DataFrame.
9. Review and edit the extracted data if necessary.
10. Download the structured data as an Excel file.

## Features

- Load PDF files containing images.
- Configure settings for image processing and OCR.
- Create bounding boxes around text areas.
- Define templates for extracting structured data.
- Apply templates to extract text and structure it into a DataFrame.
- Edit and review the extracted data.
- Download the structured data as an Excel file.

## License

This project is licensed under the BSD - 2 Clause License
