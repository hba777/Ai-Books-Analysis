import fitz  # PyMuPDF

def highlight_pdf(input_pdf, output_pdf, page_number, coords):
    """
    Highlights given coordinates in the PDF.
    
    Args:
        input_pdf (str): Path to input PDF file.
        output_pdf (str): Path to save highlighted PDF.
        page_number (int): Page number (1-based).
        coords (tuple): (x0, top, x1, bottom) coordinates.
    """
    doc = fitz.open(input_pdf)

    if page_number < 1 or page_number > len(doc):
        raise ValueError("Invalid page number")

    page = doc[page_number - 1]
    rect = fitz.Rect(coords)  # Create rectangle from coords

    # Add highlight annotation
    highlight = page.add_highlight_annot(rect)
    highlight.update()

    doc.save(output_pdf)
    doc.close()
    print(f"Highlighted PDF saved as: {output_pdf}")

# =============== RUN TEST ===============

# Example inputs (replace with your chunk coords)
pdf_path = r"The Lost War.pdf"
output_path = r"highlighted_test.pdf"

# Replace with actual chunk metadata
page_num = int(input("Enter page number: "))
x0 = float(input("Enter x0: "))
top = float(input("Enter top: "))
x1 = float(input("Enter x1: "))
bottom = float(input("Enter bottom: "))

highlight_pdf(pdf_path, output_path, page_num, (x0, top, x1, bottom))