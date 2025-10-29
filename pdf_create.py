import fitz
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
import datetime

def process_and_generate_pdf(mongo_uri: str, db_name: str, collection_name: str,
                             input_pdf_path: str, output_pdf_path: str, agent_name: str):
    """
    Connects to MongoDB, retrieves filtered chunks based on agent_name,
    and generates a new PDF with a cover page and a TOC-style links page.
    """
    try:
        # 1. Connect to MongoDB
        client = MongoClient(mongo_uri)
        client.admin.command('ismaster')
        db = client[db_name]
        collection = db[collection_name]

        print("Successfully connected to MongoDB.")

        # 2. Retrieve only chunks where classification matches agent_name
        chunks_data = []
        for doc in collection.find().sort("chunk_index", 1):
            classifications = doc.get("classification", [])
            for cls in classifications:
                if cls.get("classification", "").lower() == agent_name.lower():
                    chunks_data.append(doc)
                    break  # add once even if multiple matches

        if not chunks_data:
            print(f"No chunks found for agent '{agent_name}'. Exiting.")
            return

        # 3. Open the original PDF
        doc = fitz.open(input_pdf_path)

        # Get current date and time
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        timestamp_fontsize = 8

        # 4. Add the first page (Cover Page)
        cover_page = doc.new_page(pno=0, width=doc[0].rect.width, height=doc[0].rect.height)
        cover_text = "Classification Results"
        cover_fontsize = 28
        text_width = fitz.get_text_length(cover_text, fontname="helv", fontsize=cover_fontsize)
        cover_page.insert_text(
            ((cover_page.rect.width - text_width) / 2, cover_page.rect.height / 2),
            cover_text,
            fontname="helv",
            fontsize=cover_fontsize
        )
        timestamp_width = fitz.get_text_length(timestamp, fontname="helv", fontsize=timestamp_fontsize)
        cover_page.insert_text(
            (cover_page.rect.width - timestamp_width - 20, 20),
            timestamp,
            fontname="helv",
            fontsize=timestamp_fontsize
        )

        # 5. Add the second page for Agent and Links (TOC)
        links_page = doc.new_page(pno=1, width=doc[0].rect.width, height=doc[0].rect.height)
        links_page.insert_text(
            (links_page.rect.width - timestamp_width - 20, 20),
            timestamp,
            fontname="helv",
            fontsize=timestamp_fontsize
        )

        # ---- Title with Agent Name ----
        display_agent_name = agent_name.strip().capitalize() if agent_name.strip() else "N/A"
        title_text = f"{display_agent_name} Classification Agent"
        title_fontsize = 18
        page_width = links_page.rect.width
        text_width = fitz.get_text_length(title_text, fontname="helv", fontsize=title_fontsize)
        title_x = (page_width - text_width) / 2
        links_page.insert_text((title_x, 30), title_text, fontname="helv", fontsize=title_fontsize, color=(0,0,0))

        # ---- Table Layout (TOC) ----
        y_start = 70
        row_height = 20
        table_width = 500  # fixed width for table
        table_x_start = (page_width - table_width) / 2  # center align
        serial_number = 1

        # Column positions relative to table_x_start
        col_sn_x = table_x_start + 20
        col_chunk_x = table_x_start + 80
        col_page_x = table_x_start + 400

        # Header row with background
        header_rect = fitz.Rect(table_x_start, y_start - 6, table_x_start + table_width, y_start + row_height)
        links_page.draw_rect(header_rect, color=(0, 0, 0), fill=(0.9, 0.9, 0.9), width=1)
        links_page.insert_text((col_sn_x, y_start + row_height / 2 - 3), "S.No",
                               fontname="helv", fontsize=10, color=(0, 0, 0))
        links_page.insert_text((col_chunk_x, y_start + row_height / 2 - 3), "Chunk Preview",
                               fontname="helv", fontsize=10, color=(0, 0, 0))
        links_page.insert_text((col_page_x, y_start + row_height / 2 - 3), "Page No.",
                               fontname="helv", fontsize=10, color=(0, 0, 0))

        # Draw vertical lines in header row
        links_page.draw_line((col_chunk_x - 10, y_start - 6), (col_chunk_x - 10, y_start + row_height), color=(0,0,0), width=1)
        links_page.draw_line((col_page_x - 10, y_start - 6), (col_page_x - 10, y_start + row_height), color=(0,0,0), width=1)

        y_position = y_start + row_height

        # 6. Iterate through filtered chunks and add them as TOC rows
        for chunk in chunks_data:
            chunk_text = chunk.get("text", "Text not found")
            max_chars = 70
            if len(chunk_text) > max_chars:
                chunk_text_display = chunk_text[:max_chars].strip() + "..."
            else:
                chunk_text_display = chunk_text.strip()

            target_page_num = chunk.get("page_number", 0)
            target_coords = chunk.get("coordinates")
            if not target_coords or len(target_coords) < 4:
                continue

            # Draw row rectangle
            row_rect = fitz.Rect(table_x_start, y_position, table_x_start + table_width, y_position + row_height)
            links_page.draw_rect(row_rect, color=(0, 0, 0), width=1)

            # ---- Draw vertical lines for columns ----
            links_page.draw_line((col_chunk_x - 10, y_position), (col_chunk_x - 10, y_position + row_height), color=(0,0,0), width=1)
            links_page.draw_line((col_page_x - 10, y_position), (col_page_x - 10, y_position + row_height), color=(0,0,0), width=1)

            # Column 1: Serial number
            links_page.insert_text((col_sn_x, y_position + row_height / 2 + 1),
                                   str(serial_number), fontname="helv", fontsize=9, color=(0, 0, 0))

            # Column 2: Chunk preview (hyperlink)
            link_rect = fitz.Rect(col_chunk_x, y_position - 2, col_page_x - 10, y_position + row_height)
            links_page.insert_text(
                (col_chunk_x, y_position + row_height / 2 + 1),
                chunk_text_display,
                fontname="helv",
                fontsize=9,
                color=(0, 0, 1)
            )

            # Insert hyperlink
            links_page.insert_link({
                "kind": fitz.LINK_GOTO,
                "page": target_page_num + 1,
                "from": link_rect,
                "to": fitz.Point(target_coords[0], target_coords[1])
            })

            # Column 3: Page number
            links_page.insert_text((col_page_x, y_position + row_height / 2 + 1),
                                   str(target_page_num), fontname="helv", fontsize=9, color=(0, 0, 0))

            y_position += row_height
            serial_number += 1  # <<< serial number increment fixed here

        # 7. Save the modified PDF
        doc.save(output_pdf_path)
        doc.close()
        print(f"Successfully generated '{output_pdf_path}' for agent '{agent_name}'.")

    except ConnectionFailure:
        print("Could not connect to MongoDB. Check connection string and server.")
    except PyMongoError as e:
        print(f"A MongoDB error occurred: {e}")
    except FileNotFoundError:
        print(f"The input PDF file '{input_pdf_path}' was not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    mongo_uri = "mongodb://localhost:27017/"
    db_name = "document_classification"
    collection_name = "chunks"

    input_pdf_path = "test70_90.pdf"
    output_pdf_path = "agent_filtered_chunks.pdf"

    agent_name = input("Enter Classification Agent Name: ")
    process_and_generate_pdf(mongo_uri, db_name, collection_name, input_pdf_path, output_pdf_path, agent_name)
