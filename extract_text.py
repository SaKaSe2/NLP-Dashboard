import zipfile
import xml.etree.ElementTree as ET
import json

def extract_text_from_docx(docx_path, out_path):
    try:
        document = zipfile.ZipFile(docx_path)
        xml_content = document.read('word/document.xml')
        document.close()
        tree = ET.XML(xml_content)
        
        WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        PARA = WORD_NAMESPACE + 'p'
        TEXT = WORD_NAMESPACE + 't'
        
        paragraphs = []
        for paragraph in tree.iter(PARA):
            texts = [node.text for node in paragraph.iter(TEXT) if node.text]
            if texts:
                paragraphs.append(''.join(texts))
                
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(paragraphs))
    except Exception as e:
        print(str(e))

extract_text_from_docx(r"c:\Semester6\Teori\NLP\Tugas 2_NLP.docx", r"c:\Semester6\Teori\NLP\Tugas_2_NLP.txt")


with open(r"c:\Semester6\Teori\NLP\Data_Preprocessing_Transformation.ipynb", 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(r"c:\Semester6\Teori\NLP\notebook_output.md", 'w', encoding='utf-8') as f:
    for c in nb.get('cells', []):
        if c['cell_type'] in ('markdown', 'code'):
            f.write(f"--- {c['cell_type'].upper()} ---\n")
            f.write("".join(c.get('source', [])))
            f.write("\n\n")
