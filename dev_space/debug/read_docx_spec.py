import zipfile
import re
import os

def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as zf:
            xml_content = zf.read('word/document.xml').decode('utf-8')
            # Simple text extraction (remove xml tags)
            text = re.sub(r'<[^>]+>', '', xml_content)
            # Or just search directly in XML to avoid context loss
            return xml_content
    except Exception as e:
        return f"Error: {e}"

def search_specs():
    base_dir = r"D:\spikehunter\KRX_OPENAPI"
    files = [f for f in os.listdir(base_dir) if f.endswith(".docx")]
    
    print(f"Found {len(files)} spec files.")
    
    for f in files:
        path = os.path.join(base_dir, f)
        print(f"\nScanning: {f}...")
        content = extract_text_from_docx(path)
        
        # Search for URLs
        urls = re.findall(r'https?://[^\s<"]+|/service/[^\s<"]+', content)
        # Filter for relevant KRX URLs
        krx_urls = [u for u in urls if "krx.co.kr" in u or "/service/" in u]
        
        if krx_urls:
            print("  [URLs Found]:")
            for u in set(krx_urls):
                print(f"  - {u}")
        
        # Search for potential params (CamelCase or ALL_CAPS often used in KRX)
        # Look for common candidates
        candidates = ["AUTH_KEY", "authKey", "serviceKey", "basDd", "strtDd", "endDd", "isuCd", "mktId"]
        found_params = []
        for c in candidates:
            if c in content:
                found_params.append(c)
        
        if found_params:
            print(f"  [Params Found]: {found_params}")
            
        # Search for Output Fields (OutBlock)
        # Look for common field names in the doc
        out_candidates = [
            "BAS_DD", "IDX_CLSPRC", "CLSPRC", "IDX_OPNPRC", "OPNPRC", 
            "IDX_HGPRC", "HGPRC", "IDX_LWPRC", "LWPRC", 
            "ACC_TRDVOL", "ACC_TRDVAL", "MKTCAP", "IDX_NM"
        ]
        found_outputs = []
        for c in out_candidates:
            if c in content:
                found_outputs.append(c)
        
        if found_outputs:
            print(f"  [Output Fields Found]: {found_outputs}")

        # extract text chunks near "Request" or "요청"
        # Not easy in raw XML but let's try to print 100 chars around the URL
        if krx_urls:
            idx = content.find(list(set(krx_urls))[0])
            start = max(0, idx - 200)
            end = min(len(content), idx + 200)
            print(f"  [Context around URL]: ...{content[start:end]}...")

if __name__ == "__main__":
    search_specs()
