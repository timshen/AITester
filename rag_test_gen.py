import os
import json
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from json_repair import repair_json
import PyPDF2
import uuid
import re
from datetime import datetime

def load_cases(csv_path="test_cases.csv"):
    try:
        if not os.path.exists(csv_path):
            os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write("需求描述,测试用例\n")
            return pd.DataFrame(columns=['需求描述', '测试用例'])
            
        df = pd.read_csv(csv_path)
        
        if '需求描述' not in df.columns or '测试用例' not in df.columns:
            st.warning(f"CSV文件缺少必要的列（需要'需求描述'和'测试用例'列）。请检查文件格式。")
            return pd.DataFrame(columns=['需求描述', '测试用例'])
        
        def safe_parse_json(json_str):
            if pd.isna(json_str):
                return []
            try:
                cleaned_str = str(json_str).replace("'", "\"").strip()
                return json.loads(repair_json(cleaned_str))
            except Exception as e:
                print(f"JSON解析错误: {e}, 数据: {json_str[:100]}...")
                return []
        
        df['测试用例'] = df['测试用例'].apply(safe_parse_json)
        return df
        
    except Exception as e:
        st.error(f"CSV文件加载失败：{str(e)}（创建了新的test_cases.csv文件结构）")
        try:
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write("需求描述,测试用例\n")
            st.info("已创建新的test_cases.csv文件，可以开始添加测试用例了。")
        except Exception:
            pass
        return pd.DataFrame(columns=['需求描述', '测试用例'])

def load_knowledge_segments(csv_path="knowledge_segments.csv"):
    try:
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        else:
            return pd.DataFrame(columns=['segment_id', 'document_name', 'page_num', 'content'])
    except Exception as e:
        st.error(f"知识库加载失败：{str(e)}")
        return pd.DataFrame(columns=['segment_id', 'document_name', 'page_num', 'content'])

def process_pdf(uploaded_file):
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}.pdf"
    filepath = os.path.join("temp", filename)
    
    os.makedirs("temp", exist_ok=True)
    
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    segments = []
    try:
        pdf_reader = PyPDF2.PdfReader(filepath)
        total_pages = len(pdf_reader.pages)
        
        for page_num in range(total_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            paragraphs = re.split(r'\n\s*\n', text)
            for i, paragraph in enumerate(paragraphs):
                paragraph = paragraph.strip()
                if len(paragraph) > 20:
                    segments.append({
                        'segment_id': f"{filename}_{page_num}_{i}",
                        'document_name': uploaded_file.name,
                        'page_num': page_num + 1,
                        'content': paragraph
                    })
        
        return segments, len(segments), total_pages
    except Exception as e:
        st.error(f"PDF处理失败：{str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return [], 0, 0

def save_knowledge_segments(segments, csv_path="knowledge_segments.csv"):
    df_new = pd.DataFrame(segments)
    
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(csv_path, index=False)
    return len(df_combined)

def find_similar_cases(new_req, df, top_k=3):
    if df.empty:
        return []
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['需求描述'].tolist() + [new_req])
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    top_indices = similarity.argsort()[0][-top_k:][::-1]
    return [df.iloc[i]['测试用例'] for i in top_indices]

def find_relevant_knowledge(query, knowledge_df, top_k=3):
    if knowledge_df.empty:
        return []
    
    vectorizer = TfidfVectorizer()
    documents = knowledge_df['content'].tolist()
    try:
        tfidf_matrix = vectorizer.fit_transform(documents + [query])
        similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        top_indices = similarity.argsort()[0][-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            segment = knowledge_df.iloc[idx]
            results.append({
                'document': segment['document_name'],
                'page': segment['page_num'],
                'content': segment['content']
            })
        return results
    except Exception as e:
        st.error(f"知识搜索失败：{str(e)}")
        return []

def generate_test_cases(prompt, history_cases=None, knowledge_segments=None, max_cases=10, temp=0.7, use_enhancement=True):
    headers = {"Authorization": "Bearer sk-xxxxxx", 
               "Content-Type": "application/json"}
    
    system_prompt = ""
    
    if use_enhancement and (history_cases or knowledge_segments):
        context = "\n".join([f"历史用例{idx+1}: {case}" for idx, case in enumerate(history_cases or [])]) if history_cases else ""
        
        knowledge_context = ""
        if knowledge_segments and len(knowledge_segments) > 0:
            knowledge_context = "参考知识：\n" + "\n\n".join([
                f"文档《{item['document']}》第{item['page']}页：{item['content']}"
                for item in knowledge_segments
            ])
        
        system_prompt = f"""你是一名测试老司机，请基于以下历史用例和知识库生成{max_cases}条新用例：
{context}

{knowledge_context}

要求：
1. 输出格式为JSON数组，每个对象包含字段：
   - 用例编号（格式TC-模块-序号，如TC-LOGIN-01）
   - 步骤（简明步骤描述）
   - 预期（预期结果）
   - 优先级（1-5，1为最高）
2. 包含正向和异常场景
3. 按优先级从高到低排序
4. 仅返回合法JSON，不要额外解释"""
    else:
        system_prompt = f"""你是一名测试老司机，请为以下需求生成{max_cases}条测试用例：

要求：
1. 输出格式为JSON数组，每个对象包含字段：
   - 用例编号（格式TC-模块-序号，如TC-LOGIN-01）
   - 步骤（简明步骤描述）
   - 预期（预期结果）
   - 优先级（1-5，1为最高）
2. 包含正向和异常场景
3. 按优先级从高到低排序
4. 仅返回合法JSON，不要额外解释"""
    
    try:
        response = requests.post(
            "https://api.lkeap.cloud.tencent.com/v1",
            headers=headers,
            json={
                "model": "deepseek-r1",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temp
            },
            verify=False
        )
        content = response.json()["choices"][0]["message"]["content"]
        print(content)
        return json.loads(repair_json(content))
    except Exception as e:
        st.error(f"AI罢工了：{str(e)}（检查API_KEY是不是充话费送的？）")
        return []

def apply_custom_styles():
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            position: relative;
        }
        
        h1, h2, h3 {
            color: #1E3A8A;
        }
        
        .css-card {
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .upload-section {
            border: 2px dashed #3B82F6;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
            background-color: #F3F4F6;
        }
        
        .info-box {
            background-color: #E0F2FE;
            padding: 10px 15px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        
        .success-box {
            background-color: #D1FAE5;
            padding: 10px 15px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""

    
    <div style="
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -2;
        pointer-events: none;
        background-image: repeating-linear-gradient(
            45deg,
            rgba(180, 180, 180, 0.05),
            rgba(180, 180, 180, 0.05) 150px,
            rgba(180, 180, 180, 0.1) 150px,
            rgba(180, 180, 180, 0.1) 300px
        );
    "></div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="🤖 AI测试小秘书", layout="wide")
    apply_custom_styles()
    
    st.title("💡 AI测试用例生成器（内置知识库增强版）")
    st.markdown("<p style='color:#4B5563;'>上传专业文档，设计出更专业的测试用例</p>", unsafe_allow_html=True)
    
    if 'knowledge_segments_count' not in st.session_state:
        st.session_state.knowledge_segments_count = 0
        
    knowledge_df = load_knowledge_segments()
    if not knowledge_df.empty:
        st.session_state.knowledge_segments_count = len(knowledge_df)
    
    test_cases_df = load_cases()
    
    tab1, tab2 = st.tabs(["📝 生成测试用例", "📚 知识库管理"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<h3>输入需求描述</h3>", unsafe_allow_html=True)
            
            with st.form("magic_form"):
                user_input = st.text_area("描述你的测试需求", height=150,
                                        placeholder="示例：购物车需支持添加商品、库存不足提示、批量删除")
                
                col_button1, col_button2 = st.columns([1, 3])
                with col_button1:
                    submitted = st.form_submit_button("✨ 生成测试用例", use_container_width=True)
                with col_button2:
                    use_knowledge = st.checkbox("使用知识库增强", value=True, help="勾选后将使用知识库和历史用例增强测试用例生成")
        
        with col2:
            st.markdown("<h3>知识库状态</h3>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-box">
                <p><b>📊 知识库统计</b></p>
                <p>• 文档段落：{st.session_state.knowledge_segments_count} 条</p>
                <p>• 历史用例：{len(test_cases_df)} 条</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("如何获得更好的结果？"):
                st.markdown("""
                1. **上传领域文档**：在"知识库管理"选项卡上传相关PDF
                2. **描述需求细节**：越详细的需求描述越能获得精准的测试用例
                3. **迭代优化**：基于生成结果，调整需求描述再次生成
                """)
    
        if submitted and user_input:
            if use_knowledge:
                with st.spinner("🔍 正在搜索相关知识..."):
                    similar_cases = find_similar_cases(user_input, test_cases_df)
                    relevant_knowledge = find_relevant_knowledge(user_input, knowledge_df)
                
                with st.spinner("🤖 AI正在生成增强测试用例..."):
                    new_cases = generate_test_cases(user_input, similar_cases, relevant_knowledge, use_enhancement=True)
                
                st.success("✅ 测试用例生成完成！(使用知识库增强)")
                
                if relevant_knowledge:
                    with st.expander("📑 参考的领域知识", expanded=True):
                        for i, segment in enumerate(relevant_knowledge):
                            st.markdown(f"""
                            <div style="margin-bottom: 10px; padding: 10px; border-left: 3px solid #3B82F6; background-color: #F3F4F6;">
                                <p><b>文档：</b>{segment['document']} (第{segment['page']}页)</p>
                                <p>{segment['content']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                if similar_cases:
                    with st.expander("👉 参考的历史用例", expanded=True):
                        st.json(similar_cases)
            else:
                with st.spinner("🤖 AI正在生成基础测试用例..."):
                    new_cases = generate_test_cases(user_input, use_enhancement=False)
                
                st.success("✅ 测试用例生成完成！(仅使用原始需求)")
            
            st.subheader("🎯 生成的测试用例")
            st.dataframe(pd.DataFrame(new_cases), use_container_width=True)
    
    with tab2:
        st.markdown("<h3>📤 上传知识文档</h3>", unsafe_allow_html=True)
        
        upload_col1, upload_col2 = st.columns([3, 1])
        
        with upload_col1:
            uploaded_file = st.file_uploader("上传PDF文档（将被分割并存入知识库）", type="pdf")
            
        with upload_col2:
            if uploaded_file is not None:
                if st.button("处理文档", type="primary", use_container_width=True):
                    with st.spinner("正在处理PDF文档..."):
                        segments, segment_count, page_count = process_pdf(uploaded_file)
                        if segments:
                            total_count = save_knowledge_segments(segments)
                            st.session_state.knowledge_segments_count = total_count
                            st.success(f"✅ 文档处理完成！从 {page_count} 页中提取了 {segment_count} 个知识段落。")
                            knowledge_df = load_knowledge_segments()
        
        if not knowledge_df.empty:
            st.markdown("<h3>📚 知识库内容</h3>", unsafe_allow_html=True)
            
            docs = knowledge_df['document_name'].unique()
            st.markdown(f"当前知识库包含 **{len(docs)}** 个文档，共 **{len(knowledge_df)}** 个知识段落。")
            
            selected_doc = st.selectbox("选择要查看的文档", ["所有文档"] + list(docs))
            
            if selected_doc == "所有文档":
                display_df = knowledge_df
            else:
                display_df = knowledge_df[knowledge_df['document_name'] == selected_doc]
            
            for _, row in display_df.head(20).iterrows():
                st.markdown(f"""
                <div style="margin-bottom: 10px; padding: 15px; border-radius: 8px; background-color: #F8FAFC; border: 1px solid #E2E8F0;">
                    <p style="margin:0; color: #6B7280; font-size: 0.8rem;">文档：{row['document_name']} | 第{row['page_num']}页</p>
                    <p style="margin-top: 8px;">{row['content']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            if len(display_df) > 20:
                st.info(f"仅显示前20条记录，共 {len(display_df)} 条")

if __name__ == "__main__":
    main()
