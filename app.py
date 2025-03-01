import os
import json
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

def generate_test_cases(prompt, max_cases, temp):
    headers = {
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    system_prompt = f"""作为资深测试工程师，请生成{max_cases}条测试用例，严格遵循以下要求：
1. 输出格式为JSON数组，每个对象包含字段：
   - 用例编号（格式TC-模块-序号，如TC-LOGIN-01）
   - 步骤（简明步骤描述）
   - 预期（预期结果）
   - 优先级（1-5，1为最高）
2. 包含正向和异常场景
3. 按优先级从高到低排序"""

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
                "temperature": temp,
                "response_format": {"type": "json_object"}  # 要求返回JSON格式
            },
            verify=False
        )
        response.raise_for_status()
        
        return parse_response(response.json())
    except Exception as e:
        st.error(f"API调用失败: {str(e)}")
        return []

def parse_response(response_data):
    """解析API返回的JSON数据"""
    try:
        # 提取并验证JSON结构
        content = response_data["choices"][0]["message"]["content"]
        cases = json.loads(content)
        
        # 类型检查
        if not isinstance(cases, list):
            raise ValueError("响应不是JSON数组")
            
        # 字段验证
        required_fields = ["用例编号", "步骤", "预期", "优先级"]
        for idx, case in enumerate(cases):
            if not all(field in case for field in required_fields):
                raise ValueError(f"第{idx+1}条用例字段缺失")
            if not case["用例编号"].startswith("TC-"):
                raise ValueError(f"编号格式错误: {case['用例编号']}")
                
        return cases
    except json.JSONDecodeError:
        st.error("响应不是有效的JSON格式")
        return []
    except Exception as e:
        st.error(f"解析失败: {str(e)}")
        return []

def display_results(cases):
    """结果展示组件"""
    if not cases:
        return
        
    tab1, tab2, tab3 = st.tabs(["表格视图", "JSON数据", "统计信息"])
    
    with tab1:
        df = pd.DataFrame(cases)
        df["优先级"] = df["优先级"].astype("category")
        st.dataframe(df, use_container_width=True)
        
    with tab2:
        st.code(json.dumps(cases, indent=2, ensure_ascii=False))
        
    with tab3:
        st.subheader("用例分布")
        priority_dist = df["优先级"].value_counts().sort_index()
        st.bar_chart(priority_dist)

def main():
    """主界面"""
    st.set_page_config(page_title="DeepSeek测试用例生成器", layout="wide")
    
    # 侧边栏设置
    with st.sidebar:
        st.header("⚙️ 参数设置")
        max_cases = st.slider("生成数量", 5, 30, 10, 
                            help="建议优先生成核心用例，再补充扩展用例")
        temperature = st.slider("生成温度", 0.1, 1.0, 0.7,
                              help="值越高生成结果越多样，但可能降低准确性")
    
    # 主界面
    st.title("🧪 智能测试用例生成系统")
    
    # 需求输入
    with st.form(key="main_form"):
        user_input = st.text_area("需求描述", height=150,
                                placeholder="示例：用户登录功能需验证：\n1. 正常登录流程\n2. 密码错误处理\n3. 账户锁定机制")
        submitted = st.form_submit_button("生成用例")
    
    # 结果生成
    if submitted and user_input:
        with st.spinner("🔄 正在生成测试用例..."):
            test_cases = generate_test_cases(user_input, max_cases, temperature)
            if test_cases:
                st.success(f"✅ 成功生成 {len(test_cases)} 条测试用例！")
                display_results(test_cases)
            else:
                st.warning("⚠️ 未生成有效测试用例，请尝试调整输入描述")

if __name__ == "__main__":
    main()
