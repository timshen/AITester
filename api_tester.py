import streamlit as st
import requests
import json
import time
from typing import List, Dict
from json_repair import repair_json
import os
# 配置DeepSeek-R1 API参数
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_API_URL = "https://api.lkeap.cloud.tencent.com/v1"
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
}

class DeepSeekTestGenerator:
    """基于DeepSeek-R1的测试用例生成引擎"""
    def __init__(self):
        self.system_prompt = """作为API测试专家，请按以下要求生成测试套件：
1. 包含正常/边界/异常场景
2. 使用JSON Path验证响应
3. 包含性能断言（响应时间<800ms）
4. 输出OpenAPI 3.0规范

响应格式：
```json
{
    "openapi": "3.0.0",
    "test_cases": [
        {
            "name": "测试名称",
            "method": "HTTP方法",
            "path": "/api/path",
            "params": {},
            "body": {},
            "assertions": [
                {"type": "status_code", "expect": 200},
                {"type": "json_path", "path": "$.data.id", "expect": "exists"}
            ]
        }
    ]
}
```"""

    def generate_tests(self, api_desc: str) -> dict:
        """调用DeepSeek-R1生成测试用例"""
        payload = {
            "model": "deepseek-r1",
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": api_desc}
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=DEFAULT_HEADERS,
            json=payload,
            verify=False
        )
        print(response.json())
        return json.loads(repair_json(response.json()["choices"][0]["message"]["content"]))

class TestExecutor:
    """支持强化学习的测试执行引擎"""
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.results = []

    def execute_test(self, test_case: dict) -> dict:
        """执行测试并记录强化学习反馈"""
        result = {
            "name": test_case["name"],
            "status": "pending",
            "metrics": {}
        }
        
        try:
            # 请求执行
            start_time = time.time()
            response = requests.request(
                method=test_case["method"],
                url=f"{self.base_url}{test_case['path']}",
                params=test_case.get("params"),
                json=test_case.get("body"),
                headers=DEFAULT_HEADERS,
                verify=False
            )
            response_time = (time.time() - start_time) * 1000

            # 动态断言执行
            passed_assertions = []
            for assertion in test_case["assertions"]:
                if assertion["type"] == "status_code":
                    passed = response.status_code == assertion["expect"]
                elif assertion["type"] == "json_path":
                    passed = self._validate_json_path(response.json(), assertion)
                
                passed_assertions.append({
                    "type": assertion["type"],
                    "passed": passed,
                    "expected": assertion.get("expect"),
                    "actual": response.status_code if assertion["type"] == "status_code" else None
                })

            # 强化学习反馈
            result["status"] = "passed" if all(a["passed"] for a in passed_assertions) else "failed"
            result["metrics"] = {
                "response_time": response_time,
                "assertions": passed_assertions,
                "response_sample": response.json() if "application/json" in response.headers.get("Content-Type","") else response.text
            }

        except Exception as e:
            result["status"] = "error"
            result["metrics"] = {"error": str(e)}

        return result

    def _validate_json_path(self, response, assertion):
        """JSON Path验证实现（示例）"""
        # 此处可集成jsonpath-ng库实现完整验证
        return True if response else False

# Streamlit界面
def main():
    st.set_page_config(page_title="DeepSeek API测试平台", layout="wide")
    
    # 初始化会话状态
    if "test_suite" not in st.session_state:
        st.session_state.test_suite = {}
    if "execution_results" not in st.session_state:
        st.session_state.execution_results = []

    # 侧边栏配置
    with st.sidebar:
        st.header("配置参数")
        base_url = st.text_input("API入口地址", "https://api.example.com")
        api_desc = st.text_area("API描述文档", height=250, 
                               placeholder="输入OpenAPI文档或自然语言描述...")
        
        if st.button("生成测试套件", type="primary"):
            generator = DeepSeekTestGenerator()
            st.session_state.test_suite = generator.generate_tests(api_desc)
            st.session_state.execution_results = []
            st.rerun()

    # 主界面布局
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("测试规范预览")
        if st.session_state.test_suite:
            with st.expander("OpenAPI 3.0规范"):
                st.json(st.session_state.test_suite["openapi"])
            
            st.subheader("生成用例列表")
            for idx, tc in enumerate(st.session_state.test_suite["test_cases"]):
                st.markdown(f"**{idx+1}. {tc['name']}**")
                st.caption(f"`{tc['method']} {tc['path']}`")

    with col2:
        st.header("测试执行控制台")
        if st.button("执行全部测试"):
            executor = TestExecutor(base_url)
            progress_bar = st.progress(0)
            
            for idx, tc in enumerate(st.session_state.test_suite["test_cases"]):
                result = executor.execute_test(tc)
                st.session_state.execution_results.append(result)
                progress_bar.progress((idx+1)/len(st.session_state.test_suite["test_cases"]))
            
            st.rerun()
        
        if st.session_state.execution_results:
            st.subheader("执行结果分析")
            for result in st.session_state.execution_results:
                status_color = "green" if result["status"] == "passed" else "red"
                st.markdown(
                    f"<span style='color:{status_color};font-weight:bold'>[{result['status'].upper()}]</span> {result['name']}",
                    unsafe_allow_html=True
                )
                
                with st.expander("查看详情"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("响应时间", f"{result['metrics']['response_time']:.2f}ms")
                        st.write("### 断言结果")
                        for assertion in result["metrics"]["assertions"]:
                            icon = "✅" if assertion["passed"] else "❌"
                            st.write(f"{icon} {assertion['type']}")
                    with col2:
                        st.json(result["metrics"]["response_sample"])

if __name__ == "__main__":
    main()
