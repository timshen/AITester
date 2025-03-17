# 基于DeepSeek等LLM模型的测试用例生成器（知识库增强版）

一个LLM驱动的工具，可基于需求描述生成全面的测试用例，通过RAG（检索增强生成）技术结合领域特定知识和历史测试用例进行增强。

![AI测试用例生成器](https://img.shields.io/badge/AI-测试用例生成器-blue)

## 功能特点

- 📝 从自然语言需求描述生成测试用例
- 📚 上传PDF文档构建领域特定知识库
- 🔍 自动检索并利用相关知识来提升测试用例质量
- 💾 从历史测试用例中学习，改进未来生成效果
- 🧠 支持基于知识增强和基础测试用例生成
- 📊 基于Streamlit的简单直观用户界面

## 安装方法

1. 克隆代码仓库:
   ```bash
   git clone <仓库地址>
   cd testcasegen
   ```

2. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

3. 在脚本(`rag_test_gen.py`)中配置AI服务API密钥或使用环境变量。

## 使用方法

1. 启动应用:
   ```bash
   streamlit run rag_test_gen.py
   ```

2. 导航到"知识库管理"选项卡，上传领域特定的PDF文档。

3. 进入"生成测试用例"选项卡，输入您的需求描述。

4. 选择是否使用知识增强，然后点击"生成测试用例"。

5. 查看并导出生成的测试用例。

## 依赖项

- streamlit
- pandas
- sklearn
- PyPDF2
- json_repair
- requests
- uuid

## 知识库增强

该工具使用RAG（检索增强生成）技术来增强测试用例生成:

1. **上传PDF文档**: 系统从PDF文件中提取并分段内容
2. **TF-IDF相似度**: 生成测试用例时，检索最相关的知识片段
3. **上下文注入**: 将检索到的知识和类似的历史测试用例注入AI提示中
4. **增强生成**: AI利用丰富的上下文生成更具领域感知的测试用例

## 示例

对于登录模块需求:

```
需要实现一个登录系统，具有用户名/密码认证、
密码恢复、连续3次失败后账户锁定、
以及记住我功能。
```

该工具将生成涵盖正常和边缘情况的结构化测试用例，
每个用例都包含步骤、预期结果和优先级。

## 数据存储

- 历史测试用例存储在`test_cases.csv`中
- 上传文档的知识片段存储在`knowledge_segments.csv`中

## 许可证

[MIT许可证](LICENSE)

# AI Test Case Generator with Knowledge Enhancement

An AI-powered tool for generating comprehensive test cases from requirement descriptions, enhanced with domain-specific knowledge and historical test cases using RAG (Retrieval-Augmented Generation).

![AI Test Case Generator](https://img.shields.io/badge/AI-Test%20Case%20Generator-blue)

## Features

- 📝 Generate test cases from natural language requirement descriptions
- 📚 Upload PDF documents to build a domain-specific knowledge base
- 🔍 Automatically retrieve and leverage relevant knowledge to enhance test case quality
- 💾 Learn from historical test cases to improve future generations
- 🧠 Support for both knowledge-enhanced and basic test case generation
- 📊 Simple and intuitive user interface with Streamlit

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd testcasegen
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your AI service API key in the script (`rag_test_gen.py`) or use environment variables.

## Usage

1. Start the application:
   ```bash
   streamlit run rag_test_gen.py
   ```

2. Navigate to the "Knowledge Base Management" tab to upload domain-specific PDF documents.

3. Go to the "Generate Test Cases" tab to enter your requirement description.

4. Choose whether to use knowledge enhancement and click "Generate Test Cases".

5. Review and export the generated test cases.

## Dependencies

- streamlit
- pandas
- sklearn
- PyPDF2
- json_repair
- requests
- uuid

## Knowledge Base Enhancement

This tool uses RAG (Retrieval-Augmented Generation) to enhance test case generation:

1. **Upload PDF documents**: The system extracts and segments content from PDF files
2. **TF-IDF similarity**: When generating test cases, the most relevant knowledge segments are retrieved
3. **Context injection**: The retrieved knowledge and similar historical test cases are injected into the AI prompt
4. **Enhanced generation**: AI generates more domain-aware test cases with the enriched context

## Example

For a login module requirement:

```
Need to implement a login system with username/password authentication, 
password recovery, account lockout after 3 failed attempts, 
and remember-me functionality.
```

The tool will generate structured test cases covering normal and edge cases, 
each with steps, expected results, and priority levels.

## Data Storage

- Historical test cases are stored in `test_cases.csv`
- Knowledge segments from uploaded documents are stored in `knowledge_segments.csv`

## License

[MIT License](LICENSE)

---

