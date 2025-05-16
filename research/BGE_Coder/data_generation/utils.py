import re


def clean_content(content: str):
    if content is None:
        raise ValueError("content is None.")
    
    content = content.split('</think>')[-1].strip('\n').strip()
    
    if content.startswith('\"') and content.endswith('\"'):
        content = content[1:-1]
    
    if content.startswith("```\n") and content.endswith("\n```"):
        content = content[4:-4]
    
    return content


def clean_code(code: str, lang: str, length_threshold: int = 30) -> str:
    cleaned_code = code.strip('\ufeff').strip()
    if not cleaned_code:
        return ''

    def clean_empty_lines(text: str) -> str:
        return re.sub(r'\n\s*\n', '\n', text).strip()

    # 各语言函数/类定义检测正则表达式
    function_patterns = {
        "java": r"(?m)^(?!\s*(import|package)\b).*\b(public\s+class|class\s+\w+|void\s+main|new\s+\w+\(|@Override)\b",
        "python": r"(?m)^(?!\s*(import|from\s+\S+\s+import)\b).*\b(def\s+\w+|class\s+\w+|=\s*\S+|if\s+[:\w]|print\s+)",
        "javascript": r"(?m)^(?!\s*(import|require\(|export\s)).*\b(function\s+\w+|const\s+\w+|=>|\(\)\s*=>|console\.log)",
        "php": r"(?m)^(?!\s*(include|require|use)\b).*\b(function\s+\w+|echo\s+\S+|class\s+\w+)",
        "ruby": r"(?m)^(?!\s*(require|load)\b).*\b(class\s+\w+|def\s+\w+|puts\s+\S+)",
        "go": r"(?m)^(?!\s*import\b).*\bfunc\s+main\s*\(|type\s+\w+\s+struct",
        "c#": r"(?m)^(?!\s*using\b).*\b(class\s+\w+|void\s+Main\s*\()",
        "cplusplus": r"(?m)^(?!#include\b).*\b(int\s+main\s*\(|class\s+\w+|void\s+\w+\s*\(.*\)\s*{)",
        "c": r"(?m)^(?!#include\b).*\b(int\s+main\s*\(|void\s+\w+\s*\(.*\)\s*{)",
        "rust": r"(?m)^(?!\s*use\b).*\b(fn\s+main\s*\(|struct\s+\w+|impl\s+\w+)",
        "typescript": r"(?m)^(?!\s*(import|require\(|export\s)).*\b(interface\s+\w+|class\s+\w+|function\s+\w+)",
        "perl": r"(?m)^(?!\s*(use|require)\b).*\b(sub\s+\w+|my\s+\$\w+|print\s+\S+)",
        "shell": r"(?m)^(?!\s*(source|\.)\s).*\b(function\s+\w+|if\s+\[|\$\(|echo\s+\S+)",
        "sql": r"(?i)\b(CREATE\s+TABLE|SELECT\s+\*|INSERT\s+INTO|UPDATE\s+\w+|DELETE\s+FROM)\b",
        "batchfile": r"(?m)^(?!\s*@?call\b).*\b(echo\s+\S+|set\s+\w+|if\s+.*\s+==\s+)",
        "fortran": r"(?mi)^(?!\s*use\b).*\b(program\s+\w+|subroutine\s+\w+|do\s+\d+\s*,\s*\d+)",
        "haskell": r"(?m)^(?!\s*import\b).*\b(main\s*=\s*do|data\s+\w+|putStrLn\s+\S+)",
        "lua": r"(?m)^(?!\s*require\b).*\b(function\s+\w+|local\s+\w+|print\s*\()",
        "powershell": r"(?m)^(?!\s*Import-Module\b).*\b(function\s+\w+|Write-Host\s+\S+|\$\w+\s*=)",
        "visual_basic": r"(?m)^(?!\s*Imports\b).*\b(Module\s+\w+|Sub\s+Main|Class\s+\w+)"
    }

    # 各语言注释处理规则
    comment_patterns = {
        'java': (r'//.*?$|/\*.*?\*/|\\/\\/.*?$|\\/\*.*?\*\\/', re.DOTALL | re.MULTILINE),
        'python': (r'#.*?$', re.MULTILINE),
        'javascript': (r'//.*?$|/\*.*?\*/|\\/\\/.*?$|\\/\*.*?\*\\/', re.DOTALL | re.MULTILINE),
        'php': (r'//.*?$|#.*?$|/\*.*?\*/|\\/\\/.*?$|#.*?$|\\/\*.*?\*\\/', re.DOTALL | re.MULTILINE),
        'ruby': (r'#.*', re.MULTILINE),
        'go': (r'//.*?$|/\*.*?\*/|\\/\\/.*?$|\\/\*.*?\*\\/', re.DOTALL | re.MULTILINE),
        'csharp': (r'//.*?$|/\*.*?\*/|\\/\\/.*?$|\\/\*.*?\*\\/', re.DOTALL | re.MULTILINE),
        'cplusplus': (r'//.*?$|/\*.*?\*/|\\/\\/.*?$|\\/\*.*?\*\\/', re.DOTALL | re.MULTILINE),
        'c': (r'//.*?$|/\*.*?\*/|\\/\\/.*?$|\\/\*.*?\*\\/', re.DOTALL | re.MULTILINE),
        'rust': (r'//.*?$|/\*.*?\*/|\\/\\/.*?$|\\/\*.*?\*\\/', re.DOTALL | re.MULTILINE),
        'typescript': (r'//.*?$|/\*.*?\*/|\\/\\/.*?$|\\/\*.*?\*\\/', re.DOTALL | re.MULTILINE),
        'perl': (r'#.*', re.MULTILINE),
        'shell': (r'#.*', re.MULTILINE),
        'sql': (r'--.*?$|/\*.*?\*/', re.DOTALL),
        'batchfile': (r'^\s*(REM|@REM|::).*', re.MULTILINE | re.IGNORECASE),
        'fortran': (r'!.*', re.MULTILINE),
        'haskell': (r'--.*', re.MULTILINE),
        'lua': (r'--.*?$|--\[\[.*?\]\]', re.DOTALL),
        'powershell': (r'<#.*?#>|#.*', re.DOTALL),
        'visual_basic': (r"'.*", re.MULTILINE),
    }

    # 执行注释清理
    if lang in comment_patterns:
        pattern, flags = comment_patterns[lang]
        cleaned_code = re.sub(pattern, '', cleaned_code, flags=flags)
        cleaned_code = clean_empty_lines(cleaned_code)

    # 特殊语言处理规则
    if lang == 'fortran':
        cleaned_code = re.sub(r'^[Cc*].*', '', cleaned_code, flags=re.MULTILINE)
    elif lang == 'sql':
        cleaned_code = re.sub(r'/\*.*?\*/', '', cleaned_code, flags=re.DOTALL)
    elif lang == 'python':
        cleaned_code = re.sub(r'^\s*#.*', '', cleaned_code, flags=re.MULTILINE)

    # 函数定义检测及内容验证
    def has_valid_code(text: str, lang: str) -> bool:
        pattern = function_patterns.get(lang)
        if not pattern:
            return len(text.strip()) > 0
        
        # 增强检测逻辑
        if lang == 'batchfile':
            return bool(re.search(r'^\s*@?echo\b|:\w+', text, re.MULTILINE))
        if lang == 'shell':
            return bool(re.search(r'^\s*(if|for|while|case|echo|export|shopt|source)\b', text, re.MULTILINE))
        if lang == 'python':
            if re.search(r'^\s*(def|class)\s+\w+', text, re.MULTILINE):
                return bool(re.search(r'^\s+[^\s#]', text, re.MULTILINE))
            return False
        if lang == 'ruby':
            return bool(re.search(r'(def\s+\w+|class\s+\w+).*?\n\s+[^\s#]', text, re.MULTILINE))
        return bool(re.search(pattern, text, re.DOTALL | re.MULTILINE))

    # 最终有效性检查
    if not has_valid_code(cleaned_code, lang):
        return ''
    
    cleaned_code = cleaned_code.strip('\ufeff').strip()
    
    if len(cleaned_code) < length_threshold:
        return ''

    return cleaned_code


if __name__ == "__main__":
    test_text = "\/\/ ----------------------------------------------------------------------\n\/\/ ----------------------------------------------------------------------\n\/\/\n\/\/ File:      StrMaxProjection.h\n\/\/ Author:    mgrosso \n\/\/ Created:   Mon Jul 17 14:39:22 PDT 2006 on caliban\n\/\/ Project:   \n\/\/ Purpose:   \n\/\/ \n\/\/ $Id$\n\/\/ ----------------------------------------------------------------------\n\/\/ ----------------------------------------------------------------------\n\n#ifndef STRMAXPROJECTION_H\n#define STRMAXPROJECTION_H 1\n\n#include \"StrMinProjection.h\"\n\nclass StrMaxProjection : public StrMinProjection\n{\n    public:\n    StrMaxProjection(ExpressionPtr &operand);\n    virtual ~StrMaxProjection();\n    virtual     AbstractProjectionPtr        copy();\n\n    protected:\n    int compare(const char *lhs, const char *rhs);\n\n    private:\n    \/\/not implemented\n    StrMaxProjection();\n    StrMaxProjection( const StrMaxProjection &rhs );\n    StrMaxProjection &operator=( const StrMaxProjection &rhs );\n};\n\n#endif \/* STRMAXPROJECTION_H *\/"
    
    result = clean_code(test_text, "c", 200)
    print(result)
    
    test_text = "\/**\n * Copyright (c) Microsoft Corporation. All rights reserved.\n * Licensed under the MIT License. See License.txt in the project root for\n * license information.\n *\n * Code generated by Microsoft (R) AutoRest Code Generator.\n *\/\n\npackage com.microsoft.azure.management.datafactory.v2018_06_01;\n\nimport com.fasterxml.jackson.annotation.JsonProperty;\nimport com.fasterxml.jackson.annotation.JsonTypeInfo;\nimport com.fasterxml.jackson.annotation.JsonTypeName;\n\n\/**\n * The location of Google Cloud Storage dataset.\n *\/\n@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, include = JsonTypeInfo.As.PROPERTY, property = \"type\", defaultImpl = GoogleCloudStorageLocation.class)\n@JsonTypeName(\"GoogleCloudStorageLocation\")\npublic class GoogleCloudStorageLocation extends DatasetLocation {\n    \/**\n     * Specify the bucketName of Google Cloud Storage. Type: string (or\n     * Expression with resultType string).\n     *\/\n    @JsonProperty(value = \"bucketName\")\n    private Object bucketName;\n\n    \/**\n     * Specify the version of Google Cloud Storage. Type: string (or Expression\n     * with resultType string).\n     *\/\n    @JsonProperty(value = \"version\")\n    private Object version;\n\n    \/**\n     * Get specify the bucketName of Google Cloud Storage. Type: string (or Expression with resultType string).\n     *\n     * @return the bucketName value\n     *\/\n    public Object bucketName() {\n        return this.bucketName;\n    }\n\n    \/**\n     * Set specify the bucketName of Google Cloud Storage. Type: string (or Expression with resultType string).\n     *\n     * @param bucketName the bucketName value to set\n     * @return the GoogleCloudStorageLocation object itself.\n     *\/\n    public GoogleCloudStorageLocation withBucketName(Object bucketName) {\n        this.bucketName = bucketName;\n        return this;\n    }\n\n    \/**\n     * Get specify the version of Google Cloud Storage. Type: string (or Expression with resultType string).\n     *\n     * @return the version value\n     *\/\n    public Object version() {\n        return this.version;\n    }\n\n    \/**\n     * Set specify the version of Google Cloud Storage. Type: string (or Expression with resultType string).\n     *\n     * @param version the version value to set\n     * @return the GoogleCloudStorageLocation object itself.\n     *\/\n    public GoogleCloudStorageLocation withVersion(Object version) {\n        this.version = version;\n        return this;\n    }\n\n}"
    result = clean_code(test_text, "java", 200)
    print(result)
