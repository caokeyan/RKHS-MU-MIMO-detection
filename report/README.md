# 中期/研究报告（LaTeX）

## ShareLaTeX / Overleaf（推荐）

1. 上传项目根目录的 **`report-sharelatex.zip`**（New Project → Upload Project）  
2. Main document：`main.tex`  
3. Compiler：**XeLaTeX**  
4. Recompile 两次  

详见 `SHARELATEX上传说明.md`。

## 本地编译

```bash
cd report
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

## 文件

| 文件 | 内容 |
|------|------|
| `main.tex` | 导言、摘要、目录、参考文献 |
| `ch01`–`ch09` | 各章正文 |
| `figures/` | 中期实验图 |
