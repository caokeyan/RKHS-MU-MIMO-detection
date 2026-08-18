# 上传 ShareLaTeX / Overleaf

> ShareLaTeX 已并入 Overleaf，界面相同。

## 步骤

1. 打开 [https://www.overleaf.com](https://www.overleaf.com)（或学校 ShareLaTeX 站点）并登录  
2. **New Project → Upload Project**  
3. 选择本仓库中的 `report-sharelatex.zip`  
   （路径：项目根目录 `/RKHS优化问题/report-sharelatex.zip`）  
4. 打开项目后：
   - 菜单 **Menu**（左上）
   - **Main document**：`main.tex`
   - **Compiler**：**XeLaTeX**（不要用 pdfLaTeX）
5. 点 **Recompile**；建议再编译一次以生成目录

## 若学校仍是旧版 ShareLaTeX

同样：**New Project → Upload Project** → 选 zip → Settings 里把 compiler 设为 **XeLaTeX**。

## 注意

- 必须用 **XeLaTeX**，否则中文 `ctex` 无法编译  
- 图片已在 zip 内 `figures/` 目录，无需再改路径  
- 编译两次后目录页码才完整  
